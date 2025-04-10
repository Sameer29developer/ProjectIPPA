import requests
import io
import base64
import numpy as np
from PIL import Image
import time
import json
import logging
from PIL import Image, ImageEnhance, UnidentifiedImageError
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stable_diffusion')

def generate_image_with_local_sd(positive_prompt, negative_prompt, api_url="http://127.0.0.1:7860", 
                               model_name=None, width=768, height=768, 
                               steps=30, cfg_scale=8.5, sampler="DPM++ 2M Karras"):
    """Generate an image using a locally running Stable Diffusion API
    
    Args:
        positive_prompt (str): The positive prompt to generate an image from
        negative_prompt (str): The negative prompt (what to avoid in the image)
        api_url (str): URL of the local Stable Diffusion API
        model_name (str, optional): Model name to use. If None, uses current model.
        width (int): Output image width (768 works well for XL models)
        height (int): Output image height (768 works well for XL models)
        steps (int): Number of sampling steps
        cfg_scale (float): CFG scale (how strictly to follow the prompt)
        sampler (str): Sampler to use
        
    Returns:
        numpy.ndarray: The generated image as a numpy array, or None if generation failed
    """
    
    logger.info(f"Starting image generation for positive prompt: {positive_prompt[:50]}...")
    
    # Enhance prompt based on model type if needed
    enhanced_prompt = positive_prompt
    
    # Check if we're using a pony-specific model
    is_pony_model = False
    if model_name and ("pony" in model_name.lower()):
        is_pony_model = True
        
        # Add pony-specific terms if not already in the prompt for pony models
        pony_terms = ["pony style", "pony art", "illustration", "detailed", "high quality"]
        
        # Check if any pony terms are already in the prompt
        has_pony_term = any(term.lower() in positive_prompt.lower() for term in pony_terms)
        
        # Append pony terms if none are present
        if not has_pony_term:
            enhanced_prompt = f"{positive_prompt}, pony style, detailed illustration, high quality"
    
    # Check if we're using an XL/SDXL model
    is_xl_model = False
    if model_name and ("xl" in model_name.lower() or "XL" in model_name):
        is_xl_model = True
        
        # Add XL-specific quality boosters if not already present
        xl_boosters = ["high quality", "detailed", "best quality", "masterpiece"]
        
        # Check if any quality boosters are already in the prompt
        has_xl_booster = any(booster.lower() in enhanced_prompt.lower() for booster in xl_boosters)
        
        # Append quality boosters if none are present
        if not has_xl_booster:
            enhanced_prompt = f"{enhanced_prompt}, high quality, detailed, best quality"
    
    logger.info(f"Model type: {'Pony' if is_pony_model else ''} {'XL' if is_xl_model else ''}")
    logger.info(f"Enhanced prompt: {enhanced_prompt}")
    
    try:
        # Step 1: Check connection
        try:
            logger.info(f"Testing connection to {api_url}")
            test_response = requests.get(f"{api_url}/sdapi/v1/sd-models", timeout=5)
            if test_response.status_code != 200:
                logger.error(f"API connection test failed: {test_response.status_code}")
                return None
            else:
                logger.info("Connection successful")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Stable Diffusion API: {e}")
            return None
        
        # Step 2: Switch to the specified model if provided
        if model_name:
            logger.info(f"Setting model to: {model_name}")
            model_payload = {"sd_model_checkpoint": model_name}
            
            try:
                model_response = requests.post(
                    url=f"{api_url}/sdapi/v1/options", 
                    json=model_payload,
                    timeout=60  # Model loading can take time
                )
                
                if model_response.status_code != 200:
                    logger.warning(f"Error switching models: {model_response.text}")
                    # Continue anyway with current model
                else:
                    logger.info("Model set successfully")
                    # Give the server a moment to load the model
                    time.sleep(3)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error during model switch: {e}")
                # Continue anyway with current model
        
        # Step 3: Generate the image with optimized parameters based on model type
        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler,  # Try different parameter name for compatibility
            "sampler_index": sampler,  # Also include this for older versions
            "seed": -1,  # Random seed each time
            "batch_size": 1,
            "n_iter": 1,
            "enable_hr": False,  # Disable high-res fix to avoid potential issues
            "do_not_save_samples": True,
            "do_not_save_grid": True
        }
        
        # Add model-specific optimizations
        if is_xl_model:
            # SDXL-specific parameters
            if not any(dim >= 768 for dim in [width, height]):
                logger.info("Adjusting payload for XL model - XL models work better at higher resolutions")
                # Don't actually change the resolution, but log that this would be recommended
            
            # Some SDXL models need a slightly different parameter set
            payload.update({
                "cfg_scale": max(cfg_scale, 7.0),  # Ensure CFG scale is high enough for XL
            })
        
        logger.info(f"Sending request with payload: {json.dumps(payload)}")
        
        # Make API request for image generation with a longer timeout
        response = requests.post(
            url=f"{api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=300  # Longer timeout for XL models - 5 minutes
        )
        
        logger.info(f"Received response with status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Error generating image: {response.status_code} - {response.text}")
            return None
            
        # Parse response
        r = response.json()
        
        # Log response keys for debugging
        logger.info(f"Response keys: {list(r.keys())}")
        
        if 'images' not in r or not r['images']:
            logger.error("No images in response")
            return None
            
        # Check if images array is empty or contains empty strings
        if not r['images'][0]:
            logger.error("Empty image data returned")
            return None
            
        logger.info(f"Image data length: {len(r['images'][0])}")
        
        try:
            # Decode the base64 image with improved error handling
            image_data = base64.b64decode(r['images'][0])
            logger.info(f"Decoded image data length: {len(image_data)}")
            
            # Save raw image data for inspection
            with open("debug_image.jpg", "wb") as f:
                f.write(image_data)
            
            # Use PIL to decode the image properly
            try:
                image = Image.open(io.BytesIO(image_data))
                logger.info(f"Image opened successfully: {image.size}, {image.mode}")
                
                # Convert to numpy array for OpenCV compatibility
                img_array = np.array(image)
                
                # Check if the image is blank/dark - using more lenient threshold
                if img_array.mean() < 5:  # Very, very dark
                    logger.warning(f"Image appears to be too dark: mean pixel value = {img_array.mean()}")
                    # Try to enhance contrast to see if there's hidden content
                    enhanced_img = Image.fromarray(img_array)
                    enhanced_img = ImageEnhance.Contrast(enhanced_img).enhance(2.0)
                    enhanced_img = ImageEnhance.Brightness(enhanced_img).enhance(1.5)
                    img_array = np.array(enhanced_img)
                    logger.info(f"After enhancement: mean pixel value = {img_array.mean()}")
                
                return img_array
                
            except UnidentifiedImageError:
                logger.error("PIL could not identify the image format")
                # Try alternative approach using OpenCV
                try:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        logger.error("OpenCV could not decode the image")
                        return None
                    # Convert from BGR to RGB (OpenCV uses BGR by default)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img_rgb
                except Exception as cv_error:
                    logger.error(f"OpenCV decoding failed: {cv_error}")
                    return None
                
        except Exception as decode_error:
            logger.error(f"Error decoding image: {decode_error}")
            return None
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        # Return None to signal fallback to placeholder
        return None

def get_available_models(api_url="http://127.0.0.1:7860"):
    """Get list of available models from SD API
    
    Args:
        api_url (str): URL of the local Stable Diffusion API
        
    Returns:
        list: List of available models, or empty list if request failed
    """
    try:
        response = requests.get(f"{api_url}/sdapi/v1/sd-models", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get models: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Error fetching models: {e}")
        return []

def get_samplers(api_url="http://127.0.0.1:7860"):
    """Get list of available samplers from SD API
    
    Args:
        api_url (str): URL of the local Stable Diffusion API
        
    Returns:
        list: List of available samplers, or default list if request failed
    """
    default_samplers = ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", 
                        "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M Karras"]
    
    try:
        response = requests.get(f"{api_url}/sdapi/v1/samplers", timeout=5)
        if response.status_code == 200:
            samplers = [sampler["name"] for sampler in response.json()]
            logger.info(f"Available samplers: {samplers}")
            return samplers
        else:
            return default_samplers
    except Exception as e:
        logger.warning(f"Error fetching samplers: {e}")
        return default_samplers

def check_api_connection(api_url="http://127.0.0.1:7860"):
    """Check if the Stable Diffusion API is available
    
    Args:
        api_url (str): URL of the local Stable Diffusion API
        
    Returns:
        tuple: (bool of connection status, dict of API info or error message)
    """
    try:
        # Try to get a quick response from the API
        response = requests.get(f"{api_url}/sdapi/v1/progress", timeout=5)
        
        if response.status_code == 200:
            # Get information about current API configuration
            try:
                config_response = requests.get(f"{api_url}/sdapi/v1/options", timeout=5)
                config = config_response.json() if config_response.status_code == 200 else {}
                
                # Log current configuration for debugging
                current_model = config.get('sd_model_checkpoint', 'unknown')
                logger.info(f"Current model: {current_model}")
                
            except Exception as config_error:
                logger.warning(f"Error getting configuration: {config_error}")
                config = {}
            
            # If successful, get more detailed information
            info = {
                "status": "connected",
                "models": len(get_available_models(api_url)),
                "samplers": len(get_samplers(api_url)),
                "url": api_url,
                "current_model": config.get('sd_model_checkpoint', 'unknown')
            }
            return True, info
        else:
            return False, {"error": f"API returned status code {response.status_code}"}
    
    except requests.exceptions.ConnectionError:
        return False, {"error": "Connection refused. Make sure Stable Diffusion is running."}
    except requests.exceptions.Timeout:
        return False, {"error": "Connection timed out. Server might be busy or not responding."}
    except Exception as e:
        return False, {"error": f"Unexpected error: {str(e)}"}

def test_generation(api_url="http://127.0.0.1:7860"):
    """Test image generation with a simple prompt
    
    Args:
        api_url (str): URL of the local Stable Diffusion API
        
    Returns:
        bool: True if test was successful, False otherwise
    """
    logger.info("Running test generation")
    
    # Very simple payload for testing
    payload = {
        "prompt": "test image",
        "steps": 10,  # Low steps for quick test
        "width": 256,  # Small image for quick test
        "height": 256,
        "cfg_scale": 7
    }
    
    try:
        # Make API request
        response = requests.post(
            url=f"{api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Test generation failed: {response.status_code} - {response.text}")
            return False
            
        # Parse response
        r = response.json()
        
        if 'images' not in r or not r['images']:
            logger.error("No images in test response")
            return False
            
        # Try to decode the image
        try:
            image_data = base64.b64decode(r['images'][0])
            Image.open(io.BytesIO(image_data))
            logger.info("Test generation successful")
            return True
        except Exception as decode_error:
            logger.error(f"Error decoding test image: {decode_error}")
            return False
            
    except Exception as e:
        logger.error(f"Error during test generation: {e}")
        return False