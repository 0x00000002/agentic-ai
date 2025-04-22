import os
import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# --- Replicate Configuration ---
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_API_HOST = "https://api.replicate.com"
REPLICATE_API_VERSION = "v1"

# Model details (Using the correct SD 3.5 Medium model name)
REPLICATE_MODEL_OWNER = "stability-ai"
REPLICATE_MODEL_NAME = "stable-diffusion-3.5-medium"

# DEFAULT_SD_MODEL_VERSION is no longer needed as we use the latest version via the model endpoint

# Polling settings
POLLING_INTERVAL_SECONDS = 2
MAX_POLLING_ATTEMPTS = 150 # e.g., 150 * 2s = 300s = 5 minutes max wait

async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    # width/height are kept for compatibility, but SD 3.5 prefers aspect_ratio via kwargs
    width: int = 1024, # Default width (might be ignored if aspect_ratio is provided)
    height: int = 1024, # Default height (might be ignored if aspect_ratio is provided)
    num_inference_steps: int = 40, # Default steps matching SD 3.5 example
    # model_version parameter was removed
    **kwargs: Any # Use kwargs for model-specific params like aspect_ratio, output_format, cfg
) -> Dict[str, Any]:
    """
    Generates an image using the latest version of Stability AI's Stable Diffusion 3.5 Medium via Replicate API.
    Uses the model-specific prediction endpoint.
    Recommended kwargs for SD 3.5 Medium: aspect_ratio, output_format, output_quality, cfg, prompt_strength.

    Args:
        prompt: Text description of the image to generate.
        negative_prompt: Optional elements to avoid in the image.
        width: Image width (may be overridden by aspect_ratio in kwargs).
        height: Image height (may be overridden by aspect_ratio in kwargs).
        num_inference_steps: Number of denoising steps (default: 40).
        **kwargs: Additional parameters for the SD 3.5 Medium model (e.g., aspect_ratio="1:1", output_format="webp", cfg=5).

    Returns:
        A dictionary containing the image URL.
        Example: {"image_url": "https://.../image.png"}

    Raises:
        ValueError: If Replicate API Key is not configured.
        ConnectionError: If the API request fails or returns an error status.
        TimeoutError: If the Replicate job polling times out.
    """
    if not REPLICATE_API_TOKEN:
        logger.error("REPLICATE_API_TOKEN environment variable not set.")
        raise ValueError("Replicate API Token is not configured.")

    # Construct the model-specific prediction URL for SD 3.5 Medium
    predictions_url = (
        f"{REPLICATE_API_HOST}/{REPLICATE_API_VERSION}/models"
        f"/{REPLICATE_MODEL_OWNER}/{REPLICATE_MODEL_NAME}/predictions"
    )

    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Construct the input payload
    payload_input = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        # Include width/height, but know they might be ignored if aspect_ratio is in kwargs
        # "width": width, 
        # "height": height,
        "steps": num_inference_steps, # SD 3.5 uses 'steps' not 'num_inference_steps'
        **kwargs
    }
    # Filter out None values *before* potentially adding default width/height if aspect_ratio is missing
    payload_input = {k: v for k, v in payload_input.items() if v is not None}

    # Add default aspect_ratio if not provided in kwargs
    if 'aspect_ratio' not in payload_input:
        # Attempt to map width/height to common aspect ratios or use default
        # This mapping is approximate
        ratio = width / height if height != 0 else 1
        if abs(ratio - 1) < 0.05: payload_input['aspect_ratio'] = '1:1'
        elif abs(ratio - 16/9) < 0.05: payload_input['aspect_ratio'] = '16:9'
        elif abs(ratio - 9/16) < 0.05: payload_input['aspect_ratio'] = '9:16'
        elif abs(ratio - 4/3) < 0.05: payload_input['aspect_ratio'] = '4:3'
        elif abs(ratio - 3/4) < 0.05: payload_input['aspect_ratio'] = '3:4'
        else: payload_input['aspect_ratio'] = '1:1' # Default fallback
        logger.debug(f"Aspect ratio not provided, calculated approximate aspect_ratio: {payload_input['aspect_ratio']} from width={width}, height={height}")
    
    # Remove width/height from payload as SD 3.5 uses aspect_ratio
    # payload_input.pop('width', None)
    # payload_input.pop('height', None)

    initial_payload = {"input": payload_input}

    logger.info(f"Requesting image from Replicate: {REPLICATE_MODEL_OWNER}/{REPLICATE_MODEL_NAME} (latest version)")
    logger.debug(f"Payload input: {initial_payload['input']}")

    # Timeout for individual HTTP requests
    request_timeout = aiohttp.ClientTimeout(total=60)

    try:
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            # 1. Start the prediction (POST to model-specific URL)
            async with session.post(predictions_url, headers=headers, json=initial_payload) as response:
                if response.status >= 300:
                    error_text = await response.text()
                    logger.error(f"Replicate API error on initial request ({response.status}): {error_text}")
                    raise ConnectionError(f"Replicate API failed ({response.status}): {error_text}")
                
                initial_response_data = await response.json()
                if "urls" not in initial_response_data or "get" not in initial_response_data["urls"]:
                     logger.error(f"Replicate initial response missing status URL: {initial_response_data}")
                     raise ConnectionError("Replicate API response structure changed or invalid.")

                status_url = initial_response_data["urls"]["get"]
                prediction_id = initial_response_data.get("id")
                logger.info(f"Replicate prediction started (ID: {prediction_id}). Polling status...")

            # 2. Poll for the result
            for attempt in range(MAX_POLLING_ATTEMPTS):
                await asyncio.sleep(POLLING_INTERVAL_SECONDS)
                logger.debug(f"Polling attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS} for prediction {prediction_id}")
                async with session.get(status_url, headers=headers) as status_response:
                    if status_response.status != 200:
                        logger.warning(f"Polling failed with status {status_response.status}. Retrying...")
                        continue # Or potentially raise after several failed polls

                    status_data = await status_response.json()
                    status = status_data.get("status")

                    if status == "succeeded":
                        output = status_data.get("output")
                        if isinstance(output, list) and len(output) > 0:
                            # Assume the first URL is the desired one
                            image_url = output[0]
                            logger.info(f"Replicate prediction {prediction_id} succeeded.")
                            return {"image_url": image_url}
                        else:
                            logger.error(f"Replicate prediction {prediction_id} succeeded but output format is unexpected: {output}")
                            raise ValueError("Replicate returned success but no image URL found.")
                        
                    elif status == "failed" or status == "canceled":
                        error_detail = status_data.get("error", "Unknown error")
                        logger.error(f"Replicate prediction {prediction_id} failed or canceled: {error_detail}")
                        raise ConnectionError(f"Replicate image generation failed: {error_detail}")
                    
                    # If status is "starting" or "processing", continue polling
                    elif status not in ["starting", "processing"]:
                         logger.warning(f"Replicate prediction {prediction_id} in unexpected status: {status}")
                         # Decide whether to keep polling or fail
            
            # If loop finishes without success
            logger.error(f"Replicate prediction {prediction_id} timed out after {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL_SECONDS} seconds.")
            raise TimeoutError("Replicate image generation timed out.")

    except asyncio.TimeoutError as e:
        # Catch timeout from individual requests or the overall polling logic
        logger.error(f"Replicate API request timed out: {e}")
        raise TimeoutError(f"Replicate API request timed out: {e}")
    except aiohttp.ClientError as e:
        logger.exception(f"Error connecting to Replicate API: {e}")
        raise ConnectionError(f"Failed to connect to Replicate API: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Replicate image generation: {e}")
        raise ConnectionError(f"An unexpected error occurred: {e}")

# --- (Example Usage - commented out, needs updating) --- 
# async def main():
#     # Ensure API key is set: export REPLICATE_API_TOKEN='your_token_here'
#     logging.basicConfig(level=logging.INFO)
#     try:
#         result = await generate_image(
#             prompt="A photorealistic astronaut riding a horse on the moon, cinematic lighting",
#             negative_prompt="low quality, blurry, text, watermark, ugly, tiling",
#             width=768,
#             height=768,
#             num_inference_steps=30,
#             guidance_scale=7.5 # Example additional param
#         )
#         print(f"Replicate Result: {result}")
#     except (ValueError, ConnectionError, TimeoutError) as e:
#         print(f"Error during example usage: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())
#     pass 