# examples/openai_example.py
"""
Example script demonstrating usage of ToolEnabledAI with various OpenAI models.
"""

import asyncio
import os
from pathlib import Path
import sys

# --- Setup Path ---
# Add the project root to the Python path to allow importing from 'src'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from src.config import configure, get_config, UnifiedConfig
from src.core.tool_enabled_ai import ToolEnabledAI
from src.utils.logger import LoggerFactory, LoggingLevel
from src.exceptions import AISetupError, AIProcessingError

# --- Configuration ---
# Enable real loggers for the framework components
LoggerFactory.enable_real_loggers()

# Get a logger for this example script using the factory
logger = LoggerFactory.create("openai_example", level=LoggingLevel.INFO, use_real_logger=True)

# Configure the AI Framework (using defaults, will load models.yml etc.)
try:
    configure()
    logger.info("Framework configured.")
except AISetupError as e:
    logger.error(f"Framework configuration failed: {e}", exc_info=True)
    exit(1)

# --- Helper Function ---
async def run_openai_request(model_id: str, prompt: str):
    """Initializes ToolEnabledAI with a specific OpenAI model and runs a prompt."""
    logger.info(f"\n--- Testing Model: {model_id} ---")
    logger.info(f"Prompt: \"{prompt}\"\n")
    
    # Check for OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error(f"OPENAI_API_KEY not set. Cannot run OpenAI model '{model_id}'.")
        return
    # Also check for Replicate key if testing image generation prompt
    if "generate an image" in prompt.lower() and not os.getenv("REPLICATE_API_TOKEN"):
         logger.warning("REPLICATE_API_TOKEN not set. Image generation tool call will fail if attempted.")
        # Allow to continue to see if model attempts the call

    try:
        # Initialize ToolEnabledAI with the specified OpenAI model ID (string)
        logger.info(f"Initializing ToolEnabledAI with {model_id}...")
        ai = ToolEnabledAI(model=model_id)
        logger.info(f"ToolEnabledAI initialized successfully for {model_id}.")
        
        # Make the request
        response = await ai.request(prompt)
        
        logger.info(f"--- Response from {model_id} ---")
        logger.info(f"{response}")
        logger.info("-------------------------")
        
        # Check tool history
        tool_history = ai.get_tool_history()
        if tool_history:
            logger.info("Tool Call History:")
            for call, result in tool_history:
                logger.info(f"  - Called: {call.name}({call.arguments}) -> Success: {result.success}, Result/Error: {result.result or result.error}")
        else:
            logger.info("No tools were called for this request.")
            
    except (AISetupError, AIProcessingError) as e:
        logger.error(f"AI execution failed for model {model_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred testing model {model_id}: {e}", exc_info=True)

# --- Main Execution Logic ---
async def main():
    # Removed o3-mini due to parameter incompatibility errors
    openai_models_to_test = ["gpt-4o-mini", "gpt-4o"]
    
    simple_prompt = "Explain the concept of quantum entanglement in simple terms."
    image_prompt = "Please generate an image of a robot relaxing on a beach."
    
    # Check if models are configured before testing
    config = get_config()
    available_models = config.get_model_names()
    valid_models = [m for m in openai_models_to_test if m in available_models]
    invalid_models = [m for m in openai_models_to_test if m not in available_models]
    
    if invalid_models:
        logger.warning(f"The following OpenAI models are not defined in configuration and will be skipped: {invalid_models}")
        logger.warning(f"Available models: {available_models}")
        
    if not valid_models:
        logger.error("None of the specified OpenAI models (gpt-4o-mini, gpt-4o) are configured. Exiting.")
        return
        
    logger.info(f"Will test the following configured OpenAI models: {valid_models}")
    
    for model_id in valid_models:
        await run_openai_request(model_id, simple_prompt)
        await asyncio.sleep(1) # Small delay between requests
        await run_openai_request(model_id, image_prompt)
        await asyncio.sleep(1) # Small delay between requests

# --- Entry Point ---
if __name__ == "__main__":
    # Basic check for the primary key needed
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY environment variable is not set.")
    else:
        logger.info("Running OpenAI Example...")
        asyncio.run(main())
        logger.info("OpenAI Example finished.") 