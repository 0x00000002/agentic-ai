# examples/image_generation_example.py

import asyncio
# import logging # No longer needed directly for basic setup
import os
import uuid # Import uuid
from typing import Dict, Any

from src.config import configure, Model
from src.tools.tool_manager import ToolManager
from src.core.tool_enabled_ai import ToolEnabledAI
from src.exceptions import AIToolError, AISetupError, AIProcessingError
from src.tools.models import ToolCall
from src.utils.logger import LoggerFactory, LoggingLevel # Import necessary components

# --- Prerequisites ---
# 1. Ensure you have necessary packages installed (check requirements.txt)
# 2. Set the Replicate API token as an environment variable:
#    export REPLICATE_API_TOKEN='your_r8_..._token'

# --- Configuration ---
# Enable real loggers for the framework components
LoggerFactory.enable_real_loggers()

# Get a logger for this example script using the factory
logger = LoggerFactory.create("image_gen_example", level=LoggingLevel.INFO, use_real_logger=True)

# Configure the AI Framework (no need to pass log_level now)
try:
    configure()
except AISetupError as e:
    logger.error(f"Framework configuration failed: {e}", exc_info=True)
    exit(1)

# --- Helper Function ---
async def run_tool_directly(tool_name: str, arguments: Dict[str, Any]):
    """Demonstrates direct execution using ToolManager."""
    logger.info(f"\n--- Running Tool Directly: {tool_name} ---")
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        logger.error("REPLICATE_API_TOKEN not set. Cannot run image generation.")
        return

    try:
        tool_manager = ToolManager()
        
        tool_call_id = str(uuid.uuid4())
        tool_call_obj = ToolCall(id=tool_call_id, name=tool_name, arguments=arguments)
        
        logger.info(f"Executing tool call (ID: {tool_call_id}): {tool_call_obj.name}({tool_call_obj.arguments})")
        result = await tool_manager.execute_tool(tool_call_obj)
        
        # Check if the tool execution was successful before accessing results
        if result.success:
            logger.info(f"Tool '{tool_name}' (ID: {tool_call_id}) executed successfully.")
            # Use result.result which should contain the dict like {'image_url': '...'} 
            # based on the return value of generate_image function
            result_data = result.result 
            if isinstance(result_data, dict):
                 logger.info(f"Result: {result_data.get('image_url', 'No image_url found')}")
            else:
                 logger.info(f"Result (raw): {result_data}")
        else:
            logger.error(f"Tool '{tool_name}' (ID: {tool_call_id}) failed.")
            logger.error(f"Error: {result.error}")

    except AIToolError as e:
        # This might catch errors before ToolResult is created (e.g., tool not found)
        logger.error(f"Tool execution failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

async def run_with_agent(prompt: str):
    """Demonstrates using the tool via ToolEnabledAI."""
    logger.info(f"\n--- Running via ToolEnabledAI ---")
    logger.info(f"Prompt: \"{prompt}\"\n")

    try:
        # Use "o3-mini" as identified from the error message
        logger.info("Initializing ToolEnabledAI with o3-mini...")
        ai = ToolEnabledAI(model="o3-mini") 
        
        response = await ai.request(prompt)
        
        logger.info(f"AI Response:\n{response}")
        
        tool_history = ai.get_tool_history()
        if tool_history:
            logger.info("\nTool Call History:")
            for call, result in tool_history:
                logger.info(f"  - Called: {call.name}({call.arguments}) -> Result: {result.result_content}")
        else:
            logger.warning("AI did not seem to use any tools for this request.")
            
    except (AISetupError, AIProcessingError) as e:
        logger.error(f"AI Agent execution failed: {e}", exc_info=True) 
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent run: {e}", exc_info=True)

# --- Main Execution ---
async def main():
    # Example 1: Direct Tool Execution (Basic - uses latest SD 3.5 Medium)
    logger.info("--- Example 1: Basic execution (latest SD 3.5 Medium) ---")
    await run_tool_directly(
        tool_name="generate_image",
        arguments={
            "prompt": "A cute tabby cat wearing a small wizard hat, high quality cartoon",
            # Uses default steps (40) and approximates aspect_ratio from default width/height (1:1)
        }
    )

    # Example 2: Direct Tool Execution (SD 3.5 Medium with specific params)
    logger.info("--- Example 2: Execution with specific SD 3.5 params ---")
    await run_tool_directly(
        tool_name="generate_image",
        arguments={
            "prompt": "Epic fantasy landscape, mountains, river, detailed painting",
            "negative_prompt": "photorealistic, photograph, modern, text, watermark",
            # Explicitly passing SD 3.5 params via kwargs:
            "aspect_ratio": "16:9", 
            "output_format": "png", 
            "output_quality": 85,
            "cfg": 4.5,
            "steps": 35 # Overriding the default steps
        }
    )
    
    # Example 3: Agent-Based Execution (AI decides to use the tool)
    logger.info("--- Example 3: Agent execution (targeting SD 3.5 Medium) ---")
    await run_with_agent(
        # Prompt designed to encourage SD 3.5 Medium usage if the AI is capable
        prompt="Generate a photorealistic image of a futuristic cityscape at sunset in a synthwave style, make it 16:9 aspect ratio."
    )

# --- Entry Point ---
if __name__ == "__main__":
    # Check for API key before running async
    if not os.getenv("REPLICATE_API_TOKEN"):
        logger.critical("REPLICATE_API_TOKEN environment variable is not set.")
        logger.critical("Please set it before running this example:")
        logger.critical("  export REPLICATE_API_TOKEN='your_r8_..._token'")
    else:
        asyncio.run(main()) 