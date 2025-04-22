# examples/simple_agent_interaction.py
"""
Example demonstrating high-level interaction with the agent framework.
This hides the internal agent setup (Coordinator, ToolEnabledAI) from the main logic.
"""

import asyncio
import os
from pathlib import Path
import sys

# --- Setup Path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from src.config import configure
from src.utils.logger import LoggerFactory, LoggingLevel
# Import the main entry point agent
from src.agents.coordinator import Coordinator 
from src.exceptions import AISetupError

# --- Configuration ---
# Enable real loggers for the framework components
LoggerFactory.enable_real_loggers()

# Get a logger for this example script using the factory
logger = LoggerFactory.create("simple_agent_example", level=LoggingLevel.INFO, use_real_logger=True)

# Configure the AI Framework (using defaults, loads models.yml etc.)
# This should be called once before using the framework.
try:
    configure()
    logger.info("Framework configured successfully.")
except AISetupError as e:
    logger.error(f"Framework configuration failed: {e}", exc_info=True)
    exit(1)

# --- High-Level Interaction Function ---
async def ask_main_agent(prompt: str) -> str:
    """
    Sends a prompt to the main Coordinator agent and returns the response content.
    This function encapsulates the agent setup.
    """
    logger.info(f"\n--- Sending Prompt to Main Agent ---")
    logger.info(f"Prompt: \"{prompt}\"\n")
    
    # API Key checks (adapt based on expected tools/models the Coordinator might use)
    if ("generate an image" in prompt.lower() or "draw" in prompt.lower()) and not os.getenv("REPLICATE_API_TOKEN"):
        logger.warning("REPLICATE_API_TOKEN not set. Image generation will fail if attempted by the agent.")
        # Allow to continue to see if Coordinator handles it
    if not os.getenv("OPENAI_API_KEY"): # Assuming Coordinator might delegate to OpenAI
        logger.warning("OPENAI_API_KEY not set. Requests delegated to OpenAI models might fail.")
        # Allow to continue
        
    try:
        # 1. Instantiate the main entry point agent (Coordinator)
        #    It will internally create its dependencies (Analyzer, ToolManager, Factory)
        coordinator = Coordinator()
        logger.info("Coordinator agent instantiated.")

        # 2. Prepare the request dictionary
        request = {"prompt": prompt}

        # 3. Process the request via the Coordinator
        response_dict = await coordinator.process_request(request)
        logger.info("Coordinator processed request.")

        # 4. Extract the content (handle potential errors/structure variations)
        if isinstance(response_dict, dict):
            if response_dict.get("status") == "error":
                error_message = response_dict.get("metadata", {}).get("error_message", "Unknown error")
                logger.error(f"Agent returned an error status: {error_message}")
                # Return the formatted error content intended for the user
                return response_dict.get("content", f"Error: {error_message}") 
            else:
                content = response_dict.get("content")
                if content is None:
                    logger.warning(f"Agent response dictionary missing 'content': {response_dict}")
                    return "Error: Agent response format was unexpected (missing content)."
                return content
        else:
            # Fallback if the response wasn't a dictionary as expected
            logger.warning(f"Unexpected response type from agent: {type(response_dict)}. Content: {response_dict}")
            return f"Error: Unexpected response format from agent: {str(response_dict)}"

    except Exception as e:
        logger.error(f"An unexpected error occurred during agent interaction: {e}", exc_info=True)
        return f"System Error: An unexpected error occurred ({type(e).__name__})."

# --- Main Execution Logic ---
async def main():
    logger.info("--- Testing Simple Agent Interaction --- ")

    # Test 1: Simple Question (Should delegate to default text agent, e.g., o3-mini)
    response1 = await ask_main_agent("Explain the concept of quantum entanglement in simple terms.")
    logger.info(f"Response 1 (Text):\n{response1}")
    await asyncio.sleep(1)

    # Test 2: Image Generation Request (Should trigger direct dispatch in Coordinator)
    response2 = await ask_main_agent("Please generate an image of a robot relaxing on a beach.")
    logger.info(f"Response 2 (Image):\n{response2}")
    await asyncio.sleep(1)

    # Test 3: Meta Request (Should be handled directly by Coordinator)
    response3 = await ask_main_agent("What tools do you have available?")
    logger.info(f"Response 3 (Meta):\n{response3}")
    await asyncio.sleep(1)
    
    # Test 4: Another Image Request
    response4 = await ask_main_agent("Draw a picture of a cat programming on a laptop.")
    logger.info(f"Response 4 (Image):\n{response4}")

# --- Entry Point ---
if __name__ == "__main__":
    # Basic check for primary API keys (adjust if your default agent differs)
    keys_ok = True
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY environment variable is not set (needed for default agent).")
        keys_ok = False
    if not os.getenv("REPLICATE_API_TOKEN"):
         logger.critical("REPLICATE_API_TOKEN environment variable is not set (needed for image tool).")
         keys_ok = False
         
    if keys_ok:
        logger.info("Running Simple Agent Interaction Example...")
        asyncio.run(main())
        logger.info("Simple Agent Interaction Example finished.")
    else:
        logger.critical("Example aborted due to missing API keys.") 