#!/usr/bin/env python
"""
Standalone script to test the Orchestrator's meta-query handling.
"""
import os
import sys
import logging
from pathlib import Path
import json

# --- Path Setup ---
# Add the project root to the Python path to allow importing 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
    ]
)
logger = logging.getLogger("test_meta_query")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting meta-query test script...")

    # --- Log Anthropic SDK Version ---
    try:
        import anthropic
        logger.info(f"Anthropic SDK Version: {anthropic.__version__}")
    except ImportError:
        logger.warning("Anthropic SDK not found.")
    except Exception as e:
        logger.error(f"Could not determine Anthropic SDK version: {e}")
    # --- End Version Log ---

    try:
        # Import necessary components after path setup
        # from src.agents.orchestrator import Orchestrator # Old Import
        # from src.agents.orchestrator_v2 import OrchestratorV2 # Old V2 Import
        from src.agents.coordinator import Coordinator # New Import
        from src.config.unified_config import UnifiedConfig

        # Load configuration (ensures API keys etc. are available if needed by providers)
        # This assumes your .env file is correctly placed and configured
        try:
            config = UnifiedConfig.get_instance()
            logger.info("UnifiedConfig loaded successfully.")
            # Optional: Add checks here for specific keys like Anthropic API key
            # if not config.get_api_key("anthropic"):
            #     logger.warning("Anthropic API key might be missing!")
        except Exception as e:
             logger.error(f"Failed to load UnifiedConfig: {e}. Ensure config files/.env are set up.")
             sys.exit(1)

        # Initialize the Coordinator (using default settings)
        try:
            # orchestrator = Orchestrator() # Old Instantiation
            # orchestrator = OrchestratorV2() # Old V2 Instantiation
            coordinator = Coordinator() # New Instantiation
            logger.info("Coordinator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Coordinator: {e}", exc_info=True)
            sys.exit(1)

        # Define the meta-query request
        meta_query = {
            "prompt": "tell me what tools do you have? what agents?"
            # Add other fields like user_id if your orchestrator expects them
            # "user_id": "test_user_meta"
        }
        logger.info(f"Sending request to coordinator: {meta_query}")

        # Process the request
        try:
            response = coordinator.process_request(meta_query)
            logger.info("Received response from coordinator.")

            # Print the response nicely
            print("\n" + "-"*20 + " Orchestrator Response " + "-"*20)
            print(json.dumps(response, indent=2))
            print("-"*62 + "\n")

        except Exception as e:
            logger.error(f"Error during coordinator.process_request: {e}", exc_info=True)

    except ImportError as e:
        logger.error(f"ImportError: {e}. Make sure you run this script as a module from the project root.")
        logger.error("Example: python -m scripts.test_meta_query")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

    logger.info("Meta-query test script finished.") 