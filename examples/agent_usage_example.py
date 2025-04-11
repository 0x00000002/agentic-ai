#!/usr/bin/env python
"""
Example demonstrating direct instantiation and usage of a specific agent.
"""

import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path VERY FIRST
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Redundant if the above works

import asyncio
# import logging # Remove standard logging import

# Now import project modules
from src.config import get_config, UnifiedConfig
from src.utils.logger import LoggerFactory
# from src.core.provider_factory import ProviderFactory # No provider needed for this agent
from src.agents import AgentFactory
# Import the specific agent we want to demonstrate
from src.agents.request_analyzer import RequestAnalyzer

# Set up logging using LoggerFactory, ensuring a real logger is created
logger = LoggerFactory.create("agent_usage_example", use_real_logger=True)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("agent_usage_example")

async def main():
    # --- Configuration & Setup ---
    logger.info("Setting up environment variables (if needed for provider)")
    # Ensure necessary API keys are set (e.g., for OpenAI)
    # No API keys needed for RequestAnalyzer V2 as it uses regex
    # if "OPENAI_API_KEY" not in os.environ:
    #     logger.warning("OPENAI_API_KEY environment variable not set. Using a placeholder.")
    #     os.environ["OPENAI_API_KEY"] = "placeholder_openai_key"

    # Load configuration (may still be needed for agent init)
    try:
        config = get_config()
        logger.info("Configuration loaded.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return

    # Initialize dependencies needed by the agent
    # The RequestAnalyzer V2 does NOT need an AI provider
    # logger.info("Initializing AI Provider...")
    # try:
    #     model_id = config.get_default_model()
    #     model_config = config.get_model_config(model_id)
    #     if not model_config:
    #         raise ValueError(f"Configuration not found for model: {model_id}")
    #     provider_type = model_config.get("provider")
    #     if not provider_type:
    #         raise ValueError(f"Provider type not specified in config for model: {model_id}")
    #     provider_config = config.get_provider_config(provider_type)
    #     ai_provider = ProviderFactory.create(
    #         provider_type=provider_type,
    #         provider_config=provider_config or {},
    #         model_config=model_config,
    #         model_id=model_id,
    #         logger=logger
    #     )
    #     logger.info(f"Initialized AI Provider with model: {model_id}")
    # except Exception as e:
    #     logger.error(f"Failed to initialize AI Provider: {e}", exc_info=True)
    #     return

    # --- Agent Instantiation ---
    # logger.info("Initializing AgentFactory...") # Not using factory in this example
    try:
        # Direct Instantiation of RequestAnalyzer
        logger.info("Instantiating RequestAnalyzer directly...")
        # Pass config as it might be used internally, though optional
        request_analyzer = RequestAnalyzer(unified_config=config, logger=logger)
        logger.info("RequestAnalyzer instantiated.")
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        return

    # --- Direct Agent Interaction ---
    print("\n-------------------------")
    print("Interacting with RequestAnalyzerAgent")
    print("-------------------------")

    # Example user queries
    queries = [
        "What is the capital of France?",
        "Can you write a python function to calculate fibonacci?",
        "What tools do you have?",
        "Tell me a joke about cats."
    ]

    for query in queries:
        print(f"\nAnalyzing query: '{query}'")
        try:
            # Call the agent's primary method: classify_request_intent
            # It expects a dictionary containing the prompt
            request_data = {"prompt": query}
            # This method is synchronous in V2, no await needed
            # analysis_result = await request_analyzer.analyze(query)
            analysis_result = request_analyzer.classify_request_intent(request_data)
            
            print(f"Analysis Result (Intent): {analysis_result}")
            # Print the raw result or format it
            # The result structure depends on RequestAnalyzerAgent's implementation
            # if isinstance(analysis_result, dict):
            #     for key, value in analysis_result.items():
            #         print(f"  {key}: {value}")
            # else:
            #     print(f"  {analysis_result}")
                
            logger.info(f"Successfully analyzed query: '{query}'")

        except Exception as e:
            print(f"  Error analyzing query: {type(e).__name__}: {e}")
            # Standard logger supports exc_info
            logger.error(f"Error interacting with agent for query '{query}': {e}", exc_info=True)

    print("\n-------------------------")
    print("Agent usage example finished.")
    print("-------------------------")


if __name__ == "__main__":
    # No need for basicConfig here as LoggerFactory handles setup
    asyncio.run(main()) 