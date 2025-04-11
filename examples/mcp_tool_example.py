#!/usr/bin/env python
"""
Example demonstrating how to use tools from a configured MCP server.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_config
from src.tools import ToolManager
# from src.providers.openai import OpenAIProvider # Or any other provider
from src.core.providers.openai_provider import OpenAIProvider # Correct path
# from src.ai import ToolEnabledAI
from src.core.tool_enabled_ai import ToolEnabledAI # Correct path
from src.utils.logger import LoggerFactory

# Set up logging
logger = LoggerFactory.create("mcp_tool_example")

async def main():
    # --- Configuration & Setup ---
    logger.info("Setting up environment variables (replace with your actual keys/tokens)")
    # Set environment variables needed by the configuration
    # Replace with your actual token for the example MCP server
    os.environ["WEATHER_MCP_TOKEN"] = "dummy_mcp_token" 
    # Replace with your OpenAI API key if using the OpenAI provider
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY environment variable not set. Using a placeholder.")
        os.environ["OPENAI_API_KEY"] = "placeholder_openai_key" 

    # Load configuration (which should include mcp_servers defined in config.yml)
    # Assumes you have a config.yml with the 'weather_service_mcp' server defined
    # as shown in documentation/examples/mcp_tool_usage.md
    try:
        config = get_config()
        logger.info("Configuration loaded.")
        if not config.get_mcp_config().get('mcp_servers', {}).get('weather_service_mcp'):
             logger.error("Error: 'weather_service_mcp' not found in mcp_servers config.")
             logger.error("Please ensure your config.yml includes the MCP server definition.")
             return
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return

    # Initialize components
    logger.info("Initializing ToolManager, AIProvider, and ToolEnabledAI...")
    tool_manager = ToolManager(config=config, logger=logger)
    # Replace OpenAIProvider with your desired provider if needed
    ai_provider = OpenAIProvider(config=config, logger=logger)
    ai = ToolEnabledAI(provider=ai_provider, tool_manager=tool_manager, config=config, logger=logger)
    logger.info("Components initialized.")

    # --- List Available Tools ---
    print("\n-------------------------")
    print("Available Tools:")
    print("-------------------------")
    all_tools = tool_manager.list_available_tools()
    if not all_tools:
        print("No tools found.")
    for tool_def in all_tools:
        source_info = f"Source: {tool_def.source}"
        if tool_def.source == 'mcp':
            source_info += f", MCP Server: {tool_def.mcp_server_name}"
        print(f"- {tool_def.name} ({source_info})")
    print("-------------------------")

    # --- Example Prompt ---
    # This prompt is designed to trigger the 'get_current_weather_mcp' tool
    # which is configured to be provided by the 'weather_service_mcp' server.
    prompt = "What is the current weather like in Paris? Use fahrenheit."

    print(f"\nSending prompt: '{prompt}'")
    print("-------------------------")
    logger.info("Sending prompt to AI...")

    # --- AI Processing & Tool Call ---
    # The AI will potentially call the 'get_current_weather_mcp' tool.
    # The ToolManager will:
    # 1. Identify the tool source as 'mcp' and server as 'weather_service_mcp'.
    # 2. Request the client from MCPClientManager.
    # 3. MCPClientManager will use the config (URL, Auth type, env var) to connect.
    # 4. ToolManager will execute the call against the server.
    #
    # NOTE: This script requires:
    #   a) An MCP server actually running at the URL defined in config.yml (e.g., http://localhost:8005).
    #   b) The correct WEATHER_MCP_TOKEN environment variable set.
    #   c) A valid OpenAI API key (or relevant key for your chosen provider).
    #
    # If the server isn't running, the tool call will fail during the network request.
    try:
        response = await ai.process_prompt(prompt)
        print("\n-------------------------")
        print("Final AI Response:")
        print("-------------------------")
        print(response.content)
        print("-------------------------")
        logger.info("AI processing complete.")
        
        # --- Usage Stats --- 
        stats = tool_manager.stats_manager.get_stats()
        print("\n-------------------------")
        print("Tool Usage Stats:")
        print("-------------------------")
        print(stats)
        print("-------------------------")

    except Exception as e:
        print("\n-------------------------")
        print("An error occurred during AI processing:")
        print(f"{type(e).__name__}: {e}")
        print("-------------------------")
        logger.error(f"An error occurred: {e}", exc_info=True)
        print("\nTroubleshooting Tips:")
        print("- Is the MCP server (e.g., weather service) running at the URL configured in config.yml?")
        print("- Is the WEATHER_MCP_TOKEN environment variable set correctly?")
        print("- Is your AI Provider API key (e.g., OPENAI_API_KEY) valid?")
        print("- Check the logs for more details.")


if __name__ == "__main__":
    # Basic logging setup for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main()) 