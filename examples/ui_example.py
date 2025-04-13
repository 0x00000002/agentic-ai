#!/usr/bin/env python
"""
UI Interaction Example for Agentic AI Framework

This example demonstrates how to use the UI components of the Agentic AI framework,
specifically the SimpleChatUI.
"""
import os
import sys
import logging
import warnings
from pathlib import Path
import asyncio

# Suppress the specific NotOpenSSLWarning from urllib3
warnings.filterwarnings(
    'ignore',
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    category=Warning
)

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui_example")

from src.config import configure, get_config, UseCasePreset
from src.ui.simple_chat import SimpleChatUI
from src.agents.agent_factory import AgentFactory
from src.agents.agent_registry import AgentRegistry
from src.core.tool_enabled_ai import ToolEnabledAI


async def example_ui_interaction():
    """Example of using the UI components of the framework."""
    # Configure the framework
    logger.info("Configuring framework for UI interaction")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.CHAT,
        temperature=0.7,
        show_thinking=True
    )

    # Create an AI instance
    logger.info("Creating AI instance")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant.",
        logger=logger
    )

    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()

    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry)

    # Create a coordinator agent with AI instance
    logger.info("Creating coordinator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)

    # Create a simple chat UI with the coordinator
    logger.info("Creating simple chat UI")
    chat_ui = SimpleChatUI(coordinator=coordinator, title="Agentic AI Framework Example")

    # Build the interface
    logger.info("Building chat interface")
    interface = chat_ui.build_interface()

    # Note: In a real application, you would launch the UI here
    # For this example, we'll just simulate a conversation
    logger.info("Simulating a conversation")

    # Simulate a user message
    user_message = "Can you explain how the Agentic AI framework works?"

    # Process the message
    logger.info(f"Processing user message: {user_message}")
    history = []
    updated_history, _ = await chat_ui.process_message(user_message, history)

    # Display the response
    if updated_history and len(updated_history) > 0:
        # Get the last interaction (which is the response)
        last_interaction = updated_history[-1]
        bot_response_content = last_interaction[1]
        logger.info("Bot response:")
        print("\n" + bot_response_content + "\n")
    else:
        logger.warning("No response received or history is empty")

    logger.info("UI interaction example completed")
    logger.info("Note: In a real application, you would launch the UI with chat_ui.launch()")


if __name__ == "__main__":
    logger.info("Running UI Interaction Example...")
    asyncio.run(example_ui_interaction())
    logger.info("Example finished.") 