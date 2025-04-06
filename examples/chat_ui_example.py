#!/usr/bin/env python
"""
Chat UI Example for Agentic AI Framework

This example demonstrates how to use the SimpleChatUI to create a chat interface
for interacting with the orchestrator.
"""
import os
import sys
import logging
from pathlib import Path
import gradio as gr

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_ui_example")

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.user_config import UserConfig

def run_chat_ui():
    """Run the chat UI example."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import AI
    from src.ui import SimpleChatUI
    
    # Configure the framework
    logger.info("Configuring framework for chat UI")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.CHAT,
        temperature=0.7,
        show_thinking=True
    )
    
    # Create an AI instance
    logger.info("Creating AI instance")
    ai_instance = AI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
        logger=logger
    )
    
    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry)
    
    # Create an orchestrator agent
    logger.info("Creating orchestrator agent")
    orchestrator = agent_factory.create("orchestrator")
    
    # Set the AI instance on the orchestrator
    logger.info("Setting AI instance on orchestrator")
    orchestrator.ai_instance = ai_instance
    
    # Create a simple chat UI with the orchestrator
    logger.info("Creating simple chat UI")
    chat_ui = SimpleChatUI(
        orchestrator=orchestrator,
        title="Agentic AI Solidity Expert"
    )
    
    # Launch the interface
    logger.info("Launching chat interface")
    chat_ui.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True              # Create a public URL
    )

# Example: Basic chat function using the AI framework
def chat_function(message, history, model_choice, temperature):
    print(f"Received: {message}, History: {len(history)}, Model: {model_choice}, Temp: {temperature}")
    
    # Simulate user config override (in a real app, this might come from UI elements)
    user_cfg = UserConfig(model=model_choice, temperature=temperature)
    
    # Create AI instance with selected model (overrides default)
    ai = ToolEnabledAI(model=model_choice)
    
    # In a real app, you might load conversation history into the AI instance
    # ai.load_conversation(history) # Assuming such a method exists
    
    try:
        response = ai.request(message)
        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    run_chat_ui() 