#!/usr/bin/env python
"""
Chat UI Example for Agentic AI Framework

This example demonstrates how to use the SimpleChatUI to create a chat interface
for interacting with the coordinator agent.
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
from src.utils.logger import LoggerFactory

def run_chat_ui():
    """Run the chat UI example."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import ToolEnabledAI
    from src.ui.simple_chat import SimpleChatUI
    
    # Configure the framework
    logger.info("Configuring framework for chat UI")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get config
    config = get_config()
    
    # Create an AI instance
    logger.info("Creating AI instance")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
        logger=logger
    )
    
    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry, unified_config=config, logger=logger)
    
    # Create a coordinator agent
    logger.info("Creating coordinator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Create a simple chat UI with the coordinator
    logger.info("Creating simple chat UI")
    chat_ui = SimpleChatUI(
        coordinator=coordinator,
        title="Agentic AI Solidity Expert"
    )
    
    # Launch the interface
    logger.info("Launching chat interface")
    chat_ui.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True              # Create a public URL
    )

def run_tool_chat_ui():
    """Run the chat UI example with tool support."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import ToolEnabledAI
    from src.tools.tool_registry import ToolRegistry
    from src.tools.tool_manager import ToolManager
    from src.ui.tool_chat import ToolChatUI
    
    # Configure the framework
    logger.info("Configuring framework for tool-enabled chat UI")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get config
    config = get_config()
    
    # Set up tool registry and manager
    logger.info("Setting up tool registry and manager")
    tool_registry = ToolRegistry(logger=logger)
    tool_manager = ToolManager(unified_config=config, logger=logger, tool_registry=tool_registry)
    
    # Create an AI instance with tool support
    logger.info("Creating AI instance with tool support")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
        logger=logger,
        tool_manager=tool_manager
    )
    
    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry, unified_config=config, logger=logger)
    
    # Create a coordinator agent with tool support
    logger.info("Creating coordinator agent with tool support")
    coordinator = agent_factory.create(
        "coordinator", 
        ai_instance=ai_instance,
        tool_manager=tool_manager
    )
    
    # Create a tool-enabled chat UI with the coordinator
    logger.info("Creating tool-enabled chat UI")
    chat_ui = ToolChatUI(
        coordinator=coordinator,
        title="Agentic AI Solidity Expert with Tools"
    )
    
    # Launch the interface
    logger.info("Launching tool-enabled chat interface")
    chat_ui.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True              # Create a public URL
    )

# Example: Basic chat function using the AI framework
def chat_function(message, history, model_choice, temperature):
    """Example of a basic chat function that could be used with Gradio directly."""
    # Create logger
    logger = LoggerFactory.create("chat_function")
    logger.info(f"Received: {message}, History: {len(history)}, Model: {model_choice}, Temp: {temperature}")
    
    # Simulate user config override (in a real app, this might come from UI elements)
    user_cfg = UserConfig(model=model_choice, temperature=temperature)
    
    # Create AI instance with selected model (overrides default)
    ai = ToolEnabledAI(model=model_choice, logger=logger)
    
    # In a real app, you might load conversation history into the AI instance
    # We can now use the ConversationManager which is handled by the ToolEnabledAI class
    for h in history:
        if len(h) >= 2:
            user_msg, ai_msg = h
            ai.add_message_pair(user_msg, ai_msg)
    
    try:
        # Process the message
        response = ai.process(message)
        
        # Extract content from response
        if isinstance(response, dict):
            content = response.get("content", str(response))
        else:
            content = str(response)
            
        logger.info(f"Response generated successfully")
        return content
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tools":
        run_tool_chat_ui()
    else:
        run_chat_ui() 