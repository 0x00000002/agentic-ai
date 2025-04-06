#!/usr/bin/env python
"""
Framework Example for Agentic AI

This example demonstrates how to use the Agentic AI framework, including:
- Configuration setup
- Agent creation and orchestration
- Tool usage
- UI interaction
"""
import os
import sys
import logging
import time
from pathlib import Path
import uuid

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("framework_example")

from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.unified_config import UnifiedConfig
from src.config.user_config import UserConfig
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_manager import ToolManager
from src.agents.agent_factory import AgentFactory

def example_basic_framework_usage():
    """Example of basic framework usage with configuration and agent creation."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.coordinator import Coordinator
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import ToolEnabledAI
    
    # Configure the framework
    logger.info("Configuring framework with Claude 3.5 Sonnet and Solidity coding use case")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get the configuration instance
    config = get_config()
    logger.info(f"Using model: {config.get_default_model()}")
    
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
    agent_factory = AgentFactory(registry=registry)
    
    # Create an orchestrator agent with AI instance
    logger.info("Creating orchestrator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Process a request
    logger.info("Processing a request")
    request = {
        "prompt": "Create a simple ERC20 token contract with a fixed supply of 1,000,000 tokens.",
        "conversation_history": []
    }
    
    start_time = time.time()
    response = coordinator.process_request(request)
    end_time = time.time()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return coordinator

def example_tool_usage():
    """Example of using tools with the framework."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.coordinator import Coordinator
    from src.tools.tool_registry import ToolRegistry
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import ToolEnabledAI
    
    # Configure the framework
    logger.info("Configuring framework for tool usage")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get the tool registry
    logger.info("Getting tool registry")
    tool_registry = ToolRegistry()
    
    # List available tools
    logger.info("Available tools:")
    for tool_name, tool_info in tool_registry.get_all_tools().items():
        logger.info(f"- {tool_name}: {tool_info.get('description', 'No description')}")
    
    # Create an AI instance
    logger.info("Creating AI instance")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant specialized in coding.",
        logger=logger
    )
    
    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry)
    
    # Create an orchestrator agent with AI instance
    logger.info("Creating orchestrator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Process a request that will likely use tools
    logger.info("Processing a request that will use tools")
    request = {
        "prompt": "Search for information about the latest Solidity version and its features.",
        "conversation_history": []
    }
    
    start_time = time.time()
    response = coordinator.process_request(request)
    end_time = time.time()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return coordinator

def example_ui_interaction():
    """Example of using the UI components of the framework."""
    from src.config import configure, get_config, UseCasePreset
    from src.ui.simple_chat import SimpleChatUI
    from src.agents.agent_factory import AgentFactory
    from src.agents.agent_registry import AgentRegistry
    from src.core.tool_enabled_ai import ToolEnabledAI
    
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
    
    # Create an orchestrator agent with AI instance
    logger.info("Creating orchestrator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Create a simple chat UI with the orchestrator
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
    response = chat_ui.process_message(user_message, history)
    
    # Display the response
    if response and len(response) > 0:
        logger.info("Bot response:")
        print("\n" + response[0][1] + "\n")
    else:
        logger.warning("No response received")
    
    logger.info("UI interaction example completed")
    logger.info("Note: In a real application, you would launch the UI with chat_ui.launch()")

def example_custom_agent():
    """Example of creating and using a custom agent."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.base_agent import BaseAgent
    from src.agents.agent_registry import AgentRegistry
    from src.agents.agent_registrar import register_core_agents
    from src.core.tool_enabled_ai import ToolEnabledAI
    
    # Configure the framework
    logger.info("Configuring framework for custom agent")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Define a custom agent class
    class SolidityExpertAgent(BaseAgent):
        """A custom agent specialized in Solidity development."""
        
        def __init__(self, ai_instance=None, config=None, logger=None):
            super().__init__(ai_instance=ai_instance, config=config, logger=logger)
            self.name = "solidity_expert"
            self.description = "An agent specialized in Solidity smart contract development"
            self.system_prompt = "You are an expert Solidity developer with deep knowledge of blockchain security, gas optimization, and EVM."
        
        def process_request(self, request):
            """Process a request and return a response."""
            prompt = request.get("prompt", "")
            conversation_history = request.get("conversation_history", [])
            
            # Add Solidity-specific context to the prompt
            enhanced_prompt = f"As a Solidity expert, please help with the following: {prompt}"
            
            # Create a new request with the enhanced prompt
            enhanced_request = {
                "prompt": enhanced_prompt,
                "conversation_history": conversation_history
            }
            
            # Use the base agent's process_request method
            return super().process_request(enhanced_request)
    
    # Create an AI instance
    logger.info("Creating AI instance")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are an expert Solidity developer with deep knowledge of blockchain security, gas optimization, and EVM.",
        logger=logger
    )
    
    # Create and register the custom agent
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Register core agents
    logger.info("Registering core agents")
    register_core_agents(registry, logger)
    
    # Register the custom agent
    logger.info("Registering custom Solidity expert agent")
    registry.register("solidity_expert", SolidityExpertAgent)
    
    # Create an agent factory with the custom registry
    logger.info("Creating agent factory with custom registry")
    agent_factory = AgentFactory(registry=registry)
    
    # Create the custom agent with AI instance
    logger.info("Creating custom Solidity expert agent")
    solidity_agent = agent_factory.create("solidity_expert", ai_instance=ai_instance)
    
    # Check if agent creation was successful
    if solidity_agent is None:
        logger.error("Failed to create Solidity expert agent")
        return None
    
    # Process a request with the custom agent
    logger.info("Processing a request with the custom agent")
    request = {
        "prompt": "What are the best practices for gas optimization in Solidity?",
        "conversation_history": []
    }
    
    start_time = time.time()
    response = solidity_agent.process_request(request)
    end_time = time.time()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return solidity_agent

def simple_chat():
    print("--- Simple Chat Example ---")
    ai = ToolEnabledAI()
    response = ai.request("What is the capital of Spain?")

def chat_with_local_model():
    print("--- Chat with Local Model Example ---")
    ai = ToolEnabledAI(model="phi4")
    response = ai.request("Explain the concept of recursion in programming.")

def chat_with_override():
    print("--- Chat with User Override Example ---")
    user_cfg = UserConfig(model="gpt-4o-mini")
    ai = ToolEnabledAI(model="gpt-4o-mini")
    response = ai.request("Give me 3 ideas for a fantasy novel.")

def chat_with_tools():
    print("--- Chat with Tools Example ---")
    ai = ToolEnabledAI(auto_tool_finding=False)
    
    ai.register_tool("list_directory", list_files)

def run_examples():
    """Run all framework examples."""
    logger.info("Running framework examples")
    
    print("\n=== Basic Framework Usage ===")
    example_basic_framework_usage()
    
    print("\n=== Tool Usage ===")
    example_tool_usage()
    
    print("\n=== UI Interaction ===")
    example_ui_interaction()
    
    print("\n=== Custom Agent ===")
    example_custom_agent()
    
    print("\n=== Simple Chat Example ===")
    simple_chat()
    
    print("\n=== Chat with Local Model Example ===")
    chat_with_local_model()
    
    print("\n=== Chat with User Override Example ===")
    chat_with_override()
    
    print("\n=== Chat with Tools Example ===")
    chat_with_tools()
    
    logger.info("All framework examples completed successfully")

if __name__ == "__main__":
    run_examples() 