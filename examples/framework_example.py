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
import warnings
from pathlib import Path
import uuid
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
logger = logging.getLogger("framework_example")

from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.unified_config import UnifiedConfig
from src.config.user_config import UserConfig
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_manager import ToolManager
from src.agents.agent_factory import AgentFactory

async def example_basic_framework_usage():
    """Example of basic framework usage with configuration and agent creation."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
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
    
    # Create a coordinator agent with AI instance
    logger.info("Creating coordinator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Process a request
    logger.info("Processing a request")
    request = {
        "prompt": "Create a simple ERC20 token contract with a fixed supply of 1,000,000 tokens.",
        "conversation_history": []
    }
    
    start_time = time.monotonic()
    response = await coordinator.process_request(request)
    end_time = time.monotonic()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return coordinator

async def example_tool_usage():
    """Example of using tools with the framework."""
    from src.config import configure, get_config, UseCasePreset
    from src.agents.agent_factory import AgentFactory
    from src.agents.agent_registry import AgentRegistry
    from src.tools.tool_registry import ToolRegistry
    from src.tools.tool_manager import ToolManager
    from src.core.tool_enabled_ai import ToolEnabledAI
    
    # Configure the framework
    logger.info("Configuring framework for tool usage")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get configuration
    config = get_config()
    
    # Create tool registry and manager
    logger.info("Setting up tool registry and manager")
    tool_registry = ToolRegistry(logger=logger)
    tool_manager = ToolManager(unified_config=config, logger=logger, tool_registry=tool_registry)
    
    # List available tools
    logger.info("Available tools:")
    for tool_name, tool_def in tool_registry.get_all_tool_definitions().items():
        logger.info(f"- {tool_name}: {tool_def.description}")
    
    # Create an AI instance with tool manager
    logger.info("Creating AI instance")
    ai_instance = ToolEnabledAI(
        model="claude-3-5-sonnet",
        system_prompt="You are a helpful assistant specialized in coding.",
        logger=logger,
        tool_manager=tool_manager
    )
    
    # Create an agent registry
    logger.info("Creating agent registry")
    registry = AgentRegistry()
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry)
    
    # Create a coordinator agent with AI instance
    logger.info("Creating coordinator agent")
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance, tool_manager=tool_manager)
    
    # Process a request that will likely use tools
    logger.info("Processing a request that will use tools")
    request = {
        "prompt": "Search for information about the latest Solidity version and its features.",
        "conversation_history": []
    }
    
    start_time = time.monotonic()
    response = await coordinator.process_request(request)
    end_time = time.monotonic()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return coordinator

async def example_ui_interaction():
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

async def example_custom_agent():
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
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.7,
        show_thinking=True
    )
    
    # Define a custom agent class
    class SolidityExpertAgent(BaseAgent):
        """A specialized agent for Solidity smart contract development."""
        
        def __init__(self, ai_instance=None, tool_manager=None, unified_config=None, logger=None, agent_id=None):
            """Initialize the Solidity expert agent."""
            # Pass all parameters to the parent class
            super().__init__(
                ai_instance=ai_instance,
                tool_manager=tool_manager,
                unified_config=unified_config,
                logger=logger,
                agent_id=agent_id or "solidity_expert"
            )
        
        async def process_request(self, request):
            """Process a request asynchronously for Solidity expertise."""
            # Extract the user prompt
            prompt = request.get("prompt", "")
            conversation_history = request.get("conversation_history", [])
            
            # Add Solidity expertise to the prompt
            enhanced_prompt = f"""
            As a Solidity expert, please respond to this request: 
            
            {prompt}
            
            Focus on best practices, security considerations, and gas optimization.
            """
            
            # Log the enhanced prompt
            self.logger.debug(f"Enhanced prompt for Solidity expert: {enhanced_prompt}")
            
            # Process with the AI instance asynchronously
            if self.ai_instance:
                # Use await for the 'request' method
                response_content = await self.ai_instance.request(
                    enhanced_prompt,
                )
                # Return the standard response format
                return {
                    "content": response_content,
                    "agent_id": self.agent_id,
                    "status": "success"
                }
            else:
                # Return an error in the standard format
                return {
                    "content": "Error: No AI instance available for processing",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": "No AI instance available for processing"
                }
    
    # Get configuration
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
    
    # Register the custom agent
    logger.info("Registering custom Solidity expert agent")
    registry.register("solidity_expert", SolidityExpertAgent)
    
    # Register core agents as well
    # Note: AgentRegistry __init__ already calls register_core_agents
    # register_core_agents(registry) # This call might be redundant
    
    # Create an agent factory
    logger.info("Creating agent factory")
    agent_factory = AgentFactory(registry=registry)
    
    # Create the custom agent
    logger.info("Creating Solidity expert agent")
    solidity_agent = agent_factory.create("solidity_expert", ai_instance=ai_instance)
    
    # Process a request
    logger.info("Processing a request with the custom agent")
    request = {
        "prompt": "What are the security considerations for implementing a staking contract?",
        "conversation_history": []
    }
    
    start_time = time.monotonic()
    response = await solidity_agent.process_request(request)
    end_time = time.monotonic()
    
    logger.info(f"Response received in {end_time - start_time:.2f} seconds")
    
    # Extract and display the response content
    if isinstance(response, dict):
        content = response.get("content", str(response))
    else:
        content = str(response)
    
    logger.info("Response content:")
    print("\n" + content + "\n")
    
    return solidity_agent

async def simple_chat():
    """Run a simple chat example."""
    from src.config import configure
    configure(model="phi3-mini", use_case="chat")
    from src.ui.simple_chat import run_simple_chat
    run_simple_chat()

async def chat_with_local_model():
    """Run a chat with a local model."""
    from src.config import configure
    configure(model="llamacpp://llama3-8b", use_case="chat", system_prompt="You are a helpful AI assistant.")
    from src.ui.simple_chat import run_simple_chat
    run_simple_chat()

async def chat_with_override():
    """Run a chat with temperature override."""
    from src.config import configure
    configure(model="claude-3-haiku", use_case="chat", temperature=0.9, show_thinking=True)
    from src.ui.simple_chat import run_simple_chat
    run_simple_chat()

async def chat_with_tools():
    """Run a chat with tools enabled."""
    from src.config import configure
    configure(model="claude-3-5-sonnet", use_case="coding")
    from src.ui.tool_chat import run_tool_chat
    run_tool_chat()

async def run_examples():
    """Run all examples."""
    print("\n=== Basic Framework Usage ===\n")
    await example_basic_framework_usage()
    
    print("\n=== Tool Usage ===\n")
    await example_tool_usage()
    
    print("\n=== Custom Agent ===\n")
    await example_custom_agent()
    
    print("\n=== UI Interaction (Simulated) ===\n")
    await example_ui_interaction()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run a specific example
        example_name = sys.argv[1]
        if example_name == "basic":
            asyncio.run(example_basic_framework_usage())
        elif example_name == "tools":
            asyncio.run(example_tool_usage())
        elif example_name == "ui":
            asyncio.run(example_ui_interaction())
        elif example_name == "custom":
            asyncio.run(example_custom_agent())
        elif example_name == "chat":
            asyncio.run(simple_chat())
        elif example_name == "local":
            asyncio.run(chat_with_local_model())
        elif example_name == "override":
            asyncio.run(chat_with_override())
        elif example_name == "tool_chat":
            asyncio.run(chat_with_tools())
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: basic, tools, ui, custom, chat, local, override, tool_chat")
    else:
        # Run all examples
        asyncio.run(run_examples()) 