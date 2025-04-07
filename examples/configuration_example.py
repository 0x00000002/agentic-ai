#!/usr/bin/env python
"""
Configuration Example for Agentic AI

This example demonstrates how to use the unified configuration system.
"""
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("configuration_example")

def example_basic_usage():
    """Example of basic configuration usage."""
    from src.config import configure, get_config, UseCasePreset
    
    # Configure with specific model
    configure(model="phi4")
    
    # Get the configuration instance
    config = get_config()
    
    # Use the configuration
    print(f"Default model: {config.get_default_model()}")
    
    # Get model configuration
    model_config = config.get_model_config()
    print(f"Model details: {model_config.get('name', 'Unnamed')} ({model_config.get('provider', 'Unknown')})")
    
    # Use a predefined use case
    configure(use_case=UseCasePreset.SOLIDITY_CODING)
    
    # Get use case configuration
    use_case_config = config.get_use_case_config()
    print(f"Use case configuration: {use_case_config}")


def example_use_cases():
    """Example of using different use cases."""
    from src.config import configure, get_config, UseCasePreset
    
    # Try different use cases
    use_cases = [
        UseCasePreset.CHAT,
        UseCasePreset.CODING,
        UseCasePreset.SOLIDITY_CODING,
        UseCasePreset.DATA_ANALYSIS
    ]
    
    for use_case in use_cases:
        configure(use_case=use_case)
        config = get_config()
        use_case_config = config.get_use_case_config()
        print(f"{use_case.value}: quality={use_case_config.get('quality')}, speed={use_case_config.get('speed')}")


def example_model_listing():
    """Example of listing available models."""
    from src.config import get_config
    
    # Get the configuration instance
    config = get_config()
    
    # Get all available models
    model_names = config.get_model_names()
    
    print("Available models:")
    for model_id in model_names:
        try:
            model_config = config.get_model_config(model_id)
            print(f"- {model_id}: {model_config.get('name', 'Unnamed')} "
                  f"({model_config.get('privacy', 'unknown')} privacy, "
                  f"{model_config.get('quality', 'unknown')} quality)")
        except Exception as e:
            print(f"- {model_id}: Error retrieving details - {str(e)}")


def example_custom_settings():
    """Example of using custom configuration settings."""
    from src.config import configure, get_config
    from src.config.user_config import UserConfig
    
    # Configure with additional custom settings
    configure(
        model="claude-3-5-sonnet",
        temperature=0.8,
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
        show_thinking=True
    )
    
    # Access the configuration
    config = get_config()
    print(f"System prompt: {config.get_system_prompt()}")
    print(f"Show thinking: {config.show_thinking}")
    
    # Example of creating and applying a user config directly
    user_config = UserConfig(
        model="gpt-4o",
        temperature=0.9,
        show_thinking=True
    )
    
    # Apply the user config to the unified config
    config.apply_user_config(user_config)
    
    # Verify the changes
    print(f"Updated model: {config.get_default_model()}")
    print(f"User overrides: {config.user_overrides}")


def example_tool_config():
    """Example of accessing tool configuration."""
    from src.config import get_config
    
    # Get the configuration
    config = get_config()
    
    # Get general tool configuration
    tool_config = config.get_tool_config()
    print(f"Tool configuration: {tool_config}")
    
    # Get configuration for a specific tool (may be empty if not found)
    web_search_config = config.get_tool_config("web_search")
    print(f"Web search tool config: {web_search_config}")


def example_agent_config():
    """Example of accessing agent configuration."""
    from src.config import get_config
    
    # Get the configuration
    config = get_config()
    
    # Get configuration for a specific agent
    agent_config = config.get_agent_config("coordinator")
    print(f"Coordinator agent config: {agent_config}")
    
    # Get all agent descriptions
    agent_descriptions = config.get_agent_descriptions()
    print(f"Agent descriptions: {agent_descriptions}")


def run_examples():
    """Run all configuration examples."""
    print("\n=== Basic Usage ===")
    example_basic_usage()
    
    print("\n=== Use Cases ===")
    example_use_cases()
    
    print("\n=== Model Listing ===")
    example_model_listing()
    
    print("\n=== Custom Settings ===")
    example_custom_settings()
    
    print("\n=== Tool Configuration ===")
    example_tool_config()
    
    print("\n=== Agent Configuration ===")
    example_agent_config()


if __name__ == "__main__":
    run_examples() 