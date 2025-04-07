#!/usr/bin/env python
"""
Simple test script for the unified configuration system.
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
logger = logging.getLogger("config_test")

def test_configuration():
    """Test the configuration system."""
    from src.config import configure, get_config, UseCasePreset
    from src.config.unified_config import UnifiedConfig
    
    # Get configuration instance
    config = get_config()
    
    # List available models
    logger.info("Available models:")
    for model in config.get_model_names():
        logger.info(f"- {model}")
    
    # Configure with a specific model and use case
    logger.info("\nConfiguring with Claude 3.5 Sonnet and solidity_coding use case")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.8,
        show_thinking=True
    )
    
    # Get the configuration instance again (should be the same instance due to singleton)
    config = get_config()
    
    # Show the configuration
    logger.info(f"Default model: {config.get_default_model()}")
    
    # Get model configuration
    model_config = config.get_model_config()
    logger.info(f"Model details: {model_config.get('name', 'Unnamed')} ({model_config.get('provider', 'Unknown')})")
    logger.info(f"Model quality: {model_config.get('quality', 'Unknown')}")
    logger.info(f"Model privacy: {model_config.get('privacy', 'Unknown')}")
    
    # Get use case configuration
    use_case_config = config.get_use_case_config()
    logger.info(f"Use case configuration: {use_case_config}")
    
    # Show the system prompt
    system_prompt = config.get_system_prompt()
    logger.info(f"System prompt: {system_prompt or 'None'}")
    
    # Test with custom system prompt
    logger.info("\nConfiguring with custom system prompt")
    configure(
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development."
    )
    
    # Get the new system prompt
    system_prompt = config.get_system_prompt()
    logger.info(f"New system prompt: {system_prompt}")
    
    # Get user overrides
    logger.info(f"User overrides: {config.user_overrides}")
    
    # Test tool configuration access
    logger.info("\nAccessing tool configuration")
    tool_config = config.get_tool_config()
    logger.info(f"Tool configuration available: {bool(tool_config)}")
    
    # Test agent configuration access
    logger.info("\nAccessing agent configuration")
    agent_config = config.get_agent_config("coordinator")
    logger.info(f"Coordinator agent config: {agent_config}")
    
    # Test configuration reload
    logger.info("\nTesting configuration reload")
    config.reload()
    logger.info(f"Configuration reloaded successfully. Default model: {config.get_default_model()}")
    
    # Show thinking setting
    logger.info(f"Show thinking: {config.show_thinking}")
    
    # Test singleton behavior
    logger.info("\nTesting singleton behavior")
    another_config = UnifiedConfig.get_instance()
    assert config is another_config
    logger.info("Singleton pattern verified (same instance returned)")
    
    return None  # Success
    
def main():
    """Run the configuration test."""
    logger.info("Testing unified configuration system")
    
    success = test_configuration()
    if success is None:  # Test functions returning None is the expected behavior
        logger.info("✅ Configuration system test passed")
    else:
        logger.error("❌ Configuration system test failed")

if __name__ == "__main__":
    main() 