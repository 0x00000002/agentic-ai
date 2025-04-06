#!/usr/bin/env python
"""
Simple test script for the new configuration system.
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
    from src.config import configure, get_config, get_available_models, Model, UseCasePreset
    
    # List available models
    logger.info("Available models:")
    for model in get_available_models():
        logger.info(f"- {model}")
    
    # Configure with a specific model and use case
    logger.info("\nConfiguring with Claude 3.5 Sonnet and solidity_coding use case")
    configure(
        model="claude-3-5-sonnet",
        use_case=UseCasePreset.SOLIDITY_CODING,
        temperature=0.8,
        show_thinking=True
    )
    
    # Get the configuration instance
    config = get_config()
    
    # Show the configuration
    logger.info(f"Default model: {config.get_default_model()}")
    
    # Get model configuration
    model_config = config.get_model_config()
    logger.info(f"Model details: {model_config.get('name')} ({model_config.get('provider')})")
    logger.info(f"Model quality: {model_config.get('quality')}")
    logger.info(f"Model speed: {model_config.get('speed')}")
    
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
    
    # Test with custom kwargs
    logger.info("\nConfiguring with custom kwargs")
    configure(
        max_tokens=2000,
        custom_setting="value"
    )
    
    # Show thinking setting
    logger.info(f"Show thinking: {config.show_thinking}")
    
    # Use assert statement instead of returning True
    assert config is not None
    
def main():
    """Run the configuration test."""
    logger.info("Testing configuration system")
    
    success = test_configuration()
    if success is None:  # Test functions returning None is the expected behavior
        logger.info("✅ Configuration system test passed")
    else:
        logger.error("❌ Configuration system test failed")

if __name__ == "__main__":
    main() 