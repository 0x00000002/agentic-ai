"""
Step definitions for provider factory tests.
"""
from behave import given, when, then
from unittest.mock import MagicMock, patch
import pytest
from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.provider_factory import ProviderFactory
from src.core.models import Model

# Import shared fixtures
from tests.core.providers.fixtures import (
    create_mock_unified_config,
    create_mock_logger,
    create_provider_instance
)

@given('the AI framework is initialized')
def step_impl(context):
    """Initialize the AI framework."""
    context.unified_config = create_mock_unified_config()

@given('provider configurations are available in the unified config')
def step_impl(context):
    """Verify provider configurations are available."""
    assert context.unified_config is not None
    assert "openai" in context.unified_config.providers
    assert "anthropic" in context.unified_config.providers
    assert "gemini" in context.unified_config.providers
    assert "ollama" in context.unified_config.providers

@given('a provider type "{provider_type}"')
def step_impl(context, provider_type):
    """Set the provider type."""
    context.provider_type = provider_type

@given('a model ID "{model_id}"')
def step_impl(context, model_id):
    """Set the model ID."""
    context.model_id = model_id

@given('valid provider and model configurations')
def step_impl(context):
    """Verify valid provider and model configurations."""
    assert context.provider_type in context.unified_config.providers
    assert context.model_id in context.unified_config.providers[context.provider_type]["models"]

@given('valid model configuration')
def step_impl(context):
    """Verify valid model configuration."""
    # This is a simplified version for the invalid provider type scenario
    context.model_config = {"max_tokens": 2000, "temperature": 0.7}

@given('a custom provider class "{provider_class}"')
def step_impl(context, provider_class):
    """Set up a custom provider class."""
    class CustomProvider(BaseProvider):
        def __init__(self, model_id, provider_config, model_config, logger=None):
            super().__init__(model_id, provider_config, model_config, logger)
            
        def _format_messages(self, messages):
            return messages
            
        def _format_tool_calls(self, tool_calls):
            return tool_calls
            
        def _format_tool_results(self, tool_results):
            return tool_results
            
        def _convert_response(self, response):
            return response
            
        def _handle_streaming(self, response):
            return response
            
    context.custom_provider_class = CustomProvider

@given('a Model enum value for "{model_name}"')
def step_impl(context, model_name):
    """Set up a Model enum value."""
    # Convert the model name to the enum format (e.g., "gpt_4o" -> Model.GPT_4O)
    model_name_upper = model_name.upper().replace("-", "_")
    context.model_enum = getattr(Model, model_name_upper)
    context.model_id = model_name

@given('a custom logger instance')
def step_impl(context):
    """Set up a custom logger instance."""
    context.custom_logger = create_mock_logger()

@when('I create a provider using the factory')
def step_impl(context):
    """Create a provider using the factory."""
    try:
        context.provider = ProviderFactory.create_provider(
            provider_type=context.provider_type,
            model_id=context.model_id,
            unified_config=context.unified_config
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.provider = None

@when('I try to create a provider using the factory')
def step_impl(context):
    """Try to create a provider using the factory."""
    try:
        context.provider = ProviderFactory.create_provider(
            provider_type=context.provider_type,
            model_id=context.model_id,
            unified_config=context.unified_config
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.provider = None

@when('I register the custom provider with type "{provider_type}"')
def step_impl(context, provider_type):
    """Register the custom provider with the factory."""
    ProviderFactory.register_provider(provider_type, context.custom_provider_class)

@when('I create a provider of type "{provider_type}"')
def step_impl(context, provider_type):
    """Create a provider of the specified type."""
    try:
        context.provider = ProviderFactory.create_provider(
            provider_type=provider_type,
            model_id="test-model",
            unified_config=context.unified_config
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.provider = None

@when('I create a provider using the factory with the Model enum')
def step_impl(context):
    """Create a provider using the factory with the Model enum."""
    try:
        context.provider = ProviderFactory.create_provider_from_model(
            model=context.model_enum,
            unified_config=context.unified_config
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.provider = None

@when('I create a provider using the factory with the logger')
def step_impl(context):
    """Create a provider using the factory with the logger."""
    try:
        context.provider = ProviderFactory.create_provider(
            provider_type=context.provider_type,
            model_id=context.model_id,
            unified_config=context.unified_config,
            logger=context.custom_logger
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.provider = None

@then('the factory should return an instance of OpenAIProvider')
def step_impl(context):
    """Verify the factory returned an OpenAIProvider instance."""
    assert context.provider is not None
    assert isinstance(context.provider, OpenAIProvider)

@then('the factory should return an instance of AnthropicProvider')
def step_impl(context):
    """Verify the factory returned an AnthropicProvider instance."""
    assert context.provider is not None
    assert isinstance(context.provider, AnthropicProvider)

@then('the factory should return an instance of GeminiProvider')
def step_impl(context):
    """Verify the factory returned a GeminiProvider instance."""
    assert context.provider is not None
    assert isinstance(context.provider, GeminiProvider)

@then('the factory should return an instance of OllamaProvider')
def step_impl(context):
    """Verify the factory returned an OllamaProvider instance."""
    assert context.provider is not None
    assert isinstance(context.provider, OllamaProvider)

@then('the provider should be initialized with the correct model ID')
def step_impl(context):
    """Verify the provider was initialized with the correct model ID."""
    assert context.provider is not None
    assert context.provider.model_id == context.model_id

@then('a ValueError should be raised')
def step_impl(context):
    """Verify a ValueError was raised."""
    assert context.error is not None
    assert isinstance(context.error, ValueError)

@then('the error message should indicate the invalid provider type')
def step_impl(context):
    """Verify the error message indicates the invalid provider type."""
    assert context.error is not None
    assert context.provider_type in str(context.error)

@then('the factory should return an instance of MyCustomProvider')
def step_impl(context):
    """Verify the factory returned an instance of the custom provider."""
    assert context.provider is not None
    assert isinstance(context.provider, context.custom_provider_class)

@then('the provider should be initialized with the custom logger')
def step_impl(context):
    """Verify the provider was initialized with the custom logger."""
    assert context.provider is not None
    assert context.provider.logger == context.custom_logger

@then('log messages should be directed to the custom logger')
def step_impl(context):
    """Verify log messages are directed to the custom logger."""
    assert context.provider is not None
    context.provider.logger.info.assert_called() 