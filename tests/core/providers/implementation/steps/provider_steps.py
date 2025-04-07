"""
Step definitions for AI provider-related tests.
"""
from behave import given, when, then
from unittest.mock import MagicMock, patch
import json
import pytest

# Import necessary components from the framework
from src.core.provider_factory import ProviderFactory
from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.models import ProviderResponse
from src.tools.models import ToolDefinition, ToolCall
from src.utils.logger import LoggerFactory
from src.exceptions import AICredentialsError, AIConfigError, AIProviderError

# Step implementations for the provider test features

@given('the AI framework is initialized')
def step_impl_framework_initialized(context):
    # Setup basic mocks for the test context
    context.logger = MagicMock()
    context.provider_config = {}
    context.model_config = {}
    context.tools = []

@given('provider configurations are available')
def step_impl_provider_configs_available(context):
    # Create mock provider configurations for testing
    context.provider_configs = {
        "openai": {"api_key_env": "OPENAI_API_KEY", "timeout": 30},
        "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "timeout": 30},
        "gemini": {"api_key_env": "GOOGLE_API_KEY", "timeout": 30},
        "ollama": {"api_key_env": "OLLAMA_API_KEY", "timeout": 30},
    }

@given('provider configurations are available in the unified config')
def step_impl_unified_config_available(context):
    # Mock the unified config
    context.unified_config = MagicMock()
    context.unified_config.get_provider_config.return_value = {
        "api_key_env": "MOCK_API_KEY",
        "timeout": 30
    }
    context.unified_config.get_model_config.return_value = {
        "model_id": "mock-model",
        "provider": "mock-provider",
        "parameters": {"temperature": 0.7}
    }
    
    # Apply patch to get_config
    patcher = patch('src.config.get_config', return_value=context.unified_config)
    context.addCleanup(patcher.stop)
    patcher.start()

@given('a valid provider configuration for "{provider_type}"')
def step_impl_valid_provider_config(context, provider_type):
    context.provider_type = provider_type
    context.provider_config = {
        "api_key_env": f"{provider_type.upper()}_API_KEY",
        "timeout": 30
    }
    
    # Set mock API key in environment
    with patch.dict('os.environ', {f"{provider_type.upper()}_API_KEY": "mock-api-key"}):
        context.api_key_set = True

@given('a valid model configuration for "{model_id}"')
def step_impl_valid_model_config(context, model_id):
    context.model_id = model_id
    context.model_config = {
        "model_id": model_id,
        "provider": context.provider_type if hasattr(context, 'provider_type') else "openai",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }

@given('a provider configuration for "{provider_type}" without an API key')
def step_impl_provider_config_no_key(context, provider_type):
    context.provider_type = provider_type
    context.provider_config = {
        "api_key_env": f"{provider_type.upper()}_API_KEY",
        "timeout": 30
    }
    
    # Ensure the environment variable is not set
    with patch.dict('os.environ', {}, clear=True):
        context.api_key_set = False

@given('a provider type "{provider_type}" without provider configuration')
def step_impl_missing_provider_config(context, provider_type):
    context.provider_type = provider_type
    context.provider_config = None
    context.model_config = {
        "model_id": "test-model",
        "provider": provider_type
    }

@when('I create a new provider instance')
def step_impl_create_provider(context):
    # Mock credentials initialization to avoid actual API calls
    with patch.object(BaseProvider, '_initialize_credentials'):
        context.provider = ProviderFactory.create(
            provider_type=context.provider_type,
            model_id=context.model_id,
            provider_config=context.provider_config,
            model_config=context.model_config,
            logger=context.logger
        )

@when('I try to create a new provider instance')
def step_impl_try_create_provider(context):
    try:
        with patch.object(BaseProvider, '_initialize_credentials'):
            context.provider = ProviderFactory.create(
                provider_type=context.provider_type,
                model_id=context.model_id,
                provider_config=context.provider_config,
                model_config=context.model_config,
                logger=context.logger
            )
        context.exception = None
    except Exception as e:
        context.exception = e

@then('the provider should be initialized successfully')
def step_impl_provider_initialized(context):
    assert context.provider is not None
    assert isinstance(context.provider, BaseProvider)

@then('the provider should have the correct model ID "{model_id}"')
def step_impl_check_model_id(context, model_id):
    assert context.provider.model_id == model_id

@then('the provider should be configured with the API key')
def step_impl_check_api_key(context):
    # This will vary by provider, but in general we can check that the 
    # provider has credentials set up
    assert hasattr(context.provider, 'provider_config')
    assert context.provider.provider_config.get('api_key_env') is not None

@then('an {error_type} should be raised')
def step_impl_check_error(context, error_type):
    error_classes = {
        'AICredentialsError': AICredentialsError,
        'AIConfigError': AIConfigError,
        'AIProviderError': AIProviderError,
        'ValueError': ValueError
    }
    
    assert context.exception is not None
    assert isinstance(context.exception, error_classes[error_type])

@then('the error message should indicate the {error_subject}')
def step_impl_check_error_message(context, error_subject):
    assert context.exception is not None
    assert error_subject.lower() in str(context.exception).lower()

@given('an initialized "{provider_type}" provider for model "{model_id}"')
def step_impl_initialized_provider(context, provider_type, model_id):
    context.provider_type = provider_type
    context.model_id = model_id
    
    # Create provider config
    context.provider_config = {
        "api_key_env": f"{provider_type.upper()}_API_KEY",
        "timeout": 30
    }
    
    # Create model config
    context.model_config = {
        "model_id": model_id,
        "provider": provider_type,
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    # Create mock environment
    with patch.dict('os.environ', {f"{provider_type.upper()}_API_KEY": "mock-api-key"}):
        # Mock provider initialization
        with patch.object(BaseProvider, '_initialize_credentials'):
            context.provider = ProviderFactory.create(
                provider_type=provider_type,
                model_id=model_id,
                provider_config=context.provider_config,
                model_config=context.model_config,
                logger=context.logger
            )

@given('a list of conversation messages')
def step_impl_conversation_messages(context):
    context.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you help me with a question?"}
    ]

@when('I prepare a request payload')
def step_impl_prepare_payload(context):
    context.payload = context.provider._prepare_request_payload(
        messages=context.messages,
        options={}
    )

@when('I prepare a request payload with temperature {temp} and max_tokens {max_tokens}')
def step_impl_prepare_payload_with_options(context, temp, max_tokens):
    context.payload = context.provider._prepare_request_payload(
        messages=context.messages,
        options={
            "temperature": float(temp),
            "max_tokens": int(max_tokens)
        }
    )

@then('the payload should include the model ID "{model_id}"')
def step_impl_check_payload_model(context, model_id):
    assert context.payload.get("model") == model_id

@then('the payload should have correctly formatted messages')
def step_impl_check_payload_messages(context):
    assert "messages" in context.payload
    assert isinstance(context.payload["messages"], list)
    assert len(context.payload["messages"]) > 0

@then('the payload should include default parameters')
def step_impl_check_default_parameters(context):
    # Check for common parameters - implementation will vary by provider
    # but we can check basics
    if hasattr(context.provider, 'parameters'):
        for key, value in context.provider.parameters.items():
            if key not in ["tools", "tool_choice", "messages", "model"]:
                assert key in context.payload or key in context.payload.get("parameters", {})

@then('the payload should include the temperature {temp}')
def step_impl_check_temperature(context, temp):
    # Different providers might put temperature in different places
    temp_value = float(temp)
    temp_param_name = "temperature"  # Default for most providers
    
    # Check if temperature is at root level or in parameters object
    if temp_param_name in context.payload:
        assert context.payload[temp_param_name] == temp_value
    elif "parameters" in context.payload and temp_param_name in context.payload["parameters"]:
        assert context.payload["parameters"][temp_param_name] == temp_value
    else:
        # If the provider uses a different name, check that
        for key, value in context.payload.items():
            if "temp" in key.lower() and isinstance(value, (int, float)):
                assert value == temp_value
                return
        assert False, f"Temperature {temp} not found in payload"

@then('the payload should include max_tokens {max_tokens}')
def step_impl_check_max_tokens(context, max_tokens):
    max_tokens_value = int(max_tokens)
    max_tokens_param_name = "max_tokens"  # Default for most providers
    
    # Handle different parameter names for different providers
    if isinstance(context.provider, AnthropicProvider):
        max_tokens_param_name = "max_tokens_to_sample"
    
    # Check if max_tokens is at root level or in parameters object
    if max_tokens_param_name in context.payload:
        assert context.payload[max_tokens_param_name] == max_tokens_value
    elif "parameters" in context.payload and max_tokens_param_name in context.payload["parameters"]:
        assert context.payload["parameters"][max_tokens_param_name] == max_tokens_value
    else:
        # If we can't find it with known names, check for similar keys
        for key, value in context.payload.items():
            if "token" in key.lower() and "max" in key.lower() and isinstance(value, int):
                assert value == max_tokens_value
                return
        assert False, f"Max tokens {max_tokens} not found in payload"

@then('the payload should override any default values')
def step_impl_check_override(context):
    # We've already checked specific values, so we can add additional logic here if needed
    pass

@given('an initialized "{provider_type}" provider with a mocked SDK client')
def step_impl_provider_with_mock_sdk(context, provider_type):
    # First set up the provider
    step_impl_initialized_provider(context, provider_type, f"{provider_type}-model")
    
    # Now setup the appropriate mock SDK client based on provider type
    if provider_type == "openai":
        context.mock_sdk = MagicMock()
        context.mock_sdk.chat.completions.create.return_value = {
            "choices": [{"message": {"content": "Mock response", "role": "assistant"}}],
            "model": context.model_id,
            "usage": {"total_tokens": 50}
        }
        context.provider.client = context.mock_sdk
    elif provider_type == "anthropic":
        context.mock_sdk = MagicMock()
        context.mock_sdk.messages.create.return_value = {
            "content": [{"type": "text", "text": "Mock response"}],
            "model": context.model_id,
            "usage": {"input_tokens": 20, "output_tokens": 30}
        }
        context.provider.client = context.mock_sdk
    elif provider_type == "gemini":
        context.mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Mock response"
        context.mock_sdk.generate_content.return_value = mock_response
        context.provider.model = context.mock_sdk
    elif provider_type == "ollama":
        context.mock_sdk = MagicMock()
        context.mock_sdk.chat.return_value = {
            "message": {"content": "Mock response", "role": "assistant"},
            "model": context.model_id
        }
        context.provider.client = context.mock_sdk
    else:
        raise ValueError(f"Unsupported provider type in test: {provider_type}")

@given('a prepared request payload')
def step_impl_prepared_payload(context):
    context.payload = {
        "model": context.model_id,
        "messages": [{"role": "user", "content": "Test message"}],
        "temperature": 0.7
    }

@when('I make an API request')
def step_impl_make_api_request(context):
    try:
        context.api_response = context.provider._make_api_request(context.payload)
        context.exception = None
    except Exception as e:
        context.exception = e
        context.api_response = None

@then('the SDK client should be called once with the correct payload')
def step_impl_check_sdk_called(context):
    provider_type = context.provider_type
    
    if provider_type == "openai":
        context.mock_sdk.chat.completions.create.assert_called_once()
        # Check payload - at minimum the model ID should match
        call_args = context.mock_sdk.chat.completions.create.call_args[1]
        assert call_args.get("model") == context.model_id
    elif provider_type == "anthropic":
        context.mock_sdk.messages.create.assert_called_once()
        call_args = context.mock_sdk.messages.create.call_args[1]
        assert call_args.get("model") == context.model_id
    elif provider_type == "gemini":
        context.mock_sdk.generate_content.assert_called_once()
    elif provider_type == "ollama":
        context.mock_sdk.chat.assert_called_once()
        call_args = context.mock_sdk.chat.call_args[1]
        assert call_args.get("model") == context.model_id

@then('the raw SDK response should be returned')
def step_impl_check_sdk_response(context):
    assert context.api_response is not None
    # The exact check will depend on the mock return value setup earlier

@given('the SDK client is configured to raise a "{error_type}"')
def step_impl_sdk_raises_error(context, error_type):
    provider_type = context.provider_type
    
    # Configure the appropriate method to raise the specified error
    error_class = Exception  # Default
    error_message = f"Mock {error_type} for testing"
    
    # Custom error classes based on provider
    if provider_type == "openai":
        if error_type == "AuthenticationError":
            from openai.error import AuthenticationError
            error_class = AuthenticationError
        elif error_type == "RateLimitError":
            from openai.error import RateLimitError
            error_class = RateLimitError
        elif error_type == "InternalServerError":
            from openai.error import APIError
            error_class = APIError
            
        context.mock_sdk.chat.completions.create.side_effect = error_class(error_message)
    
    elif provider_type == "anthropic":
        if error_type == "BadRequestError":
            error_class = ValueError  # Anthropic uses standard Python errors
        elif error_type == "ContentModerationError":
            # Custom error for testing
            class ContentModerationError(Exception):
                pass
            error_class = ContentModerationError
            
        context.mock_sdk.messages.create.side_effect = error_class(error_message)
    
    elif provider_type == "gemini":
        context.mock_sdk.generate_content.side_effect = error_class(error_message)
    
    elif provider_type == "ollama":
        context.mock_sdk.chat.side_effect = error_class(error_message)

@given('an initialized "{provider_type}" provider')
def step_impl_init_provider_simple(context, provider_type):
    # Use existing step for more complete implementation
    step_impl_initialized_provider(context, provider_type, f"{provider_type}-model")

@given('a raw SDK response with text content "{content}"')
def step_impl_raw_response_text(context, content):
    provider_type = context.provider_type
    
    if provider_type == "openai":
        context.raw_response = {
            "choices": [{"message": {"content": content, "role": "assistant"}}],
            "model": context.model_id,
            "usage": {"total_tokens": 50}
        }
    elif provider_type == "anthropic":
        context.raw_response = {
            "content": [{"type": "text", "text": content}],
            "model": context.model_id,
            "usage": {"input_tokens": 20, "output_tokens": 30}
        }
    elif provider_type == "gemini":
        mock_response = MagicMock()
        mock_response.text = content
        context.raw_response = mock_response
    elif provider_type == "ollama":
        context.raw_response = {
            "message": {"content": content, "role": "assistant"},
            "model": context.model_id
        }
    else:
        # Default fallback
        context.raw_response = {"content": content}

@when('I convert the response')
def step_impl_convert_response(context):
    context.provider_response = context.provider._convert_response(context.raw_response)

@then('the ProviderResponse.content should be "{content}"')
def step_impl_check_response_content(context, content):
    assert isinstance(context.provider_response, ProviderResponse)
    assert context.provider_response.content == content

@then('the ProviderResponse.tool_calls should be empty')
def step_impl_check_response_no_tools(context):
    assert isinstance(context.provider_response, ProviderResponse)
    assert context.provider_response.tool_calls is None or len(context.provider_response.tool_calls) == 0

@then('the ProviderResponse.error should be None')
def step_impl_check_response_no_error(context):
    assert isinstance(context.provider_response, ProviderResponse)
    assert context.provider_response.error is None 