"""
Step definitions for provider error handling tests.
"""
from behave import given, when, then
from unittest.mock import MagicMock, patch, AsyncMock
import json
import asyncio
from typing import Dict, Any, List

from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.models import ProviderResponse
from src.exceptions import (
    AIProviderError,
    RateLimitError,
    NetworkError,
    InvalidResponseError,
    ConcurrentRequestError,
    AuthenticationError,
    TimeoutError
)

# Import shared fixtures
from tests.core.providers.fixtures import (
    create_mock_unified_config,
    create_mock_logger,
    create_provider_instance,
    create_mock_response,
    create_mock_tool_response,
    create_mock_tool_result
)

@given('the AI framework is initialized')
def step_impl(context):
    """Initialize the AI framework."""
    context.unified_config = create_mock_unified_config()

@given('provider configurations are available')
def step_impl(context):
    """Verify provider configurations are available."""
    assert context.unified_config is not None
    assert "openai" in context.unified_config.providers
    assert "anthropic" in context.unified_config.providers
    assert "gemini" in context.unified_config.providers
    assert "ollama" in context.unified_config.providers

@given('an OpenAI provider instance')
def step_impl(context):
    """Create an OpenAI provider instance."""
    context.provider = create_provider_instance(
        provider_type="openai",
        model_id="gpt-4",
        unified_config=context.unified_config,
        logger=create_mock_logger()
    )
    context.provider_type = "openai"

@given('a rate limit error is simulated')
def step_impl(context):
    """Simulate a rate limit error."""
    context.error = RateLimitError("Rate limit exceeded")
    context.error.status_code = 429
    context.error.retry_after = 60

@given('a network error is simulated')
def step_impl(context):
    """Simulate a network error."""
    context.error = NetworkError("Connection failed")
    context.error.status_code = 503

@given('an invalid response error is simulated')
def step_impl(context):
    """Simulate an invalid response error."""
    context.error = InvalidResponseError("Invalid response format")
    context.error.status_code = 400

@given('a concurrent request error is simulated')
def step_impl(context):
    """Simulate a concurrent request error."""
    context.error = ConcurrentRequestError("Too many concurrent requests")
    context.error.status_code = 429

@given('an authentication error is simulated')
def step_impl(context):
    """Simulate an authentication error."""
    context.error = AuthenticationError("Invalid API key")
    context.error.status_code = 401

@given('a timeout error is simulated')
def step_impl(context):
    """Simulate a timeout error."""
    context.error = TimeoutError("Request timed out")
    context.error.status_code = 408

@given('a provider-specific error is simulated')
def step_impl(context):
    """Simulate a provider-specific error."""
    if context.provider_type == "openai":
        context.error = AIProviderError("OpenAI specific error")
    elif context.provider_type == "anthropic":
        context.error = AIProviderError("Anthropic specific error")
    elif context.provider_type == "gemini":
        context.error = AIProviderError("Gemini specific error")
    elif context.provider_type == "ollama":
        context.error = AIProviderError("Ollama specific error")
    context.error.status_code = 500

@given('a transient failure is simulated')
def step_impl(context):
    """Simulate a transient failure."""
    context.error = NetworkError("Temporary network issue")
    context.error.status_code = 503
    context.error.is_transient = True

@given('a slow response is simulated')
def step_impl(context):
    """Simulate a slow response."""
    context.error = TimeoutError("Response too slow")
    context.error.status_code = 408
    context.error.elapsed_time = 30

@when('I make a request')
def step_impl(context):
    """Make a request that will trigger the error."""
    with patch.object(context.provider, '_make_request', side_effect=context.error):
        try:
            context.response = context.provider.generate(
                messages=[{"role": "user", "content": "Hello"}]
            )
            context.error_occurred = False
        except Exception as e:
            context.error_occurred = True
            context.caught_error = e

@when('I make a streaming request')
def step_impl(context):
    """Make a streaming request that will trigger the error."""
    mock_stream = AsyncMock()
    mock_stream.__aiter__.side_effect = context.error
    
    with patch.object(context.provider, '_make_streaming_request', return_value=mock_stream):
        try:
            context.streaming_response = asyncio.run(context.provider.generate_streaming(
                messages=[{"role": "user", "content": "Hello"}]
            ))
            context.error_occurred = False
        except Exception as e:
            context.error_occurred = True
            context.caught_error = e

@when('I make multiple concurrent requests')
def step_impl(context):
    """Make multiple concurrent requests."""
    async def make_request():
        try:
            return await context.provider.generate_async(
                messages=[{"role": "user", "content": "Hello"}]
            )
        except Exception as e:
            return e

    async def run_concurrent_requests():
        tasks = [make_request() for _ in range(5)]
        return await asyncio.gather(*tasks)

    with patch.object(context.provider, '_make_request', side_effect=context.error):
        try:
            context.responses = asyncio.run(run_concurrent_requests())
            context.error_occurred = False
        except Exception as e:
            context.error_occurred = True
            context.caught_error = e

@when('I retry the request')
def step_impl(context):
    """Retry the request after an error."""
    mock_response = create_mock_response("Hello, world!")
    
    with patch.object(context.provider, '_make_request', return_value=mock_response):
        try:
            context.response = context.provider.generate(
                messages=[{"role": "user", "content": "Hello"}]
            )
            context.error_occurred = False
        except Exception as e:
            context.error_occurred = True
            context.caught_error = e

@then('the error should be handled gracefully')
def step_impl(context):
    """Verify the error is handled gracefully."""
    assert context.error_occurred
    assert isinstance(context.caught_error, type(context.error))
    assert context.caught_error.status_code == context.error.status_code

@then('the error should be logged appropriately')
def step_impl(context):
    """Verify the error is logged appropriately."""
    assert context.provider.logger.error.called
    error_message = context.provider.logger.error.call_args[0][0]
    assert str(context.error) in error_message
    assert str(context.error.status_code) in error_message

@then('no sensitive information should be exposed')
def step_impl(context):
    """Verify no sensitive information is exposed."""
    error_message = context.provider.logger.error.call_args[0][0]
    assert "api_key" not in error_message.lower()
    assert "secret" not in error_message.lower()
    assert "token" not in error_message.lower()

@then('the error should be retried')
def step_impl(context):
    """Verify the error is retried."""
    assert not context.error_occurred
    assert context.response is not None
    assert isinstance(context.response, ProviderResponse)
    assert context.response.content == "Hello, world!"

@then('the retry should be logged')
def step_impl(context):
    """Verify the retry is logged."""
    assert context.provider.logger.info.called
    info_message = context.provider.logger.info.call_args[0][0]
    assert "retry" in info_message.lower()
    assert str(context.error.status_code) in info_message

@then('the concurrent requests should be handled')
def step_impl(context):
    """Verify concurrent requests are handled."""
    assert context.error_occurred
    assert isinstance(context.caught_error, ConcurrentRequestError)
    assert context.caught_error.status_code == 429

@then('the transient failure should be retried')
def step_impl(context):
    """Verify transient failure is retried."""
    assert not context.error_occurred
    assert context.response is not None
    assert isinstance(context.response, ProviderResponse)

@then('the slow response should be handled')
def step_impl(context):
    """Verify slow response is handled."""
    assert context.error_occurred
    assert isinstance(context.caught_error, TimeoutError)
    assert context.caught_error.status_code == 408
    assert context.caught_error.elapsed_time == 30 