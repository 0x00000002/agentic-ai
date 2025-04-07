"""
Step definitions for provider implementation tests.
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
from src.exceptions import AIProviderError

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

@given('an Anthropic provider instance')
def step_impl(context):
    """Create an Anthropic provider instance."""
    context.provider = create_provider_instance(
        provider_type="anthropic",
        model_id="claude-3-opus",
        unified_config=context.unified_config,
        logger=create_mock_logger()
    )
    context.provider_type = "anthropic"

@given('a Gemini provider instance')
def step_impl(context):
    """Create a Gemini provider instance."""
    context.provider = create_provider_instance(
        provider_type="gemini",
        model_id="gemini-pro",
        unified_config=context.unified_config,
        logger=create_mock_logger()
    )
    context.provider_type = "gemini"

@given('an Ollama provider instance')
def step_impl(context):
    """Create an Ollama provider instance."""
    context.provider = create_provider_instance(
        provider_type="ollama",
        model_id="llama3",
        unified_config=context.unified_config,
        logger=create_mock_logger()
    )
    context.provider_type = "ollama"

@given('a system message')
def step_impl(context):
    """Set up a system message."""
    context.system_message = "You are a helpful assistant."

@given('a user message')
def step_impl(context):
    """Set up a user message."""
    context.user_message = "Hello, how are you?"

@given('a conversation with system and user messages')
def step_impl(context):
    """Set up a conversation with system and user messages."""
    context.messages = [
        {"role": "system", "content": context.system_message},
        {"role": "user", "content": context.user_message}
    ]

@given('a tool call')
def step_impl(context):
    """Set up a tool call."""
    context.tool_call = {
        "name": "search",
        "arguments": json.dumps({"query": "test query"})
    }

@given('a tool result')
def step_impl(context):
    """Set up a tool result."""
    context.tool_result = {
        "name": "search",
        "content": "Search results for test query"
    }

@given('a conversation with tool calls and results')
def step_impl(context):
    """Set up a conversation with tool calls and results."""
    context.messages = [
        {"role": "system", "content": context.system_message},
        {"role": "user", "content": context.user_message},
        {"role": "assistant", "content": None, "tool_calls": [context.tool_call]},
        {"role": "tool", "content": context.tool_result["content"], "name": context.tool_result["name"]}
    ]

@given('a streaming response')
def step_impl(context):
    """Set up a streaming response."""
    context.streaming_response = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ", "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]}
    ]

@given('a multi-part content response')
def step_impl(context):
    """Set up a multi-part content response."""
    context.multi_part_response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello, "},
                        {"type": "image", "image_url": {"url": "https://example.com/image.jpg"}},
                        {"type": "text", "text": "world!"}
                    ],
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ]
    }

@when('I make a request with system and user messages')
def step_impl(context):
    """Make a request with system and user messages."""
    # Mock the provider's request method
    mock_response = create_mock_response("Hello, how are you?")
    
    with patch.object(context.provider, '_make_request', return_value=mock_response):
        try:
            context.response = context.provider.generate_with_system(
                system_message=context.system_message,
                prompt=context.user_message
            )
            context.error = None
        except Exception as e:
            context.error = e
            context.response = None

@when('I format tool calls')
def step_impl(context):
    """Format tool calls."""
    try:
        context.formatted_tool_calls = context.provider._format_tool_calls([context.tool_call])
        context.error = None
    except Exception as e:
        context.error = e
        context.formatted_tool_calls = None

@when('I format tool results')
def step_impl(context):
    """Format tool results."""
    try:
        context.formatted_tool_results = context.provider._format_tool_results([context.tool_result])
        context.error = None
    except Exception as e:
        context.error = e
        context.formatted_tool_results = None

@when('I make a request with tool calls and results')
def step_impl(context):
    """Make a request with tool calls and results."""
    # Mock the provider's request method
    mock_response = create_mock_response("Based on the search results, here's what I found.")
    
    with patch.object(context.provider, '_make_request', return_value=mock_response):
        try:
            context.response = context.provider.generate_with_tools(
                messages=context.messages[:-2],  # Exclude the tool call and result
                tool_calls=[context.tool_call],
                tool_results=[context.tool_result]
            )
            context.error = None
        except Exception as e:
            context.error = e
            context.response = None

@when('I make a streaming request')
def step_impl(context):
    """Make a streaming request."""
    # Mock the provider's streaming request method
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = context.streaming_response
    
    with patch.object(context.provider, '_make_streaming_request', return_value=mock_stream):
        try:
            context.streaming_response = asyncio.run(context.provider.generate_streaming(
                messages=context.messages
            ))
            context.error = None
        except Exception as e:
            context.error = e
            context.streaming_response = None

@when('I convert a response')
def step_impl(context):
    """Convert a response."""
    mock_response = create_mock_response("Hello, world!")
    
    try:
        context.converted_response = context.provider._convert_response(mock_response)
        context.error = None
    except Exception as e:
        context.error = e
        context.converted_response = None

@when('I handle a streaming response')
def step_impl(context):
    """Handle a streaming response."""
    try:
        context.handled_streaming = context.provider._handle_streaming(context.streaming_response)
        context.error = None
    except Exception as e:
        context.error = e
        context.handled_streaming = None

@when('I convert a multi-part content response')
def step_impl(context):
    """Convert a multi-part content response."""
    try:
        context.converted_multi_part = context.provider._convert_response(context.multi_part_response)
        context.error = None
    except Exception as e:
        context.error = e
        context.converted_multi_part = None

@then('the provider should handle the system message correctly')
def step_impl(context):
    """Verify the provider handles the system message correctly."""
    assert context.response is not None
    assert hasattr(context.response, 'content')
    assert context.response.content == "Hello, how are you?"

@then('the provider should format tool calls correctly')
def step_impl(context):
    """Verify the provider formats tool calls correctly."""
    assert context.formatted_tool_calls is not None
    
    if context.provider_type == "openai":
        assert "function" in context.formatted_tool_calls[0]
        assert context.formatted_tool_calls[0]["function"]["name"] == "search"
    elif context.provider_type == "anthropic":
        assert "type" in context.formatted_tool_calls[0]
        assert context.formatted_tool_calls[0]["type"] == "function"
    elif context.provider_type == "gemini":
        assert "function_declarations" in context.formatted_tool_calls
    elif context.provider_type == "ollama":
        assert "name" in context.formatted_tool_calls[0]
        assert context.formatted_tool_calls[0]["name"] == "search"

@then('the provider should format tool results correctly')
def step_impl(context):
    """Verify the provider formats tool results correctly."""
    assert context.formatted_tool_results is not None
    
    if context.provider_type == "openai":
        assert "role" in context.formatted_tool_results[0]
        assert context.formatted_tool_results[0]["role"] == "tool"
    elif context.provider_type == "anthropic":
        assert "type" in context.formatted_tool_results[0]
        assert context.formatted_tool_results[0]["type"] == "tool_result"
    elif context.provider_type == "gemini":
        assert "content" in context.formatted_tool_results[0]
    elif context.provider_type == "ollama":
        assert "role" in context.formatted_tool_results[0]
        assert context.formatted_tool_results[0]["role"] == "tool"

@then('the provider should handle tool calls and results correctly')
def step_impl(context):
    """Verify the provider handles tool calls and results correctly."""
    assert context.response is not None
    assert hasattr(context.response, 'content')
    assert context.response.content == "Based on the search results, here's what I found."

@then('the provider should handle streaming correctly')
def step_impl(context):
    """Verify the provider handles streaming correctly."""
    assert context.streaming_response is not None
    assert len(context.streaming_response) == 4
    assert context.streaming_response[-1]["choices"][0]["finish_reason"] == "stop"

@then('the provider should convert the response correctly')
def step_impl(context):
    """Verify the provider converts the response correctly."""
    assert context.converted_response is not None
    assert isinstance(context.converted_response, ProviderResponse)
    assert context.converted_response.content == "Hello, world!"

@then('the provider should handle the streaming response correctly')
def step_impl(context):
    """Verify the provider handles the streaming response correctly."""
    assert context.handled_streaming is not None
    assert len(context.handled_streaming) == 4
    assert context.handled_streaming[-1]["choices"][0]["finish_reason"] == "stop"

@then('the provider should convert the multi-part content correctly')
def step_impl(context):
    """Verify the provider converts the multi-part content correctly."""
    assert context.converted_multi_part is not None
    assert isinstance(context.converted_multi_part, ProviderResponse)
    assert "Hello, " in context.converted_multi_part.content
    assert "world!" in context.converted_multi_part.content
    assert "https://example.com/image.jpg" in context.converted_multi_part.content 