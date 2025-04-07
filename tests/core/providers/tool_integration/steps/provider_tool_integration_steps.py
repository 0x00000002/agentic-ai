"""
Step definitions for provider tool integration tests.
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
from src.tools.models import ToolDefinition, ToolCall, ToolResult
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

@given('a tool definition')
def step_impl(context):
    """Create a tool definition."""
    context.tool_definition = ToolDefinition(
        name="search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    )

@given('a tool call')
def step_impl(context):
    """Create a tool call."""
    context.tool_call = ToolCall(
        id="call-123",
        name="search",
        arguments={"query": "test query"}
    )

@given('a tool result')
def step_impl(context):
    """Create a tool result."""
    context.tool_result = ToolResult(
        tool_call_id="call-123",
        name="search",
        content="Search results for test query"
    )

@given('a conversation with tool calls')
def step_impl(context):
    """Create a conversation with tool calls."""
    context.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for information about AI."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-123",
                "name": "search",
                "arguments": json.dumps({"query": "AI information"})
            }]
        }
    ]

@given('a conversation with tool results')
def step_impl(context):
    """Create a conversation with tool results."""
    context.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for information about AI."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-123",
                "name": "search",
                "arguments": json.dumps({"query": "AI information"})
            }]
        },
        {
            "role": "tool",
            "content": "Search results for AI information",
            "name": "search"
        }
    ]

@given('a streaming tool response')
def step_impl(context):
    """Create a streaming tool response."""
    context.streaming_response = [
        {"choices": [{"delta": {"content": "Based on the search results"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ", here's what I found"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "..."}, "finish_reason": "stop"}]}
    ]

@given('a tool execution error')
def step_impl(context):
    """Create a tool execution error."""
    context.tool_error = AIProviderError("Tool execution failed")
    context.tool_error.status_code = 500

@given('a tool timeout error')
def step_impl(context):
    """Create a tool timeout error."""
    context.tool_error = AIProviderError("Tool execution timed out")
    context.tool_error.status_code = 408

@when('I execute a tool call')
def step_impl(context):
    """Execute a tool call."""
    mock_response = create_mock_response("Based on the search results, here's what I found.")
    
    with patch.object(context.provider, '_make_request', return_value=mock_response):
        try:
            context.response = context.provider.generate_with_tools(
                messages=context.messages[:-1],  # Exclude the tool call
                tool_calls=[context.tool_call]
            )
            context.error = None
        except Exception as e:
            context.error = e
            context.response = None

@when('I format a tool result')
def step_impl(context):
    """Format a tool result."""
    try:
        context.formatted_result = context.provider._format_tool_result(context.tool_result)
        context.error = None
    except Exception as e:
        context.error = e
        context.formatted_result = None

@when('I validate a tool call')
def step_impl(context):
    """Validate a tool call."""
    try:
        context.is_valid = context.provider._validate_tool_call(context.tool_call)
        context.error = None
    except Exception as e:
        context.error = e
        context.is_valid = False

@when('I track tool usage')
def step_impl(context):
    """Track tool usage."""
    try:
        context.provider._track_tool_usage(context.tool_call)
        context.error = None
    except Exception as e:
        context.error = e

@when('I check tool availability')
def step_impl(context):
    """Check tool availability."""
    try:
        context.is_available = context.provider._is_tool_available(context.tool_definition)
        context.error = None
    except Exception as e:
        context.error = e
        context.is_available = False

@when('I handle a tool execution error')
def step_impl(context):
    """Handle a tool execution error."""
    with patch.object(context.provider, '_make_request', side_effect=context.tool_error):
        try:
            context.response = context.provider.generate_with_tools(
                messages=context.messages[:-1],
                tool_calls=[context.tool_call]
            )
            context.error = None
        except Exception as e:
            context.error = e
            context.response = None

@when('I make a streaming tool request')
def step_impl(context):
    """Make a streaming tool request."""
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = context.streaming_response
    
    with patch.object(context.provider, '_make_streaming_request', return_value=mock_stream):
        try:
            context.streaming_response = asyncio.run(context.provider.generate_streaming_with_tools(
                messages=context.messages[:-1],
                tool_calls=[context.tool_call]
            ))
            context.error = None
        except Exception as e:
            context.error = e
            context.streaming_response = None

@when('I handle a tool timeout')
def step_impl(context):
    """Handle a tool timeout."""
    with patch.object(context.provider, '_make_request', side_effect=context.tool_error):
        try:
            context.response = context.provider.generate_with_tools(
                messages=context.messages[:-1],
                tool_calls=[context.tool_call],
                timeout=1
            )
            context.error = None
        except Exception as e:
            context.error = e
            context.response = None

@then('the tool call should be executed correctly')
def step_impl(context):
    """Verify the tool call is executed correctly."""
    assert context.response is not None
    assert isinstance(context.response, ProviderResponse)
    assert context.response.content == "Based on the search results, here's what I found."

@then('the tool result should be formatted correctly')
def step_impl(context):
    """Verify the tool result is formatted correctly."""
    assert context.formatted_result is not None
    
    if context.provider_type == "openai":
        assert "role" in context.formatted_result
        assert context.formatted_result["role"] == "tool"
    elif context.provider_type == "anthropic":
        assert "type" in context.formatted_result
        assert context.formatted_result["type"] == "tool_result"
    elif context.provider_type == "gemini":
        assert "content" in context.formatted_result
    elif context.provider_type == "ollama":
        assert "role" in context.formatted_result
        assert context.formatted_result["role"] == "tool"

@then('the tool call should be valid')
def step_impl(context):
    """Verify the tool call is valid."""
    assert context.is_valid
    assert context.error is None

@then('the tool usage should be tracked')
def step_impl(context):
    """Verify the tool usage is tracked."""
    assert context.error is None
    assert context.provider.logger.info.called
    info_message = context.provider.logger.info.call_args[0][0]
    assert "tool usage" in info_message.lower()
    assert context.tool_call.name in info_message

@then('the tool should be available')
def step_impl(context):
    """Verify the tool is available."""
    assert context.is_available
    assert context.error is None

@then('the tool execution error should be handled')
def step_impl(context):
    """Verify the tool execution error is handled."""
    assert context.error is not None
    assert isinstance(context.error, AIProviderError)
    assert context.error.status_code == 500
    assert context.provider.logger.error.called
    error_message = context.provider.logger.error.call_args[0][0]
    assert "tool execution" in error_message.lower()

@then('the streaming tool response should be handled')
def step_impl(context):
    """Verify the streaming tool response is handled."""
    assert context.streaming_response is not None
    assert len(context.streaming_response) == 3
    assert context.streaming_response[-1]["choices"][0]["finish_reason"] == "stop"

@then('the tool timeout should be handled')
def step_impl(context):
    """Verify the tool timeout is handled."""
    assert context.error is not None
    assert isinstance(context.error, AIProviderError)
    assert context.error.status_code == 408
    assert context.provider.logger.error.called
    error_message = context.provider.logger.error.call_args[0][0]
    assert "timeout" in error_message.lower() 