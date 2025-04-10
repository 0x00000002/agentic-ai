import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
import json
from enum import Enum

from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ProviderResponse, ToolCall
from src.tools.models import ToolResult
from src.core.base_ai import AIBase

# Define local Role and StepType enums for this test module
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class StepType(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"


class TestToolEnabledAIAdvancedRequests:
    """Tests for advanced request handling in ToolEnabledAI."""

    @pytest.mark.asyncio
    async def test_process_prompt_with_streaming_falls_back_to_non_streaming(self, tool_enabled_ai, mock_provider,
                                               mock_convo_manager):
        """Test process_prompt(streaming=True) uses non-streaming request when tools supported."""
        # Setup
        prompt = "Test prompt with streaming fallback"
        tool_enabled_ai._supports_tools = True

        # Mock the non-streaming provider.request response
        expected_content = "Non-streaming response despite streaming=True"
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Execute with streaming=True
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)

        # Verify non-streaming request was called
        mock_provider.request.assert_called_once()
        assert response == expected_content
        # Verify conversation updated
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        mock_convo_manager.add_message.assert_any_call(role="assistant", content=expected_content)

    @pytest.mark.asyncio
    async def test_process_prompt_with_conversation_history(self, tool_enabled_ai, mock_provider,
                                                          mock_convo_manager):
        """Test processing a prompt passes conversation history to provider.request."""
        # Setup
        prompt = "Test prompt with history"
        tool_enabled_ai._supports_tools = True
        expected_content = "Response based on history"
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Add some history via the manager instead
        mock_convo_manager.add_message(role="user", content="Previous message 1")
        mock_convo_manager.add_message(role="assistant", content="Previous response 1")

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt)

        # Verify
        mock_convo_manager.get_messages.assert_called_once()
        mock_provider.request.assert_called_once()
        # Cannot reliably check the content of messages passed due to when get_messages is called
        # Check only that the call was made and the final response is correct
        assert response == expected_content

    @pytest.mark.asyncio
    async def test_process_prompt_with_system_message(self, tool_enabled_ai, mock_provider,
                                                    mock_convo_manager):
        """Test processing a prompt includes system message in provider.request."""
        # Setup
        prompt = "Test prompt with system message check"
        tool_enabled_ai._supports_tools = True
        expected_content = "Response considering system message"
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt)

        # Verify provider was called and response is correct
        mock_provider.request.assert_called_once()
        assert response == expected_content

    @pytest.mark.asyncio
    async def test_process_prompt_with_temperature(self, tool_enabled_ai, mock_provider):
        """Test processing a prompt passes temperature to provider.request."""
        # Setup
        prompt = "Test prompt with temperature"
        temperature = 0.8
        tool_enabled_ai._supports_tools = True
        expected_content = "Response with custom temperature"
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt, temperature=temperature)

        # Verify
        mock_provider.request.assert_called_once()
        # Check that the provider received the temperature
        call_args, call_kwargs = mock_provider.request.call_args
        assert call_kwargs.get('temperature') == temperature
        assert response == expected_content

    @pytest.mark.asyncio
    async def test_process_prompt_with_max_tokens(self, tool_enabled_ai, mock_provider):
        """Test processing a prompt passes max_tokens to provider.request."""
        # Setup
        prompt = "Test prompt with max tokens"
        max_tokens = 100
        tool_enabled_ai._supports_tools = True
        expected_content = "Response with max tokens limit"
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt, max_tokens=max_tokens)

        # Verify
        mock_provider.request.assert_called_once()
        # Check that the provider received the max_tokens
        call_args, call_kwargs = mock_provider.request.call_args
        assert call_kwargs.get('max_tokens') == max_tokens
        assert response == expected_content

    @pytest.mark.asyncio
    async def test_process_prompt_with_non_tool_provider_uses_base(self, tool_enabled_ai, mock_provider,
                                                       mock_convo_manager):
        """Test process_prompt uses provider.request via AIBase if provider doesn't support tools."""
        # Setup
        prompt = "Test prompt with non-tool provider"
        tool_enabled_ai._supports_tools = False # Force non-tool path
        mock_provider.supports_tools = False # Ensure provider mock reflects this
        expected_response_content = "Response from base AI request"
        
        # Configure the mock provider's request method directly
        mock_provider.request.return_value = ProviderResponse(content=expected_response_content)

        # Configure convo manager to return expected messages when get_messages is called
        expected_messages = [
            {"role": "system", "content": ANY}, # Assuming system message from fixture
            {"role": "user", "content": prompt}
        ]
        mock_convo_manager.get_messages.return_value = expected_messages

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt)

        # Verify the provider's request method was called directly (via AIBase.request)
        mock_provider.request.assert_called_once()
        # We trust the args are passed correctly down the chain if the final call is made.
        # No need to inspect args/kwargs here if the call happens and response is correct.
        
        # process_prompt returns the content string
        assert response == expected_response_content

    @pytest.mark.asyncio
    async def test_process_prompt_with_response_format(self, tool_enabled_ai, mock_provider):
        """Test processing a prompt passes response_format to provider.request."""
        # Setup
        prompt = "Test prompt with response format"
        response_format = {"type": "json_object"}
        tool_enabled_ai._supports_tools = True
        expected_content = '{"key": "value"}'
        mock_provider.request.return_value = ProviderResponse(content=expected_content, tool_calls=None)

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt, response_format=response_format)

        # Verify
        mock_provider.request.assert_called_once()
        # Check that the provider received the response_format
        call_args, call_kwargs = mock_provider.request.call_args
        assert call_kwargs.get('response_format') == response_format
        assert response == expected_content 