import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
import json
from enum import Enum

from src.core.base_ai import AIBase
from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ProviderResponse, ToolCall
from src.tools.models import ToolResult
# Import necessary exceptions
from src.exceptions import AIProviderError, AIProcessingError

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


class TestToolEnabledAIErrorHandling:
    """Tests for error handling in ToolEnabledAI."""

    @pytest.mark.asyncio
    async def test_provider_request_raises_exception(self, tool_enabled_ai, mock_provider,
                                                  mock_convo_manager):
        """Test handling when mock_provider.request raises an exception, returning error message."""
        # Setup
        prompt = "Test provider exception"
        mock_provider.supports_tools = True
        provider_error_msg = "Provider communication failed"
        provider_error = AIProviderError(provider_error_msg)
        mock_provider.request.side_effect = provider_error

        # Execute - process_prompt should catch the error and return a formatted string
        response = await tool_enabled_ai.process_prompt(prompt)
        
        # Verify error message is in the response (including PROVIDER prefix)
        assert f"[Error processing request: PROVIDER: {provider_error_msg}]" in response

        # Verify provider was called once (no retry success)
        mock_provider.request.assert_called_once()
        # Verify conversation state
        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_has_calls([
            call(role="system", content=ANY),
            call(role="user", content=prompt)
        ], any_order=False)
        assistant_calls = [
            c for c in mock_convo_manager.add_message.call_args_list
            if c.kwargs.get('role') == 'assistant'
        ]
        assert not assistant_calls
        assert tool_enabled_ai._tool_history == []


    @pytest.mark.asyncio
    async def test_provider_returns_error_in_response(self, tool_enabled_ai, mock_provider,
                                                     mock_convo_manager):
        """Test handling when provider.request returns a ProviderResponse with an error."""
        # Setup
        prompt = "Test provider response error"
        mock_provider.supports_tools = True
        error_response = ProviderResponse(error="Internal provider error")
        mock_provider.request.return_value = error_response

        # Execute and verify
        response = await tool_enabled_ai.process_prompt(prompt)

        assert "[Error from provider: Internal provider error]" in response

        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_has_calls([
            call(role="system", content=ANY),
            call(role="user", content=prompt)
        ], any_order=False)
        assistant_calls = [
            c for c in mock_convo_manager.add_message.call_args_list
            if c.kwargs.get('role') == 'assistant'
        ]
        assert not assistant_calls
        assert tool_enabled_ai._tool_history == []


    @pytest.mark.asyncio
    async def test_streaming_provider_request_raises_exception(self, tool_enabled_ai, mock_provider,
                                                 mock_convo_manager):
        """Test handling when provider.request raises error during streaming=True (returns error msg)."""
        # Setup
        prompt = "Test streaming provider exception"
        mock_provider.supports_tools = True
        provider_error_msg = "Streaming provider communication failed"
        provider_error = AIProviderError(provider_error_msg)
        mock_provider.request.side_effect = provider_error

        # Execute - process_prompt should catch the error and return a formatted string
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)

        # Verify error message is in the response (including PROVIDER prefix)
        assert f"[Error processing request: PROVIDER: {provider_error_msg}]" in response

        # Verify provider was called once
        mock_provider.request.assert_called_once()
        # Verify conversation state
        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_has_calls([
            call(role="system", content=ANY),
            call(role="user", content=prompt)
        ], any_order=False)
        assistant_calls = [
            c for c in mock_convo_manager.add_message.call_args_list
            if c.kwargs.get('role') == 'assistant'
        ]
        assert not assistant_calls
        assert tool_enabled_ai._tool_history == []

    @pytest.mark.asyncio
    async def test_tool_manager_not_initialized_raises_value_error(self, tool_enabled_ai, mock_provider,
                                              mock_convo_manager):
        """Test ValueError is raised if tool manager is None during tool loop."""
        # Setup
        prompt = "Test with no tool manager"
        mock_provider.supports_tools = True

        mock_tool_call = ToolCall(id="call_abc123", name="test_tool", arguments={"param1": "value"})
        provider_response = ProviderResponse(content=None, tool_calls=[mock_tool_call])
        mock_provider.request.return_value = provider_response
        tool_enabled_ai._tool_manager = None

        with pytest.raises(ValueError, match="Tool manager not initialized"):
            await tool_enabled_ai.process_prompt(prompt)

        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_has_calls([
            call(role="system", content=ANY),
            call(role="user", content=prompt)
        ], any_order=False)

    @pytest.mark.asyncio
    async def test_tool_not_found_in_tool_manager(self, tool_enabled_ai, mock_provider,
                                mock_convo_manager, mock_tool_manager):
        """Test handling when a tool name from provider is not found in ToolManager."""
        # Setup
        prompt = "Test tool not found"
        mock_provider.supports_tools = True

        # Mock provider response requesting a non-existent tool
        mock_tool_call = ToolCall(id="call_abc123", name="nonexistent_tool", arguments={"param1": "value"})
        provider_response_1 = ProviderResponse(content=None, tool_calls=[mock_tool_call])

        # Mock ToolManager's execute_tool to simulate "not found" error internally (as ToolManager handles this)
        # ToolManager's execute_tool should return a ToolResult with success=False
        tool_not_found_result = ToolResult(success=False, error="Tool not found: nonexistent_tool", tool_name="nonexistent_tool")
        mock_tool_manager.execute_tool.return_value = tool_not_found_result
        
        # Mock provider response for the *second* call (after submitting tool results)
        final_response_content = "Okay, I couldn't find that tool."
        provider_response_2 = ProviderResponse(content=final_response_content)
        
        mock_provider.request.side_effect = [provider_response_1, provider_response_2]
        
        # Mock provider's ability to format the error result back into a message
        # Assume _add_tool_message returns a list of dicts for history
        mock_provider._add_tool_message = MagicMock(return_value=[{
            "role": "tool", 
            "tool_call_id": "call_abc123", 
            "content": tool_not_found_result.error # Content is the error message
        }])

        # Execute
        final_response = await tool_enabled_ai.process_prompt(prompt)

        # Verify
        assert mock_provider.request.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(tool_name="nonexistent_tool", param1="value", request_id=tool_enabled_ai._request_id)
        
        # Check provider was called to add the tool result message
        mock_provider._add_tool_message.assert_called_once()
        add_tool_args = mock_provider._add_tool_message.call_args[1]
        assert add_tool_args['tool_call_id'] == "call_abc123"
        assert tool_not_found_result.error in add_tool_args['content']

        # Check final response is from the second provider call
        assert final_response == final_response_content
        
        # Check tool history has the error entry
        assert len(tool_enabled_ai._tool_history) == 1
        history_entry = tool_enabled_ai._tool_history[0]
        assert history_entry['tool_name'] == "nonexistent_tool"
        assert history_entry['error'] == tool_not_found_result.error
        assert history_entry['result'] is None
        assert history_entry['tool_call_id'] == "call_abc123"


    @pytest.mark.asyncio
    async def test_invalid_tool_call_format_from_provider(self, tool_enabled_ai, mock_provider,
                                          mock_convo_manager, mock_tool_manager):
        """Test handling when provider returns something not matching ToolCall model."""
        # Setup
        prompt = "Test invalid tool call format"
        mock_provider.supports_tools = True
        invalid_tool_call_dict = {"id": "call_invalid", "arguments": {"p": 1}}
        mock_response_with_invalid_tool_call = MagicMock(spec=ProviderResponse)
        mock_response_with_invalid_tool_call.content = "Content before invalid tool call"
        mock_response_with_invalid_tool_call.tool_calls = [invalid_tool_call_dict]
        mock_response_with_invalid_tool_call.error = None
        # Add stop_reason attribute to avoid subsequent error
        mock_response_with_invalid_tool_call.stop_reason = "mocked_stop"
        
        mock_provider.request.side_effect = [mock_response_with_invalid_tool_call]
        # This is the error logged when the format is bad
        logged_error_content = "Invalid tool call format from provider: item is not a ToolCall object."
        # This is the error message potentially returned if accessing response fails later (like stop_reason)
        # We expect the loop to break and return based on the *first* response content now.
        mock_provider._add_tool_message = MagicMock(return_value=[{
            "role": "tool", "tool_call_id": "call_invalid", "content": logged_error_content
        }])

        # Execute
        final_response = await tool_enabled_ai.process_prompt(prompt)

        # Verify - Expect only one provider call because loop breaks
        mock_provider.request.assert_called_once()
        assert not mock_tool_manager.execute_tool.called
        # Check the final response - should be the error message from the internal exception
        assert "[Error processing request: 'dict' object has no attribute 'id']" in final_response
        # Check tool history - Should be empty as error happens before execution/history append
        assert tool_enabled_ai._tool_history == []


    @pytest.mark.asyncio
    async def test_tool_execution_raises_exception(self, tool_enabled_ai, mock_provider,
                                             mock_convo_manager, mock_tool_manager):
        """Test handling when ToolManager.execute_tool raises an unexpected exception."""
        # Setup
        prompt = "Test tool execution exception"
        mock_provider.supports_tools = True

        # Mock provider response requesting a valid tool
        mock_tool_call = ToolCall(id="call_exec_fail", name="failing_tool", arguments={"p": "go"})
        provider_response_1 = ProviderResponse(content=None, tool_calls=[mock_tool_call])

        # Mock ToolManager's execute_tool to raise an exception
        execution_exception = Exception("Core tool logic failed!")
        mock_tool_manager.execute_tool.side_effect = execution_exception

        # Mock provider response for the second call
        final_response_content = "There was an error running the tool."
        provider_response_2 = ProviderResponse(content=final_response_content)

        mock_provider.request.side_effect = [provider_response_1, provider_response_2]
        
        # Mock provider's _add_tool_message for the error result
        # The error is caught in _execute_tool_call and returned as ToolResult(success=False)
        error_content = f"Unexpected error in ToolEnabledAI._execute_tool_call: {execution_exception}"
        mock_provider._add_tool_message = MagicMock(return_value=[{
            "role": "tool",
            "tool_call_id": "call_exec_fail",
            "content": error_content
        }])

        # Execute
        final_response = await tool_enabled_ai.process_prompt(prompt)

        # Verify
        assert mock_provider.request.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(tool_name="failing_tool", p="go", request_id=tool_enabled_ai._request_id)

        # Check provider was called to add the error message
        mock_provider._add_tool_message.assert_called_once()
        add_tool_args = mock_provider._add_tool_message.call_args[1]
        assert add_tool_args['tool_call_id'] == "call_exec_fail"
        assert error_content in add_tool_args['content']

        assert final_response == final_response_content

        # Check tool history
        assert len(tool_enabled_ai._tool_history) == 1
        history_entry = tool_enabled_ai._tool_history[0]
        assert history_entry['tool_name'] == "failing_tool"
        assert error_content in history_entry['error'] # Error generated in _execute_tool_call
        assert history_entry['result'] is None
        assert history_entry['tool_call_id'] == "call_exec_fail"


    @pytest.mark.asyncio
    async def test_non_tool_provider_falls_back_to_base_request(self, tool_enabled_ai, mock_provider,
                                                  mock_convo_manager):
        """Test process_prompt uses provider.request via AIBase if provider doesn't support tools."""
        # Setup
        prompt = "Test non-tool provider fallback"
        mock_provider.supports_tools = False
        expected_response_content = "Response from basic AI request"
        
        # Configure the mock provider's request method directly
        mock_provider.request.return_value = ProviderResponse(content=expected_response_content)

        # Configure convo manager to return expected messages when get_messages is called
        expected_messages = [
            {"role": "system", "content": ANY}, # Assuming system message from fixture
            {"role": "user", "content": prompt}
        ]
        mock_convo_manager.get_messages.return_value = expected_messages

        # Execute
        response = await tool_enabled_ai.process_prompt(prompt, temperature=0.5)

        # Verify the provider's request method was called directly (via AIBase.request)
        mock_provider.request.assert_called_once()
        args, kwargs = mock_provider.request.call_args
        # Check messages were passed to the provider as a KEYWORD argument
        assert not args # Ensure args is empty
        assert 'messages' in kwargs
        messages_arg = kwargs['messages']
        assert isinstance(messages_arg, list) # Check the kwarg is a list (messages)
        # Check system message is likely the first in the list
        assert messages_arg[0]['role'] == 'system'
        # Check user prompt is likely the last
        assert messages_arg[-1]['role'] == 'user'
        assert messages_arg[-1]['content'] == prompt
        # Check other kwargs
        assert kwargs.get('temperature') == 0.5
        # process_prompt returns the content string
        assert response == expected_response_content

        # Check conversation manager was called
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        assert tool_enabled_ai._tool_history == []