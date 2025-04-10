import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
import json
from enum import Enum
from typing import AsyncGenerator

from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ProviderResponse, ToolCall
from src.tools.models import ToolResult, ToolDefinition
from src.exceptions import AIProviderError

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


@pytest.fixture
def mock_tool_call_obj():
    """Fixture to create a mock ToolCall object."""
    return ToolCall(
        id="call_abc123", 
        name="test_tool", 
        arguments={"param1": "test_value"}
    )


class TestToolEnabledAIStreaming:
    """Tests for streaming behavior (or lack thereof) in ToolEnabledAI when tools are enabled."""

    @pytest.mark.asyncio
    async def test_streaming_true_uses_non_streaming_request_when_no_tools_called(self, tool_enabled_ai, mock_provider, 
                                              mock_convo_manager):
        """Test process_prompt(streaming=True) uses provider.request if tools are supported, even if no tools are called."""
        # Setup
        prompt = "Test streaming=True without tool calls"
        tool_enabled_ai._supports_tools = True # Ensure tool support is enabled

        # Mock the non-streaming provider.request response
        expected_content = "Final response without tools"
        provider_response = ProviderResponse(content=expected_content, tool_calls=None)
        mock_provider.request.return_value = provider_response
        
        # Execute process_prompt with streaming=True
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)
        
        # Verify that the non-streaming request method was called
        mock_provider.request.assert_called_once()
        # Verify streaming methods were NOT called
        assert not hasattr(mock_provider, 'get_streaming_completion_with_tools') or not mock_provider.get_streaming_completion_with_tools.called
        # Verify the final content is returned
        assert response == expected_content
        # Verify conversation history update
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        mock_convo_manager.add_message.assert_any_call(role="assistant", content=expected_content)
        assert tool_enabled_ai._tool_history == []

    @pytest.mark.asyncio
    async def test_streaming_true_uses_non_streaming_request_with_tool_call(self, tool_enabled_ai, mock_provider, 
                                          mock_convo_manager, mock_tool_manager,
                                          mock_tool_call_obj):
        """Test process_prompt(streaming=True) uses provider.request loop when a tool is called."""
        # Setup
        prompt = "Test streaming=True with tool call"
        tool_enabled_ai._supports_tools = True

        # Mock the provider responses (first with tool call, second with final content)
        provider_response_1 = ProviderResponse(content=None, tool_calls=[mock_tool_call_obj])
        final_content = "Final response after tool execution"
        provider_response_2 = ProviderResponse(content=final_content, tool_calls=None)
        mock_provider.request.side_effect = [provider_response_1, provider_response_2]

        # Mock tool execution
        tool_result_content = "Tool execution result"
        mock_tool_manager.execute_tool.return_value = ToolResult(success=True, result=tool_result_content, tool_name=mock_tool_call_obj.name)
        
        # Mock provider's ability to format the tool result message
        mock_provider._add_tool_message = MagicMock(return_value=[{
            "role": "tool", 
            "tool_call_id": mock_tool_call_obj.id, 
            "content": tool_result_content
        }])
        
        # Execute process_prompt with streaming=True
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)
        
        # Verify that the non-streaming request method was called twice
        assert mock_provider.request.call_count == 2
        # Verify streaming methods were NOT called
        assert not hasattr(mock_provider, 'get_streaming_completion_with_tools') or not mock_provider.get_streaming_completion_with_tools.called
        # Verify tool execution happened
        mock_tool_manager.execute_tool.assert_called_once_with(tool_name=mock_tool_call_obj.name, **mock_tool_call_obj.arguments, request_id=tool_enabled_ai._request_id)
        mock_provider._add_tool_message.assert_called_once()
        # Verify the final content is returned
        assert response == final_content
        assert len(tool_enabled_ai._tool_history) == 1
        assert tool_enabled_ai._tool_history[0]["result"] == tool_result_content


    @pytest.mark.asyncio
    async def test_streaming_true_uses_non_streaming_request_with_multiple_tool_calls(self, tool_enabled_ai, mock_provider, 
                                                   mock_convo_manager, mock_tool_manager,
                                                   mock_tool_call_obj):
        """Test process_prompt(streaming=True) uses provider.request loop with multiple tool calls."""
        # Setup
        prompt = "Test streaming=True with multiple tool calls"
        tool_enabled_ai._supports_tools = True

        # Create a second tool call object
        mock_tool_call_obj_2 = ToolCall(id="call_def456", name="second_tool", arguments={"p2": 100})

        # Mock the provider responses
        provider_response_1 = ProviderResponse(content="Okay, calling tools...", tool_calls=[mock_tool_call_obj, mock_tool_call_obj_2])
        final_content = "Final response after multiple tools"
        provider_response_2 = ProviderResponse(content=final_content, tool_calls=None)
        mock_provider.request.side_effect = [provider_response_1, provider_response_2]

        # Mock tool execution results
        result_1 = ToolResult(success=True, result="Result 1", tool_name=mock_tool_call_obj.name)
        result_2 = ToolResult(success=True, result="Result 2", tool_name=mock_tool_call_obj_2.name)
        # Use a side effect to return different results based on tool name
        async def execute_side_effect(tool_name, **kwargs):
            if tool_name == mock_tool_call_obj.name:
                return result_1
            elif tool_name == mock_tool_call_obj_2.name:
                return result_2
            return ToolResult(success=False, error="Unknown tool")
        mock_tool_manager.execute_tool.side_effect = execute_side_effect

        # Mock provider's ability to format tool result messages
        mock_provider._add_tool_message = MagicMock()
        mock_provider._add_tool_message.side_effect = [
             [{ "role": "tool", "tool_call_id": mock_tool_call_obj.id, "content": result_1.result }],
             [{ "role": "tool", "tool_call_id": mock_tool_call_obj_2.id, "content": result_2.result }]
        ]

        # Execute process_prompt with streaming=True
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)

        # Verify
        assert mock_provider.request.call_count == 2
        # Verify streaming methods were NOT called
        assert not hasattr(mock_provider, 'get_streaming_completion_with_tools') or not mock_provider.get_streaming_completion_with_tools.called
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(tool_name=mock_tool_call_obj.name, **mock_tool_call_obj.arguments, request_id=tool_enabled_ai._request_id)
        mock_tool_manager.execute_tool.assert_any_call(tool_name=mock_tool_call_obj_2.name, **mock_tool_call_obj_2.arguments, request_id=tool_enabled_ai._request_id)
        assert mock_provider._add_tool_message.call_count == 2 # Should be called once per tool result
        assert response == final_content
        assert len(tool_enabled_ai._tool_history) == 2
        assert tool_enabled_ai._tool_history[0]["result"] == result_1.result
        assert tool_enabled_ai._tool_history[1]["result"] == result_2.result


    @pytest.mark.asyncio
    async def test_streaming_true_uses_non_streaming_request_with_tool_error(self, tool_enabled_ai, mock_provider, 
                                                    mock_convo_manager, mock_tool_manager,
                                                    mock_tool_call_obj):
        """Test process_prompt(streaming=True) uses provider.request loop when a tool execution fails."""
        # Setup
        prompt = "Test streaming=True with tool error"
        tool_enabled_ai._supports_tools = True

        # Mock the provider responses
        provider_response_1 = ProviderResponse(content="Calling tool...", tool_calls=[mock_tool_call_obj])
        final_content = "Final response after tool error"
        provider_response_2 = ProviderResponse(content=final_content, tool_calls=None)
        mock_provider.request.side_effect = [provider_response_1, provider_response_2]

        # Mock tool execution failure
        error_message = "Tool execution failed!"
        failed_result = ToolResult(success=False, error=error_message, tool_name=mock_tool_call_obj.name)
        mock_tool_manager.execute_tool.return_value = failed_result

        # Mock provider's ability to format the error result message
        mock_provider._add_tool_message = MagicMock(return_value=[{
            "role": "tool", 
            "tool_call_id": mock_tool_call_obj.id, 
            "content": error_message # Content should be the error
        }])

        # Execute process_prompt with streaming=True
        response = await tool_enabled_ai.process_prompt(prompt, streaming=True)

        # Verify
        assert mock_provider.request.call_count == 2
        # Verify streaming methods were NOT called
        assert not hasattr(mock_provider, 'get_streaming_completion_with_tools') or not mock_provider.get_streaming_completion_with_tools.called
        mock_tool_manager.execute_tool.assert_called_once_with(tool_name=mock_tool_call_obj.name, **mock_tool_call_obj.arguments, request_id=tool_enabled_ai._request_id)
        mock_provider._add_tool_message.assert_called_once()
        assert response == final_content
        assert len(tool_enabled_ai._tool_history) == 1
        history_entry = tool_enabled_ai._tool_history[0]
        assert history_entry['error'] == error_message
        assert history_entry['result'] is None

    # Removed test_streaming_with_partial_tool_calls as it's no longer relevant
    # since process_prompt bypasses streaming when tools are involved. 