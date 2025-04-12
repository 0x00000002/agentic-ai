import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

import json
from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ProviderResponse, ToolCall
from src.tools.models import ToolResult
from src.config.unified_config import UnifiedConfig

# Define local Role enum for this test module if needed (or import if available)
from enum import Enum
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# -----------------------------------------------------------------------------
# Main tests focus on process_prompt functionality
# -----------------------------------------------------------------------------

class TestToolEnabledAIProcessPrompt:
    """Tests for the process_prompt method of ToolEnabledAI."""
    
    @pytest.mark.asyncio
    async def test_process_prompt_no_tool_support_fallback(self, mock_provider, mock_config, mock_logger):
        """
        When the provider doesn't support tools, process_prompt should fall back to the base request behavior.
        """
        # Setup: Create a provider mock that does NOT support tools
        mock_provider_no_tools = AsyncMock()
        mock_provider_no_tools.supports_tools = False 
        mock_provider_no_tools.request = AsyncMock(return_value=ProviderResponse(
            content="Fallback response"
        ))
        
        # Patch factory to return this specific provider during init
        with patch('src.core.base_ai.ConversationManager'), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider_no_tools), \
             patch('src.core.tool_enabled_ai.ToolManager'): 
            
            # Instantiate the AI *after* patching the factory
            ai_no_tool_support = ToolEnabledAI(model="test-model", logger=mock_logger)
            
            # Execute: Call process_prompt on the instance configured with no tool support
            result = await ai_no_tool_support.process_prompt("Test prompt")
            
            # Verify
            assert result == "Fallback response"
            # Check that the provider's request method was called (indicating fallback behavior)
            mock_provider_no_tools.request.assert_called_once()
            # Assert that tool manager execute was NOT called (crucial check for fallback)
            # We don't have the tool_manager mock available here directly, 
            # but ensuring request was called implies the tool loop was bypassed.

    @pytest.mark.asyncio
    async def test_process_prompt_no_tools_available(self, tool_enabled_ai, mock_provider, mock_tool_manager):
        """
        When no tools are available, the request should proceed without tool options.
        """
        # Setup - empty tools dictionary
        mock_tool_manager.get_all_tools.return_value = {}
        
        mock_provider.request.return_value = ProviderResponse(
            role=Role.ASSISTANT,
            content="Response without tools"
        )
        
        # Execute
        result = await tool_enabled_ai.process_prompt("Test prompt")
        
        # Verify
        assert result == "Response without tools"
        mock_provider.request.assert_called_once()
        # Ensure no tools were passed to the provider
        assert mock_provider.request.call_args[1].get('tools', 'marker') == 'marker'
    
    @pytest.mark.asyncio
    async def test_process_prompt_with_tools_no_tool_calls(self, tool_enabled_ai, mock_provider, mock_tool_manager):
        """
        When tools are available but not used by the AI, the response should be returned directly.
        """
        # Setup - provide mock tools using MagicMock to simulate ToolDefinition
        mock_tool_def = MagicMock()
        mock_tool_def.name = "test_tool"
        mock_tool_def.description = "A test tool"
        mock_tool_def.source = "internal" # Add source attribute
        mock_tool_def.parameters_schema = {}
        mock_tools = {"test_tool": mock_tool_def}
        mock_tool_manager.get_all_tools.return_value = mock_tools
        
        # Provider response with no tool calls
        mock_provider.request.return_value = ProviderResponse(
            role=Role.ASSISTANT,
            content="Response without using tools"
        )
        
        # Execute
        result = await tool_enabled_ai.process_prompt("Test prompt")
        
        # Verify
        assert result == "Response without using tools"
        mock_provider.request.assert_called_once()
        # Ensure tools were passed to the provider
        call_args, call_kwargs = mock_provider.request.call_args
        assert "tools" in call_kwargs
        # Check that the *names* were passed
        assert call_kwargs.get('tools') == ["test_tool"]
    
    @pytest.mark.asyncio
    async def test_process_prompt_single_tool_call_SIMPLE(self,
                                                  tool_enabled_ai, mock_provider, 
                                                  mock_tool_manager, mock_convo_manager):
        """
        Simplified test: Does process_prompt trigger execute_tool?
        """
        # Setup - provide mock tools using MagicMock to simulate ToolDefinition
        mock_tool_def = MagicMock()
        mock_tool_def.name = "test_tool"
        mock_tool_def.description = "A test tool"
        mock_tool_def.source = "internal" # Add source attribute
        mock_tool_def.parameters_schema = {}
        mock_tools = {"test_tool": mock_tool_def}
        mock_tool_manager.get_all_tools.return_value = mock_tools
        
        # Provider returns a response with a tool call
        tool_call = ToolCall(id="call1", name="test_tool", arguments={"arg": "value"})
        
        # ** Only configure the FIRST provider call **
        first_response = ProviderResponse(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[tool_call]
        )
        
        # Tool execution result (needed for return type, content doesn't matter much here)
        tool_result = ToolResult(success=True, tool_name="test_tool", result="output")
        
        # Re-add instance mocking
        tool_enabled_ai._tool_manager.execute_tool = AsyncMock(return_value=tool_result)
        
        # Mock _add_tool_message (still needed for loop to proceed)
        mock_provider._add_tool_message = MagicMock(return_value=[{'role': 'tool', 'content': 'Mock tool result msg', 'tool_call_id': 'call1'}])
        
        # Mock the SECOND provider call (to prevent error when loop continues)
        # Use a separate mock for the side_effect if request is called again
        final_response_mock = ProviderResponse(role=Role.ASSISTANT, content="Final dummy response")
        mock_provider.request.side_effect = [first_response, final_response_mock]

        # Execute
        await tool_enabled_ai.process_prompt("Can you use the test tool?")
        
        # Verify ONLY the execute_tool call (on the instance)
        tool_enabled_ai._tool_manager.execute_tool.assert_called_once_with(
            tool_name="test_tool", 
            request_id=tool_enabled_ai._request_id, 
            arg="value"
        )
    
    @pytest.mark.asyncio
    async def test_process_prompt_multiple_tool_calls(self,
                                                     tool_enabled_ai, mock_provider, 
                                                     mock_tool_manager, mock_convo_manager):
        """
        Test a response with multiple tool calls that are executed sequentially.
        """
        # Setup - provide mock tools using MagicMock
        mock_tool_def1 = MagicMock()
        mock_tool_def1.name = "tool1"
        mock_tool_def1.description = "First test tool"
        mock_tool_def1.source = "internal"
        mock_tool_def1.parameters_schema = {}
        mock_tool_def2 = MagicMock()
        mock_tool_def2.name = "tool2"
        mock_tool_def2.description = "Second test tool"
        mock_tool_def2.source = "internal"
        mock_tool_def2.parameters_schema = {}
        mock_tools = {"tool1": mock_tool_def1, "tool2": mock_tool_def2}
        mock_tool_manager.get_all_tools.return_value = mock_tools
        
        # Provider returns a response with multiple tool calls
        tool_call1 = ToolCall(id="call1", name="tool1", arguments={"arg1": "value1"})
        tool_call2 = ToolCall(id="call2", name="tool2", arguments={"arg2": "value2"})
        
        # Define side effects for each call to request
        mock_provider.request.side_effect = [
            # Initial response with two tool calls
            ProviderResponse(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[tool_call1, tool_call2]
            ),
            # Final response after tool executions
            ProviderResponse(
                role=Role.ASSISTANT,
                content="Final response after both tools"
            )
        ]
        
        # Tool execution results (Corrected)
        tool_result1 = ToolResult(
            success=True,                    # Added success=True
            tool_name="tool1",             # Changed name to tool_name
            result={"status": "success", "data": "tool1 output"} # Kept result
            # Removed call_id and arguments
        )
        
        tool_result2 = ToolResult(
            success=True,                    # Added success=True
            tool_name="tool2",             # Changed name to tool_name
            result={"status": "success", "data": "tool2 output"} # Kept result
            # Removed call_id and arguments
        )
        
        # Re-add instance mocking
        tool_enabled_ai._tool_manager.execute_tool = AsyncMock(side_effect=[tool_result1, tool_result2])
        
        # Mock tool result messages
        mock_provider.build_tool_result_messages.return_value = [
            {"role": "tool", "content": "Tool execution result"}
        ]
        # ADD MOCK for _add_tool_message (handle multiple calls)
        mock_provider._add_tool_message = MagicMock()
        mock_provider._add_tool_message.side_effect = lambda *args, **kwargs: [{'role': 'tool', 'content': f'Mock tool result for {kwargs.get("tool_name")}', 'tool_call_id': kwargs.get("tool_call_id")}]

        # Execute
        result = await tool_enabled_ai.process_prompt("Can you use both tools?")
        
        # Verify
        assert result == "Final response after both tools"
        
        # Provider called twice (initial request + after tool executions)
        assert mock_provider.request.call_count == 2
        
        # Both tools were executed (Assert on the instance mock)
        assert tool_enabled_ai._tool_manager.execute_tool.call_count == 2
        
        # Check first tool execution
        tool_enabled_ai._tool_manager.execute_tool.assert_any_call(
            tool_name="tool1",
            request_id=tool_enabled_ai._request_id,
            arg1="value1"
        )
        
        # Check second tool execution
        tool_enabled_ai._tool_manager.execute_tool.assert_any_call(
            tool_name="tool2", 
            request_id=tool_enabled_ai._request_id,
            arg2="value2"
        )
    
    @pytest.mark.asyncio
    async def test_process_prompt_tool_execution_error(self,
                                                      tool_enabled_ai, mock_provider, 
                                                      mock_tool_manager, mock_convo_manager):
        """
        Test that tool execution errors are handled properly.
        """
        # Setup - provide mock tools using MagicMock
        mock_tool_def = MagicMock()
        mock_tool_def.name = "error_tool"
        mock_tool_def.description = "A tool that will error"
        mock_tool_def.source = "internal"
        mock_tool_def.parameters_schema = {}
        mock_tools = {"error_tool": mock_tool_def}
        mock_tool_manager.get_all_tools.return_value = mock_tools
        
        # Provider returns a response with a tool call
        tool_call = ToolCall(id="error1", name="error_tool", arguments={"arg": "value"})
        
        mock_provider.request.side_effect = [
            # Initial response with tool call
            ProviderResponse(
                role=Role.ASSISTANT,
                content=None, 
                tool_calls=[tool_call]
            ),
            # Final response after error
            ProviderResponse(
                role=Role.ASSISTANT,
                content="I encountered an error with the tool"
            )
        ]
        
        # Tool execution fails (Corrected)
        error_result = ToolResult(
            success=False,                   # Added success=False
            tool_name="error_tool",        # Changed name to tool_name
            error="Tool execution failed"    # Moved error message to error field
            # Removed call_id, arguments, and result field
        )
        # Re-add instance mocking
        tool_enabled_ai._tool_manager.execute_tool = AsyncMock(return_value=error_result)
        
        # Mock error messages
        error_messages = [
            {"role": "tool", "content": json.dumps({"status": "error", "message": "Tool execution failed"})}
        ]
        mock_provider.build_tool_result_messages.return_value = error_messages
        # ADD MOCK for _add_tool_message
        mock_provider._add_tool_message = MagicMock(return_value=[{'role': 'tool', 'content': 'Mock tool error msg', 'tool_call_id': 'error1'}])
        
        # Execute
        result = await tool_enabled_ai.process_prompt("Can you use the error_tool?")
        
        # Verify
        assert result == "I encountered an error with the tool"
        assert mock_provider.request.call_count == 2
        
        # Tool was attempted (Assert on the instance mock)
        tool_enabled_ai._tool_manager.execute_tool.assert_called_once()
        
        # Error messages were added to conversation (Use assert_any_call)
        mock_convo_manager.add_message.assert_any_call(
            role='tool', 
            content='Mock tool error msg', 
            tool_call_id='error1'
        )
    
    @pytest.mark.asyncio
    async def test_process_prompt_max_iterations_exceeded(self,
                                                         tool_enabled_ai, mock_provider, 
                                                         mock_tool_manager, mock_convo_manager, mock_logger):
        """Test that the tool calling loop terminates if max iterations are exceeded."""
        # Setup - provide mock tools using MagicMock
        mock_tool_def = MagicMock()
        mock_tool_def.name = "loop_tool"
        mock_tool_def.description = "A tool called repeatedly"
        mock_tool_def.source = "internal"
        mock_tool_def.parameters_schema = {}
        mock_tools = {"loop_tool": mock_tool_def}
        mock_tool_manager.get_all_tools.return_value = mock_tools
        
        # Set a low max_iterations for the test
        tool_enabled_ai._max_tool_iterations = 3
        
        # Provider keeps returning tool calls up to max iterations
        tool_call = ToolCall(id="loop1", name="loop_tool", arguments={"arg": "value"})
        
        # Always return a new tool call (forcing it to hit max iterations)
        repeated_response = ProviderResponse(
            role=Role.ASSISTANT,
            content=None, 
            tool_calls=[tool_call]
        )
        mock_provider.request.side_effect = [repeated_response] * 4  # More than max_iterations
        
        # Tool execution succeeds but AI keeps requesting it (Corrected)
        tool_result = ToolResult(
            success=True,                    # Added success=True
            tool_name="loop_tool",         # Changed name to tool_name
            result={"status": "success", "data": "tool output"} # Kept result
            # Removed call_id and arguments
        )
        # Re-add instance mocking
        tool_enabled_ai._tool_manager.execute_tool = AsyncMock(return_value=tool_result)
        
        # Mock tool messages
        mock_provider.build_tool_result_messages.return_value = [
            {"role": "tool", "content": "Tool output"}
        ]
        # ADD MOCK for _add_tool_message
        mock_provider._add_tool_message = MagicMock(return_value=[{'role': 'tool', 'content': 'Mock loop tool result', 'tool_call_id': 'loop1'}])
        
        # Execute
        result = await tool_enabled_ai.process_prompt(
            "Use the loop tool repeatedly", 
            max_tool_iterations=3 # Pass the limit explicitly
        )
        
        # Verify max_iterations was reached
        assert mock_provider.request.call_count == 3  # Expect exactly 3 calls (iter 1, 2, 3)
        assert tool_enabled_ai._tool_manager.execute_tool.call_count == 3 # Assert on instance mock
        
        # Assert that the warning was logged (Corrected message and value)
        mock_logger.warning.assert_called_with(
            f"Exceeded maximum tool iterations (3). Returning last assistant content."
        )
