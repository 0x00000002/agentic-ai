import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
import json
from enum import Enum

from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ToolCall, ProviderResponse
from src.tools.models import ToolResult, ToolDefinition
from src.exceptions import AIToolError

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

# -----------------------------------------------------------------------------
# Tests for helper methods in ToolEnabledAI
# -----------------------------------------------------------------------------

class TestToolEnabledAIHelpers:
    """Tests for the helper methods of ToolEnabledAI."""
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, tool_enabled_ai, mock_tool_manager):
        """Test that _execute_tool_call correctly executes a tool and returns the result."""
        # Setup
        tool_call_args = {"arg": "value"}
        # Use correct ToolCall model fields: id, name, arguments (dict)
        tool_call = ToolCall(id="test_id", name="test_tool", arguments=tool_call_args) 
        
        # ToolResult from ToolManager is expected to have success/error/result
        expected_manager_result = ToolResult(
            success=True, 
            result=json.dumps({"status": "success", "data": "test output"}), 
            error=None
        )
        mock_tool_manager.execute_tool = AsyncMock(return_value=expected_manager_result)
        
        # Execute _execute_tool_call
        actual_result = await tool_enabled_ai._execute_tool_call(tool_call)
        
        # Verify _execute_tool_call returns the ToolResult from the manager
        assert actual_result == expected_manager_result
        # Verify ToolManager was called correctly
        mock_tool_manager.execute_tool.assert_called_once_with(
            tool_name="test_tool", # ToolManager expects tool_name
            request_id=ANY,        # Add ANY to match request_id
            **tool_call_args      # Pass arguments as kwargs
        )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_manager_error(self, tool_enabled_ai, mock_tool_manager, mock_logger):
        """Test that errors from the tool manager are handled properly by _execute_tool_call."""
        # Setup
        tool_call_args = {"arg": "value"}
        tool_call = ToolCall(id="error_id", name="error_tool", arguments=tool_call_args)
        error_message = "Tool manager failed execution"
        # Simulate ToolManager returning an error result
        manager_error_result = ToolResult(success=False, error=error_message, result=None)
        mock_tool_manager.execute_tool = AsyncMock(return_value=manager_error_result)
        
        # Execute
        actual_result = await tool_enabled_ai._execute_tool_call(tool_call)
        
        # Verify _execute_tool_call returns the error result from the manager
        assert actual_result == manager_error_result
        assert actual_result.success is False
        assert actual_result.error == error_message
        mock_tool_manager.execute_tool.assert_called_once_with(
            tool_name="error_tool", 
            request_id=ANY,       # Add ANY to match request_id
            **tool_call_args
        )
        # Logger might be called within ToolManager, not necessarily in _execute_tool_call itself
        # mock_logger.error.assert_called_once() 
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_unexpected_error(self, tool_enabled_ai, mock_tool_manager, mock_logger):
        """Test that unexpected errors *within* _execute_tool_call are handled."""
        # Setup: Simulate an error *before* calling the manager, e.g., invalid ToolCall structure
        # (Although Pydantic should catch this earlier normally)
        # Let's simulate the manager raising an unexpected exception instead
        tool_call_args = {"arg": "value"}
        tool_call = ToolCall(id="crash_id", name="crash_tool", arguments=tool_call_args)
        unexpected_exception = TypeError("Something unexpected broke")
        mock_tool_manager.execute_tool = AsyncMock(side_effect=unexpected_exception)
        
        # Execute
        actual_result = await tool_enabled_ai._execute_tool_call(tool_call)
        
        # Verify _execute_tool_call catches the exception and returns an error ToolResult
        assert actual_result.success is False
        assert actual_result.error == f"Unexpected error in ToolEnabledAI._execute_tool_call: {unexpected_exception}"
        assert actual_result.tool_name == "crash_tool" # Should include tool name
        mock_tool_manager.execute_tool.assert_called_once_with(
            tool_name="crash_tool", 
            request_id=ANY, # Add ANY check for request_id
            **tool_call_args
        )
        # Logger should be called by the exception handler in _execute_tool_call
        mock_logger.error.assert_called_once()
        
    def test_get_tool_history_no_tools(self, tool_enabled_ai):
        """Test that get_tool_history returns an empty list when no tools have been used."""
        # Setup - no tools used
        tool_enabled_ai._tool_history = []
        
        # Execute
        result = tool_enabled_ai.get_tool_history()
        
        # Verify
        assert result == []
        assert result is not tool_enabled_ai._tool_history  # Should return a copy
    
    def test_get_tool_history_returns_internal_list(self, tool_enabled_ai):
        """Test that get_tool_history returns a copy of the internal tool history."""
        # Setup - simulate tool history with dicts (as stored internally)
        history_item1 = {
            "tool_name": "history_tool", 
            "arguments": {"arg": "value"}, 
            "result": "history output", 
            "error": None,
            "tool_call_id": "history1"
        }
        history_item2 = {
            "tool_name": "error_tool", 
            "arguments": {},
            "result": None,
            "error": "Some error",
            "tool_call_id": "history2"
        }
        
        tool_enabled_ai._tool_history = [history_item1, history_item2]
        
        # Execute
        result = tool_enabled_ai.get_tool_history()
        
        # Verify
        assert len(result) == 2
        assert result[0] == history_item1
        assert result[1] == history_item2
        assert result is not tool_enabled_ai._tool_history  # Should return a copy
    
    def test_get_available_tools_no_manager_raises_error(self, basic_ai, mock_logger):
        """Test get_available_tools raises AttributeError on AIBase (or returns empty/logs warning)."""
        # AIBase does not have ToolManager or get_available_tools
        with pytest.raises(AttributeError): # Expecting AttributeError
             basic_ai.get_available_tools()
        # Alternatively, if it's designed to fail gracefully:
        # result = basic_ai.get_available_tools() 
        # assert result == {}
        # mock_logger.warning.assert_called_once() 
    
    def test_get_available_tools_success(self, tool_enabled_ai, mock_tool_manager):
        """Test that get_available_tools returns tools from the manager."""
        # Setup
        expected_tools = {
            "tool1": MagicMock(spec=ToolDefinition), # Use ToolDefinition mock
            "tool2": MagicMock(spec=ToolDefinition)
        }
        mock_tool_manager.get_all_tools.return_value = expected_tools
        
        # Execute
        result = tool_enabled_ai.get_available_tools()
        
        # Verify
        assert result == expected_tools
        mock_tool_manager.get_all_tools.assert_called_once() 