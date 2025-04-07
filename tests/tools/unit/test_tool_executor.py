# tests/tools/unit/test_tool_executor.py
"""
Unit tests for the ToolExecutor class.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, call
import time

# Import necessary components from src/tools
from src.tools.models import ToolCall, ToolResult, ToolDefinition
from src.tools.tool_executor import ToolExecutor, timeout_handler, TimeoutError
from src.exceptions import AIToolError

# --- Test Functions --- 
def simple_sync_tool(a: int, b: int) -> int:
    """A simple synchronous tool function."""
    return a + b

async def simple_async_tool(name: str) -> str:
    """A simple asynchronous tool function."""
    await asyncio.sleep(0.01) # Simulate async work
    return f"Hello, {name}"

def error_tool() -> None:
    """A tool function that raises an exception."""
    raise ValueError("Tool execution failed!")

def timeout_tool(duration: float) -> str:
    """A tool function that takes time to execute."""
    time.sleep(duration)
    return "Finished sleeping"

# --- Test Suite --- 
class TestToolExecutor:
    """Test suite for ToolExecutor."""

    @pytest.fixture
    def executor(self) -> ToolExecutor:
        """Provides a ToolExecutor instance."""
        # ToolExecutor __init__ does not take registry
        # mock_registry = MagicMock()
        # return ToolExecutor(registry=mock_registry)
        return ToolExecutor() # Initialize without registry
        
    # Remove this fixture as it's no longer needed if registry isn't passed
    # @pytest.fixture
    # def mock_registry(self, executor: ToolExecutor) -> MagicMock:
    #     """Provides access to the mock registry used by the executor fixture."""
    #     return executor._registry # Access the mock registry used in the executor

    # Modify tests to mock registry lookup externally if needed, 
    # but execute_tool takes ToolDefinition directly now.
    
    # --- Execution Tests --- 
    @pytest.mark.asyncio
    async def test_execute_sync_tool_success(self, executor: ToolExecutor): # Remove mock_registry
        """Test successful execution of a synchronous tool."""
        # ToolExecutor.execute takes ToolDefinition directly, no need to mock registry lookup here
        tool_def = ToolDefinition(name="simple_sync_tool", description="", 
                                  parameters_schema={}, function=simple_sync_tool)
        
        # The ToolExecutor.execute method is not async, but we called it via async wrapper? 
        # Let's assume ToolExecutor should have an async execute method or test the sync execute.
        # Reading execute method again...
        # Reading reveals execute is SYNC, but _execute_with_timeout uses signals -> BAD for async/threads
        # Refactoring test to call execute directly and check ToolResult
        
        # RETHINK: The original tests used execute_tool which IS async. Let's assume that exists.
        # Need to read ToolExecutor again if execute_tool isn't the method.
        # Re-checking execute signature: `execute(self, tool_definition: ToolDefinition, **args) -> ToolResult` -> SYNC
        # The tests were written for an async `execute_tool` which doesn't seem to exist. 
        # Let's assume the tests meant to call the sync `execute` method.
        # We will need to adjust the tests to be synchronous.
        
        # Create a dummy ToolDefinition
        tool_def = ToolDefinition(name="simple_sync_tool", description="Test", parameters_schema={}, function=simple_sync_tool)
        result = executor.execute(tool_def, a=5, b=3)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == 8 # 5 + 3
        
        # Test async function (requires running event loop or modifying test structure)
        # This suggests ToolExecutor.execute needs to handle async functions properly, 
        # maybe using asyncio.run or checking iscoroutinefunction. Let's test the sync path first.

    # Adjusting all tests to call the synchronous execute method and check ToolResult
    # Removing @pytest.mark.asyncio

    def test_execute_sync_tool_success_sync(self, executor: ToolExecutor):
        """Test successful execution of a synchronous tool (Sync Test)."""
        tool_def = ToolDefinition(name="simple_sync_tool", description="Test", parameters_schema={}, function=simple_sync_tool)
        result = executor.execute(tool_def, a=5, b=3)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == 8


    def test_execute_tool_internal_error(self, executor: ToolExecutor):
        """Test execution when the tool function itself raises an error (Sync Test)."""
        tool_def = ToolDefinition(name="error_tool", description="Test", parameters_schema={}, function=error_tool)
        result = executor.execute(tool_def)
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Tool execution failed!" in result.error
        # assert isinstance(result.error_details, ValueError) # Assuming error_details is not part of ToolResult model

    def test_execute_tool_missing_arguments(self, executor: ToolExecutor):
        """Test execution when required arguments are missing (Sync Test)."""
        tool_def = ToolDefinition(name="simple_sync_tool", description="Test", parameters_schema={}, function=simple_sync_tool)
        # The execute method catches the TypeError and returns a ToolResult
        # with pytest.raises(TypeError, match="missing 1 required positional argument: 'b'"):
        #      executor.execute(tool_def, a=5)
        result = executor.execute(tool_def, a=5)
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "missing 1 required positional argument: 'b'" in result.error

    # --- Timeout Tests (Need careful handling due to signals) ---
    # Signals don't work well with pytest, especially across threads/async. 
    # Consider refactoring ToolExecutor to use asyncio.wait_for or threading if async support needed.
    # For now, test the timeout logic carefully, possibly patching time.sleep or signal.

    @patch('src.tools.tool_executor.signal') # Patch the whole signal module
    def test_execute_tool_timeout(self, mock_signal, executor: ToolExecutor):
        """Test tool execution timing out (Sync Test)."""
        # Mock the timeout handler raising the exception
        def alarm_side_effect(duration):
            if duration > 0: # If setting the alarm
                # Simulate the timeout happening by raising the error directly
                # Use the imported TimeoutError
                raise TimeoutError("Tool execution timed out") 
            # Ignore signal.alarm(0)
            
        mock_signal.alarm.side_effect = alarm_side_effect
        executor.timeout = 1 # Set a nominal timeout
        
        tool_def = ToolDefinition(name="timeout_tool", description="Test", parameters_schema={}, function=timeout_tool)
        result = executor.execute(tool_def, duration=5) 
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Tool execution timed out" in result.error
        # Use the imported timeout_handler in assertion
        mock_signal.signal.assert_called_with(mock_signal.SIGALRM, timeout_handler)
        # Assert alarm was set, don't check for alarm(0) which isn't reached
        # mock_signal.alarm.assert_has_calls([call(executor.timeout), call(0)])
        mock_signal.alarm.assert_any_call(executor.timeout)

    @patch('src.tools.tool_executor.signal')
    def test_execute_tool_no_timeout(self, mock_signal, executor: ToolExecutor):
        """Test tool execution completing before timeout (Sync Test)."""
        executor.timeout = 5
        tool_def = ToolDefinition(name="timeout_tool", description="Test", parameters_schema={}, function=timeout_tool)
        
        # Ensure the mock alarm doesn't raise TimeoutError prematurely
        def alarm_side_effect(duration):
            pass # Do nothing, let the tool run
        mock_signal.alarm.side_effect = alarm_side_effect
        
        # duration is short, should complete
        result = executor.execute(tool_def, duration=0.01)
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == "Finished sleeping"
        mock_signal.alarm.assert_has_calls([call(executor.timeout), call(0)])

    # Add tests for retry logic