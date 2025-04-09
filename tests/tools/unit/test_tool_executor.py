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
from src.exceptions import AIToolError, RetryableToolError

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
        """Provides a ToolExecutor instance with test-friendly settings."""
        # Use much shorter timeouts for testing
        return ToolExecutor(timeout=1, max_retries=0)  # Disable retries for most tests

    @pytest.fixture
    def retry_executor(self) -> ToolExecutor:
        """Provides a ToolExecutor instance configured to test retry logic."""
        return ToolExecutor(timeout=1, max_retries=2)  # Short timeout, 2 retries for testing

    # --- Execution Tests --- 
    @pytest.mark.asyncio
    async def test_execute_sync_tool_success(self, executor: ToolExecutor): # Remove mock_registry
        """Test successful execution of a synchronous tool."""
        tool_def = ToolDefinition(
            name="simple_sync_tool",
            description="",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
        # Call execute directly, it's synchronous
        result = executor.execute(tool_def, a=1, b=2)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.result == 3 # 1 + 2 = 3

    def test_execute_sync_tool_success_sync(self, executor: ToolExecutor):
        """Test successful execution of a synchronous tool (Sync Test)."""
        tool_def = ToolDefinition(
            name="simple_sync_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
        result = executor.execute(tool_def, a=1, b=2)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == 3


    def test_execute_tool_internal_error(self, executor: ToolExecutor):
        """Test execution when the tool function itself raises an error (Sync Test)."""
        tool_def = ToolDefinition(
            name="error_tool",
            description="Test",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="error_tool",
            function=error_tool
        )
        # Call execute without the unexpected argument 'x'
        result = executor.execute(tool_def)
        assert isinstance(result, ToolResult)
        assert result.success is False
        # Check for the specific error raised by the error_tool function
        assert "Tool execution failed!" in result.error

    def test_execute_tool_missing_arguments(self, executor: ToolExecutor):
        """Test execution when required arguments are missing (Sync Test)."""
        tool_def = ToolDefinition(
            name="simple_sync_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
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

    @patch('src.tools.tool_executor.signal')
    @patch('src.tools.tool_executor.time.sleep')  # Patch time.sleep to avoid actual waiting
    def test_execute_tool_timeout(self, mock_sleep, mock_signal, executor: ToolExecutor):
        """Test tool execution timing out (Sync Test)."""
        # Mock the timeout handler raising the exception
        def alarm_side_effect(duration):
            if duration > 0: # If setting the alarm
                # Simulate the timeout happening by raising the error directly
                # Use the imported TimeoutError
                raise TimeoutError("Tool execution timed out") 
            # Ignore signal.alarm(0)
            
        mock_signal.alarm.side_effect = alarm_side_effect
        # timeout is already set to 1 in the fixture
        
        tool_def = ToolDefinition(
            name="timeout_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="timeout_tool",
            function=timeout_tool
        )
        result = executor.execute(tool_def, duration=5) 
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Tool execution timed out" in result.error
        # Use the imported timeout_handler in assertion
        mock_signal.signal.assert_called_with(mock_signal.SIGALRM, timeout_handler)
        # Assert alarm was set
        mock_signal.alarm.assert_any_call(executor.timeout)
        # Verify sleep wasn't called (since max_retries is 0)
        mock_sleep.assert_not_called()

    @patch('src.tools.tool_executor.signal')
    @patch('time.sleep')  # Patch the global time.sleep to avoid actual waiting
    def test_execute_tool_no_timeout(self, mock_sleep, mock_signal, executor: ToolExecutor):
        """Test tool execution completing before timeout (Sync Test)."""
        # timeout is already set to 1 in the fixture
        tool_def = ToolDefinition(
            name="timeout_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="timeout_tool",
            function=timeout_tool
        )
        
        # Ensure the mock alarm doesn't raise TimeoutError prematurely
        def alarm_side_effect(duration):
            pass # Do nothing, let the tool run
        mock_signal.alarm.side_effect = alarm_side_effect
        
        # duration is short, should complete
        result = executor.execute(tool_def, duration=0.5)
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == "Finished sleeping"
        mock_signal.alarm.assert_has_calls([call(executor.timeout), call(0)])
        # Our implementation of time.sleep in the tool itself is still being called,
        # so we can't assert that sleep wasn't called

    @patch('src.tools.tool_executor.time.sleep')  # Patch sleep to avoid delays
    def test_execute_tool_with_retries(self, mock_sleep, retry_executor):
        """Test that retry logic works properly but doesn't cause delays."""
        # Create a tool that fails on first two calls, succeeds on third
        call_count = [0]
        
        def flaky_tool():
            call_count[0] += 1
            if call_count[0] <= 2:  # Fail first two calls
                # Use RetryableToolError to correctly test retry logic
                raise RetryableToolError(f"Failing on attempt {call_count[0]}", tool_name="flaky_tool") 
            return "Success on third try"
        
        tool_def = ToolDefinition(
            name="flaky_tool", 
            description="Test",
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="flaky_tool",
            function=flaky_tool
        )
        
        result = retry_executor.execute(tool_def)
        
        assert result.success is True
        assert result.result == "Success on third try"
        assert call_count[0] == 3 # Should be called 3 times (2 failures + 1 success)
        # Check sleep was called twice (after first and second failures)
        assert mock_sleep.call_count == 2 
        # Check backoff delays (optional)
        # mock_sleep.assert_has_calls([
        #     call(min(2 ** 1, 10)),
        #     call(min(2 ** 2, 10))
        # ])