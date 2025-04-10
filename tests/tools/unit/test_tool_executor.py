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
from src.tools.tool_executor import ToolExecutor
from src.exceptions import AIToolError, RetryableToolError
from asyncio import TimeoutError # Import TimeoutError from asyncio

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
    async def test_execute_sync_tool_success(self, executor: ToolExecutor):
        """Test successful execution of a synchronous tool (run via async execute)."""
        tool_def = ToolDefinition(
            name="simple_sync_tool",
            description="",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
        # Call execute with await as the execute method is async
        result = await executor.execute(tool_def, a=1, b=2)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.result == 3 # 1 + 2 = 3
        assert result.error is None

    @pytest.mark.asyncio # Mark as async
    async def test_execute_sync_tool_success_sync(self, executor: ToolExecutor):
        """Test successful execution of a synchronous tool (run via async execute)."""
        tool_def = ToolDefinition(
            name="simple_sync_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
        result = await executor.execute(tool_def, a=1, b=2) # Await the async execute method
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == 3


    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_internal_error(self, executor: ToolExecutor):
        """Test execution when the tool function itself raises an error (run via async execute)."""
        tool_def = ToolDefinition(
            name="error_tool",
            description="Test",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="error_tool",
            function=error_tool
        )
        result = await executor.execute(tool_def) # Await the async execute method
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.result is None
        assert "Tool execution failed!" in result.error

    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_missing_arguments(self, executor: ToolExecutor):
        """Test execution when required arguments are missing (run via async execute)."""
        tool_def = ToolDefinition(
            name="simple_sync_tool", 
            description="Test", 
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_sync_tool",
            function=simple_sync_tool
        )
        result = await executor.execute(tool_def, a=5) # Await the async execute method
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.result is None
        assert result.error is not None
        assert "missing 1 required positional argument: \'b\'" in result.error

    # --- Timeout Tests (Need careful handling due to signals) ---
    # Signals don't work well with pytest, especially across threads/async. 
    # Consider refactoring ToolExecutor to use asyncio.wait_for or threading if async support needed.
    # For now, test the timeout logic carefully, possibly patching time.sleep or signal.

    # Remove tests relying on the old signal-based timeout_handler and signal patching
    # @patch('src.tools.tool_executor.signal')
    # @patch('src.tools.tool_executor.time.sleep')
    # def test_execute_tool_timeout(self, mock_sleep, mock_signal, executor: ToolExecutor):
    #     ...

    # @patch('src.tools.tool_executor.signal')
    # @patch('time.sleep')
    # def test_execute_tool_no_timeout(self, mock_sleep, mock_signal, executor: ToolExecutor):
    #     ...

    # New test for asyncio timeout
    @pytest.mark.asyncio
    async def test_execute_asyncio_timeout(self, executor: ToolExecutor):
        """Test tool execution timing out using asyncio.wait_for."""
        tool_def = ToolDefinition(
            name="timeout_tool",
            description="Test",
            parameters_schema={}, 
            module_path="tests.tools.unit.test_tool_executor",
            function_name="timeout_tool",
            function=timeout_tool
        )
        # Set a duration longer than the executor's timeout (1s)
        result = await executor.execute(tool_def, duration=1.5)

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "TimeoutError: Tool execution exceeded" in result.error # Check for asyncio timeout message

    @pytest.mark.asyncio
    async def test_execute_asyncio_no_timeout(self, executor: ToolExecutor):
        """Test tool execution completing before asyncio timeout."""
        tool_def = ToolDefinition(
            name="timeout_tool",
            description="Test",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="timeout_tool",
            function=timeout_tool
        )
        # Set a duration shorter than the executor's timeout (1s)
        result = await executor.execute(tool_def, duration=0.1)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.error is None
        assert result.result == "Finished sleeping"

    @pytest.mark.asyncio
    async def test_execute_async_tool_success(self, executor: ToolExecutor):
        """Test successful execution of an asynchronous tool."""
        tool_def = ToolDefinition(
            name="simple_async_tool",
            description="Test",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="simple_async_tool",
            function=simple_async_tool
        )
        result = await executor.execute(tool_def, name="World")
        assert result.success is True
        assert result.result == "Hello, World"
        
    @pytest.mark.asyncio
    async def test_execute_tool_internal_error_async(self, executor: ToolExecutor):
        """Test execution when an async tool function raises an error."""
        async def async_error_tool():
            raise ValueError("Async tool error!")
            
        tool_def = ToolDefinition(
            name="async_error_tool", 
            description="Test", 
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor", # Placeholder
            function_name="async_error_tool",
            function=async_error_tool
        )
        result = await executor.execute(tool_def)
        assert result.success is False
        assert "Async tool error!" in result.error
        
    @pytest.mark.asyncio
    @patch('src.tools.tool_executor.asyncio.sleep') # Patch asyncio sleep
    async def test_execute_tool_with_retries_async(self, mock_sleep, retry_executor):
        """Test async retry logic with patched asyncio.sleep."""
        call_count = [0]
        async def async_flaky_tool():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RetryableToolError(f"Failing async on attempt {call_count[0]}", tool_name="async_flaky_tool")
            return "Async Success"

        tool_def = ToolDefinition(
            name="async_flaky_tool",
            description="Test",
            parameters_schema={},
            module_path="tests.tools.unit.test_tool_executor",
            function_name="async_flaky_tool",
            function=async_flaky_tool
        )

        result = await retry_executor.execute(tool_def)

        assert result.success is True
        assert result.result == "Async Success"
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2 # Check asyncio.sleep was called twice