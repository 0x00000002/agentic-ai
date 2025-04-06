"""
Tool executor for handling tool execution.
"""
from typing import Any, Dict, Optional
import time
from functools import wraps
import signal
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError
from .models import ToolDefinition, ToolResult, ToolExecutionStatus


class TimeoutError(Exception):
    """Exception raised when a tool execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Tool execution timed out")


class ToolExecutor:
    """Executor for handling tool execution."""
    
    def __init__(self, logger: Optional[LoggerInterface] = None, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the tool executor.
        
        Args:
            logger: Logger instance
            timeout: Execution timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self._logger = logger or LoggerFactory.create("tool_executor")
        self.timeout = timeout
        self.max_retries = max_retries
    
    def _execute_with_timeout(self, tool, args):
        """
        Execute a tool with timeout.
        
        Args:
            tool: The tool to execute
            args: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            TimeoutError: If execution times out
        """
        # Set signal handler for SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            # Execute the function stored in the definition directly
            # 'tool' here is actually the tool_definition object
            result = tool.function(**args)
            return result
        finally:
            # Cancel the alarm
            signal.alarm(0)
    
    def execute(self, tool_definition: ToolDefinition, **args) -> ToolResult:
        """
        Execute a tool function defined in ToolDefinition with timeout and retries.
        
        Args:
            tool_definition: The tool definition object containing the function.
            args: Arguments to pass to the tool function.
            
        Returns:
            Tool execution result
        """
        tool_name = tool_definition.name
        # We pass the whole definition to _execute_with_timeout now
        # tool = tool_definition 
        
        # Extract request_id if present
        request_id = args.pop("request_id", None)
        
        # Try to execute the tool with retries
        retries = 0
        last_error = None
        execution_time_ms = None
        
        while retries <= self.max_retries:
            start_time = time.time()
            try:
                # Try to execute with timeout
                try:
                    result = self._execute_with_timeout(tool_definition, args)
                    end_time = time.time()
                    execution_time_ms = int((end_time - start_time) * 1000)
                    
                    # Successful execution
                    return ToolResult(
                        status=ToolExecutionStatus.SUCCESS,
                        result=result,
                        error=None
                    )
                except TimeoutError as e:
                    self._logger.warning(f"Tool {tool_name} execution timed out after {self.timeout} seconds")
                    last_error = str(e)
                    
            except Exception as e:
                self._logger.warning(f"Tool {tool_name} execution failed (attempt {retries+1}/{self.max_retries+1}): {str(e)}")
                last_error = str(e)
            
            # Increment retry counter
            retries += 1
            
            # If we have more retries, wait before trying again
            if retries <= self.max_retries:
                time.sleep(min(2 ** retries, 10))  # Exponential backoff with max of 10 seconds
        
        # If we get here, all retries failed
        self._logger.error(f"Tool {tool_name} execution failed after {self.max_retries+1} attempts")
        return ToolResult(
            status=ToolExecutionStatus.ERROR,
            result=None,
            error=last_error or "Unknown error"
        ) 