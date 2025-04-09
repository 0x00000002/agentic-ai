"""
Tool executor for handling tool execution.
"""
from typing import Any, Dict, Optional, Callable
import time
from functools import wraps
import signal
import importlib  # Added for lazy loading
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError, RetryableToolError
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
    
    def _get_tool_function(self, tool_definition: ToolDefinition) -> Callable[..., Any]:
        """Lazily loads and returns the tool function."""
        if tool_definition.function is None:
            self._logger.debug(f"Lazily loading function '{tool_definition.function_name}' from module '{tool_definition.module_path}'")
            try:
                module = importlib.import_module(tool_definition.module_path)
                function_callable = getattr(module, tool_definition.function_name)
                # Cache the loaded function back into the definition
                tool_definition.function = function_callable
            except ImportError:
                self._logger.error(f"Failed to import module '{tool_definition.module_path}' for tool '{tool_definition.name}'.")
                raise AIToolError(f"Tool '{tool_definition.name}' module not found.", tool_name=tool_definition.name)
            except AttributeError:
                self._logger.error(f"Failed to find function '{tool_definition.function_name}' in module '{tool_definition.module_path}' for tool '{tool_definition.name}'.")
                raise AIToolError(f"Tool '{tool_definition.name}' function not found.", tool_name=tool_definition.name)
            except Exception as e:
                self._logger.error(f"Unexpected error loading tool '{tool_definition.name}': {e}", exc_info=True)
                raise AIToolError(f"Failed to load tool '{tool_definition.name}'.", tool_name=tool_definition.name)
        
        return tool_definition.function

    def _execute_with_timeout(self, tool_definition, args):
        """
        Execute a tool with timeout.
        
        Args:
            tool_definition: The tool definition object
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
            # Get the function (lazy load if needed)
            tool_function = self._get_tool_function(tool_definition)
            # Execute the function
            result = tool_function(**args)
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
            should_retry = False
            try:
                # Try to execute with timeout
                result = self._execute_with_timeout(tool_definition, args)
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                
                # Successful execution - return immediately
                return ToolResult(
                    success=True,
                    result=result,
                    error=None # No error details needed on success
                )
                
            except TimeoutError as e:
                self._logger.warning(f"Tool {tool_name} execution timed out (attempt {retries+1}/{self.max_retries+1})")
                last_error = f"{type(e).__name__}: {str(e)}"
                should_retry = True # Timeouts are retryable
                
            except RetryableToolError as e:
                 self._logger.warning(f"Tool {tool_name} encountered retryable error (attempt {retries+1}/{self.max_retries+1}): {str(e)}")
                 last_error = f"{type(e).__name__}: {str(e)}"
                 should_retry = True # Explicitly retryable
                 
            except (ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
                 # Specific non-retryable errors from within the tool logic or import issues
                 self._logger.error(f"Tool {tool_name} failed with non-retryable error: {type(e).__name__}: {str(e)}")
                 last_error = f"{type(e).__name__}: {str(e)}"
                 should_retry = False # Do not retry these
                 break # Exit the while loop immediately

            except Exception as e:
                # Catch any other unexpected exception
                self._logger.warning(f"Tool {tool_name} failed with unexpected error (attempt {retries+1}/{self.max_retries+1}): {type(e).__name__}: {str(e)}")
                last_error = f"{type(e).__name__}: {str(e)}"
                # Decide whether to retry unexpected errors. Current logic retries. Let's keep that for now.
                should_retry = True 
            
            # Increment retry counter only if we are going to retry
            if should_retry and retries < self.max_retries:
                retries += 1
                try:
                    # Exponential backoff with max of 10 seconds
                    delay = min(2 ** retries, 10) 
                    self._logger.info(f"Retrying tool {tool_name} in {delay} seconds...")
                    time.sleep(delay)
                except Exception as sleep_e: # Catch potential interruption during sleep
                     self._logger.warning(f"Sleep interrupted: {sleep_e}")
                     last_error = f"Retry interrupted during sleep: {sleep_e}"
                     break # Stop retrying if sleep fails
            else:
                 # If should_retry is False, or we've exhausted retries
                 break # Exit the while loop
        
        # If we get here, execution failed definitively (either non-retryable or all retries exhausted)
        self._logger.error(f"Tool {tool_name} execution failed definitively. Last error: {last_error}")
        return ToolResult(
            success=False,
            result=None,
            error=last_error or "Unknown error"
            # Consider adding error_details here if model is updated
        ) 