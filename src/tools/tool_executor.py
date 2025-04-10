"""
Tool executor for handling the execution of **internal** Python functions.
"""
from typing import Any, Dict, Optional, Callable
import time
from functools import wraps, partial
import signal
import importlib
import asyncio
import inspect
import functools

from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError, RetryableToolError
from .models import ToolDefinition, ToolResult

# Default timeout for tool execution in seconds
DEFAULT_TOOL_TIMEOUT = 30

class ToolExecutor:
    """
    Executor for handling internal Python tool execution with timeouts and retries.
    Assumes it receives a ToolDefinition for an internal tool.
    """
    
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
        """Lazily loads and returns the tool function from an internal ToolDefinition."""
        # Add check for internal source
        if tool_definition.source != 'internal':
            # This check should ideally happen in ToolManager before calling executor
            raise AIToolError(f"Only internal tools can be executed by this executor: {tool_definition.name}")
        
        # Check for function caching first
        if tool_definition._function_cache is not None:
            return tool_definition._function_cache
            
        # Support both naming conventions (module/function and module_path/function_path)
        module_path = getattr(tool_definition, 'module_path', None) or getattr(tool_definition, 'module', None)
        function_path = getattr(tool_definition, 'function_path', None) or getattr(tool_definition, 'function', None)
        
        # Check if both naming conventions are used simultaneously with different values
        if hasattr(tool_definition, 'module') and hasattr(tool_definition, 'module_path') and \
           tool_definition.module and tool_definition.module_path and \
           tool_definition.module != tool_definition.module_path:
            raise AIToolError(f"Tool definition {tool_definition.name} has inconsistent module/function path naming")
            
        if hasattr(tool_definition, 'function') and hasattr(tool_definition, 'function_path') and \
           tool_definition.function and tool_definition.function_path and \
           tool_definition.function != tool_definition.function_path:
            raise AIToolError(f"Tool definition {tool_definition.name} has inconsistent module/function path naming")
        
        # Add check for missing module/function path
        if not module_path or not function_path:
             raise AIToolError(f"Internal tool definition {tool_definition.name} is missing module or function path.")
             
        # Function loading logic
        self._logger.debug(f"Lazily loading function '{function_path}' from module '{module_path}'")
        try:
            module = importlib.import_module(module_path)
            function_callable = getattr(module, function_path)
            # Cache the loaded function
            tool_definition._function_cache = function_callable
        except ImportError:
            self._logger.error(f"Failed to import module '{module_path}' for tool '{tool_definition.name}'.")
            raise AIToolError(f"Tool '{tool_definition.name}' module not found.", tool_name=tool_definition.name)
        except AttributeError:
            self._logger.error(f"Failed to find function '{function_path}' in module '{module_path}' for tool '{tool_definition.name}'.")
            raise AIToolError(f"Tool '{tool_definition.name}' function not found.", tool_name=tool_definition.name)
        except Exception as e:
            self._logger.error(f"Unexpected error loading tool '{tool_definition.name}': {e}", exc_info=True)
            raise AIToolError(f"Failed to load tool '{tool_definition.name}'.", tool_name=tool_definition.name)
        
        return tool_definition._function_cache

    async def execute(
        self,
        tool_definition: ToolDefinition,
        parameters: Dict[str, Any],
        retry_count: int = 0,
        timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> ToolResult:
        """Execute a tool with the given parameters. May retry on retryable errors."""
        # Check if source attribute exists
        if not hasattr(tool_definition, 'source'):
            error_msg = f"Source attribute is required for tool execution: {tool_definition.name}"
            self._logger.error(error_msg)
            return ToolResult(
                tool_name=getattr(tool_definition, 'name', 'unknown'),
                success=False,
                error=error_msg,
                output=None,
            )

        # Ensure we're only executing internal tools
        if tool_definition.source != 'internal':
            error_msg = f"Only internal tools can be executed by this executor: {tool_definition.name}"
            self._logger.error(error_msg)
            return ToolResult(
                tool_name=tool_definition.name,
                success=False, 
                error=error_msg,
                output=None,
            )

        tool_name = tool_definition.name
        attempt = 0
        last_exception = None
        
        while attempt <= self.max_retries:
            try:
                self._logger.debug(f"Executing tool '{tool_name}' (Attempt {attempt + 1})")
                tool_function = self._get_tool_function(tool_definition)
                
                if asyncio.iscoroutinefunction(tool_function):
                    # Await async tool function with timeout
                    result = await asyncio.wait_for(
                        tool_function(**parameters), 
                        timeout=self.timeout
                    )
                else:
                    # Run sync tool function in executor with timeout
                    loop = asyncio.get_running_loop()
                    partial_func = functools.partial(tool_function, **parameters)
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, partial_func),
                        timeout=self.timeout
                    )
                
                self._logger.info(f"Tool '{tool_name}' executed successfully.")
                return ToolResult(success=True, result=result, tool_name=tool_name)
            
            except (RetryableToolError, TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    self._logger.warning(f"Tool '{tool_name}' failed with {type(e).__name__}. Retrying ({attempt + 1}/{self.max_retries})...")
                    # Exponential backoff
                    delay = 2 ** (attempt + 1)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue # Retry the loop
                else:
                    # Max retries reached
                    self._logger.error(f"Tool '{tool_name}' failed after {self.max_retries + 1} attempts due to {type(e).__name__}.")
                    break # Exit retry loop
            
            except Exception as e:
                # Non-retryable error
                self._logger.error(f"Tool '{tool_name}' failed with non-retryable error: {e}")
                last_exception = e
                break # Exit retry loop
                
        # If loop finished due to retries or non-retryable error
        error_message = f"{type(last_exception).__name__}: {str(last_exception)}"
        # Add tool_name prefix to RetryableToolError message for clarity if needed
        if isinstance(last_exception, RetryableToolError) and not str(last_exception).startswith(tool_name):
             error_message = f"{type(last_exception).__name__}: {tool_name}: {str(last_exception)}"
             
        return ToolResult(success=False, error=error_message, tool_name=tool_name) 