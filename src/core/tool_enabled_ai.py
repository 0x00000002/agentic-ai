"""
AI implementation with tool-calling capabilities.
Extends the base AI with the ability to call tools to handle complex tasks.
"""
from typing import Dict, List, Any, Optional, Union, Callable
import json
import inspect
from .base_ai import AIBase
from ..exceptions import AIProcessingError, AIToolError, ErrorHandler
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.dynamic_models import Model
from ..tools.tool_manager import ToolManager
from ..tools.models import ToolDefinition, ToolResult, ToolCall
from ..prompts.prompt_template import PromptTemplate
# Import provider interface and response model for type checking
from .interfaces import ProviderInterface
from .models import ProviderResponse
# Import ToolCapableProviderInterface for type checking
from .interfaces import ToolCapableProviderInterface 
# Import specific providers to check for Anthropic type if needed
from .providers.anthropic_provider import AnthropicProvider
import asyncio # Add asyncio import


class ToolEnabledAI(AIBase):
    """
    AI implementation with tool-calling capabilities.
    Orchestrates interaction with providers and tool execution via ToolManager.
    """
    
    def __init__(self, 
                 model: Optional[Union[Model, str]] = None, 
                 system_prompt: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None,
                 tool_manager: Optional[ToolManager] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 **kwargs): # Allow extra kwargs for base class
        """
        Initialize the tool-enabled AI.
        
        Args:
            model: The model to use (Model enum or string ID)
            system_prompt: Custom system prompt (or None for default)
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            tool_manager: Optional ToolManager instance. If None, a default one is created.
            prompt_template: Optional Prompt template for the AI
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            logger=logger,
            request_id=request_id,
            prompt_template=prompt_template,
            **kwargs
        )
        
        unified_config = UnifiedConfig.get_instance()
        # Set up tool manager
        self._tool_manager = tool_manager or ToolManager(
            unified_config=unified_config,
            logger=self._logger
        )
        
        # Check if the provider supports tool calling (based on interface/attribute)
        self._supports_tools = isinstance(self._provider, ToolCapableProviderInterface) or \
                              (hasattr(self._provider, 'supports_tools') and self._provider.supports_tools)

        if not self._supports_tools:
            self._logger.warning(f"Provider {type(self._provider).__name__} for model {self._model_config.get('model_id', 'N/A')} may not fully support tool calling based on configuration.")
            
        # Access model_id from the model config dictionary stored by AIBase
        model_id_for_log = self._model_config.get('model_id', 'UNKNOWN')
        self._logger.info(f"Initialized {self.__class__.__name__} with model {model_id_for_log}. Tool support: {self._supports_tools}")
        # Tool history tracking - perhaps per request? Resetting here for now.
        self._tool_history = [] 
    
    async def request_basic(self, prompt: str, **options) -> ProviderResponse: # Changed to async def
        """
        Makes a basic request to the underlying provider asynchronously, returning the standardized response.

        Handles formatting messages with history and sending to the provider.
        Does NOT automatically execute tool calls. Returns the standardized ProviderResponse object
        which may contain 'content' and/or 'tool_calls'.

        Args:
            prompt: The user prompt string
            **options: Additional options for the provider request

        Returns:
            The standardized ProviderResponse object from the provider.
        """
        # Add the current user prompt to history
        self._conversation_manager.add_message(role="user", content=prompt)
        messages = self._conversation_manager.get_messages() 
        
        self._logger.debug(f"ToolEnabledAI.request_basic: Calling provider with {len(messages)} messages. Options: {options}")
        
        try:
            # Call provider's async request method
            # Assume provider.request is also made async or wrapped appropriately
            provider_response = await self._provider.request(messages=messages, **options) # Use await
            
            # Check for errors in the response object itself
            if provider_response.error:
                 self._logger.error(f"Provider returned an error in ProviderResponse: {provider_response.error}")
                 # Re-raise as an exception or handle appropriately?
                 # For now, let's add a marker to content and return the object
                 # A better approach might be to raise AIProcessingError here.
                 # provider_response.content = f"[Error: {provider_response.error}]"
                 raise AIProcessingError(f"Provider error: {provider_response.error}")
            
            # Add AI response message (content and tool_calls) to history
            assistant_message = {
                "role": "assistant",
                "content": provider_response.content # Use content from model
            }
            # Include tool calls if the provider returned them
            if provider_response.tool_calls:
                 assistant_message["tool_calls"] = provider_response.tool_calls
                 
            # Add message only if it has content or tool calls
            if assistant_message["content"] is not None or assistant_message.get("tool_calls"):
                 self._conversation_manager.add_message(**assistant_message)
            else:
                 self._logger.warning("Provider response had neither content nor tool_calls. Assistant message not added to history.")

            self._logger.debug(f"ToolEnabledAI.request_basic: Received ProviderResponse. Content: {bool(provider_response.content)}, Tool Calls: {len(provider_response.tool_calls or [])}")
            return provider_response
            
        except Exception as e:
            self._logger.error(f"Error during ToolEnabledAI basic request: {e}", exc_info=True)
            # If the provider itself raised an exception, convert to ProviderResponse error obj
            # This path might be less common now if provider.request catches errors.
            # return ProviderResponse(error=str(e))
            raise AIProcessingError(f"Failed processing basic request: {e}") from e

    async def process_prompt(self, # Changed to async def
                       prompt: str, 
                       max_tool_iterations: int = 5, 
                       **options) -> str:
        """
        Processes a prompt, automatically handling the tool-calling loop.

        Calls the provider, checks for tool calls, executes them via ToolManager,
        adds results back to the conversation history using provider-specific formatting,
        and continues calling the provider until a final text response is received
        or the maximum iterations are reached.

        Args:
            prompt: The user prompt string.
            max_tool_iterations: Maximum number of tool execution rounds.
            **options: Additional options for the provider request (e.g., temperature).

        Returns:
            The final AI response string after any tool execution.
        """
        if not self._supports_tools:
             self._logger.warning("Provider does not support tools. Performing basic request using AIBase.request.")
             # Fallback to base class request if tools aren't supported
             response_str = await super().request(prompt, **options) # Use await
             return response_str

        self._logger.info(f"Processing prompt with tool support: '{prompt[:50]}...' (Max Iterations: {max_tool_iterations})")
        self._tool_history = [] # Reset tool history for this request

        # Add initial user prompt to history
        self._conversation_manager.add_message(role="user", content=prompt)

        iteration_count = 0
        last_assistant_content = "" # Store the last text content received

        while iteration_count < max_tool_iterations:
            iteration_count += 1
            self._logger.info(f"Tool Loop Iteration {iteration_count}/{max_tool_iterations}")

            # >>> ADD CHECK HERE <<<
            if not self._tool_manager:
                # This state should ideally not be reached if _supports_tools is True,
                # but adding a safeguard.
                self._logger.error("Tool manager is not initialized, cannot proceed with tool loop.")
                raise ValueError("Tool manager not initialized")
            # >>> END CHECK <<<

            current_messages = self._conversation_manager.get_messages()

            # --- Prepare tools for the provider ---
            available_tools_defs = self._tool_manager.get_all_tools()
            provider_tools_param = None
            tool_choice_param = None
            if available_tools_defs:
                 provider_tools_param = available_tools_defs 
                 tool_choice_param = options.get("tool_choice", "auto") 
            else:
                 self._logger.info("No tools registered in ToolManager. Proceeding without tools for this turn.")
                 
            provider_options = options.copy()
            if provider_tools_param:
                 provider_options["tools"] = provider_tools_param
                 provider_options["tool_choice"] = tool_choice_param
                 
            try:
                # --- Call Provider ---
                self._logger.debug(f"Calling provider. Request messages count: {len(current_messages)}")
                # Provider now returns a ProviderResponse object - assume request is async
                provider_response = await self._provider.request(messages=current_messages, **provider_options) # Use await
                self._logger.debug(f"Provider response received. Stop reason: {provider_response.stop_reason}")

                # --- Check for Errors in Response Object --- 
                if provider_response.error:
                     self._logger.error(f"Provider returned error in response object: {provider_response.error}")
                     # Return last good content or the error message
                     return last_assistant_content or f"[Error from provider: {provider_response.error}]"

                # --- Process Provider Response ---
                assistant_content = provider_response.content
                tool_calls = provider_response.tool_calls # List[ToolCall] or None

                # --- Store Assistant Message (Important for Anthropic) ---
                assistant_message_for_history = {"role": "assistant"}
                if assistant_content is not None:
                    assistant_message_for_history["content"] = assistant_content
                if tool_calls:
                    assistant_message_for_history["tool_calls"] = tool_calls 
                    
                # Add this assistant message to history *before* adding tool results
                # Only add if there is content or tool calls to avoid empty messages
                if assistant_content is not None or tool_calls:
                    self._conversation_manager.add_message(**assistant_message_for_history)
                    self._logger.debug(f"Added assistant message to history. Content: {bool(assistant_content)}, Tool Calls: {len(tool_calls or [])}")
                else:
                    self._logger.warning("Provider response had neither content nor tool_calls.")
                    return last_assistant_content or "" # Return previous content or empty

                if assistant_content is not None:
                     last_assistant_content = assistant_content # Update last good text content

                # --- Check if Tool Calls Were Made ---
                if not tool_calls:
                    self._logger.info("No tool calls requested by the model. Finishing process.")
                    return assistant_content or "" # Final response
                
                self._logger.info(f"Model requested {len(tool_calls)} tool calls.")
                
                # --- Execute Tools ---
                tool_results: List[ToolResult] = []
                # If multiple tool calls, consider executing them concurrently
                # For simplicity now, execute sequentially
                tool_execution_tasks = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, ToolCall):
                        self._logger.error(f"Provider returned invalid tool call object type: {type(tool_call)}. Skipping.")
                        # Add an error result directly
                        tool_results.append(ToolResult(success=False, error="Invalid tool call format from provider", tool_name="unknown"))
                        continue
                        
                    # Create an awaitable task for each tool call
                    tool_execution_tasks.append(self._execute_tool_call(tool_call))
                    
                # Execute tool calls concurrently using asyncio.gather
                if tool_execution_tasks:
                    self._logger.info(f"Executing {len(tool_execution_tasks)} tool calls concurrently...")
                    executed_results = await asyncio.gather(*tool_execution_tasks, return_exceptions=True)
                    self._logger.info("Concurrent tool execution finished.")
                    
                    # Process results (handle potential exceptions returned by gather)
                    for i, result_or_exception in enumerate(executed_results):
                        tool_call = tool_calls[i] # Get corresponding tool call
                        if isinstance(result_or_exception, Exception):
                             self._logger.error(f"Exception during concurrent execution of tool '{tool_call.name}': {result_or_exception}", exc_info=result_or_exception)
                             result = ToolResult(success=False, error=f"Execution exception: {result_or_exception}", tool_name=tool_call.name)
                        elif not isinstance(result_or_exception, ToolResult):
                            # Should not happen if _execute_tool_call always returns ToolResult
                            self._logger.error(f"Unexpected result type from concurrent execution of tool '{tool_call.name}': {type(result_or_exception)}")
                            result = ToolResult(success=False, error="Unexpected execution result type", tool_name=tool_call.name)
                        else:
                             result = result_or_exception # It's a valid ToolResult
                             
                        tool_results.append(result)
                        self._tool_history.append({
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "result": result.result if result.success else None,
                            "error": result.error,
                            "tool_call_id": tool_call.id # Include ID if available
                        })
                
                # --- Add Tool Results to History ---
                needs_last_assistant_message = isinstance(self._provider, AnthropicProvider)
                messages_to_add_to_history = []
                for i, tool_call in enumerate(tool_calls):
                     result = tool_results[i]
                     tool_content = str(result.result) if result.success else str(result.error or "Tool execution failed")
                     
                     add_tool_args = {
                         "tool_call_id": tool_call.id,
                         "tool_name": tool_call.name,
                         "content": tool_content
                     }
                     if needs_last_assistant_message:
                          add_tool_args["last_assistant_message"] = assistant_message_for_history
                          
                     if not tool_call.id:
                         self._logger.error(f"Tool call for '{tool_call.name}' missing ID. Cannot add result to history via provider.")
                         continue
                         
                     try:
                          if not hasattr(self._provider, '_add_tool_message'):
                              raise NotImplementedError(f"Provider {type(self._provider).__name__} missing _add_tool_message method.")
                          
                          returned_messages = self._provider._add_tool_message(**add_tool_args)
                          
                          if not returned_messages:
                               self._logger.warning(f"Provider '_add_tool_message' returned empty list for tool call {tool_call.id}. Result might not be added correctly.")
                               continue
                               
                          messages_to_add_to_history.extend(returned_messages)
                          
                     except NotImplementedError as nie:
                          self._logger.error(f"Cannot add tool result: {nie}")
                          return last_assistant_content or f"[Error: Provider cannot handle tool results: {nie}]"
                     except Exception as e:
                          self._logger.error(f"Error calling provider _add_tool_message for {tool_call.name}: {e}", exc_info=True)
                          return last_assistant_content or f"[Error: Failed to format tool result for {tool_call.name}]"
                
                if messages_to_add_to_history:
                     self._logger.debug(f"Adding {len(messages_to_add_to_history)} tool result message(s) to history.")
                     for msg_dict in messages_to_add_to_history:
                          self._conversation_manager.add_message(**msg_dict)
                else:
                     self._logger.warning("No tool result messages were generated by the provider calls.")

            except AIProcessingError as e: # Catch errors from provider.request or base request_basic
                self._logger.error(f"AI Processing error during tool loop iteration {iteration_count}: {e}", exc_info=True)
                return last_assistant_content or f"[Error during processing: {e}]"
            except Exception as e:
                self._logger.error(f"Unexpected error during tool loop iteration {iteration_count}: {e}", exc_info=True)
                if iteration_count == 1:
                     return f"[Error processing request: {e}]"
                return last_assistant_content or f"[Error during tool execution: {e}]"

        # --- Max Iterations Reached ---
        self._logger.warning(f"Exceeded maximum tool iterations ({max_tool_iterations}). Returning last assistant content.")
        return last_assistant_content or "" # Return the last text content received

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        self._logger.critical(f"--- Entered _execute_tool_call for {tool_call.name}") # Log entry
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        self._logger.info(f"Executing tool: {tool_call.name} with arguments: {tool_call.arguments}")
        
        if not isinstance(tool_args, dict):
            self._logger.warning(f"Tool arguments for '{tool_name}' are not a dictionary ({type(tool_args)}). Attempting execution anyway.")
            # ToolManager expects kwargs, wrap if not dict? Or let execute_tool handle it?
            # For now, pass as is, ToolExecutor might handle it or fail.
            
        try:
            self._logger.critical("--- Entered try block") # Log try entry
            exec_args = tool_call.arguments.copy()
            self._logger.critical("--- Copied args") # Log copy
            exec_args['request_id'] = self._request_id # Use protected attribute
            self._logger.critical(f"--- Added request_id: {self._request_id}") # Log request_id add
            
            self._logger.critical(f"--- About to await tool_manager.execute_tool for {tool_name}") # Existing debug log
            # Check if self._tool_manager is valid right before the call
            if self._tool_manager:
                 self._logger.critical(f"--- self._tool_manager type: {type(self._tool_manager)}")
            else:
                 self._logger.critical("--- self._tool_manager is None!!!")
                 
            result: ToolResult = await self._tool_manager.execute_tool( # Use await
                tool_name=tool_name,
                **exec_args
            )
            self._logger.critical("--- Awaited execute_tool successfully") # Log success
            return result
        except Exception as e:
            self._logger.critical(f"--- EXCEPTION in _execute_tool_call: {e}", exc_info=True) # Log exception
            # Handle unexpected errors during the execution call itself (less likely now with ToolManager handling)
            self._logger.error(f"Unexpected error executing tool '{tool_call.name}': {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Unexpected error in ToolEnabledAI._execute_tool_call: {str(e)}",
                tool_name=tool_call.name
            )

    def get_tool_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of tool usage for the last process_prompt call.
        
        Returns:
            List of tool usage records for the current request.
        """
        return self._tool_history.copy()

    def get_available_tools(self) -> Dict[str, ToolDefinition]:
        """Gets all tools currently registered with the ToolManager."""
        if not self._tool_manager:
             self._logger.warning("ToolManager not initialized.")
             return {}
        try:
             return self._tool_manager.get_all_tools()
        except Exception as e:
             self._logger.error(f"Error retrieving tools from ToolManager: {e}")
             return {}