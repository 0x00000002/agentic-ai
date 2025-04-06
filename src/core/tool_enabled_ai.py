"""
AI implementation with tool-calling capabilities.
Extends the base AI with the ability to call tools to handle complex tasks.
"""
from typing import Dict, List, Any, Optional, Union, Callable
import json
import inspect
from .base_ai import AIBase
from ..exceptions import AIProcessingError, AIToolError, ErrorHandler
from ..tools.tool_call import ToolCall
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.dynamic_models import Model
from ..tools.tool_manager import ToolManager
from ..tools.models import ToolDefinition, ToolResult
from ..prompts.prompt_template import PromptTemplate


class AI(AIBase):
    """
    AI implementation with tool-calling capabilities.
    Can register and call tools based on the model's output.
    """
    
    def __init__(self, 
                 model=None, 
                 system_prompt=None,
                 logger=None,
                 request_id=None,
                 auto_find_tools=False,
                 tool_manager=None,
                 auto_tool_finding=False,
                 use_structured_tools=True,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the tool-enabled AI.
        
        Args:
            model: The model to use (Model enum or string ID)
            system_prompt: Custom system prompt (or None for default)
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            auto_find_tools: Whether to automatically find tools in the current scope
            tool_manager: Tool manager for handling tools
            auto_tool_finding: Whether to automatically find tools
            use_structured_tools: Whether to use structured tool format
            prompt_template: Prompt template for the AI
        """
        # Use UnifiedConfig instead of ConfigFactory
        unified_config = UnifiedConfig.get_instance()
        
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            logger=logger,
            request_id=request_id,
            prompt_template=prompt_template
        )
        
        self._tools = {}
        self._tool_schemas = {}
        
        # Check if the provider supports tool calling
        self._supports_tools = hasattr(self._provider, "supports_tools") and self._provider.supports_tools
        if not self._supports_tools:
            self._logger.warning(f"Provider {self._model_config.get('provider', 'unknown')} does not fully support tools")
        
        # Automatically find tools if requested
        if auto_find_tools:
            self._find_tools()
        
        # Set up tool manager
        self._tool_manager = tool_manager or ToolManager(
            unified_config=unified_config,
            logger=self._logger
        )
        
        # Tool configuration
        self._auto_tool_finding = auto_tool_finding
        self._use_structured_tools = use_structured_tools
        
        # Internal state for tool calls
        self._tool_history = []
        
        # Initialize auto tool finding if requested
        if auto_tool_finding:
            self._enable_auto_tool_finding()
            
        self._logger.info(f"Initialized {self.__class__.__name__} with model {self.get_model_info().get('model_id')}")
    
    def register_tool(self, 
                     tool_name: str, 
                     tool_function: Callable, 
                     description: str = None, 
                     parameters_schema: Dict = None) -> None:
        """
        Register a tool that can be called by the AI.
        
        Args:
            tool_name: Name of the tool
            tool_function: Function to call
            description: Tool description
            parameters_schema: JSON schema for parameters (or None to infer)
        """
        self._tools[tool_name] = tool_function
        
        # If no parameters schema provided, try to infer from function signature
        if parameters_schema is None:
            parameters_schema = self._infer_schema(tool_function)
        
        # Use function docstring as description if not provided
        if description is None and tool_function.__doc__:
            description = tool_function.__doc__.strip()
        
        self._tool_schemas[tool_name] = {
            "name": tool_name,
            "description": description or f"Call {tool_name} function",
            "parameters": parameters_schema
        }
        
        self._logger.info(f"Registered tool: {tool_name}")
    
    def _infer_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Infer JSON schema from function signature.
        
        Args:
            func: Function to inspect
            
        Returns:
            JSON schema for function parameters
        """
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            # Skip self for methods
            if name == "self":
                continue
                
            properties[name] = {"type": "string"}
            
            # Handle type annotations
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    properties[name]["type"] = "integer"
                elif param.annotation == float:
                    properties[name]["type"] = "number"
                elif param.annotation == bool:
                    properties[name]["type"] = "boolean"
                elif param.annotation == list:
                    properties[name]["type"] = "array"
                elif param.annotation == dict:
                    properties[name]["type"] = "object"
            
            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _find_tools(self) -> None:
        """
        Find tools in the current scope.
        Placeholder for future implementation.
        """
        self._logger.info("Tool discovery not yet implemented")
    
    def _call_tool(self, tool_call: ToolCall) -> str:
        """
        Call a registered tool.
        
        Args:
            tool_call: Tool call object
            
        Returns:
            Tool response as string
            
        Raises:
            AIToolError: If tool execution fails
        """
        tool_name = tool_call.name
        
        if tool_name not in self._tools:
            raise AIToolError(f"Tool not found: {tool_name}", tool_name=tool_name)
        
        try:
            # Parse arguments - handle both string and dict formats
            if isinstance(tool_call.arguments, str):
                try:
                    args = json.loads(tool_call.arguments)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as a string
                    self._logger.warning(f"Invalid JSON arguments: {tool_call.arguments}, trying to parse as is")
                    args = tool_call.arguments
            else:
                # Already a dict
                args = tool_call.arguments
            
            # Call the tool function
            self._logger.info(f"Calling tool: {tool_name} with args: {args}")
            result = self._tools[tool_name](**args)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                result = json.dumps(result)
                
            return result
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AIToolError(f"Failed to execute tool {tool_name}: {str(e)}", tool_name=tool_name),
                self._logger
            )
            self._logger.error(f"Tool execution error: {error_response['message']}")
            raise
    
    def request(self, prompt: str, **options) -> str:
        """
        Send a request to the AI model, with potential tool usage.
        
        Args:
            prompt: The user prompt
            **options: Additional options for the request, including system_prompt
            
        Returns:
            Response from the AI
        """
        # Clear tool history for new request
        self._tool_history = []
        
        # Check for tools that might be relevant to this request
        tools_to_use = []
        
        if self._auto_tool_finding:
            try:
                self._logger.debug("Finding relevant tools...")
                relevant_tools = self._tool_manager.find_tools(prompt)
                self._logger.info(f"Found {len(relevant_tools)} relevant tools: {relevant_tools}")
                tools_to_use = relevant_tools
            except Exception as e:
                self._logger.error(f"Error finding relevant tools: {str(e)}")
                # Continue without tools
        
        # Process request with tools if available
        if tools_to_use:
            return self._request_with_tools(prompt, tools_to_use, **options)
        else:
            # No tools needed or available
            return super().request(prompt, **options)
    
    def _request_with_tools(self, 
                           prompt: str, 
                           tools: List[str], 
                           **options) -> str:
        """
        Process a request with tools and handle tool calls.
        
        Args:
            prompt: The user prompt
            tools: List of tool IDs to use
            **options: Additional options for the request, including system_prompt
            
        Returns:
            Final response from the AI after tool usage
        """
        try:
            # Get tool definitions
            tool_definitions = []
            for tool_id in tools:
                tool_def = self._tool_manager.tool_registry.get_tool(tool_id)
                if tool_def:
                    tool_definitions.append(tool_def)
            
            if not tool_definitions:
                self._logger.warning("No valid tool definitions found, processing without tools")
                return super().request(prompt, **options)
            
            # Process with tools
            return self._process_with_tools(prompt, tool_definitions, **options)
            
        except AIToolError as e:
            self._logger.error(f"Tool error: {str(e)}")
            return f"Error using tools: {str(e)}"
        except Exception as e:
            self._logger.error(f"Error processing with tools: {str(e)}")
            return f"Error: {str(e)}"
    
    def _process_with_tools(self, 
                           prompt: str, 
                           tool_definitions: List[ToolDefinition], 
                           max_iterations: int = 3,
                           **options) -> str:
        """
        Process a request with tools, handling multiple tool calls if needed.
        
        Args:
            prompt: The user prompt
            tool_definitions: List of tool definitions
            max_iterations: Maximum number of tool use iterations
            **options: Additional options for the request, including system_prompt
            
        Returns:
            Final response from the AI
        """
        # Prepare tool definitions for the provider
        provider_tools = []
        for tool_def in tool_definitions:
            provider_tools.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.parameters
            })
        
        current_prompt = prompt
        conversation = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self._logger.info(f"Tool iteration {iteration}/{max_iterations}")
            
            try:
                # Send request to provider with tools
                response = self._provider.request_with_tools(
                    current_prompt, 
                    provider_tools,
                    conversation=conversation,
                    system_prompt=options.get("system_prompt"),
                    structured_tools=self._use_structured_tools
                )
                
                # Check if tool calls were made
                if not response.tool_calls:
                    # No tool calls, just return the content
                    self._logger.info("No tool calls made, returning final response")
                    return response.content
                
                # Process tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    result = self._execute_tool_call(tool_call)
                    tool_results.append(result)
                    self._tool_history.append({
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": result.result if result.status == "success" else result.error
                    })
                
                # Create new prompt with tool results
                current_prompt = self._create_tool_followup_prompt(
                    response.content,
                    response.tool_calls,
                    tool_results
                )
                
                # Update conversation for context
                conversation.append(("user", prompt))
                conversation.append(("assistant", response.content))
                conversation.append(("user", current_prompt))
                
                # If this was just a tool use without further questions, we're done
                if "The AI has completed its use of tools." in current_prompt:
                    self._logger.info("Tools used successfully, returning final response")
                    return response.content
                
            except Exception as e:
                self._logger.error(f"Error in tool processing iteration {iteration}: {str(e)}")
                if iteration == 1:
                    # If we fail on first iteration, try without tools
                    self._logger.info("Falling back to regular request without tools")
                    return super().request(prompt, **options)
                else:
                    # Return what we have so far
                    return f"Error completing the request with tools: {str(e)}. Here's what was determined so far: {current_prompt}"
        
        # If we've exceeded max iterations, return the current state
        self._logger.warning(f"Exceeded maximum tool iterations ({max_iterations})")
        return f"I've reached the maximum number of tool-use iterations. Here's what I've determined so far: {current_prompt}"
    
    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool result
        """
        self._logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        
        try:
            # Convert arguments to kwargs
            kwargs = {}
            if isinstance(tool_call.arguments, dict):
                kwargs = tool_call.arguments
            elif isinstance(tool_call.arguments, str):
                try:
                    kwargs = json.loads(tool_call.arguments)
                except json.JSONDecodeError:
                    self._logger.warning(f"Failed to parse tool arguments as JSON: {tool_call.arguments}")
                    # Try to create a simple dict with a single 'query' parameter
                    kwargs = {"query": tool_call.arguments}
            
            # Execute the tool
            result = self._tool_manager.execute_tool(tool_call.name, **kwargs)
            self._logger.info(f"Tool result status: {result.status}")
            return result
            
        except Exception as e:
            self._logger.error(f"Error executing tool {tool_call.name}: {str(e)}")
            return ToolResult(
                status="error",
                error=f"Error executing tool: {str(e)}",
                result=None
            )
    
    def _create_tool_followup_prompt(self, 
                                    ai_response: str, 
                                    tool_calls: List[ToolCall], 
                                    tool_results: List[ToolResult]) -> str:
        """
        Create a follow-up prompt with tool results.
        
        Args:
            ai_response: The AI's response containing tool calls
            tool_calls: List of tool calls made
            tool_results: List of tool results
            
        Returns:
            Follow-up prompt with tool results
        """
        followup = "I've executed the tools you requested. Here are the results:\n\n"
        
        for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results)):
            followup += f"Tool: {tool_call.name}\n"
            followup += f"Arguments: {json.dumps(tool_call.arguments)}\n"
            
            if result.status == "success":
                followup += f"Result: {result.result}\n"
            else:
                followup += f"Error: {result.error}\n"
                
            if i < len(tool_calls) - 1:
                followup += "\n"
        
        followup += "\nPlease continue based on these results."
        
        # Check if this seems like the end of the conversation
        if len(ai_response.split()) > 50 and not any(marker in ai_response.lower() for marker in ["i need to", "i'll use", "let me use"]):
            followup += " The AI has completed its use of tools."
            
        return followup
    
    def _enable_auto_tool_finding(self) -> None:
        """Enable automatic tool finding using the tool manager."""
        self._logger.info("Enabling automatic tool finding")
        try:
            self._tool_manager.enable_agent_based_tool_finding(self)
            self._auto_tool_finding = True
        except Exception as e:
            self._logger.error(f"Error enabling auto tool finding: {str(e)}")
            self._auto_tool_finding = False
    
    def register_tool(self, name: str, tool_definition: ToolDefinition) -> None:
        """
        Register a tool for use with this AI.
        
        Args:
            name: Tool name
            tool_definition: Tool definition
        """
        self._tool_manager.register_tool(name, tool_definition)
        self._logger.info(f"Registered tool: {name}")
    
    def set_auto_tool_finding(self, enabled: bool) -> None:
        """
        Enable or disable automatic tool finding.
        
        Args:
            enabled: Whether auto tool finding should be enabled
        """
        if enabled and not self._auto_tool_finding:
            self._enable_auto_tool_finding()
        elif not enabled:
            self._auto_tool_finding = False
            self._logger.info("Disabled automatic tool finding")
    
    def get_tool_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of tool usage for the last request.
        
        Returns:
            List of tool usage records
        """
        return self._tool_history.copy()
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools and their schemas.
        
        Returns:
            Dictionary mapping tool names to their schemas
        """
        return self._tool_schemas
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a specific tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return tool_name in self._tools