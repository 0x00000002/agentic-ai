"""
Tool management for AI providers.
Handles tool formatting, validation, and provider-specific tool interfaces.
"""
from typing import Dict, Any, Optional, List, Set, Union
from ...utils.logger import LoggerInterface, LoggerFactory
from ...tools.models import ToolCall, ToolDefinition, ToolResult
from ...tools.tool_registry import ToolRegistry


class ToolManager:
    """
    Manages tool-related functionality for AI providers.
    
    This class is responsible for:
    - Formatting tools for different providers
    - Handling tool calls in responses
    - Building tool result messages
    - Managing tool choice settings
    """
    
    def __init__(self, 
                 provider_name: str,
                 logger: Optional[LoggerInterface] = None,
                 tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the tool manager.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            logger: Optional logger instance
            tool_registry: Optional tool registry instance
        """
        self.provider_name = provider_name
        self.logger = logger or LoggerFactory.create(name=f"{provider_name}_tools")
        self.tool_registry = tool_registry or ToolRegistry()
        
    def format_tools(self, tool_names: Union[List[str], Set[str]]) -> List[Dict[str, Any]]:
        """
        Format tools for the provider's API.
        
        Args:
            tool_names: List or set of tool names to format
            
        Returns:
            List of formatted tool definitions
        """
        try:
            # Convert to set if it's a list
            tool_name_set = set(tool_names) if isinstance(tool_names, list) else tool_names
            
            # Use tool registry to format tools for the provider
            formatted_tools = self.tool_registry.format_tools_for_provider(
                self.provider_name.upper(),  # Convert to uppercase for registry lookup
                tool_name_set
            )
            
            return formatted_tools or []
            
        except Exception as e:
            self.logger.error(f"Failed to format tools for provider '{self.provider_name}': {e}", exc_info=True)
            return []
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from a provider response.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            List of ToolCall objects
        """
        # Basic implementation - override in provider-specific subclasses if needed
        tool_calls = []
        
        # Standard format - look for 'tool_calls' key
        if 'tool_calls' in response:
            raw_tool_calls = response['tool_calls']
            if not raw_tool_calls:
                return []
                
            for raw_call in raw_tool_calls:
                # Handle dict format
                if isinstance(raw_call, dict):
                    try:
                        # Get the required fields
                        call_id = raw_call.get('id', '')
                        name = raw_call.get('function', {}).get('name', '')
                        arguments_str = raw_call.get('function', {}).get('arguments', '{}')
                        
                        # Convert string arguments to dict if needed
                        if isinstance(arguments_str, str):
                            try:
                                import json
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse arguments JSON: {arguments_str}")
                                arguments = {"_raw_args": arguments_str}
                        else:
                            arguments = arguments_str
                            
                        # Create the ToolCall object
                        tool_call = ToolCall(
                            id=call_id,
                            name=name,
                            arguments=arguments
                        )
                        tool_calls.append(tool_call)
                    except Exception as e:
                        self.logger.error(f"Error creating ToolCall: {e}")
                # Handle object format (e.g., OpenAI SDK objects)
                elif hasattr(raw_call, 'function') and hasattr(raw_call.function, 'name'):
                    try:
                        # Similar approach for object attributes
                        call_id = getattr(raw_call, 'id', '')
                        name = raw_call.function.name
                        arguments_str = getattr(raw_call.function, 'arguments', '{}')
                        
                        # Convert string arguments to dict if needed
                        if isinstance(arguments_str, str):
                            try:
                                import json
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to parse arguments JSON: {arguments_str}")
                                arguments = {"_raw_args": arguments_str}
                        else:
                            arguments = arguments_str
                            
                        tool_call = ToolCall(
                            id=call_id,
                            name=name,
                            arguments=arguments
                        )
                        tool_calls.append(tool_call)
                    except Exception as e:
                        self.logger.error(f"Error creating ToolCall from object: {e}")
                    
        return tool_calls
    
    def format_tool_choice(self, tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the tool_choice parameter for the provider.
        
        Args:
            tool_choice: Tool choice parameter value
            
        Returns:
            Formatted tool_choice value
        """
        # Basic implementation - pass through for most providers
        # Override in provider-specific subclasses if needed
        return tool_choice
    
    def add_tool_message(self, 
                       messages: List[Dict[str, Any]],
                       name: str,
                       content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the messages list.
        
        Args:
            messages: Current list of messages
            name: Tool name
            content: Tool response content
            
        Returns:
            Updated list of messages
        """
        # Basic implementation
        tool_message = {
            "role": "tool",
            "name": name,
            "content": str(content)
        }
        
        # Create a new list to avoid modifying the original
        return messages + [tool_message]
    
    def build_tool_result_messages(self, 
                                 tool_calls: List[ToolCall], 
                                 tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Build messages from tool results.
        
        Args:
            tool_calls: List of tool calls
            tool_results: List of tool results
            
        Returns:
            List of messages representing tool results
        """
        if not tool_calls or not tool_results:
            return []
            
        # Build a mapping of tool_call ids to tool_calls for easier lookup
        call_map = {call.id: call for call in tool_calls if call.id}
        
        # Build messages
        messages = []
        for result in tool_results:
            # Find the matching tool call if possible
            tool_name = result.tool_name
            content = str(result.result) if result.success else str(result.error or "Error")
            
            for call_id, call in call_map.items():
                if tool_name == call.name:
                    message = {
                        "role": "tool",
                        "name": call.name,
                        "content": content,
                        "tool_call_id": call_id
                    }
                    messages.append(message)
                    break
                
        return messages
    
    def has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Check if a response contains tool calls.
        
        Args:
            response: Provider response
            
        Returns:
            True if response contains tool calls, False otherwise
        """
        # Standard format - look for 'tool_calls' key with non-empty value
        return bool(response.get('tool_calls', []))
    
    def get_tool_by_name(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolDefinition or None if not found
        """
        return self.tool_registry.get_tool(tool_name) 