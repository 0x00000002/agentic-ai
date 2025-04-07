"""
Message formatting for AI providers.
Handles standardization and provider-specific message format conversions.
"""
from typing import List, Dict, Any, Optional, Union, Callable
from ...utils.logger import LoggerInterface, LoggerFactory


class MessageFormatter:
    """
    Handles message formatting for AI providers.
    
    This class is responsible for converting between the standard message format
    and provider-specific formats, including role mapping and special field handling.
    """
    
    def __init__(self, 
                 role_mapping: Optional[Dict[str, str]] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the message formatter.
        
        Args:
            role_mapping: Optional mapping from standard roles to provider-specific roles
            logger: Optional logger instance
        """
        self.logger = logger or LoggerFactory.create(name="message_formatter")
        # Default role mapping (standard roles)
        self._role_map = role_mapping or {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool"
        }
        
    def map_role(self, role: str) -> str:
        """
        Map a standard role to a provider-specific role.
        
        Args:
            role: Standard role name (system, user, assistant, tool)
            
        Returns:
            Provider-specific role name
        """
        return self._role_map.get(role, "user")  # Default to user if role not found
    
    def format_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single message for the provider.
        
        Args:
            message: Message in standard format
            
        Returns:
            Formatted message for the provider
        """
        role = self.map_role(message.get("role", "user"))
        content = message.get("content", "")
        
        # Create base formatted message
        formatted_message = {
            "role": role,
            "content": content
        }
        
        # Add special fields as needed
        if "name" in message and role == self.map_role("tool"):
            formatted_message["name"] = message["name"]
            
        if "tool_calls" in message and role == self.map_role("assistant"):
            formatted_message["tool_calls"] = message["tool_calls"]
            
        # Additional custom processing
        return self.post_process_message(formatted_message, message)
    
    def post_process_message(self, 
                            formatted_message: Dict[str, Any], 
                            original_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform provider-specific post-processing on a formatted message.
        
        Subclasses can override this method to perform additional formatting.
        
        Args:
            formatted_message: Message formatted by format_message
            original_message: Original message before formatting
            
        Returns:
            Post-processed message
        """
        # Base implementation does nothing
        return formatted_message
    
    def format_messages(self, 
                       messages: List[Dict[str, Any]],
                       system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format a list of messages for the provider.
        
        Args:
            messages: List of messages in standard format
            system_prompt: Optional system prompt to prepend
            
        Returns:
            List of formatted messages
        """
        formatted_messages = []
        
        # Add system prompt if provided and not already present
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            formatted_messages.append(self.format_message({
                "role": "system",
                "content": system_prompt
            }))
            
        # Format each message
        for message in messages:
            formatted_messages.append(self.format_message(message))
            
        return formatted_messages
    
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
        tool_message = {
            "role": "tool",
            "name": name,
            "content": str(content)
        }
        
        # Format the tool message according to provider requirements
        formatted_tool_message = self.format_message(tool_message)
        
        # Create a new list to avoid modifying the original
        return messages + [formatted_tool_message]
    
    def adapt_for_provider(self, 
                         standard_messages: List[Dict[str, Any]], 
                         provider_name: str,
                         custom_formatter: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Adapt standard messages for a specific provider.
        
        Args:
            standard_messages: List of messages in standard format
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            custom_formatter: Optional custom formatter function
            
        Returns:
            Messages adapted for the provider
        """
        if custom_formatter:
            return custom_formatter(standard_messages)
        
        # Default adaptation is just regular formatting
        return self.format_messages(standard_messages) 