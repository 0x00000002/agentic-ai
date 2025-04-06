"""
Base provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import AIProviderError
from ...config import get_config
from ...config import UnifiedConfig


class BaseProvider(ProviderInterface):
    """Base implementation for AI providers with common message handling."""
    
    def __init__(self, 
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the base provider.
        
        Args:
            model_id: The model identifier
            provider_config: The provider-specific configuration dictionary
            model_config: The model-specific configuration dictionary
            logger: Logger instance
        """
        # Assign arguments
        self.model_id = model_id
        self.provider_config = provider_config # Use passed provider-specific config
        self.model_config = model_config       # Use passed model-specific config
        self.logger = logger or LoggerFactory.create(name="base_provider")
        self.config = UnifiedConfig.get_instance() # Still need global config access
        
        # Initialize credentials (can use self.provider_config and self.config)
        self._initialize_credentials()
        
        self.logger.info(f"Initialized {self.__class__.__name__} for model {model_id}")
    
    def _initialize_credentials(self) -> None:
        """Initialize credentials for the provider. Override in subclasses if needed."""
        pass
    
    def _map_role(self, role: str) -> str:
        """
        Map a standard role to the provider-specific role name.
        
        Args:
            role: Standard role name ("system", "user", "assistant", etc.)
            
        Returns:
            Provider-specific role name
        """
        # By default, use the same role names
        # Subclasses should override this if they use different role names
        return role
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for the provider API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted messages
        """
        # By default, just map the roles
        formatted_messages = []
        
        for message in messages:
            # Map the role
            role = self._map_role(message.get("role", "user"))
            
            # Create formatted message
            formatted_message = {
                "role": role,
                "content": message.get("content", "")
            }
            
            # Add any additional fields from the original message
            for key, value in message.items():
                if key not in ["role", "content"]:
                    formatted_message[key] = value
            
            formatted_messages.append(formatted_message)
            
        return formatted_messages
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to provider-specific format.
        Override this method in provider implementations.
        
        Args:
            messages: List of messages in standard format
            
        Returns:
            List of messages in provider-specific format
        """
        return messages
    
    def _convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert provider response to standard format.
        Override this method in provider implementations.
        
        Args:
            response: Provider-specific response
            
        Returns:
            Response in standard format
        """
        return response
    
    def _add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, 
                         content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message in provider-specific format.
        Override this method in provider implementations.
        
        Args:
            messages: Current conversation messages
            name: Tool name
            content: Tool response content
            
        Returns:
            Updated messages list
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content)
        })
        return messages
        
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content from a provider response.
        Override this method in provider implementations if needed.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            Content string
        """
        return response.get('content', '')
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Check if response contains tool calls.
        Override this method in provider implementations if needed.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            True if response contains tool calls
        """
        return bool(response.get('tool_calls', []))
    
    def standardize_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ensure all provider responses have a consistent format.
        
        Args:
            response: Provider response (string or dictionary)
            
        Returns:
            Standardized response dictionary with at least a 'content' key
        """
        # If the response is already a dictionary, ensure it has a content key
        if isinstance(response, dict):
            if 'content' not in response:
                response['content'] = response.get('text', '')
            return response
        
        # If the response is a string, convert it to a dictionary with content key
        if isinstance(response, str):
            return {'content': response, 'tool_calls': []}
            
        # Default case - empty response
        return {'content': '', 'tool_calls': []}
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the AI model.
        Override this method in provider implementations, but maintain the return contract.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Either a string response (when no tool calls are needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        # This is a base implementation - providers should override this
        return "This is a base implementation. Override in provider classes."
        
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the AI model.
        Override this method in provider implementations.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        # This is a base implementation - providers should override this
        return "This is a base implementation. Override in provider classes." 