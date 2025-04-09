"""
Base provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Type
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import AIProviderError
from ...config import get_config
from ...config import UnifiedConfig
# Import Tool models for type hinting
from ...tools.models import ToolCall, ToolDefinition, ToolResult
# Import the new model
from ..models import ProviderResponse

# Import helper classes
from .message_formatter import MessageFormatter
from .parameter_manager import ParameterManager
from .credential_manager import CredentialManager
from .provider_tool_handler import ProviderToolHandler


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
        self.provider_config = provider_config
        self.model_config = model_config
        self.logger = logger or LoggerFactory.create(name="base_provider")
        self.config = UnifiedConfig.get_instance()
        
        # Extract provider name from class name for helper components
        provider_name = self.__class__.__name__.replace("Provider", "").lower()
        
        # Initialize helper components
        self._initialize_helpers(provider_name)
        
        # Initialize credentials through credential manager
        self._initialize_credentials()
        
        self.logger.debug(f"Initialized {self.__class__.__name__} for model {model_id}")
    
    def _initialize_helpers(self, provider_name: str) -> None:
        """
        Initialize helper components.
        
        Args:
            provider_name: Provider name for helper components
        """
        # Create message formatter with role mapping from subclass if available
        role_mapping = getattr(self.__class__, "_ROLE_MAP", None)
        self.message_formatter = MessageFormatter(role_mapping=role_mapping, logger=self.logger)
        
        # Create parameter manager with default parameters
        default_params = {}
        # Get runtime parameters (temp, etc.) merged by UnifiedConfig
        model_runtime_params = self.model_config.get("runtime_parameters", {}) 
        allowed_params = set()
        param_mapping = {}
        
        # Allow subclasses to provide provider-specific defaults/allowed/mapping
        if hasattr(self.__class__, "DEFAULT_PARAMETERS"):
            default_params = self.__class__.DEFAULT_PARAMETERS.copy()
        if hasattr(self.__class__, "ALLOWED_PARAMETERS"):
            allowed_params = set(self.__class__.ALLOWED_PARAMETERS)
        if hasattr(self.__class__, "PARAMETER_MAPPING"):
            param_mapping = self.__class__.PARAMETER_MAPPING.copy()
            
        self.parameter_manager = ParameterManager(
            default_parameters=default_params,
            model_parameters=model_runtime_params, # Use the runtime parameters dict
            allowed_parameters=allowed_params,
            parameter_mapping=param_mapping,
            logger=self.logger
        )
        
        # Create credential manager
        self.credential_manager = CredentialManager(
            provider_name=provider_name, 
            provider_config=self.provider_config,
            logger=self.logger,
            config=self.config
        )
        
        # Create tool manager (renamed)
        self.tool_manager = ProviderToolHandler(
            provider_name=provider_name,
            logger=self.logger
        )
    
    def _initialize_credentials(self) -> None:
        """Initialize credentials for the provider."""
        # Use credential manager to load credentials
        try:
            self.credential_manager.load_credentials()
        except Exception as e:
            self.logger.error(f"Failed to initialize credentials: {e}", exc_info=True)
            raise
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for the provider API.
        
        Args:
            messages: List of message dictionaries in standard format
            
        Returns:
            Formatted messages suitable for the provider's API
        """
        return self.message_formatter.format_messages(messages)

    def _prepare_request_payload(self, 
                               messages: List[Dict[str, Any]], 
                               options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the request payload for the provider API.
        
        Args:
            messages: List of conversation messages
            options: Runtime options for the request
            
        Returns:
            Complete request payload dictionary
        """
        self.logger.debug(f"Preparing request payload with options: {options}")
        
        # Special keys to handle separately
        special_keys = ["system_prompt", "tools", "tool_choice"]
        
        # Process parameters
        params, special_params = self.parameter_manager.prepare_request_payload(
            runtime_options=options,
            special_keys=special_keys
        )
        
        # Format messages (potentially with system prompt)
        system_prompt = special_params.get("system_prompt")
        formatted_messages = self.message_formatter.format_messages(
            messages=messages,
            system_prompt=system_prompt
        )
        
        # Start building payload
        payload = {
            "messages": formatted_messages,
            "model": self.model_id
        }
        
        # Add remaining parameters
        payload.update(params)
        
        # Handle tools if present
        tools = special_params.get("tools")
        tool_choice = special_params.get("tool_choice")
        
        if tools:
            try:
                formatted_tools = self.tool_manager.format_tools(tools)
                if formatted_tools:
                    payload["tools"] = formatted_tools
                    
                    # Add tool_choice if specified
                    if tool_choice:
                        formatted_choice = self.tool_manager.format_tool_choice(tool_choice)
                        payload["tool_choice"] = formatted_choice
            except Exception as e:
                self.logger.error(f"Failed to format tools: {e}", exc_info=True)
        
        self.logger.debug(f"Final request payload keys: {list(payload.keys())}")
        return payload

    def _add_tool_message(self, messages: List[Dict[str, Any]], 
                        name: str, 
                        content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation.
        
        Args:
            messages: Current conversation messages
            name: Tool name
            content: Tool response content
            
        Returns:
            Updated messages list
        """
        return self.tool_manager.add_tool_message(messages, name, content)
        
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content from a provider response.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            Content string
        """
        return response.get('content', '')
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Check if response contains tool calls.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            True if response contains tool calls
        """
        return self.tool_manager.has_tool_calls(response)
    
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
    
    def _convert_response(self, raw_response: Any) -> ProviderResponse:
        """
        Convert the raw provider response to a standardized ProviderResponse.
        
        Args:
            raw_response: Raw response from the provider's API
            
        Returns:
            Standardized ProviderResponse object
        """
        raise NotImplementedError(f"Subclasses must implement _convert_response")
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> ProviderResponse:
        """
        Make a request to the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional request options
            
        Returns:
            Standardized ProviderResponse object
        """
        if isinstance(messages, str):
             # Convert simple prompt string to messages list
             messages = [{"role": "user", "content": messages}]
        
        # Prepare the payload
        payload = self._prepare_request_payload(messages, options)
        
        try:
            # Make the API request
            raw_response = self._make_api_request(payload)
            
            # Convert to standardized response
            provider_response = self._convert_response(raw_response)
            
            # Ensure the return type is ProviderResponse
            if not isinstance(provider_response, ProviderResponse):
                 self.logger.error(f"{self.__class__.__name__}._convert_response did not return a ProviderResponse object (returned {type(provider_response)}). Attempting conversion.")
                 # Attempt to create the model from the dict if possible
                 try:
                      provider_response = ProviderResponse(**provider_response)
                 except Exception as conversion_error:
                      self.logger.error(f"Failed to convert provider response dict to ProviderResponse model: {conversion_error}")
                      raise AIProviderError(f"Invalid response format from {self.__class__.__name__}._convert_response")
            
            return provider_response
            
        except Exception as e:
             # Catch potential API errors or conversion errors
             self.logger.error(f"Error during {self.__class__.__name__} request: {e}", exc_info=True)
             return ProviderResponse(error=str(e))
             
    def _make_api_request(self, payload: Dict[str, Any]) -> Any:
        """
        Make the actual API request to the provider.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            Raw response from the provider's API
        """
        raise NotImplementedError(f"Subclasses must implement _make_api_request")
        
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        raise NotImplementedError(f"Subclasses must implement stream")
    
    def build_tool_result_messages(self, 
                                 tool_calls: List[ToolCall], 
                                 tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Build formatted messages for tool results.
        
        Args:
            tool_calls: List of tool calls
            tool_results: List of tool results
            
        Returns:
            List of messages representing tool results
        """
        return self.tool_manager.build_tool_result_messages(tool_calls, tool_results) 