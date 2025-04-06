"""
Base provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import AIProviderError
from ...config import get_config
from ...config import UnifiedConfig
# Import Tool models for type hinting
from ...tools.models import ToolCall, ToolDefinition
# Import ToolRegistry to format tools
from ...tools.tool_registry import ToolRegistry
# Import the new model
from ..models import ProviderResponse


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
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats messages, mapping roles and potentially handling provider-specific structures.
        Subclasses should override if complex formatting (e.g., system prompt placement) is needed.
        
        Args:
            messages: List of message dictionaries in standard format.
            
        Returns:
            Formatted messages suitable for the provider's API.
        """
        formatted_messages = []
        for message in messages:
            role = self._map_role(message.get("role", "user"))
            content = message.get("content", "")
            
            # Basic structure
            formatted_message = {
                "role": role,
                "content": content
            }
            
            # Include other relevant fields (like name for tool messages)
            if "name" in message and role == self._map_role("tool"):
                 formatted_message["name"] = message["name"]
            if "tool_calls" in message and role == self._map_role("assistant"):
                 formatted_message["tool_calls"] = message["tool_calls"]
                 
            # Allow subclasses to add/modify fields if necessary
            formatted_messages.append(self._post_process_formatted_message(formatted_message, message))
            
        return formatted_messages

    def _post_process_formatted_message(self, formatted_message: Dict[str, Any], original_message: Dict[str, Any]) -> Dict[str, Any]:
        """ Hook for subclasses to modify a formatted message before it's added to the list. """
        # Base implementation does nothing
        return formatted_message

    def _prepare_request_payload(self, 
                                 messages: List[Dict[str, Any]], 
                                 options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the final request payload dictionary for the provider's API.
        Handles merging default/model/runtime parameters, system prompt, tools, etc.
        Subclasses might override parts, especially for provider-specific parameter names.
        
        Args:
            messages: The list of conversation messages (already role-mapped if needed).
            options: Runtime options for the request (e.g., temperature, max_tokens, tools).
            
        Returns:
            The complete request payload dictionary.
        """
        self.logger.debug(f"Preparing request payload with options: {options}")
        
        # 1. Parameter Merging (Defaults < Model Config < Runtime Options)
        # Start with provider/model defaults (subclass should populate self.parameters)
        payload_params = self.parameters.copy() if hasattr(self, 'parameters') else {}
        # Merge model-specific config parameters (already handled in subclass __init__ usually)
        # payload_params.update(self.model_config.get("parameters", {}))
        # Merge runtime options, overriding defaults/model config
        payload_params.update(options)

        # 2. Handle System Prompt
        system_prompt = payload_params.pop("system_prompt", None)
        # How system prompt is added depends on provider (handled in _format_messages or here)
        # Example: some providers might take it as a top-level parameter
        # if system_prompt:
        #     payload["system"] = system_prompt 
        # Others expect it as the first message (handled in _format_messages override)

        # 3. Format Messages (using potentially overridden _format_messages)
        # Note: Pass system_prompt to _format_messages if it needs to handle it
        formatted_messages = self._format_messages(messages)
        payload = {"messages": formatted_messages}

        # 4. Handle Tools
        tools = payload_params.pop("tools", None)
        tool_choice = payload_params.pop("tool_choice", None)
        if tools:
            # Requires ToolRegistry access - maybe pass it in or get instance?
            # Assuming ToolRegistry is available or passed during init
            # This part needs refinement based on ToolRegistry availability
            try:
                tool_registry = ToolRegistry() # Or get instance/passed registry
                formatted_tools = tool_registry.format_tools_for_provider(
                    self.__class__.__name__.upper(), # Derive provider name
                    set(tools) if isinstance(tools, list) else None # Assume tools is list of names
                )
                if formatted_tools:
                    payload["tools"] = formatted_tools
                    if tool_choice:
                        # Add tool_choice logic (provider-specific formatting needed)
                        # payload["tool_choice"] = self._format_tool_choice(tool_choice)
                        payload["tool_choice"] = tool_choice # Placeholder - needs provider formatting
            except Exception as e:
                self.logger.error(f"Failed to format tools for provider: {e}", exc_info=True)

        # 5. Add remaining merged parameters to the payload
        # Subclasses might need to rename keys (e.g., max_tokens vs max_tokens_to_sample)
        final_payload = self._map_payload_parameters(payload_params)
        payload.update(final_payload)
        
        # Add model ID
        payload["model"] = self.model_id
        
        self.logger.debug(f"Final request payload keys: {list(payload.keys())}")
        # Avoid logging full messages/tools payload by default unless needed
        # self.logger.debug(f"Final request payload: {payload}") 
        return payload

    def _map_payload_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ Hook for subclasses to rename parameters to provider-specific names. """
        # Base implementation returns parameters as is
        return params

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
    
    def _convert_response(self, raw_response: Any) -> ProviderResponse:
        """
        Abstract method for subclasses to convert the raw provider response 
        into a standardized ProviderResponse object.
        """
        raise NotImplementedError(f"Subclasses must implement _convert_response")
    
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
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> ProviderResponse:
        """
        Make a request to the AI model.
        This base method now handles payload preparation.
        Subclasses MUST implement _make_api_request().
        
        Args:
            messages: The conversation messages (List[Dict]) or a simple string prompt.
            options: Additional request options (temperature, tools, etc.).
            
        Returns:
            Standardized ProviderResponse object.
        """
        if isinstance(messages, str):
             # Convert simple prompt string to messages list
             messages = [{"role": "user", "content": messages}]
        
        # Prepare the payload using the new standardized method
        payload = self._prepare_request_payload(messages, options)
        
        try:
            # Subclasses implement the actual API call
            raw_response = self._make_api_request(payload)
            
            # Subclasses implement response conversion to standard ProviderResponse object
            provider_response = self._convert_response(raw_response)
            
            # Ensure the return type is ProviderResponse
            if not isinstance(provider_response, ProviderResponse):
                 self.logger.error(f"{self.__class__.__name__}._convert_response did not return a ProviderResponse object (returned {type(provider_response)}). Attempting conversion.")
                 # Attempt to create the model from the dict if possible
                 try:
                      provider_response = ProviderResponse(**provider_response)
                 except Exception as conversion_error:
                      self.logger.error(f"Failed to convert provider response dict to ProviderResponse model: {conversion_error}")
                      # Raise a more specific error or return a default error response?
                      # For now, re-raise the original error from _make_api_request if it failed
                      raise AIProviderError(f"Invalid response format from {self.__class__.__name__}._convert_response")
            
            return provider_response
            
        except Exception as e:
             # Catch potential API errors or conversion errors
             self.logger.error(f"Error during {self.__class__.__name__} request: {e}", exc_info=True)
             # TODO: Map to specific AIProviderError subclass
             # Return an error ProviderResponse object
             return ProviderResponse(error=str(e))
             
    def _make_api_request(self, payload: Dict[str, Any]) -> Any:
        """ Abstract method for subclasses to implement the actual API call. 
            Should return the raw response object from the provider's SDK.
        """
        raise NotImplementedError(f"Subclasses must implement _make_api_request")
        
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