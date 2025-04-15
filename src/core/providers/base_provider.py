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

import asyncio # Add asyncio
import abc # Import Abstract Base Classes
from abc import abstractmethod # Import abstractmethod decorator


class BaseProvider(ProviderInterface, abc.ABC):
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
        tool_names = special_params.get("tools")
        tool_choice = special_params.get("tool_choice")
        
        formatted_tools = None
        if tool_names:
            try:
                # Check if 'tool_names' is actually a list of names (strings)
                if isinstance(tool_names, list) and all(isinstance(name, str) for name in tool_names):
                    # If it's names, use the handler to format them
                    formatted_tools = self.tool_manager.format_tools(tool_names)
                # Check if it's the already formatted list of dicts from ToolEnabledAI
                elif isinstance(tool_names, list) and all(isinstance(item, dict) for item in tool_names):
                    self.logger.debug("Received pre-formatted tools list. Using directly.")
                    formatted_tools = tool_names # Use the pre-formatted list
                else:
                    self.logger.warning(f"Received 'tools' option in unexpected format: {type(tool_names)}. Expected List[str] or List[Dict]. Skipping tool formatting.")

                if formatted_tools:
                    payload["tools"] = formatted_tools
                    # Add tool_choice if specified and tools were successfully formatted/provided
                    if tool_choice:
                        # Assume tool_manager handles formatting tool_choice regardless
                        formatted_choice = self.tool_manager.format_tool_choice(tool_choice)
                        payload["tool_choice"] = formatted_choice

            except Exception as e:
                self.logger.error(f"Failed to format tools: {e}", exc_info=True)
        elif tool_names:
            # Log a warning if 'tools' is present but not a list (unexpected format)
            self.logger.warning(f"Received 'tools' option in unexpected format: {type(tool_names)}. Expected List. Skipping tool formatting.")

        self.logger.debug(f"Final request payload keys after tool processing: {list(payload.keys())}")
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
        Converts the raw response from the underlying provider API into a standardized ProviderResponse object.
        This method MUST be implemented by subclasses.
        
        Args:
            raw_response: The raw data returned by the provider's API call.
            
        Returns:
            A ProviderResponse object.
        """
        raise NotImplementedError("Subclasses must implement _convert_response")
    
    @abstractmethod
    def _get_error_map(self) -> Dict[Type[Exception], Type[AIProviderError]]:
        """
        Returns a mapping from provider-specific exceptions to framework exceptions.
        This MUST be implemented by subclasses.
        
        Returns:
            Dict[Type[Exception], Type[AIProviderError]]: Mapping dictionary.
        """
        pass
        
    def _handle_api_error(self, error: Exception, payload: Dict[str, Any]) -> AIProviderError:
        """
        Handles errors raised during the _make_api_request call, mapping them to 
        standard framework exceptions using the subclass-specific error map.
        
        Args:
            error: The original exception raised by the provider SDK.
            payload: The request payload (for context like model ID).
        
        Returns:
            An instance of a framework exception (subclass of AIProviderError).
        """
        error_map = self._get_error_map()
        provider_name = self.__class__.__name__.replace("Provider", "").lower()
        model_id = payload.get("model", self.model_id)
        status_code = getattr(error, 'status_code', None) # Attempt to get status code

        for provider_exception_type, framework_exception_type in error_map.items():
            if isinstance(error, provider_exception_type):
                # Log the original error before re-raising mapped version
                self.logger.error(
                    f"Provider Error ({provider_name}, model: {model_id}): Encountered {type(error).__name__}. Mapping to {framework_exception_type.__name__}. Original error: {error}", 
                    exc_info=True # Include traceback for original error
                )
                
                # Create mapped exception instance
                # Pass relevant context like provider name, status code, model_id
                kwargs = {'provider': provider_name, 'status_code': status_code}
                if issubclass(framework_exception_type, ModelNotFoundError):
                    kwargs['model_id'] = model_id
                
                # Remove None values from kwargs before instantiation
                final_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                
                # Instantiate with original error message and context
                mapped_exception = framework_exception_type(str(error), **final_kwargs)
                # Preserve the original error cause
                raise mapped_exception from error
                
        # If no specific mapping found, log and raise a generic AIProviderError
        self.logger.error(
            f"Unmapped Provider Error ({provider_name}, model: {model_id}): Encountered {type(error).__name__}: {error}", 
            exc_info=True
        )
        raise AIProviderError(
            f"Provider {provider_name} encountered an unmapped error: {error}", 
            provider=provider_name,
            status_code=status_code
        ) from error

    async def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> ProviderResponse:
        """
        Make an asynchronous request to the AI model provider.
        Handles payload preparation, calling the provider-specific API request, 
        error handling/mapping, and response conversion.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional request options
            
        Returns:
            Standardized ProviderResponse object
        
        Raises:
            AIProviderError (or subclass): If the provider request fails after handling.
        """
        if isinstance(messages, str):
             # Convert simple prompt string to messages list
             messages = [{"role": "user", "content": messages}]
        
        payload = {}
        try:
            # Prepare payload (remains synchronous)
            payload = self._prepare_request_payload(messages, options)
            
            # Make the API request asynchronously using the subclass implementation
            # (The retry decorator is applied within _make_api_request)
            raw_response = await self._make_api_request(payload)
            
            # Convert raw response to standardized ProviderResponse (remains synchronous)
            provider_response = self._convert_response(raw_response)
            
            return provider_response
            
        except Exception as e:
             # Centralized error handling and mapping
             # This will catch errors from _prepare_request_payload OR _make_api_request
             # It calls the subclass's error map via _handle_api_error
             # _handle_api_error logs appropriately and raises the mapped framework exception
             # We simply re-raise the exception returned/raised by _handle_api_error
             raise self._handle_api_error(e, payload)

    @abstractmethod
    async def _make_api_request(self, payload: Dict[str, Any]) -> Any:
        """
        Makes the actual asynchronous API request to the provider.
        This method MUST be implemented by subclasses.
        
        Args:
            payload: The request payload.
            
        Returns:
            The raw response from the provider API.
        """
        raise NotImplementedError("Subclasses must implement _make_api_request")
        
    async def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response asynchronously from the AI model provider.
        NOTE: This base implementation raises NotImplementedError. Subclasses must provide
        the actual async streaming logic.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        # Basic message handling
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        # Subclasses need to implement the actual async streaming logic
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming.")
    
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