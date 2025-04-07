"""
Base AI implementation that handles common functionality.
Implements the AIInterface and provides core conversation features.
"""
from typing import Dict, List, Any, Optional, Union
from .interfaces import AIInterface, ProviderInterface
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig
from ..exceptions import AISetupError, AIProcessingError, ErrorHandler
from ..conversation.conversation_manager import ConversationManager, Message
from .provider_factory import ProviderFactory
from .providers.base_provider import BaseProvider
import uuid
from ..prompts.prompt_template import PromptTemplate
from ..config.unified_config import AIConfigError

# Default system prompt if none provided
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions accurately and concisely."""


class AIBase(AIInterface):
    """
    Base implementation of the AI interface.
    Handles provider management, conversation history, and common operations.
    """
    
    def __init__(self, 
                 model: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the base AI implementation.
        
        Args:
            model: The model to use (string ID, or None for default)
            system_prompt: Custom system prompt (or None for default)
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            prompt_template: PromptTemplate service for generating prompts
            
        Raises:
            AISetupError: If initialization fails
        """
        try:
            self._request_id = request_id or str(uuid.uuid4())
            # Create a logger with the appropriate name
            self._logger = logger or LoggerFactory.create(name=f"ai_framework.{self._request_id[:8]}")
            
            # Get unified configuration
            self._config = UnifiedConfig.get_instance()
            
            # --- Detailed Init Logging (moved to DEBUG level) --- 
            model_key = model or self._config.get_default_model()
            self._logger.debug(f"AIBase Init: Requested model key='{model_key}'")
            
            # Store the model key for later reference
            self._model_key = model_key
            
            try:
                self._model_config = self._config.get_model_config(model_key)
                self._logger.debug(f"AIBase Init: Fetched model config: {self._model_config}")
            except Exception as e:
                 self._logger.error(f"AIBase Init: Failed to get model config for '{model_key}': {e}", exc_info=True)
                 raise AISetupError(f"Missing model configuration for '{model_key}'") from e

            provider_model_id = self._model_config.get("model_id", model_key)
            provider_name = self._model_config.get("provider") # Get provider, error if missing
            self._logger.debug(f"AIBase Init: Determined provider_name='{provider_name}', provider_model_id='{provider_model_id}'")
            
            if not provider_name:
                 self._logger.error(f"AIBase Init: Provider name missing in model config for '{model_key}'")
                 raise AISetupError(f"Provider not specified in model configuration for '{model_key}'")

            try:
                provider_config = self._config.get_provider_config(provider_name)
                self._logger.debug(f"AIBase Init: Fetched provider config for '{provider_name}'")
            except AIConfigError as e:
                 self._logger.error(f"AIBase Init: Failed to get provider config for '{provider_name}': {e}", exc_info=True)
                 raise AISetupError(f"Missing provider configuration for '{provider_name}'") from e

            self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
            
            self._logger.debug(f"AIBase Init: Creating provider via ProviderFactory for type='{provider_name}', model_id='{provider_model_id}'")
            self._provider = ProviderFactory.create(
                provider_type=provider_name,
                model_id=provider_model_id, 
                provider_config=provider_config,
                model_config=self._model_config,
                logger=self._logger
            )
            self._logger.debug(f"AIBase Init: ProviderFactory returned instance of type: {type(self._provider).__name__}")
            # --- End Detailed Init Logging ---

            # Set up conversation manager
            self._conversation_manager = ConversationManager()
            
            # Set system prompt (config, parameter, or default)
            self._system_prompt = system_prompt or self._config.get_system_prompt() or self._get_default_system_prompt()
            
            # Add initial system message
            self._conversation_manager.add_message(
                role="system",
                content=self._system_prompt
            )
            
            self._logger.info(f"Initialized AI with model: {model_key}")
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AISetupError(f"Failed to initialize AI: {str(e)}", component="AIBase"),
                self._logger
            )
            self._logger.error(f"Initialization error: {error_response['message']}")
            # Raise AISetupError, chaining the original exception
            raise AISetupError(f"Failed to initialize AI: {str(e)}", component="AIBase") from e
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt based on model configuration.
        Uses the template system if available.
        
        Returns:
            Default system prompt string
        """
        try:
            # Try to use template
            prompt, _ = self._prompt_template.render_prompt(
                template_id="base_ai",
                variables={"model_id": self._model_config.get("model_id", "")}
            )
            return prompt
        except (ValueError, AttributeError):
            # Fallback to hardcoded prompt
            self._logger.warning("System prompt template not found, using fallback")
            return f"You are a helpful AI assistant using the {self._model_config.get('model_id', 'default')} model."
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response as a string
            
        Raises:
            AIProcessingError: If the request fails
        """
        try:
            self._logger.debug(f"Processing request: {prompt[:50]}...")
            
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Get response from provider
            response = self._provider.request(self._conversation_manager.get_messages(), **options)
            
            # --- Corrected Response Handling ---
            content = ""
            if hasattr(response, 'content'): # Check if it has a content attribute
                content = response.content
            elif isinstance(response, dict): # Handle if it's unexpectedly a dict
                content = response.get('content', '')
                self._logger.warning("Provider returned a dict instead of ProviderResponse object.")
            elif isinstance(response, str): # Handle if it's unexpectedly a string
                content = response
                self._logger.warning("Provider returned a string instead of ProviderResponse object.")
            else:
                self._logger.error(f"Received unexpected response type from provider: {type(response)}")
            # ----------------------------------
            
            # Add assistant message with thoughts handling
            self._conversation_manager.add_message(
                role="assistant",
                content=content,
                extract_thoughts=True,
                show_thinking=self._config.show_thinking
            )
            
            return content
            
        except Exception as e:
            # Create the specific error type we want to handle/log
            ai_error = AIProcessingError(f"Request failed: {str(e)}", component="AIBase")

            # Use error handler for standardized error handling, passing the specific error type
            error_response = ErrorHandler.handle_error(
                ai_error, 
                self._logger
            )
            self._logger.error(f"Request error: {error_response['message']}")
            
            # Raise the specific error type, chaining the original exception cause
            raise ai_error from e
    
    def stream(self, prompt: str, **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The complete streamed response as a string
            
        Raises:
            AIProcessingError: If streaming fails
        """
        try:
            self._logger.debug(f"Processing streaming request: {prompt[:50]}...")
            
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Stream the response
            response = self._provider.stream(self._conversation_manager.get_messages(), **options)
            
            # --- Corrected Response Handling ---
            content = ""
            if hasattr(response, 'content'): # Check if it has a content attribute
                content = response.content
            elif isinstance(response, dict): # Handle if it's unexpectedly a dict
                content = response.get('content', '')
                self._logger.warning("Provider returned a dict instead of ProviderResponse object during streaming.")
            elif isinstance(response, str): # Handle if it's unexpectedly a string
                content = response
                self._logger.warning("Provider returned a string instead of ProviderResponse object during streaming.")
            else:
                self._logger.error(f"Received unexpected response type from provider during streaming: {type(response)}")
            # ----------------------------------
            
            # Add assistant message with thoughts handling
            self._conversation_manager.add_message(
                role="assistant", 
                content=content,
                extract_thoughts=True,
                show_thinking=self._config.show_thinking
            )
            
            return content
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AIProcessingError(f"Streaming failed: {str(e)}", component="AIBase"),
                self._logger
            )
            self._logger.error(f"Streaming error: {error_response['message']}")
            raise
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation history.
        
        Clears all messages and restores the system prompt.
        """
        self._conversation_manager.reset()
        self._logger.debug("Conversation history reset")
        
        # Restore system prompt if it exists
        if self._system_prompt:
            self._conversation_manager.add_message(
                role="system",
                content=self._system_prompt
            )
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self._conversation_manager.get_messages()
    
    def get_system_prompt(self) -> Optional[str]:
        """Get the current system prompt."""
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set a new system prompt.
        
        Args:
            system_prompt: New system prompt
        """
        self._system_prompt = system_prompt
        self._conversation_manager.set_system_prompt(system_prompt)
        self._logger.debug("System prompt updated")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self._model_config.get("model_id", ""),
            "provider": self._model_config.get("provider", ""),
            "quality": self._model_config.get("quality", ""),
            "speed": self._model_config.get("speed", ""),
            "parameters": self._model_config.get("parameters", {}),
            "privacy": self._model_config.get("privacy", ""),
            "short_key": getattr(self, "_model_key", "")  # Add short key
        }