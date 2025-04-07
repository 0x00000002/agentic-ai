"""
Ollama provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, BinaryIO
try:
    import ollama
except ImportError:
    ollama = None

from ..interfaces import ProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import (AIRequestError, AICredentialsError, AIProviderError, AISetupError, 
                         InvalidRequestError, AIAuthenticationError, ModelNotFoundError)
from ...tools.models import ToolCall
from ..models import ProviderResponse, TokenUsage
from .base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama local models."""
    
    # Explicitly state no native tool support
    supports_tools = False

    def __init__(self,
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            model_id: The model identifier
            provider_config: Configuration dictionary for the Ollama provider (e.g., host)
            model_config: Configuration dictionary for the specific model
            logger: Logger instance
        """
        super().__init__(model_id=model_id,
                         provider_config=provider_config,
                         model_config=model_config,
                         logger=logger)
        
        if ollama is None:
            raise AISetupError(
                "Ollama SDK not installed. Please install with 'pip install ollama'.",
                component="ollama"
            )
            
        # Store relevant Ollama options from model_config
        self.ollama_options = {}
        ollama_params = ["temperature", "top_p", "top_k", "num_ctx", "num_predict", "stop"]
        for key in ollama_params:
            if key in self.model_config:
                self.ollama_options[key] = self.model_config[key]
                
        # Consider adding base_url if provided in provider_config
        self.client_options = {}
        if self.provider_config.get("base_url"):
             self.client_options['host'] = self.provider_config["base_url"]
             self._client = ollama.Client(**self.client_options)
             self.logger.info(f"Using Ollama host: {self.client_options['host']}")
        else:
             self._client = ollama.Client() # Use default host
             self.logger.info("Using default Ollama host.")
             
        self.logger.info(f"Initialized Ollama provider for model: {self.model_id}")
        self.logger.debug(f"Ollama Default Options: {self.ollama_options}")

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Ollama API (system, user, assistant)."""
        formatted = []
        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", "")) # Ensure content is string
            
            if role in ["user", "assistant", "system"]:
                formatted.append({"role": role, "content": content})
            elif role == "tool":
                 # How to represent tool results? Append as user message?
                 self.logger.warning("Cannot natively represent tool messages for Ollama. Appending as user message.")
                 formatted.append({
                      "role": "user", 
                      "content": f"[Tool Result for '{msg.get('name', 'unknown')}']: {content}"
                  })
            else:
                 self.logger.warning(f"Unsupported role '{role}' for Ollama, treating as user.")
                 formatted.append({"role": "user", "content": content})
        
        return formatted

    def _prepare_request_payload(self, 
                                 messages: List[Dict[str, Any]], 
                                 options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the Ollama payload for the ollama.chat call.
        Merges parameters into the 'options' dictionary.
        """
        self.logger.debug("Preparing Ollama request payload...")
        
        formatted_messages = self._format_messages(messages)
        
        # Merge default Ollama options with runtime options
        request_options = self.ollama_options.copy()
        # Map standard names if needed (e.g., max_tokens -> num_predict)
        runtime_mapped = {
            "temperature": options.get("temperature"),
            "top_p": options.get("top_p"),
            "top_k": options.get("top_k"),
            "num_ctx": options.get("num_ctx"),
            "num_predict": options.get("max_tokens") or options.get("num_predict"),
            "stop": options.get("stop")
        }
        request_options.update({k: v for k, v in runtime_mapped.items() if v is not None})

        payload = {
            "model": self.model_id,
            "messages": formatted_messages,
            "options": request_options,
            "stream": False # Explicitly set for non-streaming request
        }
        
        # Tools are not natively supported, ignore options["tools"]
        if options.get("tools"):
             self.logger.warning("Ollama provider received 'tools' option, but does not natively support tool calling. Ignoring.")

        self.logger.debug(f"Ollama payload prepared. Messages: {len(payload['messages'])}, Options: {payload['options']}")
        return payload

    # --- IMPLEMENT Required Abstract Methods --- 

    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Makes the actual API call to the Ollama chat endpoint."""
        self.logger.debug("Making Ollama API request...")
        try:
            # Use the configured client (_client handles custom host if set)
            response = self._client.chat(**payload)
            self.logger.debug("Received Ollama API response.")
            return response
        except ollama.ResponseError as e:
             # Handle specific Ollama response errors based on status code
             self.logger.error(f"Ollama Response Error (Status: {e.status_code}): {e}", exc_info=True)
             if e.status_code == 401: # Unauthorized
                  raise AIAuthenticationError(f"Ollama authentication error: {e}", provider="ollama") from e
             elif e.status_code == 404: # Not Found (Model?)
                  raise ModelNotFoundError(f"Ollama model not found?: {e}", provider="ollama", model_id=payload.get("model")) from e
             elif e.status_code == 400: # Bad Request
                  raise InvalidRequestError(f"Invalid request to Ollama: {e}", provider="ollama", status_code=400) from e
             else: # Other HTTP errors
                  raise AIProviderError(f"Ollama API returned status {e.status_code}: {e}", provider="ollama", status_code=e.status_code) from e
        except ollama.RequestError as e: # Network or connection errors
             self.logger.error(f"Ollama Request/Connection Error: {e}", exc_info=True)
             raise AIProviderError(f"Ollama connection error: {e}", provider="ollama") from e
        except Exception as e:
             # Catch unexpected errors
             self.logger.error(f"Unexpected error during Ollama API request: {e}", exc_info=True)
             raise AIProviderError(f"Unexpected error making Ollama request: {e}", provider="ollama") from e

    def _convert_response(self, raw_response: Dict[str, Any]) -> ProviderResponse:
        """Converts the raw Ollama response dictionary into a standardized ProviderResponse model."""
        self.logger.debug("Converting Ollama response to standard ProviderResponse...")
        content = raw_response.get("message", {}).get("content") # Use None if missing
        stop_reason = "stop" if raw_response.get("done") else None
        model_id = raw_response.get("model", self.model_id)
        error = None
        
        # Extract usage if available
        usage_data = None
        prompt_tokens = raw_response.get("prompt_eval_count")
        completion_tokens = raw_response.get("eval_count")
        # Sometimes total_tokens is provided, otherwise calculate
        total_tokens = raw_response.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
            
        if prompt_tokens is not None or completion_tokens is not None:
             usage_data = TokenUsage(
                  prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens,
                  total_tokens=total_tokens
              )

        # Create and return ProviderResponse model instance
        provider_response = ProviderResponse(
            content=content,
            tool_calls=None, # Ollama does not return structured tool calls
            stop_reason=stop_reason,
            usage=usage_data,
            model=model_id,
            error=error, 
            raw_response=None # Exclude raw response by default
        )
        self.logger.debug(f"Standardized response created: {provider_response.model_dump(exclude_none=True, exclude={'raw_response'})}")
        return provider_response
        
    # --- Override Tool-related Helpers --- 
    
    def _add_tool_message(self, 
                         tool_call_id: str, 
                         tool_name: str, 
                         content: str,
                         last_assistant_message: Optional[Dict[str, Any]] = None
                         ) -> List[Dict[str, Any]]:
        """
        Constructs message(s) for tool results for Ollama.
        Since Ollama doesn't have a native tool role, we format it as user message.
        """
        self.logger.warning(f"Constructing tool result for '{tool_name}' as a user message for Ollama.")
        # Format as a user message explaining the tool result
        tool_message = {
            "role": "user", 
            "content": f"[Tool Result for '{tool_name}' (ID: {tool_call_id})]: {str(content)}"
        }
        return [tool_message] # Return as a list containing one dictionary

    def stream(self, messages: List[Dict[str, Any]], **options) -> str:
        """
        Stream a response from the Ollama API.
        (NEEDS REFACTORING to use _prepare_request_payload, _make_api_request structure)
        """
        self.logger.warning("Ollama stream() method needs refactoring.")
        # Placeholder implementation - requires significant changes
        try:
            payload = self._prepare_request_payload(messages, options)
            payload["stream"] = True # Ensure stream is set
            
            # Use the configured client
            stream_response = self._client.chat(**payload)
            
            chunks = []
            for chunk in stream_response:
                if "message" in chunk and "content" in chunk["message"]:
                    chunks.append(chunk["message"]["content"])
            return "".join(chunks)
            
        except Exception as e:
            self.logger.error(f"Ollama streaming failed: {str(e)}", exc_info=True)
            raise AIRequestError(f"Ollama streaming error: {str(e)}", provider="ollama") from e 