"""
Anthropic provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from ..interfaces import ProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError, AIAuthenticationError, AIRateLimitError, ModelNotFoundError, ContentModerationError, InvalidRequestError
from ...tools.models import ToolResult, ToolCall
from ..models import ProviderResponse, TokenUsage
from .base_provider import BaseProvider
import json
import os
import httpx
import asyncio # Add asyncio

# Import the retry decorator
from ...utils.retry import async_retry_with_backoff

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic # Import Async client
    from anthropic.types import Message
    # Import specific Anthropic errors for retrying
    from anthropic import (
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        # APIStatusError, # Avoid retrying all status errors for now
        # APITimeoutError # Seems less common/often wrapped
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Message = None # Define Message as None if import fails
    # Define dummy exceptions if import fails, so the tuple below doesn't break
    APIConnectionError = RateLimitError = InternalServerError = Exception

# Define exceptions specific to Anthropic that warrant a retry
ANTHROPIC_RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


class AnthropicProvider(BaseProvider, ToolCapableProviderInterface):
    """Provider implementation for Anthropic Claude models with tools support."""
    
    # Add property for tool support
    supports_tools = True
    
    # Role mapping for Anthropic API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        # Role for tool results, user message containing tool_result content blocks
        "tool": "user" 
    }
    
    # --- Add Class Attributes for Parameter Management ---
    # Parameters allowed by the Anthropic API
    ALLOWED_PARAMETERS = {
        "temperature", "max_tokens", "top_p", "top_k", "stop_sequences"
    }
    
    # Default parameters for Anthropic API
    DEFAULT_PARAMETERS = {
        "temperature": 0.7,
        "max_tokens": 4096  # Default fallback
    }
    # --------------------------------------------------
    
    def __init__(self, 
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_id: Model identifier
            provider_config: Provider configuration dictionary
            model_config: Model configuration dictionary
            logger: Logger instance
        """
        if not ANTHROPIC_AVAILABLE:
            raise AIProviderError("Anthropic Python SDK not installed. Run 'pip install anthropic'")
            
        super().__init__(model_id, provider_config, model_config, logger)
        
        self.logger.info(f"Initialized Anthropic provider with model {model_id}")
    
    def _initialize_credentials(self) -> None:
        """Initialize Anthropic API credentials."""
        try:
            # Get API key from configuration
            api_key = self.provider_config.get("api_key")
            if not api_key:
                api_key = self.config.get_api_key("anthropic")
                
            if not api_key:
                # Try getting from environment directly
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                
            if not api_key:
                self.logger.error("No Anthropic API key found in configuration or environment")
                raise AICredentialsError("No Anthropic API key found")
                
            self.logger.info("Found Anthropic API key")
                
            # Set up Anthropic client - Use Async
            try:
                self.client = AsyncAnthropic(api_key=api_key) # Use AsyncAnthropic
                self.logger.info("Successfully initialized AsyncAnthropic client")
            except Exception as e:
                self.logger.error(f"Failed to create AsyncAnthropic client: {str(e)}")
                raise AICredentialsError(f"Failed to create AsyncAnthropic client: {str(e)}")
            
        except anthropic.AuthenticationError as e:
             self.logger.error(f"Anthropic Authentication Error: {e}")
             raise AIAuthenticationError(f"Anthropic authentication failed: {e}", provider="anthropic") from e
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic credentials: {str(e)}")
            raise AICredentialsError(f"Failed to initialize Anthropic credentials: {str(e)}", provider="anthropic") from e
    
    def _map_role(self, role: str) -> str:
        """Map standard role to Anthropic role."""
        return self._ROLE_MAP.get(role, "user")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formats messages for Anthropic, excluding the system message."""
        formatted = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue # Skip system message, handled separately
                
            anthropic_role = self._map_role(role)
            content = message.get("content", "")
            
            # Handle potential tool calls in assistant messages
            # And tool results in user messages (added by _add_tool_message)
            if role == "assistant" and message.get("tool_calls"):
                # Anthropic expects tool_calls in a specific content block structure
                content_blocks = []
                if content: # Add text content if it exists
                    content_blocks.append({"type": "text", "text": content})
                for tc in message["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments # Anthropic uses 'input'
                    })
                formatted.append({"role": anthropic_role, "content": content_blocks})
            elif role == "user" and message.get("is_tool_result"):
                 # Tool results are formatted by _add_tool_message
                 # We need to ensure they are passed through correctly
                 content_blocks = [{
                     "type": "tool_result",
                     "tool_use_id": message["tool_use_id"],
                     "content": str(content) # Content should be the result string
                 }]
                 formatted.append({"role": anthropic_role, "content": content_blocks})
            else:
                # Standard text message
                 formatted.append({"role": anthropic_role, "content": content})
                 
        return formatted

    def _extract_system_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extracts the system message content from the list."""
        for message in messages:
            if message.get("role") == "system":
                return message.get("content")
        return None

    def _prepare_request_payload(self, 
                                 messages: List[Dict[str, Any]], 
                                 options: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the Anthropic payload, handling the system message separately."""
        self.logger.debug("Preparing Anthropic request payload...")
        
        # Extract system message before standard preparation
        system_prompt = self._extract_system_message(messages)
        # Filter out system message from list passed to standard formatting
        filtered_messages = [m for m in messages if m.get("role") != "system"]
        
        # Use BaseProvider's logic for param merging, tool formatting, etc.
        payload = super()._prepare_request_payload(filtered_messages, options)
        
        # Add system prompt if it exists
        if system_prompt:
            payload["system"] = system_prompt
            self.logger.debug("Added system prompt to payload.")
        
        # Remove messages key if empty (Anthropic requires at least one message)
        if not payload.get("messages"):
             raise AIRequestError("Anthropic request must contain at least one message.")

        self.logger.debug(f"Anthropic payload prepared. Keys: {list(payload.keys())}")
        return payload

    def _map_payload_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ Maps standard parameter names to Anthropic specific names if needed. """
        # Currently, major params like max_tokens, temperature seem aligned
        # No mapping needed for now, but kept as hook
        mapped_params = params.copy()
        # Example mapping if needed:
        # if "max_tokens" in mapped_params:
        #     mapped_params["max_tokens_to_sample"] = mapped_params.pop("max_tokens")
        return mapped_params
        
    def _get_error_map(self) -> Dict[Type[Exception], Type[AIProviderError]]:
        """Returns the specific error mapping for Anthropic."""
        if not ANTHROPIC_AVAILABLE:
            return {}
        # Import relevant exceptions here
        from anthropic import (
            APIConnectionError, RateLimitError, InternalServerError,
            AuthenticationError, PermissionDeniedError, NotFoundError, 
            BadRequestError, APIStatusError, APIError
        )
        # Import framework exceptions
        from ...exceptions import (
            AIProviderError, AIAuthenticationError, AIRateLimitError, 
            ModelNotFoundError, InvalidRequestError, ContentModerationError
        )
        
        return {
            # Map SDK exceptions to framework exceptions
            AuthenticationError: AIAuthenticationError,
            PermissionDeniedError: AIAuthenticationError,
            RateLimitError: AIRateLimitError,
            NotFoundError: ModelNotFoundError, # Usually model not found
            BadRequestError: InvalidRequestError, # Content moderation handled in _make_api_request
            APIConnectionError: AIProviderError, # Map to generic (retry handles transient)
            InternalServerError: AIProviderError, # Map 5xx to generic 
            APIStatusError: AIProviderError, # Map other status errors to generic
            APIError: AIProviderError # Map base Anthropic API error to generic
        }

    # --- IMPLEMENT Required Abstract Methods --- 

    @async_retry_with_backoff(retry_on_exceptions=ANTHROPIC_RETRYABLE_EXCEPTIONS)
    async def _make_api_request(self, payload: Dict[str, Any]) -> Message: # Changed to async def
        """Makes the actual asynchronous API call to Anthropic messages endpoint.
           Simplified to let exceptions propagate for central handling.
        """
        if not ANTHROPIC_AVAILABLE:
             raise AISetupError("Anthropic SDK not available.", component="anthropic")
             
        self.logger.debug(f"Making async Anthropic API request with model {self.model_id}...")
        try:
            response = await self.client.messages.create(**payload) # Use await
            # Check for content moderation based on stop_reason *after* successful call
            # This is different from OpenAI where it's an exception
            if response.stop_reason == 'max_tokens':
                 # Check if max_tokens was hit due to moderation (heuristic)
                 # Anthropic might return max_tokens AND have a moderation notice?
                 # Need to check SDK documentation or observe behavior.
                 # For now, assume only BadRequestError or API response content indicate moderation explicitly.
                 pass # Let it be handled by _convert_response if needed
            elif response.stop_reason == 'error': # Anthropic might signal errors this way too?
                 # Unlikely for standard errors caught by exceptions, but possible
                 self.logger.warning("Anthropic response stop_reason is 'error', check raw response.")
                 # Might need to raise AIProviderError here based on response content
            
            self.logger.debug(f"Received Anthropic API response. Stop Reason: {response.stop_reason}")
            return response
        except anthropic.BadRequestError as e:
             # Specific check for Content Moderation *within* BadRequestError
             # Look for specific phrasing in the error message or body
             err_str = str(e).lower()
             body_str = str(getattr(e, 'body', '')).lower()
             if "safety" in err_str or "moderation" in body_str or "content filter" in err_str:
                 self.logger.error(f"Anthropic Content Filter Error detected: {e}")
                 # Raise the specific framework exception here
                 raise ContentModerationError(f"Anthropic content moderation block: {e}", provider="anthropic") from e
             else:
                 # If it's a different BadRequestError, let it propagate to be mapped
                 raise e
        # --- Other exceptions propagate to BaseProvider.request for mapping --- 
        # Removed the extensive try...except block mapping exceptions here

    def _convert_response(self, raw_response: Message) -> ProviderResponse:
        """Converts the raw Anthropic Message object into a standardized ProviderResponse model."""
        if not ANTHROPIC_AVAILABLE:
             # Return error response if SDK not available
             return ProviderResponse(error="Anthropic SDK not available.")
             
        self.logger.debug("Converting Anthropic response to standard ProviderResponse...")
        content = None
        tool_calls_list = []
        text_content = ""
        stop_reason = raw_response.stop_reason
        model_id = raw_response.model
        usage_data = raw_response.usage

        try:
            for block in raw_response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls_list.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input # Arguments are already parsed dict
                        )
                    )
                else:
                     self.logger.warning(f"Unsupported content block type from Anthropic: {block.type}")
                     
            # Assign text content if it exists
            if text_content:
                 content = text_content
                 
        except Exception as e:
             self.logger.error(f"Error parsing Anthropic response content blocks: {e}", exc_info=True)
             # Return an error response object if parsing fails
             return ProviderResponse(error=f"Error parsing Anthropic response: {e}", model=model_id)

        # Create TokenUsage model
        usage = None
        if usage_data:
             usage = TokenUsage(
                prompt_tokens=usage_data.input_tokens,
                completion_tokens=usage_data.output_tokens,
                total_tokens=usage_data.input_tokens + usage_data.output_tokens
             )

        # Create and return ProviderResponse model instance
        provider_response = ProviderResponse(
            content=content,
            tool_calls=tool_calls_list if tool_calls_list else None,
            stop_reason=stop_reason,
            usage=usage,
            model=model_id,
            error=None,
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
        Constructs the two-message sequence (assistant tool_use + user tool_result) for Anthropic.
        
        Args:
            tool_call_id: The ID of the tool call this result corresponds to.
            tool_name: The name of the tool that was called (used for logging).
            content: The result content from the tool execution (as a string).
            last_assistant_message: The previous assistant message containing the corresponding 
                                    tool_use block. THIS IS REQUIRED for Anthropic.
            
        Returns:
            A list containing the two message dictionaries (assistant, user) required by Anthropic,
            or an empty list if the required previous assistant message is missing.
        """
        self.logger.debug(f"Constructing tool result messages for Anthropic call_id: {tool_call_id}")
        messages_to_add = []
        
        # 1. Add the Assistant's message containing the tool_use block(s)
        if last_assistant_message and last_assistant_message.get("role") == "assistant" and last_assistant_message.get("tool_calls"):
             # We need to format this message correctly for the *next* API call
             # The _format_messages method already handles converting our standard
             # {"role": "assistant", "tool_calls": [...]} structure into Anthropic's 
             # content block format. So, we just need to pass the standard dict.
             messages_to_add.append(last_assistant_message)
             self.logger.debug("Included previous assistant message with tool_calls.")
        else:
             self.logger.error("Cannot construct Anthropic tool result: Missing or invalid 'last_assistant_message' containing tool_calls.")
             # Return empty list as we cannot construct the required sequence
             return [] 

        # 2. Add the User message containing the tool_result block
        #    Mark this message with is_tool_result so _format_messages can handle it.
        #    Note: Anthropic API expects the role to be 'user' for tool results.
        tool_result_user_message = {
            "role": "user", 
            "content": str(content), # The actual result content
            "is_tool_result": True, # Custom flag for _format_messages
            "tool_use_id": tool_call_id
        }
        messages_to_add.append(tool_result_user_message)
        self.logger.debug("Constructed user message with tool_result block.")
        
        return messages_to_add

    async def stream(self, messages: List[Dict[str, str]], **options) -> str: # Changed to async def
        """
        Stream a response asynchronously from the Anthropic API.
        NOTE: Aggregates the response. For true streaming, handle the async generator.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        if not ANTHROPIC_AVAILABLE:
             raise AIProviderError("Anthropic SDK not available.")
             
        # Prepare payload (sync)
        payload = self._prepare_request_payload(messages, options)
        payload["stream"] = True
        
        self.logger.debug(f"Making async streaming call to Anthropic. Payload keys: {list(payload.keys())}")
        
        try:
            full_response_content = ""
            async with self.client.messages.stream(**payload) as stream: # Use async context manager
                async for text in stream.text_stream: # Iterate through text stream
                    full_response_content += text
            
            # Note: This simple aggregation misses potential tool_use events in the stream.
            # A more robust implementation would need to handle different stream event types.
            self.logger.debug(f"Anthropic stream finished. Aggregated length: {len(full_response_content)}")
            return full_response_content
            
        except Exception as e:
             # Catch errors during streaming
             self.logger.error(f"Error streaming from Anthropic: {e}", exc_info=True)
             raise AIProviderError(f"Error streaming from Anthropic: {e}", provider="anthropic") from e
    
    def build_tool_result_messages(self,
                                  tool_calls: List[ToolCall],
                                  tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Builds Anthropic message containing tool results.
        Groups all results into a single 'user' role message with 'tool_result' content blocks.
        """
        content_blocks = []
        for call, result in zip(tool_calls, tool_results):
            # Ensure call has an ID, needed for tool_use_id
            if not hasattr(call, 'id') or not call.id:
                self.logger.warning(f"ToolCall object for tool '{call.name}' is missing an ID. Cannot generate Anthropic tool result message.")
                continue

            # Create the tool_result content block
            block = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": str(result.result) if result.status == "success" else str(result.error or "Tool execution failed")
            }
            # Add error flag if necessary
            if result.status == "error":
                block["is_error"] = True
            content_blocks.append(block)

        # Return a single message dictionary containing all blocks
        if content_blocks:
            return [
                {
                    "role": "user",
                    "content": content_blocks
                }
            ]
        else:
            # Return empty list if no valid results (e.g., all calls lacked IDs)
            self.logger.warning("No valid tool result content blocks generated for Anthropic.")
            return []

    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history (Anthropic format).
        DEPRECATED? - build_tool_result_messages is likely preferred.
        Appends a user message with a tool_result content block.
        """
        # This logic for finding tool_use_id seems complex and potentially fragile
        tool_use_id = "unknown_id" # Default
        try:
             # Try to find the corresponding assistant message asking for this tool
             for msg in reversed(messages):
                 if msg.get("role") == "assistant":
                     assistant_content = msg.get("content")
                     if isinstance(assistant_content, list):
                         for block in assistant_content:
                             if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == name:
                                 tool_use_id = block.get("id", tool_use_id)
                                 # Found the most recent call for this tool name
                                 break
                     if tool_use_id != "unknown_id":
                        break # Stop searching once found
        except Exception as e:
            self.logger.warning(f"Error finding tool_use_id in add_tool_message: {e}")

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(content)
                }
            ]
        })
        return messages

    # --- ToolCapableProviderInterface Methods --- 

    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history (Anthropic format).
        Appends a user message with a tool_result content block.
        """
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": next((msg.get("content", [{}])[0].get("id") 
                                      for msg in reversed(messages) 
                                      if msg.get("role") == "assistant" 
                                      and isinstance(msg.get("content"), list) 
                                      and msg["content"][0].get("type") == "tool_use" 
                                      and msg["content"][0].get("name") == name), "unknown_id"),
                    "content": str(content)
                }
            ]
        })
        return messages