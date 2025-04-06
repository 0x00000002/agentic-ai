"""
Anthropic provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
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

try:
    import anthropic
    from anthropic import Anthropic
    from anthropic.types import Message
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Message = None # Define Message as None if import fails


class AnthropicProvider(BaseProvider, ToolCapableProviderInterface):
    """Provider implementation for Anthropic Claude models with tools support."""
    
    # Add property for tool support
    supports_tools = True
    
    # Role mapping for Anthropic API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "assistant"  # Anthropic has no tool role
    }
    
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
        
        # Initialize model parameters from self.model_config
        self.parameters = {}
        # Anthropic uses max_tokens, temperature, top_p, top_k
        anthropic_params = ["temperature", "max_tokens", "top_p", "top_k", "stop_sequences"]
        
        # Use output_limit from config for max_tokens if present
        if 'output_limit' in model_config:
            self.parameters['max_tokens'] = model_config['output_limit']
            
        for key in anthropic_params:
            # Don't overwrite max_tokens if output_limit was used
            if key in model_config and not (key == 'max_tokens' and 'max_tokens' in self.parameters):
                self.parameters[key] = model_config[key]
        
        # Provide Anthropic defaults if not specified
        self.parameters.setdefault('temperature', 0.7)
        self.parameters.setdefault('max_tokens', 4096) # Default
        
        self.logger.info(f"Initialized Anthropic provider with model {model_id}")
        self.logger.debug(f"Anthropic Parameters set: {self.parameters}")
    
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
                
            # Set up Anthropic client
            try:
                self.client = Anthropic(api_key=api_key)
                self.logger.info("Successfully initialized Anthropic client")
            except Exception as e:
                self.logger.error(f"Failed to create Anthropic client: {str(e)}")
                raise AICredentialsError(f"Failed to create Anthropic client: {str(e)}")
            
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
        
    # --- IMPLEMENT Required Abstract Methods --- 

    def _make_api_request(self, payload: Dict[str, Any]) -> Message:
        """Makes the actual API call to Anthropic messages endpoint."""
        if not ANTHROPIC_AVAILABLE:
             raise AIProviderError("Anthropic SDK not available.")
             
        self.logger.debug(f"Making Anthropic API request with model {self.model_id}...")
        try:
            response = self.client.messages.create(**payload)
            self.logger.debug(f"Received Anthropic API response. Stop Reason: {response.stop_reason}")
            return response
        # --- Specific Anthropic Error Handling --- 
        except anthropic.AuthenticationError as e:
             self.logger.error(f"Anthropic Authentication Error: {e}")
             raise AIAuthenticationError(f"Anthropic authentication failed: {e}", provider="anthropic") from e
        except anthropic.PermissionDeniedError as e:
             self.logger.error(f"Anthropic Permission Denied Error: {e}")
             raise AIAuthenticationError(f"Anthropic permission denied: {e}", provider="anthropic") from e
        except anthropic.RateLimitError as e:
             self.logger.error(f"Anthropic Rate Limit Error: {e}")
             # TODO: Extract retry_after if available
             raise AIRateLimitError(f"Anthropic rate limit exceeded: {e}", provider="anthropic") from e
        except anthropic.NotFoundError as e: # Often indicates model not found
            self.logger.error(f"Anthropic Not Found Error (likely model): {e}")
            raise ModelNotFoundError(f"Model or endpoint not found for Anthropic: {e}", provider="anthropic", model_id=payload.get("model")) from e
        except anthropic.BadRequestError as e:
             # Check for content filtering
             if "safety" in str(e).lower() or (hasattr(e, 'body') and e.body and "moderation" in str(e.body).lower()):
                 self.logger.error(f"Anthropic Content Filter Error: {e}")
                 raise ContentModerationError(f"Anthropic content moderation block: {e}", provider="anthropic") from e
             else:
                 self.logger.error(f"Anthropic Bad Request Error: {e}")
                 raise InvalidRequestError(f"Invalid request to Anthropic: {e}", provider="anthropic", status_code=e.status_code) from e
        except anthropic.APIConnectionError as e:
             self.logger.error(f"Anthropic API Connection Error: {e}")
             raise AIProviderError(f"Anthropic connection error: {e}", provider="anthropic") from e
        except anthropic.APIStatusError as e: # Catch other non-2xx status codes
             self.logger.error(f"Anthropic API Status Error ({e.status_code}): {e}")
             raise AIProviderError(f"Anthropic API returned status {e.status_code}: {e}", provider="anthropic", status_code=e.status_code) from e
        except anthropic.APIError as e: # Catch-all for other Anthropic API errors
            self.logger.error(f"Anthropic API Error: {e}", exc_info=True)
            raise AIProviderError(f"Anthropic API error: {e}", provider="anthropic", status_code=getattr(e, 'status_code', None)) from e
        # --- End Specific Anthropic Error Handling ---
        except Exception as e:
            self.logger.error(f"Unexpected error during Anthropic API request: {e}", exc_info=True)
            raise AIProviderError(f"Unexpected error making Anthropic request: {e}", provider="anthropic") from e

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

    def stream(self, messages: List[Dict[str, str]], **options) -> str:
        """
        Stream a response from the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Format messages for Anthropic API
            formatted_messages = self._format_messages(messages)
            
            # Extract system message
            system = self._extract_system_message(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-Anthropic parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "top_k", 
                              "stop_sequences"]:
                    del params[key]
            
            # Make the streaming API call
            chunks = []
            with self.client.messages.stream(
                model=self.model_id,
                messages=formatted_messages,
                system=system,
                max_tokens=params.get("max_tokens", 4096),
                temperature=params.get("temperature", 0.7),
                # Add other params? stop_sequences, top_p, top_k?
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
                    
            # Join all chunks
            return "".join(chunks)
            
        except anthropic.APIError as e:
            raise AIRequestError(f"Anthropic API error in streaming: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error streaming from Anthropic: {str(e)}")
    
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