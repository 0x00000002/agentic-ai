"""
Gemini provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
# Import specific exceptions
from ...exceptions import (AIRequestError, AICredentialsError, AIProviderError, 
                         AIAuthenticationError, AIRateLimitError, InvalidRequestError,
                         ModelNotFoundError, ContentModerationError)
from ...tools.models import ToolResult, ToolCall
from ...tools.tool_registry import ToolRegistry
from .base_provider import BaseProvider
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
import json
import re # Import regex for tool call detection
# Import the ProviderResponse model
from ..models import ProviderResponse, TokenUsage


class GeminiProvider(BaseProvider, ToolCapableProviderInterface):
    """Provider implementation for Google's Gemini AI."""
    
    # Add property for tool support
    supports_tools = True
    
    # Role mapping (less critical for Gemini's format but good practice)
    _ROLE_MAP = {
        "system": "system", # Handled specially
        "user": "user",
        "assistant": "model",
        "tool": "function" # Role for tool results
    }

    def __init__(self, 
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            model_id: The model identifier
            provider_config: Configuration for the Gemini provider (e.g., API key)
            model_config: Configuration for the specific model
            logger: Logger instance
        """
        super().__init__(model_id=model_id,
                         provider_config=provider_config,
                         model_config=model_config,
                         logger=logger)
        
        # API key configuration is handled by _initialize_credentials
        self._initialize_credentials()
        
        # Set up generation config from model parameters
        self.generation_config = {
            "temperature": self.model_config.get('temperature', 0.7),
            "top_p": self.model_config.get('top_p', None),
            "top_k": self.model_config.get('top_k', None),
            "max_output_tokens": self.model_config.get('output_limit', None) or self.model_config.get('max_tokens', None),
            "stop_sequences": self.model_config.get('stop', None) or self.model_config.get('stop_sequences', None)
        }
        # Filter out None values
        self.generation_config = {k: v for k, v in self.generation_config.items() if v is not None}
        
        try:
            # Pass generation_config during model initialization
            self._model = genai.GenerativeModel(
                 model_name=self.model_id,
                 generation_config=self.generation_config
            )
            self.logger.info(f"Successfully initialized Gemini model {self.model_id}")
            self.logger.debug(f"Gemini Generation Config: {self.generation_config}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise AICredentialsError(f"Failed to initialize Gemini model: {str(e)}")

    def _initialize_credentials(self) -> None:
        """Initialize Google API credentials."""
        api_key = self.provider_config.get("api_key") or self.config.get_api_key("google")
        if not api_key:
            self.logger.error("Gemini API key not found in provider_config or global config.")
            raise AICredentialsError("No Gemini API key found.")
        try:
            genai.configure(api_key=api_key)
            self.logger.info("Gemini API key configured successfully.")
        except Exception as e:
             self.logger.error(f"Failed to configure Gemini API key: {e}")
             # Assume configuration errors are related to credentials/auth setup
             raise AICredentialsError(f"Failed to configure Gemini API key: {e}", provider="gemini") from e

    def _map_role(self, role: str) -> str:
        return self._ROLE_MAP.get(role, "user")

    def _format_parts(self, content: Any) -> List[Any]:
        """Ensures content is wrapped in a list for Gemini parts."""
        # TODO: Handle multimedia content properly here
        if isinstance(content, list):
             return content # Assume it's already formatted correctly (e.g., multimedia)
        return [str(content)] # Wrap simple string content

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to Gemini's history format (list of {'role': 'user'/'model', 'parts': [...]}).
        Handles system prompts by prepending to the first user message.
        Handles tool calls and results by formatting them into specific structures.
        """
        gemini_history = []
        system_prompt_content = None

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            mapped_role = self._map_role(role)

            if role == "system":
                system_prompt_content = content # Store system prompt
                continue # Skip adding system message directly to history

            # Prepend system prompt to the *first* user message parts
            if role == "user" and system_prompt_content:
                parts = self._format_parts(content)
                if parts and isinstance(parts[0], str):
                    parts[0] = f"System: {system_prompt_content}\n\n{parts[0]}"
                else: 
                     parts.insert(0, f"System: {system_prompt_content}")
                content = parts
                system_prompt_content = None
            
            message_data = {"role": mapped_role}
            
            # Handle different message types
            if role == "assistant" and msg.get("tool_calls"):
                 # For history, just include the text part of the assistant's turn.
                 # The function calls themselves aren't added back this way.
                 # The subsequent 'function' role message provides the result context.
                 parts = []
                 if content: 
                      parts.append(genai.types.Part(text=str(content)))
                 # If no text content, maybe add a placeholder? Or skip?
                 # Let's add the message only if there was text content.
                 if parts:
                     message_data["parts"] = parts
                 else:
                      self.logger.debug(f"Assistant message with tool calls but no text content. Skipping for Gemini history.")
                      continue # Skip adding empty assistant message
            elif role == "tool":
                 # Format tool result using 'function' role
                 tool_call_id = msg.get("tool_call_id") 
                 tool_name = msg.get("name", "unknown_tool")
                 message_data["role"] = "function"
                 # Ensure content is stringified for the response part
                 try:
                      content_str = content if isinstance(content, str) else json.dumps(content)
                 except TypeError:
                      content_str = str(content)
                 message_data["parts"] = [
                     genai.types.Part(function_response=genai.types.FunctionResponse(
                         name=tool_name,
                         response={ "content": content_str }
                     ))
                 ]
            else: # User message or standard assistant message without tool calls
                message_data["parts"] = self._format_parts(content)

            gemini_history.append(message_data)
            
        if system_prompt_content:
             gemini_history.insert(0, {"role": "user", "parts": [f"System: {system_prompt_content}"]})

        return gemini_history

    def _prepare_request_payload(self, 
                                 messages: List[Dict[str, Any]], 
                                 options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the Gemini payload.
        Handles message formatting and generation config, but NOT tool injection 
        as Gemini API handles tools via FunctionCallingConfig.
        """
        self.logger.debug("Preparing Gemini request payload...")
        
        # 1. Parameter Merging (Use self.generation_config merged with runtime options)
        request_gen_config = self.generation_config.copy()
        # Map runtime options to Gemini config names if needed
        runtime_config = {
             "temperature": options.get("temperature"),
             "top_p": options.get("top_p"),
             "top_k": options.get("top_k"),
             "max_output_tokens": options.get("max_tokens") or options.get("output_limit"),
             "stop_sequences": options.get("stop") or options.get("stop_sequences")
         }
        request_gen_config.update({k: v for k, v in runtime_config.items() if v is not None})

        # 2. Format Messages (includes system prompt handling)
        formatted_messages = self._format_messages(messages)
        
        # 3. Handle Tools (using FunctionCallingConfig)
        tools_param = options.get("tools") # Expects Dict[str, ToolDefinition]
        tool_config = None
        if tools_param and isinstance(tools_param, dict):
             try:
                  # Requires ToolRegistry access - maybe pass it in or get instance?
                  tool_registry = ToolRegistry() # Or get instance/passed registry
                  # Format tools specifically for Gemini FunctionCallingConfig
                  gemini_tools = tool_registry.format_tools_for_provider("GEMINI", set(tools_param.keys()))
                  if gemini_tools:
                       # Potentially set tool_config based on gemini_tools and tool_choice
                       # Example (adjust based on actual API needs):
                       # tool_config = genai.types.ToolConfig(function_calling_config=...) 
                       self.logger.debug(f"Formatted tools for Gemini: {gemini_tools}")
                       # Placeholder: Gemini models usually use genai.GenerativeModel(tools=...) 
                       # We might need to re-initialize the model or handle this differently.
                       # For now, payload doesn't include tools directly, assume handled at model level.
                       pass # Tool setup is usually part of model init or a specific param
             except Exception as e:
                  self.logger.error(f"Failed to format tools for Gemini provider: {e}", exc_info=True)

        # 4. Construct final payload for send_message/generate_content
        # Gemini's send_message takes content directly, history is handled by chat object.
        # generate_content takes the full list.
        # We'll structure payload assuming generate_content for flexibility.
        payload = {
             "contents": formatted_messages,
             "generation_config": request_gen_config,
             # "tools": gemini_tools, # <-- Add formatted tools here if generate_content accepts them
             # "tool_config": tool_config, # <-- Add tool config here if needed
         }
        
        self.logger.debug(f"Gemini payload prepared. Content messages: {len(payload['contents'])}")
        return payload

    # --- IMPLEMENT Required Abstract Methods --- 

    def _make_api_request(self, payload: Dict[str, Any]) -> GenerateContentResponse:
        """Makes the actual API call to Gemini generate_content."""
        self.logger.debug("Making Gemini API request...")
        try:
            response = self._model.generate_content(**payload)
            # Check for safety blocks immediately after the call if possible
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name
                 self.logger.error(f"Gemini request blocked due to prompt feedback. Reason: {block_reason}")
                 raise ContentModerationError(f"Gemini prompt blocked by safety filter: {block_reason}", provider="gemini", reason=block_reason)
            # Also check candidate finish reason for safety
            if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                 self.logger.error(f"Gemini response candidate blocked due to safety.")
                 # Potentially extract safety ratings if needed
                 raise ContentModerationError(f"Gemini response blocked by safety filter.", provider="gemini", reason="SAFETY")
                 
            self.logger.debug("Received Gemini API response.")
            return response
        # --- Specific Google / Gemini Error Handling --- 
        # Wrap google.api_core.exceptions as appropriate
        # Note: Specific exception types might vary based on SDK version
        except ImportError: # Handle case where google exceptions aren't available
             self.logger.warning("Cannot import google.api_core exceptions for specific Gemini error handling.")
             # Fallback to generic handling
             pass
        except Exception as e:
             # Attempt to map common Google API errors
             # This requires importing google.api_core exceptions, which might fail
             try:
                  from google.api_core import exceptions as google_exceptions
                  
                  if isinstance(e, google_exceptions.PermissionDenied):
                       self.logger.error(f"Google Permission Denied Error: {e}")
                       raise AIAuthenticationError(f"Google API permission denied: {e}", provider="gemini") from e
                  elif isinstance(e, google_exceptions.Unauthenticated):
                       self.logger.error(f"Google Unauthenticated Error: {e}")
                       raise AIAuthenticationError(f"Google API authentication failed: {e}", provider="gemini") from e
                  elif isinstance(e, google_exceptions.ResourceExhausted): # Often maps to rate limits
                       self.logger.error(f"Google Resource Exhausted Error (Rate Limit?): {e}")
                       raise AIRateLimitError(f"Google API resource exhausted (rate limit?): {e}", provider="gemini") from e
                  elif isinstance(e, google_exceptions.NotFound): # Model not found?
                       self.logger.error(f"Google Not Found Error (Model?): {e}")
                       raise ModelNotFoundError(f"Model or resource not found for Google API: {e}", provider="gemini", model_id=payload.get("model")) from e
                  elif isinstance(e, google_exceptions.InvalidArgument): # Invalid request params
                       self.logger.error(f"Google Invalid Argument Error: {e}")
                       raise InvalidRequestError(f"Invalid request to Google API: {e}", provider="gemini", status_code=400) from e
                  elif isinstance(e, google_exceptions.FailedPrecondition): # Often indicates API not enabled or billing issues
                       self.logger.error(f"Google Failed Precondition Error: {e}")
                       raise AIProviderError(f"Google API precondition failed (API enabled? Billing?): {e}", provider="gemini", status_code=400) from e # Treat as provider config issue
                  elif isinstance(e, google_exceptions.GoogleAPIError): # Catch-all for other Google API errors
                       status_code = getattr(e, 'code', None)
                       self.logger.error(f"Google API Error (Code: {status_code}): {e}", exc_info=True)
                       raise AIProviderError(f"Google API error: {e}", provider="gemini", status_code=status_code) from e
                  
             except ImportError: # Handle case where google exceptions aren't available
                  pass # Already logged warning, fall through to generic exception
             except Exception as mapping_err: # Catch errors during the mapping itself
                  self.logger.error(f"Error occurred while mapping Google API exception: {mapping_err}", exc_info=True)
                  # Fall through to generic exception
                  
             # --- Generic Fallback --- 
             error_details = str(e)
             if hasattr(e, 'message'): # Try to get a more specific message
                  error_details = e.message
             self.logger.error(f"Gemini API error (unmapped or google.api_core not available): {e}", exc_info=True)
             raise AIRequestError(f"Gemini API error: {error_details}", provider="gemini") from e

    def _convert_response(self, raw_response: GenerateContentResponse) -> ProviderResponse:
        """Converts the raw Gemini GenerateContentResponse into a standardized ProviderResponse model."""
        self.logger.debug("Converting Gemini response to standard ProviderResponse...")
        content = None
        tool_calls_list = []
        stop_reason = None
        usage_data = None
        model_id = self._model.model_name # Use the initialized model name
        error = None

        try:
            if raw_response.candidates:
                 candidate = raw_response.candidates[0]
                 # Map Gemini finish reason to our standard terms if possible
                 finish_reason_map = {
                     "STOP": "stop",
                     "MAX_TOKENS": "max_tokens",
                     "SAFETY": "safety",
                     "RECITATION": "recitation",
                     # Add other mappings as needed
                 }
                 stop_reason = finish_reason_map.get(candidate.finish_reason.name, candidate.finish_reason.name) if candidate.finish_reason else None
                 
                 text_content = ""
                 if candidate.content and candidate.content.parts:
                     for part in candidate.content.parts:
                         if hasattr(part, 'text') and part.text:
                              text_content += part.text
                         elif hasattr(part, 'function_call') and part.function_call:
                              fc = part.function_call
                              args = fc.args if isinstance(fc.args, dict) else {}
                              tool_calls_list.append(
                                   ToolCall(
                                        id=f"gemini-tool-{fc.name}-{len(tool_calls_list)}", 
                                        name=fc.name,
                                        arguments=args
                                   )
                              )
                 if text_content:
                      content = text_content # Assign collected text content
                      
            elif raw_response.prompt_feedback and raw_response.prompt_feedback.block_reason:
                 block_reason = raw_response.prompt_feedback.block_reason.name
                 self.logger.error(f"Gemini request blocked. Reason: {block_reason}")
                 error = f"Content blocked by Gemini API: {block_reason}"
                 stop_reason = "safety" # Set stop reason for blocked content
            else:
                 self.logger.warning("Gemini response missing candidates and prompt feedback.")
                 error = "Gemini response was empty or missing expected content."

            # Extract usage metadata if available
            if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                 usage_meta = raw_response.usage_metadata
                 usage_data = TokenUsage(
                      prompt_tokens=usage_meta.prompt_token_count,
                      completion_tokens=usage_meta.candidates_token_count,
                      total_tokens=usage_meta.total_token_count
                  )

        except Exception as e:
             self.logger.error(f"Error parsing Gemini response content: {e}", exc_info=True)
             error = f"Error parsing Gemini response: {e}"

        # Create and return ProviderResponse model instance
        provider_response = ProviderResponse(
            content=content,
            tool_calls=tool_calls_list if tool_calls_list else None,
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
                         tool_call_id: str, # Not directly used by Gemini API format
                         tool_name: str, 
                         content: str,
                         last_assistant_message: Optional[Dict[str, Any]] = None # Not used by Gemini
                         ) -> List[Dict[str, Any]]:
        """
        Constructs the 'function' role message for Gemini tool results.
        Uses the logic previously in format_tool_result.
        """
        self.logger.debug(f"Constructing Gemini tool result message for tool: {tool_name}")
        
        # Ensure content is serializable string for the API part
        if not isinstance(content, str):
             # Basic serialization for common types, might need refinement
             try:
                  content_str = json.dumps(content)
             except TypeError:
                  content_str = str(content)
        else:
             content_str = content
             
        # Construct the message using Gemini SDK types if possible, or dict
        try:
             tool_result_part = genai.types.Part(function_response=genai.types.FunctionResponse(
                 name=tool_name,
                 response={ "content": content_str } 
             ))
             tool_message = { "role": "function", "parts": [tool_result_part] }
        except Exception as e:
             # Fallback to raw dict if SDK types fail (e.g., older version)
             self.logger.warning(f"Could not create Gemini SDK types for tool result, using dict fallback: {e}")
             tool_message = {
                 "role": "function", 
                 "parts": [
                     {
                         "function_response": {
                             "name": tool_name,
                             "response": { "content": content_str }
                         }
                     }
                 ]
             }
             
        return [tool_message] # Return as a list containing one dictionary

    # --- Keep format_tool_result for potential internal use or remove if redundant --- 
    def format_tool_result(self, tool_call_id: str, tool_name: str, result_content: Any) -> Dict[str, Any]:
        """
        DEPRECATED: Use _add_tool_message which returns the required list format.
        Formats a tool result for the Gemini API.
        """
        self.logger.warning("format_tool_result is deprecated, use _add_tool_message instead.")
        result_messages = self._add_tool_message(tool_call_id, tool_name, result_content)
        return result_messages[0] if result_messages else {} # Return first message dict or empty

    # --- REMOVE existing request method --- 

    # --- Keep existing stream method (needs refactoring similar to request) --- 
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the Gemini API.
        (NEEDS REFACTORING to use _prepare_request_payload, _make_api_request structure)
        """
        self.logger.warning("Gemini stream() method needs refactoring.")
        # Placeholder implementation - requires significant changes
        try:
            # Basic string handling
            if isinstance(messages, str):
                response = self._model.generate_content(messages, stream=True, generation_config=self.generation_config)
                return "".join([chunk.text for chunk in response if chunk.text])
            else:
                 # History handling for streaming is more complex
                 # payload = self._prepare_request_payload(messages, options)
                 # response = self._model.generate_content(contents=payload['contents'], stream=True, generation_config=payload['generation_config'])
                 # return "".join([chunk.text for chunk in response if chunk.text])
                 raise NotImplementedError("Streaming with history not fully refactored for Gemini.")
        except Exception as e:
            self.logger.error(f"Gemini streaming failed: {str(e)}")
            raise AIRequestError(f"Failed to stream Gemini response: {str(e)}", provider="gemini", original_error=e)

    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content)
        })
        return messages

    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Gemini API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Either a string response (when no tools were needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        try:
            # Check for tools
            tools = options.pop("tools", None)
            
            # For Gemini models, we can't directly pass tools to the API
            # Instead, we'll inject a prompt with the tool descriptions and usage instructions
            if tools and isinstance(messages, list):
                # Extract the last user message
                last_user_msg = None
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        last_user_msg = msg
                        break
                
                if last_user_msg:
                    # Create a tool description
                    tool_desc = "You have access to the following tools:\n\n"
                    for tool in tools:
                        tool_desc += f"- {tool['name']}: {tool['description']}\n"
                        if 'parameters' in tool and 'properties' in tool['parameters']:
                            tool_desc += "  Parameters:\n"
                            for param_name, param_details in tool['parameters']['properties'].items():
                                param_desc = param_details.get('description', 'No description')
                                param_type = param_details.get('type', 'string')
                                tool_desc += f"    - {param_name} ({param_type}): {param_desc}\n"
                    
                    tool_desc += "\nIf you need to use a tool, respond ONLY with JSON in this exact format:\n"
                    tool_desc += '{"tool": "<tool_name>", "parameters": {"param1": "value1", "param2": "value2"}}\n'
                    tool_desc += "\nDO NOT include any explanation, code blocks, or text before or after the JSON. ONLY output the raw JSON if you need to use a tool."
                    
                    # Append to the last user message
                    last_user_msg["content"] = f"{last_user_msg['content']}\n\n{tool_desc}"
                    self.logger.debug("Enhanced user message with tool instructions")
            
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                chat = self._model.start_chat()
                response = chat.send_message(messages)
            else:
                # Convert messages to Gemini format
                gemini_messages = self._format_messages(messages)
                
                # Create chat session
                chat = self._model.start_chat(history=gemini_messages[:-1])
                
                # Send the last message to get the response
                last_message = gemini_messages[-1]
                response = chat.send_message(last_message["parts"][0])
            
            # Extract content
            content = response.text
            
            # Try to identify tool calls in the text response
            if tools and "{" in content and "}" in content:
                try:
                    # First try to parse the entire response as JSON
                    content_cleaned = content.strip()
                    # Remove any markdown code block markers
                    if content_cleaned.startswith("```") and content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[3:-3].strip()
                    if content_cleaned.startswith("```json") and content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[7:-3].strip()
                    
                    # Try to parse cleaned content as JSON
                    try:
                        data = json.loads(content_cleaned)
                        if "tool" in data and "parameters" in data:
                            tool_name = data["tool"]
                            params = data["parameters"]
                            
                            # Create a tool call
                            tool_call = ToolCall(
                                name=tool_name,
                                arguments=params,  # Pass as dict, not JSON string
                                id=f"tool-{tool_name}"
                            )
                            
                            self.logger.info(f"Detected tool call in Gemini response: {tool_name}")
                            self.logger.debug(f"Tool parameters: {params}")
                            # Return with the tool call
                            return {
                                "content": content,
                                "tool_calls": [tool_call]
                            }
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON using a better regex
                        # This regex handles nested braces properly
                        json_pattern = r'({(?:[^{}]|(?R))*})'
                        # Fallback to simpler pattern if the recursive one isn't supported
                        try:
                            json_matches = re.findall(json_pattern, content, re.DOTALL)
                        except re.error:
                            # Use non-recursive pattern as fallback
                            json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
                            json_matches = re.findall(json_pattern, content, re.DOTALL)
                        
                        for json_str in json_matches:
                            try:
                                data = json.loads(json_str)
                                if "tool" in data and "parameters" in data:
                                    tool_name = data["tool"]
                                    params = data["parameters"]
                                    
                                    # Create a tool call
                                    tool_call = ToolCall(
                                        name=tool_name,
                                        arguments=params,  # Pass as dict, not JSON string
                                        id=f"tool-{tool_name}"
                                    )
                                    
                                    self.logger.info(f"Detected tool call in Gemini response: {tool_name}")
                                    self.logger.debug(f"Tool parameters: {params}")
                                    # Return with the tool call
                                    return {
                                        "content": content,
                                        "tool_calls": [tool_call]
                                    }
                            except Exception as json_err:
                                self.logger.debug(f"Failed to parse JSON chunk: {json_err}")
                                continue
                except Exception as e:
                    self.logger.warning(f"Failed to parse potential tool calls: {str(e)}")
            
            # No tool calls detected, return just the content
            return self.standardize_response(content)
            
        except Exception as e:
            self.logger.error(f"Gemini request failed: {str(e)}")
            raise AIRequestError(
                f"Failed to make Gemini request: {str(e)}",
                provider="gemini",
                original_error=e
            ) 