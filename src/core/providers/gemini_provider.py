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
from .base_provider import BaseProvider
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse, FunctionDeclaration, Tool
import json
import re # Import regex for tool call detection
import asyncio # Add asyncio
# Import the ProviderResponse model
from ..models import ProviderResponse, TokenUsage

# Define Part dictionary type hint for clarity
PartDict = Dict[str, Any]

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
                 # Tools are potentially configured here or passed to generate_content_async
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

    def _format_parts(self, content: Any) -> List[PartDict]:
        """Ensures content is wrapped in a list of Gemini Parts (dictionaries)."""
        parts = []
        if isinstance(content, list):
            # If it's a list, iterate and try to create Part dicts
            for item in content:
                try:
                    # Check if item is already a dict likely representing a Part
                    if isinstance(item, dict) and any(k in item for k in ["text", "inline_data", "function_call", "function_response"]):
                        parts.append(item)
                    elif item is not None:
                        parts.append({"text": str(item)}) # Convert other items to text Part dict
                except Exception as e:
                     self.logger.warning(f"Could not convert item to Gemini Part dict: {item}. Error: {e}. Using str().")
                     parts.append({"text": str(item)}) # Fallback
        elif content is not None:
             # Simple content, treat as text
             parts.append({"text": str(content)})
        # else: content is None, parts remains empty

        return parts

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to Gemini's history format (list of {'role': 'user'/'model', 'parts': [...]}).
        Handles system prompts by prepending to the first user message.
        Handles tool calls and results by formatting them into specific structures.
        """
        gemini_history = []
        system_prompt_content = None
        first_user_message_handled = False

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            mapped_role = self._map_role(role)

            if role == "system":
                system_prompt_content = content # Store system prompt
                continue # Skip adding system message directly to history

            # Prepend system prompt to the *first* user message parts
            if role == "user" and system_prompt_content and not first_user_message_handled:
                self.logger.debug("Prepending system prompt to first user message.")
                # Add system prompt as a separate text part dict
                current_parts = [{"text": f"System Prompt: {system_prompt_content}"}]
                # Add original user content parts after system prompt
                user_content_parts = self._format_parts(content)
                current_parts.extend(user_content_parts)
                first_user_message_handled = True
            # --- Handle tool result message ('function' role for Gemini) ---
            elif mapped_role == "tool":
                 tool_name = msg.get("name", "unknown_tool")
                 # Content is the result payload, expected to be dict by Gemini FunctionResponse
                 response_payload = {}
                 # Ensure content is serializable for the response part
                 try:
                     # Gemini expects a dictionary response, not just a string
                     result_payload = { "content": content } 
                     if isinstance(content, (dict, list)): # If content is already structured
                          result_payload = content
                     elif isinstance(content, str):
                          try:
                               parsed_json = json.loads(content)
                               if isinstance(parsed_json, (dict, list)): # Use parsed if valid JSON structure
                                    result_payload = parsed_json
                               # else keep as { "content": string }
                          except json.JSONDecodeError:
                               pass # Keep as { "content": string } if not valid JSON
                               
                     response_payload = result_payload
                 except Exception as e:
                      self.logger.warning(f"Could not create Gemini SDK types for tool result, using dict fallback: {e}")
                      response_payload = { "content": content }
                 # Construct the FunctionResponse Part dictionary directly
                 current_parts = [
                     {
                          "function_response": {
                               "name": tool_name,
                               "response": response_payload
                          }
                     }
                 ]
            # --- Handle assistant message that *made* tool calls ---
            elif mapped_role == "model" and msg.get("tool_calls"):
                 # Add text part first if content exists
                 if isinstance(content, str) and content:
                      current_parts = [{"text": content}]
                 # Add function call parts as dicts
                 tool_calls = msg.get("tool_calls")
                 for tc in tool_calls:
                      # Ensure tc is ToolCall object or dict
                      tool_name = getattr(tc, 'name', tc.get('name') if isinstance(tc, dict) else None)
                      tool_args = getattr(tc, 'arguments', tc.get('arguments') if isinstance(tc, dict) else None)
                      if tool_name and isinstance(tool_args, dict):
                           try:
                               # Create the FunctionCall Part dictionary directly
                               current_parts.append({"function_call": {"name": tool_name, "args": tool_args}})
                           except Exception as e:
                                self.logger.warning(f"Could not create Gemini FunctionCall Part dict: {e}. Skipping call {tool_name}.")
                      else:
                            self.logger.warning(f"Skipping invalid tool_call data for Gemini history: {tc}")
            # --- Standard user message or assistant message without tool calls ---
            else:
                 # Only format parts if content is not None
                 if content is not None:
                     current_parts = self._format_parts(content)
                 elif mapped_role == "user": # User message should generally have content
                     self.logger.warning("User message has None content. Creating empty text part.")
                     current_parts = [{"text": ""}]
                 # else: Assistant message with None content and no tool calls - skip

            # --- Add message to history if it has parts ---
            if current_parts:
                gemini_history.append({"role": mapped_role, "parts": current_parts})
            # else: logger.debug(f"Skipping message role '{mapped_role}' as it resulted in no parts.")

        # --- Final check for unhandled system prompt ---
        if system_prompt_content and not first_user_message_handled:
             self.logger.warning("System prompt provided but no user message found. Adding as initial user message.")
             gemini_history.insert(0, {"role": "user", "parts": [{"text": f"System Prompt: {system_prompt_content}"}]})

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
        
        # 3. Handle Tools (Function Declarations)
        # Gemini expects tools formatted as google.generativeai.types.Tool
        gemini_tools_list: Optional[List[Tool]] = None
        tools_option = options.get("tools") # Expected: List[Dict[str, Any]] from ToolManager.format_tools
        tool_choice = options.get("tool_choice") # String like "auto", "any", "none" or specific tool name

        if tools_option and isinstance(tools_option, list):
             try:
                  function_declarations = []
                  for tool_dict in tools_option:
                       # Assuming tools_option contains dicts like
                       # {"function_declaration": {"name": ..., "description": ..., "parameters": ...}}
                       if "function_declaration" in tool_dict:
                            decl = tool_dict["function_declaration"]
                            # Basic validation
                            if isinstance(decl, dict) and decl.get("name") and decl.get("parameters") is not None:
                                 function_declarations.append(FunctionDeclaration(**decl))
                            else:
                                 self.logger.warning(f"Skipping invalid Gemini tool structure: {tool_dict}")
                       else:
                            self.logger.warning(f"Tool dictionary missing 'function_declaration': {tool_dict}")
                  
                  if function_declarations:
                       gemini_tools_list = [Tool(function_declarations=function_declarations)]
                       self.logger.debug(f"Formatted {len(function_declarations)} tools for Gemini.")
             except Exception as e:
                  self.logger.error(f"Failed to format tools for Gemini provider: {e}", exc_info=True)

        # 4. Construct final payload for generate_content_async
        payload = {
             "contents": formatted_messages,
             "generation_config": genai.types.GenerationConfig(**request_gen_config), # Use the specific type
             "tools": gemini_tools_list, # Pass the list of Tools
             # TODO: Handle tool_config (tool choice) - check SDK for exact parameter
             # "tool_config": ... 
         }
         # Remove None values before sending
        payload = {k: v for k, v in payload.items() if v is not None}
        
        self.logger.debug(f"Gemini payload prepared. Content messages: {len(payload.get('contents',[]))}")
        return payload

    # --- IMPLEMENT Required Abstract Methods --- 

    async def _make_api_request(self, payload: Dict[str, Any]) -> GenerateContentResponse: # Changed to async def
        """Makes the actual asynchronous API call to Gemini generate_content_async."""
        self.logger.debug("Making async Gemini API request...")
        try:
            response = await self._model.generate_content_async(**payload) # Use await and async method
            
            # Check for safety blocks immediately after the call
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
            
        # --- Specific Google API Error Handling (add more as needed) ---
        # Import Google API core exceptions if not already done
        except ImportError:
             # Handle case where google-api-core is not installed (though genai depends on it)
             self.logger.warning("google.api_core.exceptions not available for specific error handling.")
             # Fall through to generic Exception
             pass 
        except Exception as google_err:
             # Attempt to import dynamically for specific handling
             try:
                  from google.api_core.exceptions import GoogleAPIError, Unauthenticated, PermissionDenied, ResourceExhausted, InvalidArgument, NotFound
                  
                  if isinstance(google_err, Unauthenticated):
                       self.logger.error(f"Gemini Unauthenticated Error: {google_err}")
                       raise AIAuthenticationError(f"Gemini authentication failed (check API key): {google_err}", provider="gemini") from google_err
                  elif isinstance(google_err, PermissionDenied):
                       self.logger.error(f"Gemini Permission Denied Error: {google_err}")
                       raise AIAuthenticationError(f"Gemini permission denied (check API key/roles): {google_err}", provider="gemini") from google_err
                  elif isinstance(google_err, ResourceExhausted): # Rate limiting
                       self.logger.error(f"Gemini Resource Exhausted (Rate Limit) Error: {google_err}")
                       raise AIRateLimitError(f"Gemini rate limit exceeded: {google_err}", provider="gemini") from google_err
                  elif isinstance(google_err, InvalidArgument):
                       self.logger.error(f"Gemini Invalid Argument Error: {google_err}")
                       raise InvalidRequestError(f"Invalid request to Gemini: {google_err}", provider="gemini") from google_err
                  elif isinstance(google_err, NotFound): # Model not found?
                       self.logger.error(f"Gemini Not Found Error: {google_err}")
                       # Check if message indicates model not found
                       if "model" in str(google_err).lower() and "not found" in str(google_err).lower():
                            raise ModelNotFoundError(f"Gemini model not found: {self.model_id} - {google_err}", provider="gemini", model=self.model_id) from google_err
                       else:
                            raise AIRequestError(f"Gemini resource not found: {google_err}", provider="gemini") from google_err
                  elif isinstance(google_err, GoogleAPIError):
                       # Catch other Google API errors
                       self.logger.error(f"Gemini GoogleAPIError: {google_err}", exc_info=True)
                       raise AIProviderError(f"Gemini API error: {google_err}", provider="gemini") from google_err
                  else:
                       # If it wasn't a Google specific error, re-raise to be caught below
                       raise google_err
             except ImportError:
                  # If google-api-core not installed, raise the original error
                  raise google_err
        # --- End Specific Google Error Handling --- 

    def _convert_response(self, raw_response: GenerateContentResponse) -> ProviderResponse:
        """Converts the raw Gemini GenerateContentResponse object into a standardized ProviderResponse."""
        self.logger.debug("Converting Gemini response...")
        try:
            text_content = ""
            tool_calls_list = []
            stop_reason = None
            usage = None

            # Check for blocked prompt first
            if raw_response.prompt_feedback and raw_response.prompt_feedback.block_reason:
                 block_reason_name = raw_response.prompt_feedback.block_reason.name
                 error_msg = f"Gemini prompt blocked by safety filter: {block_reason_name}"
                 self.logger.error(error_msg)
                 # Return error immediately, maybe ContentModerationError should be raised earlier?
                 return ProviderResponse(error=error_msg)

            # Process candidates (usually only one)
            if raw_response.candidates:
                 candidate = raw_response.candidates[0]
                 stop_reason = candidate.finish_reason.name
                 
                 # Check for safety block in candidate finish reason
                 if stop_reason == "SAFETY":
                      safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                      error_msg = f"Gemini response candidate blocked by safety filter. Ratings: {safety_ratings_str}"
                      self.logger.error(error_msg)
                      return ProviderResponse(error=error_msg)

                 # Extract content parts (text and function calls)
                 if candidate.content and candidate.content.parts:
                     for part in candidate.content.parts:
                         if hasattr(part, 'text'):
                              text_content += part.text
                         elif hasattr(part, 'function_call'):
                            fc = part.function_call
                             # Gemini function_call.args is already a dict
                            tool_calls_list.append(ToolCall(
                                 id=f"gemini-call-{fc.name}",
                                 name=fc.name,
                                 arguments=dict(fc.args) if fc.args else {} # Ensure it's a dict
                             ))
                         else:
                              self.logger.warning(f"Gemini response part has unexpected type: {part}")
            else:
                 self.logger.warning("Gemini response has no candidates.")
                 # Check if response text might be available directly (older API versions?)
                 try:
                      text_content = raw_response.text
                      self.logger.warning("Using raw_response.text as content.")
                 except (AttributeError, ValueError):
                      self.logger.error("Could not extract content from Gemini response.")
                      return ProviderResponse(error="No content found in Gemini response")
            
            # Map Gemini stop reason to standard ones
            stop_reason_map = {
                 "STOP": "stop",
                 "MAX_TOKENS": "length",
                 "SAFETY": "safety", # Should have been caught above, but include mapping
                 "RECITATION": "recitation",
                 "OTHER": "unknown",
                 "FUNCTION_CALL": "tool_calls" # Map this correctly
            }
            standard_stop_reason = stop_reason_map.get(stop_reason, stop_reason)
            
            # Extract usage from metadata if available (check SDK docs for exact location)
            # Placeholder - Gemini usage reporting might be different or require separate calls
            # if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
            #     usage = TokenUsage(...)
            # Let's simulate potential location based on other providers
            prompt_token_count = getattr(raw_response, 'usage_metadata', {}).get('prompt_token_count')
            candidates_token_count = getattr(raw_response, 'usage_metadata', {}).get('candidates_token_count')
            total_token_count = getattr(raw_response, 'usage_metadata', {}).get('total_token_count')
            if prompt_token_count is not None and candidates_token_count is not None and total_token_count is not None:
                 usage = TokenUsage(
                     prompt_tokens=prompt_token_count,
                     completion_tokens=candidates_token_count,
                     total_tokens=total_token_count
                 )

            return ProviderResponse(
                content=text_content,
                tool_calls=tool_calls_list if tool_calls_list else None,
                stop_reason=standard_stop_reason,
                usage=usage,
                model=self.model_id, # Use configured model ID
                # raw_response=raw_response # Optional
            )
        except Exception as e:
             self.logger.error(f"Error converting Gemini response: {e}", exc_info=True)
             return ProviderResponse(error=f"Error processing Gemini response: {str(e)}")

    # Removed _add_tool_message

    # Removed format_tool_result (handled by build_tool_result_messages)
    
    async def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str: # Changed to async def
        """
        Stream a response asynchronously from the Gemini API.
        Aggregates text content. Tool calls during streaming are complex and not fully handled here.
        """
        # Prepare payload (sync)
        payload = self._prepare_request_payload(messages, options)
        
        self.logger.debug("Making async Gemini streaming request...")
        
        try:
            # Stream the response asynchronously
            stream = await self._model.generate_content_async(**payload, stream=True) # Use await and stream=True
            
            full_response_text = ""
            async for chunk in stream: # Use async for
                 # Check for safety issues in chunks
                 if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                      block_reason_name = chunk.prompt_feedback.block_reason.name
                      error_msg = f"Gemini stream prompt blocked by safety filter: {block_reason_name}"
                      self.logger.error(error_msg)
                      raise ContentModerationError(error_msg, provider="gemini", reason=block_reason_name)
                 if chunk.candidates and chunk.candidates[0].finish_reason.name == "SAFETY":
                      safety_ratings_str = str(getattr(chunk.candidates[0], 'safety_ratings', 'N/A'))
                      error_msg = f"Gemini stream response blocked by safety filter. Ratings: {safety_ratings_str}"
                      self.logger.error(error_msg)
                      raise ContentModerationError(error_msg, provider="gemini", reason="SAFETY")
                      
                 # Aggregate text parts
                 try:
                     # Access text safely, handling potential errors if chunk structure is unexpected
                      if hasattr(chunk, 'text'):
                           full_response_text += chunk.text
                      elif chunk.parts:
                           for part in chunk.parts:
                                if hasattr(part, 'text'):
                                     full_response_text += part.text
                 except (AttributeError, ValueError, TypeError) as e:
                      self.logger.warning(f"Could not extract text from stream chunk: {chunk}. Error: {e}")
                      
                 # TODO: Handle function calls appearing mid-stream if needed (complex)

            self.logger.debug(f"Gemini stream finished. Aggregated length: {len(full_response_text)}")
            return full_response_text
            
        except Exception as e:
            self.logger.error(f"Gemini streaming failed: {str(e)}")
            raise AIRequestError(f"Failed to stream Gemini response: {str(e)}", provider="gemini", original_error=e)

    # Keep build_tool_result_messages synchronous
    def build_tool_result_messages(self,
                                  tool_calls: List[ToolCall],
                                  tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Builds Gemini message containing tool results (FunctionResponse parts).
        Gemini requires results in a specific 'function' role message.
        """
        parts = []
        if len(tool_calls) != len(tool_results):
            self.logger.error(f"Mismatch between tool calls ({len(tool_calls)}) and results ({len(tool_results)}). Cannot build Gemini result messages.")
            return []
            
        for call, result in zip(tool_calls, tool_results):
             # Gemini FunctionResponse expects a dictionary in the 'response' field.
             response_payload = {}
             if result.success:
                  if isinstance(result.result, dict):
                       response_payload = result.result
                  elif isinstance(result.result, list): # Wrap list in a standard key? e.g., 'result'
                       response_payload = { "result": result.result }
                  elif result.result is not None:
                       # Attempt to parse if string looks like JSON, otherwise wrap in 'content'
                       content_str = str(result.result)
                       try:
                            parsed = json.loads(content_str)
                            if isinstance(parsed, dict):
                                 response_payload = parsed
                            else: # Parsed but not dict, wrap it
                                 response_payload = { "content": parsed }
                       except json.JSONDecodeError:
                             response_payload = { "content": content_str }
                  else:
                       response_payload = { "content": None } # Explicit None content
             else:
                  # Include error information in the response payload
                  response_payload = {
                       "error": {
                            "message": str(result.error) if result.error else "Tool execution failed."
                       }
                  }
             
             # Use the dictionary structure directly
             try:
                  parts.append({
                       "function_response": {
                            "name": call.name,
                            "response": response_payload
                       }
                  })
             except Exception as e:
                  # Should be less likely now with direct dict construction
                  self.logger.warning(f"Error constructing function_response dict for {call.name}. Error: {e}")

        if parts:
            # Return the single message dict structure expected by _format_messages
            return [
                {
                    "role": "function", # Use 'function' role for Gemini
                    "parts": parts
                }
            ]
        else:
             self.logger.warning("No valid parts generated for Gemini tool results.")
             return []

    # Removed redundant add_tool_message
    # Removed redundant request method

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