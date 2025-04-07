"""
OpenAI provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from ..interfaces import ProviderInterface, MultimediaProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError, AIAuthenticationError, AIRateLimitError, ModelNotFoundError, ContentModerationError, InvalidRequestError
from ...tools.models import ToolResult, ToolCall, ToolDefinition
# Import the ProviderResponse model
from ..models import ProviderResponse, TokenUsage
from .base_provider import BaseProvider
import openai
import os
import json


class OpenAIProvider(BaseProvider, MultimediaProviderInterface, ToolCapableProviderInterface):
    """Provider implementation for OpenAI models with multimedia capabilities."""
    
    # Role mapping for OpenAI API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
    
    def __init__(self, 
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_id: The model identifier
            provider_config: Configuration for the OpenAI provider (API key, base URL, etc.)
            model_config: Configuration for the specific model (parameters)
            logger: Logger instance
        """
        super().__init__(model_id=model_id,
                         provider_config=provider_config,
                         model_config=model_config,
                         logger=logger)
        
        # --- Correctly initialize parameters dictionary --- 
        self.parameters = {}
        # Extract known OpenAI parameters directly from model_config
        openai_params = ["temperature", "max_tokens", "top_p", 
                         "frequency_penalty", "presence_penalty", "stop", 
                         "response_format"]
        
        # Use output_limit from config for max_tokens if present
        if 'output_limit' in model_config:
            self.parameters['max_tokens'] = model_config['output_limit']
            
        for key in openai_params:
            if key in model_config and key != 'max_tokens': # Avoid overwriting if output_limit was used
                self.parameters[key] = model_config[key]
        
        # Provide defaults if not specified in config
        self.parameters.setdefault('temperature', 0.7)
        self.parameters.setdefault('max_tokens', 4096) # Default fallback
        # Add other defaults as needed
        # --- End parameter initialization ---
        
        self.logger.info(f"Initialized OpenAI provider with model {self.model_id}")
        self.logger.debug(f"OpenAI Parameters set: {self.parameters}") # Log the parameters
    
    def _initialize_credentials(self) -> None:
        """Initialize OpenAI API credentials."""
        try:
            # Get API key from configuration
            api_key = self.provider_config.get("api_key")
            if not api_key:
                api_key = self.config.get_api_key("openai")
                
            # Set up OpenAI client
            self.client = openai.OpenAI(api_key=api_key)
            
            # Get base URL if specified (for Azure, etc.)
            base_url = self.provider_config.get("base_url")
            if base_url:
                self.client.base_url = base_url
                
            # Get organization if specified
            org_id = self.provider_config.get("organization")
            if org_id:
                self.client.organization = org_id
                
        except openai.AuthenticationError as e:
             self.logger.error(f"OpenAI Authentication Error: {e}")
             raise AIAuthenticationError(f"OpenAI authentication failed: {e}", provider="openai") from e
        except openai.PermissionDeniedError as e: # Often indicates API key issue
             self.logger.error(f"OpenAI Permission Denied Error: {e}")
             raise AIAuthenticationError(f"OpenAI permission denied (check API key/org): {e}", provider="openai") from e
        except Exception as e:
            # Catch other potential setup errors (network, config)
             self.logger.error(f"Failed to initialize OpenAI credentials: {str(e)}", exc_info=True)
             raise AICredentialsError(f"Failed to initialize OpenAI credentials: {str(e)}", provider="openai") from e
    
    def _map_role(self, role: str) -> str:
        """Map standard role to OpenAI role."""
        return self._ROLE_MAP.get(role, "user")
        
    def request(self, messages: List[Dict[str, str]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the OpenAI API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Response content as a string, or a dictionary with content and tool calls
        """
        try:
            # Format messages for OpenAI API
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # --- Format tools if present ---
            formatted_tools = None
            if "tools" in params:
                tools = params.pop("tools") # Remove from params to avoid sending raw
                if isinstance(tools, list):
                    formatted_tools = []
                    for tool in tools:
                        if isinstance(tool, dict) and "function" in tool and "type" not in tool:
                             # Assume function type if not specified
                            formatted_tools.append({"type": "function", "function": tool["function"]})
                        elif isinstance(tool, dict) and "name" in tool and "parameters" in tool:
                             # Adapt basic structure if needed
                             formatted_tools.append({
                                 "type": "function",
                                 "function": {
                                     "name": tool["name"],
                                     "description": tool.get("description", ""),
                                     "parameters": tool["parameters"]
                                 }
                             })
                        else:
                             # Keep tool as is if format is unexpected or already correct
                            formatted_tools.append(tool) 
            
            # Remove any non-OpenAI parameters from the main params
            allowed_params = {"temperature", "max_tokens", "top_p", "frequency_penalty", 
                              "presence_penalty", "stop", "tool_choice", "response_format"}
            params_to_send = {k: v for k, v in params.items() if k in allowed_params}
            
            # Prepare arguments for the API call
            api_args = {
                "model": self.model_id,
                "messages": formatted_messages,
                **params_to_send  # Add filtered standard parameters
            }
            
            # Add formatted tools and tool_choice if they exist
            if formatted_tools:
                api_args["tools"] = formatted_tools
                if "tool_choice" in params: # Check original params for tool_choice
                    api_args["tool_choice"] = params["tool_choice"]
                else:
                     # Default tool_choice if tools are provided but no choice specified
                     # Might need adjustment based on desired behavior (e.g., 'auto')
                     pass 

            # Make the API call
            response = self.client.chat.completions.create(**api_args)
            
            # Extract tool calls if present
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                            id=tool_call.id
                        )
                    )
            
            # If tool calls are present, return a dictionary with content and tool calls
            if tool_calls:
                return {
                    "content": response.choices[0].message.content or "",
                    "tool_calls": tool_calls
                }
            
            # Otherwise, return just the content
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making OpenAI request: {str(e)}")
    
    def request_with_tools(self, 
                          prompt: str, 
                          tools: List[Dict[str, Any]],
                          conversation: Optional[List[Tuple[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          structured_tools: bool = True) -> Dict[str, Any]:
        """
        Make a request that can use tools.
        
        Args:
            prompt: User prompt
            tools: List of tool definitions
            conversation: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt
            structured_tools: Whether to use structured tools format
            
        Returns:
            Dictionary with content and tool calls
        """
        try:
            # Build messages
            messages = []
            
            # Add system message
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            if conversation:
                for role, content in conversation:
                    messages.append({"role": role, "content": content})
                    
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Format tools for OpenAI API
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                formatted_tools.append(formatted_tool)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=self._format_messages(messages),
                tools=formatted_tools,
                tool_choice="auto"
            )
            
            # Process response
            content = response.choices[0].message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                            id=tool_call.id
                        )
                    )
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making OpenAI tools request: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]], **options) -> str:
        """
        Stream a response from the OpenAI API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Format messages for OpenAI API
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-OpenAI parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "frequency_penalty", 
                            "presence_penalty", "stop", "tools", "tool_choice", "response_format"]:
                    del params[key]
            
            # Make the streaming API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=formatted_messages,
                stream=True,
                **params
            )
            
            # Collect the chunks
            chunks = []
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    
            # Join all chunks
            return "".join(chunks)
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in streaming: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error streaming from OpenAI: {str(e)}")
    
    def analyze_image(self, image_data: Union[str, BinaryIO], prompt: str) -> str:
        """
        Analyze an image with the model.
        
        Args:
            image_data: Image data (file path, URL, or file-like object)
            prompt: Prompt describing what to analyze in the image
            
        Returns:
            Analysis as a string
        """
        try:
            # Create message with image
            if isinstance(image_data, str) and (image_data.startswith("http") or image_data.startswith("https")):
                # Handle URL
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            else:
                # Handle base64 or file-like object
                if isinstance(image_data, str):
                    # Assume it's a file path
                    with open(image_data, "rb") as f:
                        # Encode to base64
                        import base64
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                else:
                    # File-like object
                    import base64
                    base64_image = base64.b64encode(image_data.read()).decode("utf-8")
                    
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
                
            # Create messages with image
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    image_message
                ]}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages
            )
            
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in image analysis: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error analyzing image with OpenAI: {str(e)}")
    
    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
        """
        Generate an image with DALL-E.
        
        Args:
            prompt: Prompt describing the image to generate
            size: Image size (1024x1024, 512x512, or 256x256)
            quality: Image quality (standard or hd)
            
        Returns:
            URL of the generated image
        """
        try:
            # Check if model is DALL-E
            if "dall-e" not in self.model_id.lower():
                self.logger.warning(f"Attempting to generate image with non-DALL-E model: {self.model_id}")
            
            # Make API call
            response = self.client.images.generate(
                model=self.model_id,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Return image URL
            if response.data and response.data[0].url:
                return response.data[0].url
            else:
                raise AIProviderError("No image URL returned from OpenAI")
                
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in image generation: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error generating image with OpenAI: {str(e)}")
    
    def transcribe_audio(self, audio_data: Union[str, BinaryIO]) -> str:
        """
        Transcribe audio using the OpenAI API.
        
        Args:
            audio_data: Audio file path or file-like object
            
        Returns:
            Transcription as a string
        """
        try:
            # Make the transcription request
            if isinstance(audio_data, str):
                # Assume it's a file path
                with open(audio_data, "rb") as f:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f
                    )
            else:
                # Handle file-like object
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data
                )
            
            return transcript.text
            
        except Exception as e:
            self.logger.error(f"OpenAI transcription failed: {str(e)}")
            raise AIRequestError(f"OpenAI transcription failed: {str(e)}")
    
    def text_to_speech(self, 
                      text: str, 
                      **options) -> Union[bytes, str]:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text to convert to speech
            options: Additional options including:
                - voice: Voice to use (default: alloy)
                - model: TTS model to use (default: tts-1)
                - output_path: Path to save audio file (optional)
                - format: Audio format (default: mp3)
            
        Returns:
            Audio data as bytes or path to saved audio file
        """
        try:
            # Set default options
            voice = options.get("voice", "alloy")
            model = options.get("model", "tts-1")
            output_path = options.get("output_path", None)
            
            # Generate speech
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Save to file if output path is provided
            if output_path:
                response.stream_to_file(output_path)
                self.logger.info(f"Speech saved to file: {output_path}")
                return output_path
            
            # Otherwise return audio data
            audio_data = response.content
            self.logger.info("Speech generation successful")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"OpenAI text-to-speech failed: {str(e)}")
            raise AIRequestError(
                f"Failed to generate speech: {str(e)}",
                provider="openai",
                original_error=e
            )
    
    def _add_tool_message(self, 
                         tool_call_id: str, 
                         tool_name: str, 
                         content: str,
                         last_assistant_message: Optional[Dict[str, Any]] = None # Not used by OpenAI
                         ) -> List[Dict[str, Any]]:
        """
        Constructs the single 'tool' role message for OpenAI.
        
        Args:
            tool_call_id: The ID of the tool call this is a result for.
            tool_name: The name of the tool that was called.
            content: The result content of the tool execution (as a string).
            last_assistant_message: Not used by OpenAI.
            
        Returns:
            A list containing a single tool result message dictionary.
        """
        self.logger.debug(f"Constructing tool result message for call_id: {tool_call_id}")
        # OpenAI expects a single message with role 'tool'
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id, 
            "name": tool_name, # Although optional, OpenAI examples include it
            "content": str(content) # Ensure content is string
        }
        return [tool_message] # Return as a list containing one dictionary

    def build_tool_result_messages(self, 
                                  tool_calls: List[ToolCall], 
                                  tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Builds OpenAI 'tool' role messages for each tool result.
        """
        messages = []
        for call, result in zip(tool_calls, tool_results):
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name, 
                "content": str(result.result) if result.status == "success" else str(result.error or "Tool execution failed")
            })
        return messages

    def _make_api_request(self, payload: Dict[str, Any]) -> openai.types.chat.ChatCompletion:
        """Makes the actual API call to OpenAI chat completions."""
        self.logger.debug(f"Making OpenAI API request with model {self.model_id}...")
        try:
            response = self.client.chat.completions.create(**payload)
            self.logger.debug(f"Received OpenAI API response.")
            return response
        # --- Specific OpenAI Error Handling --- 
        except openai.AuthenticationError as e:
             self.logger.error(f"OpenAI Authentication Error: {e}")
             raise AIAuthenticationError(f"OpenAI authentication failed: {e}", provider="openai") from e
        except openai.PermissionDeniedError as e:
             self.logger.error(f"OpenAI Permission Denied Error: {e}")
             raise AIAuthenticationError(f"OpenAI permission denied: {e}", provider="openai") from e
        except openai.RateLimitError as e:
             self.logger.error(f"OpenAI Rate Limit Error: {e}")
             # TODO: Extract retry_after if available from headers
             raise AIRateLimitError(f"OpenAI rate limit exceeded: {e}", provider="openai") from e
        except openai.NotFoundError as e: # Often indicates model not found
            self.logger.error(f"OpenAI Not Found Error (likely model): {e}")
            raise ModelNotFoundError(f"Model or endpoint not found for OpenAI: {e}", provider="openai", model_id=payload.get("model")) from e
        except openai.BadRequestError as e:
             # Check if it's a content filter error
             if e.code == 'content_filter':
                  self.logger.error(f"OpenAI Content Filter Error: {e}")
                  raise ContentModerationError(f"OpenAI content moderation block: {e}", provider="openai", reason=e.code) from e
             else:
                  # General invalid request
                  self.logger.error(f"OpenAI Bad Request Error: {e}")
                  raise InvalidRequestError(f"Invalid request to OpenAI: {e}", provider="openai", status_code=e.status_code) from e
        except openai.APIConnectionError as e:
             self.logger.error(f"OpenAI API Connection Error: {e}")
             raise AIProviderError(f"OpenAI connection error: {e}", provider="openai") from e # Generic provider error for connection issues
        except openai.APIStatusError as e: # Catch other non-2xx status codes
             self.logger.error(f"OpenAI API Status Error ({e.status_code}): {e}")
             raise AIProviderError(f"OpenAI API returned status {e.status_code}: {e}", provider="openai", status_code=e.status_code) from e
        except openai.APIError as e: # Catch-all for other OpenAI API errors
            self.logger.error(f"OpenAI API Error: {e}", exc_info=True)
            raise AIProviderError(f"OpenAI API error: {e}", provider="openai", status_code=getattr(e, 'status_code', None)) from e
        # --- End Specific OpenAI Error Handling ---
        except Exception as e:
            # Catch unexpected errors during the request
            self.logger.error(f"Unexpected error during OpenAI API request: {e}", exc_info=True)
            raise AIProviderError(f"Unexpected error making OpenAI request: {e}", provider="openai") from e

    def _convert_response(self, raw_response: openai.types.chat.ChatCompletion) -> ProviderResponse:
        """Converts the raw OpenAI ChatCompletion object into a standardized ProviderResponse model."""
        self.logger.debug("Converting OpenAI response to standard ProviderResponse...")
        if not raw_response.choices:
            # Return an error response object
            return ProviderResponse(error="OpenAI response missing 'choices'.")
            
        message = raw_response.choices[0].message
        stop_reason = raw_response.choices[0].finish_reason
        usage_data = raw_response.usage
        model_id = raw_response.model

        content = message.content or None # Use None if empty
        tool_calls_list = []

        if message.tool_calls:
            self.logger.debug(f"Detected {len(message.tool_calls)} tool calls in response.")
            for tc in message.tool_calls:
                if tc.type == "function":
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON arguments for tool {tc.function.name}: {tc.function.arguments}")
                        arguments = { "_raw_args": tc.function.arguments } # Store raw args if parsing fails
                        
                    tool_calls_list.append(
                        ToolCall(
                            id=tc.id, 
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )
                else:
                     self.logger.warning(f"Unsupported tool call type: {tc.type}")
        else:
            self.logger.debug("No tool calls detected in response.")

        # Create TokenUsage model
        usage = None
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens
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

# End of OpenAIProvider class


# --- Utility Functions ---

def load_openai_client():
    """Load the OpenAI client with API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

# Ensure no trailing code or comments outside definitions 