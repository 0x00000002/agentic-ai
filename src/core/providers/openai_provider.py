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
# Import async client
from openai import AsyncOpenAI # Add async client
import os
import json
import asyncio # Add asyncio


class OpenAIProvider(BaseProvider, MultimediaProviderInterface, ToolCapableProviderInterface):
    """Provider implementation for OpenAI models with multimedia capabilities."""
    
    # Role mapping for OpenAI API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
    
    # Parameter mapping for OpenAI API
    PARAMETER_MAPPING = {}  # OpenAI uses standard parameter names
    
    # Set of parameters allowed by OpenAI API
    ALLOWED_PARAMETERS = {
        "temperature", "max_tokens", "top_p", "frequency_penalty", 
        "presence_penalty", "stop", "response_format"
    }
    
    # Default parameters for OpenAI API
    DEFAULT_PARAMETERS = {
        "temperature": 0.7,
        "max_tokens": 4096  # Default fallback
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
        
        # Use output_limit from config for max_tokens if present
        if 'output_limit' in model_config:
            self.parameter_manager.update_defaults({'max_tokens': model_config['output_limit']})
            
        # Client will be initialized asynchronously if needed, or keep sync init?
        # For now, let's assume sync initialization is fine, but API calls are async.
        # self.client: Optional[AsyncOpenAI] = None # Initialize as None
        self.logger.debug(f"Initialized OpenAI provider with model {self.model_id}")
    
    def _initialize_credentials(self) -> None:
        """Initialize OpenAI API credentials."""
        try:
            # Get client initialization arguments
            client_kwargs = self.credential_manager.get_client_kwargs()
            
            # Set up OpenAI client - Use AsyncOpenAI
            self.client = AsyncOpenAI(**client_kwargs) # Changed to AsyncOpenAI
            
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
    
    async def _make_api_request(self, payload: Dict[str, Any]) -> openai.types.chat.ChatCompletion: # Changed to async def
        """
        Make an asynchronous request to the OpenAI API.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            ChatCompletion object from the OpenAI SDK
        """
        try:
            # Remove any special keys not supported by OpenAI
            for key in list(payload.keys()):
                if key not in ["model", "messages", "temperature", "max_tokens", "top_p", 
                               "frequency_penalty", "presence_penalty", "stop", 
                               "tools", "tool_choice", "response_format"]:
                    del payload[key]
            
            # Make the API call using the async client
            self.logger.debug(f"Making async API call to OpenAI with payload: {payload}")
            response = await self.client.chat.completions.create(**payload) # Use await
            self.logger.debug("Async API call to OpenAI completed.")
            return response
            
        except openai.APIConnectionError as e:
             self.logger.error(f"OpenAI API Connection Error: {e}")
             raise AIProviderError(f"Failed to connect to OpenAI API: {e}", provider="openai") from e
        except openai.RateLimitError as e:
             self.logger.error(f"OpenAI Rate Limit Error: {e}")
             raise AIRateLimitError(f"OpenAI rate limit exceeded: {e}", provider="openai") from e
        except openai.AuthenticationError as e: # Catch auth errors during request too
             self.logger.error(f"OpenAI Authentication Error during request: {e}")
             raise AIAuthenticationError(f"OpenAI authentication failed during request: {e}", provider="openai") from e
        except openai.NotFoundError as e: # Model not found
             self.logger.error(f"OpenAI Model Not Found Error: {e}")
             raise ModelNotFoundError(f"OpenAI model not found: {self.model_id} - {e}", provider="openai", model=self.model_id) from e
        except openai.BadRequestError as e:
             # Differentiate between content moderation and other bad requests
             if "content_filter" in str(e):
                 self.logger.error(f"OpenAI Content Moderation Error: {e}")
                 raise ContentModerationError(f"OpenAI content moderation triggered: {e}", provider="openai") from e
             else:
                 self.logger.error(f"OpenAI Invalid Request Error: {e}")
                 raise InvalidRequestError(f"OpenAI invalid request: {e}", provider="openai") from e
        except openai.APIStatusError as e:
             # Catch broader API status errors (like 500s)
             self.logger.error(f"OpenAI API Status Error: {e.status_code} - {e.response}")
             raise AIRequestError(f"OpenAI API returned status {e.status_code}: {e}", provider="openai") from e
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected error making OpenAI async request: {str(e)}", exc_info=True)
            raise AIProviderError(f"Unexpected error during OpenAI request: {str(e)}", provider="openai") from e
    
    def _convert_response(self, raw_response: openai.types.chat.ChatCompletion) -> ProviderResponse:
        """
        Convert OpenAI ChatCompletion to ProviderResponse.
        
        Args:
            raw_response: ChatCompletion object from the OpenAI SDK
            
        Returns:
            Standardized ProviderResponse object with properly formatted fields
            
        Notes:
            Tool call arguments from OpenAI are automatically parsed from JSON strings
            into Python dictionaries for easier consumption by tool handlers.
            If JSON parsing fails, the raw string is preserved in a '_raw_args' field.
        """
        try:
            # Extract basic fields - make sure content is never None
            content = raw_response.choices[0].message.content or ""
            
            # Extract tool calls if present
            tool_calls = []
            if hasattr(raw_response.choices[0].message, 'tool_calls') and raw_response.choices[0].message.tool_calls:
                for tc in raw_response.choices[0].message.tool_calls:
                    # Parse JSON arguments
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON arguments for tool {tc.function.name}: {tc.function.arguments}")
                        arguments = {"_raw_args": tc.function.arguments}  # Store raw args if parsing fails
                    
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments  # Now using parsed dictionary
                        )
                    )
            
            # Create usage info if available
            usage = None
            if hasattr(raw_response, 'usage') and raw_response.usage:
                usage = TokenUsage(
                    prompt_tokens=raw_response.usage.prompt_tokens,
                    completion_tokens=raw_response.usage.completion_tokens,
                    total_tokens=raw_response.usage.total_tokens
                )
            
            # Create standardized response
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,  # Always provide the list, even if empty
                stop_reason=raw_response.choices[0].finish_reason,
                usage=usage,
                model=raw_response.model,
                raw_response=raw_response
            )
            
        except Exception as e:
            self.logger.error(f"Error converting OpenAI response: {str(e)}", exc_info=True)
            return ProviderResponse(
                content="",
                error=f"Error processing response: {str(e)}"
            )
    
    async def stream(self, messages: List[Dict[str, str]], **options) -> str: # Changed to async def
        """
        Stream a response asynchronously from the OpenAI API.
        NOTE: Returns the aggregated response string. For true streaming,
        use the stream property of the ChatCompletion object.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Prepare the payload
            payload = self._prepare_request_payload(messages, options)
            
            # Add streaming flag
            payload["stream"] = True
            
            # Make the async streaming API call
            stream = await self.client.chat.completions.create(**payload) # Use await
            
            # Collect the chunks asynchronously
            chunks = []
            
            async for chunk in stream: # Use async for
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    
            # Join all chunks
            return "".join(chunks)
            
        except Exception as e:
            raise AIProviderError(f"Error streaming from OpenAI: {str(e)}")
    
    async def analyze_image(self, image_data: Union[str, BinaryIO], prompt: str) -> str: # Changed to async def
        """
        Analyze an image with the model asynchronously.
        
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
            
            # Make the async API call
            response = await self.client.chat.completions.create( # Use await
                model=self.model_id, # Ensure correct model is used
                messages=messages
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise AIProviderError(f"Error analyzing image with OpenAI: {str(e)}")
    
    async def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard") -> str: # Changed to async def
        """
        Generate an image using DALL-E asynchronously.
        
        Args:
            prompt: Prompt describing the image to generate
            size: Image size (1024x1024, 512x512, or 256x256)
            quality: Image quality (standard or hd)
            
        Returns:
            URL of the generated image
        """
        try:
            # Make the async API call
            response = await self.client.images.generate( # Use await
                model=self.model_config.get("image_generation_model", "dall-e-3"), # Use configured or default model
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Return URL
            return response.data[0].url
            
        except Exception as e:
            raise AIProviderError(f"Error generating image with OpenAI: {str(e)}")
    
    async def transcribe_audio(self, audio_data: Union[str, BinaryIO]) -> str: # Changed to async def
        """
        Transcribe audio using Whisper model asynchronously.
        
        Args:
            audio_data: Audio data (file path or file-like object)
            
        Returns:
            Transcribed text
        """
        try:
            # Ensure audio_data is a file-like object for the client
            if isinstance(audio_data, str):
                # If it's a path, open it
                with open(audio_data, "rb") as audio_file:
                    # Make the async API call
                    transcript = await self.client.audio.transcriptions.create( # Use await
                        model=self.model_config.get("audio_transcription_model", "whisper-1"), 
                        file=audio_file
                    )
            else:
                # Assume it's already a file-like object (e.g., BytesIO)
                 transcript = await self.client.audio.transcriptions.create( # Use await
                     model=self.model_config.get("audio_transcription_model", "whisper-1"), 
                     file=audio_data
                 )
            
            return transcript.text
            
        except Exception as e:
            raise AIProviderError(f"Error transcribing audio with OpenAI: {str(e)}")
    
    async def text_to_speech(self, # Changed to async def
                      text: str, 
                      **options) -> Union[bytes, str]:
        """
        Convert text to speech using OpenAI TTS model asynchronously.

        Args:
            text: Text to convert.
            **options: Additional options like 'voice' (alloy, echo, fable, onyx, nova, shimmer)
                       and 'output_format' (mp3, opus, aac, flac). Can also include 'output_path' to save to file.

        Returns:
            Audio data as bytes if 'output_path' is not provided, otherwise path to saved file.
        """
        try:
            voice = options.get("voice", "alloy")
            output_format = options.get("output_format", "mp3")
            output_path = options.get("output_path")

            self.logger.debug(f"Generating speech for text: '{text[:30]}...' with voice: {voice}")
            
            # Make the async API call
            response = await self.client.audio.speech.create( # Use await
                model=self.model_config.get("tts_model", "tts-1"), 
                voice=voice,
                input=text,
                response_format=output_format
            )

            # Handle output
            if output_path:
                 # Stream response content to a file asynchronously
                 await response.astream_to_file(output_path)
                 self.logger.info(f"Saved speech output to {output_path}")
                 return output_path
            else:
                # Read the entire content into bytes asynchronously
                audio_bytes = await response.aread()
                self.logger.info(f"Generated speech audio bytes (length: {len(audio_bytes)})")
                return audio_bytes

        except Exception as e:
             self.logger.error(f"Error generating speech with OpenAI: {e}", exc_info=True)
             raise AIProviderError(f"Error generating speech with OpenAI: {str(e)}")


def load_openai_client():
    """Legacy function to load an OpenAI client from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AICredentialsError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

# End of OpenAIProvider class


# --- Utility Functions ---

def load_openai_client():
    """Load the OpenAI client with API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

# Ensure no trailing code or comments outside definitions 