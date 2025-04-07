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
            
        self.logger.debug(f"Initialized OpenAI provider with model {self.model_id}")
    
    def _initialize_credentials(self) -> None:
        """Initialize OpenAI API credentials."""
        try:
            # Get client initialization arguments
            client_kwargs = self.credential_manager.get_client_kwargs()
            
            # Set up OpenAI client
            self.client = openai.OpenAI(**client_kwargs)
            
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
    
    def _make_api_request(self, payload: Dict[str, Any]) -> openai.types.chat.ChatCompletion:
        """
        Make a request to the OpenAI API.
        
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
            
            # Make the API call
            response = self.client.chat.completions.create(**payload)
            return response
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making OpenAI request: {str(e)}")
    
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
            # Prepare the payload
            payload = self._prepare_request_payload(messages, options)
            
            # Add streaming flag
            payload["stream"] = True
            
            # Make the streaming API call
            response = self.client.chat.completions.create(**payload)
            
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
            # Make API call to DALL-E
            response = self.client.images.generate(
                model="dall-e-3",  # or other model id
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Return URL
            return response.data[0].url
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in image generation: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error generating image with OpenAI: {str(e)}")
    
    def transcribe_audio(self, audio_data: Union[str, BinaryIO]) -> str:
        """
        Transcribe audio using Whisper API.
        
        Args:
            audio_data: Audio data (file path or file-like object)
            
        Returns:
            Transcribed text
        """
        try:
            # Handle file path or file-like object
            if isinstance(audio_data, str):
                # File path
                with open(audio_data, "rb") as f:
                    audio_file = f
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            else:
                # File-like object
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data
                )
            
            return response.text
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in transcription: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error transcribing audio with OpenAI: {str(e)}")
    
    def text_to_speech(self, 
                      text: str, 
                      **options) -> Union[bytes, str]:
        """
        Convert text to speech using OpenAI TTS API.
        
        Args:
            text: Text to convert to speech
            **options: Additional options
                voice: Voice to use (default: alloy)
                model: TTS model ID (default: tts-1)
                response_format: Format of the response (default: mp3)
                output_file: Optional path to save output
                
        Returns:
            Audio data as bytes or path to saved file
        """
        try:
            # Get options
            voice = options.get("voice", "alloy")
            model = options.get("model", "tts-1")
            response_format = options.get("response_format", "mp3")
            output_file = options.get("output_file")
            
            # Make API call
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format
            )
            
            # Save to file if output_file is specified
            if output_file:
                # Make sure the directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Write to file
                with open(output_file, "wb") as f:
                    f.write(response.content)
                return output_file
            
            # Otherwise, return the bytes
            return response.content
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in text-to-speech: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error with OpenAI text-to-speech: {str(e)}")


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