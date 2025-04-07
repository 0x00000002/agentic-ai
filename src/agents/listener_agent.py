"""
Listener Agent for processing audio input and converting it to text.
Handles speech recognition and audio metadata extraction.
"""
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import os
import tempfile
import json
from ..utils.logger import LoggerInterface, LoggerFactory
from .base_agent import BaseAgent
from ..core.provider_factory import ProviderFactory
from ..config.unified_config import UnifiedConfig
from ..core.interfaces import MultimediaProviderInterface
from ..exceptions import AIAgentError, AIRequestError, ErrorHandler

class ListenerAgent(BaseAgent):
    """
    Agent for processing audio input and converting it to text.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the listener agent.
        
        Args:
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id="listener", **kwargs)
        
        # Load configuration
        self.default_speech_model = self.agent_config.get("default_speech_model", "whisper-1")
        self.language_detection = self.agent_config.get("language_detection", True) if self.agent_config else True
        self.speaker_diarization = self.agent_config.get("speaker_diarization", False) if self.agent_config else False
        self.emotion_detection = self.agent_config.get("emotion_detection", False) if self.agent_config else False
        
        # Try to get multimedia provider capabilities
        try:
            unified_config = kwargs.get("unified_config")
            # Use the provider from the framework's provider system
            provider = ProviderFactory.create(
                provider_type="openai",  # Default to OpenAI for now
                model_id=self.default_speech_model,
                logger=self.logger
            )
            
            # Check if the provider supports multimedia capabilities
            if isinstance(provider, MultimediaProviderInterface):
                self.multimedia_provider = provider
                self.has_multimedia_provider = True
                self.logger.info(f"Initialized with multimedia provider: {type(provider).__name__}")
            else:
                self.multimedia_provider = None
                self.has_multimedia_provider = False
                self.logger.warning(f"Provider {type(provider).__name__} does not support multimedia capabilities")
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to initialize multimedia provider: {str(e)}", agent_id=self.agent_id),
                self.logger
            )
            self.logger.warning(f"Provider initialization error: {error_response['message']}")
            self.multimedia_provider = None
            self.has_multimedia_provider = False
        
        try:
            # Try to import whisper for local processing
            import whisper
            self.whisper_model = whisper.load_model("base")
            self.has_whisper = True
        except (ImportError, Exception) as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Local Whisper not available: {str(e)}", agent_id=self.agent_id),
                self.logger
            )
            self.logger.warning(f"Whisper initialization error: {error_response['message']}")
            self.has_whisper = False
        
        self.logger.info(f"Listener agent initialized with speech model: {self.default_speech_model}")
        
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an audio request and convert it to text.
        
        Args:
            request: Request containing audio data or file path
            
        Returns:
            Response with transcribed text and metadata
        """
        try:
            self.logger.info("Processing audio request")
            
            # Handle different input types
            if 'audio' in request and isinstance(request['audio'], np.ndarray):
                # Process numpy array audio data
                text, metadata = self._process_audio_numpy(request['audio'], request.get('sample_rate', 16000))
            elif 'audio_path' in request and isinstance(request['audio_path'], str):
                # Extract language code from request
                language = request.get("language", None) # None means auto-detect for whisper
                self.logger.debug(f"Language hint from request: {language}")
                # Process audio file path
                text, metadata = self._process_audio_file(request['audio_path'], language=language)
            else:
                raise AIAgentError(
                    "Invalid audio input format. Provide either 'audio' numpy array or 'audio_path' string.",
                    agent_id=self.agent_id
                )
            
            # Return the transcribed text and metadata
            return {
                "text": text,
                "metadata": metadata,
                "content": text,  # For compatibility with standard agent response format
                "agent_id": self.agent_id,
                "status": "success"
            }
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Error processing audio request: {str(e)}", agent_id=self.agent_id),
                self.logger
            )
            self.logger.error(f"Request error: {error_response['message']}")
            
            return {
                "error": error_response['message'],
                "agent_id": self.agent_id,
                "status": "error"
            }
    
    def _process_audio_numpy(self, 
                            audio_data: np.ndarray, 
                            sample_rate: int) -> Tuple[str, Dict[str, Any]]:
        """
        Process audio data from numpy array.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        self.logger.info(f"Processing audio with shape {audio_data.shape} at {sample_rate}Hz")
        
        # Convert float to int16 for file writing
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_path, sample_rate, audio_data)
                self.logger.info(f"Saved audio to temporary file: {temp_path}")
                
                # Process the audio file
                return self._process_audio_file(temp_path)
                
            except Exception as e:
                raise AIAgentError(f"Failed to process audio data: {str(e)}", agent_id=self.agent_id)
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
    
    def _process_audio_file(self, audio_path: str, language: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process audio from a file path.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code hint (e.g., "en", "ru"). If None, auto-detect is used.
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise AIAgentError(f"Audio file does not exist: {audio_path}", agent_id=self.agent_id)
            
        # Initialize metadata
        metadata = {
            "model": self.default_speech_model,
            "language_detection": self.language_detection,
            "detected_language": None,
            "speaker_diarization": self.speaker_diarization,
            "emotion_detection": self.emotion_detection,
            "file_path": audio_path,
            "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
        }
        
        # Check which transcription method to use
        if self.has_multimedia_provider:
            # Use the multimedia provider interface
            return self._transcribe_with_multimedia_provider(audio_path, metadata)
        elif self.has_whisper:
            # Use local Whisper model, passing the language hint
            return self._transcribe_with_local_whisper(audio_path, metadata, language=language)
        else:
            # Fallback to basic transcription
            self.logger.warning("No speech recognition libraries available. Using AI to generate mock response.")
            prompt = "Simulate transcribing an audio recording of someone asking about the weather."
            mock_text = self.ai_instance.request(prompt)
            metadata["note"] = "Mock transcription (no speech recognition libraries available)"
            return mock_text, metadata
    
    def _transcribe_with_multimedia_provider(self, audio_path: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio using a multimedia provider interface.
        
        Args:
            audio_path: Path to the audio file
            metadata: Initial metadata dictionary
            
        Returns:
            Tuple of (transcribed_text, updated_metadata)
        """
        self.logger.info(f"Transcribing with multimedia provider: {type(self.multimedia_provider).__name__}")
        
        try:
            # Set transcription options
            options = {
                "model": self.default_speech_model,
                "response_format": "verbose_json" if self.language_detection else "text",
            }
            
            # Use the provider to transcribe
            transcription, provider_metadata = self.multimedia_provider.transcribe_audio(
                audio_file=audio_path,
                **options
            )
            
            # Merge metadata
            metadata.update(provider_metadata)
            
            self.logger.info(f"Transcription successful: {transcription[:50]}...")
            return transcription, metadata
                
        except AIRequestError as e:
            # This is already a framework error, so we don't need to wrap it
            error_response = ErrorHandler.handle_error(e, self.logger)
            self.logger.error(f"Multimedia provider transcription failed: {error_response['message']}")
            
            # Fallback to simulated transcription
            prompt = "Transcribe audio of someone asking a simple question. Generate a realistic transcription."
            transcription = self.ai_instance.request(prompt)
            
            metadata["note"] = f"Simulated transcription (provider failed: {error_response['message']})"
            metadata["error"] = error_response['message']
            return transcription, metadata
        except Exception as e:
            # Wrap generic exceptions in AIAgentError
            error = AIAgentError(f"Multimedia provider transcription failed: {str(e)}", agent_id=self.agent_id)
            error_response = ErrorHandler.handle_error(error, self.logger)
            
            # Fallback to simulated transcription
            prompt = "Transcribe audio of someone asking a simple question. Generate a realistic transcription."
            transcription = self.ai_instance.request(prompt)
            
            metadata["note"] = f"Simulated transcription (provider failed: {error_response['message']})"
            metadata["error"] = error_response['message']
            return transcription, metadata
    
    def _transcribe_with_local_whisper(self, audio_path: str, metadata: Dict[str, Any], language: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio using local Whisper model.
        
        Args:
            audio_path: Path to the audio file
            metadata: Initial metadata dictionary
            language: Optional language code hint (e.g., "en", "ru"). If None, auto-detect.
            
        Returns:
            Tuple of (transcribed_text, updated_metadata)
        """
        self.logger.info("Transcribing with local Whisper model")
        if language:
             self.logger.info(f"Using language hint: {language}")
        else:
             self.logger.info("No language hint provided, using auto-detect.")

        try:
            # Prepare transcribe options
            transcribe_options = {}
            if language:
                transcribe_options["language"] = language

            # Load audio and transcribe, passing options
            result = self.whisper_model.transcribe(audio_path, **transcribe_options)
            
            transcription = result["text"]
            metadata["detected_language"] = result.get("language", language or "unknown")
            
            self.logger.info(f"Local transcription successful: {transcription[:50]}...")
            return transcription, metadata
            
        except Exception as e:
            error = AIAgentError(f"Local Whisper transcription failed: {str(e)}", agent_id=self.agent_id)
            error_response = ErrorHandler.handle_error(error, self.logger)
            self.logger.error(f"Whisper error: {error_response['message']}")
            
            error_msg = f"Sorry, I couldn't transcribe the audio. Error: {error_response['message']}"
            metadata["error"] = error_response['message']
            return error_msg, metadata
    
    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            request: The request object
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Check if this is an audio request
        if 'type' in request and request['type'] == 'audio':
            return 1.0
        
        if 'audio' in request and isinstance(request['audio'], (np.ndarray, bytes)):
            return 1.0
            
        if 'audio_path' in request and isinstance(request['audio_path'], str):
            return 1.0
        
        # Check for audio-related keywords in the text
        if 'prompt' in request and isinstance(request['prompt'], str):
            audio_keywords = ['transcribe', 'listen', 'speech', 'recording', 'audio', 'voice']
            prompt = request['prompt'].lower()
            
            for keyword in audio_keywords:
                if keyword in prompt:
                    return 0.7
        
        return 0.1