"""
Core interfaces for AI and provider implementations.
"""
from typing import Protocol, List, Dict, Any, Optional, Union, Tuple, BinaryIO
from typing_extensions import runtime_checkable
# Import necessary types for tool handling
from src.tools.models import ToolCall
from ..tools.models import ToolResult


@runtime_checkable
class AIInterface(Protocol):
    """Interface for synchronous AI interactions."""
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response (after any tool calls have been resolved)
        """
        ...
    
    def stream(self, prompt: str, **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The complete streamed response
        """
        ...
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        ...
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages
        """
        ...


@runtime_checkable
class ProviderInterface(Protocol):
    """Interface for AI providers."""
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Either a string response (when no tools are called) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        ...
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        ...


@runtime_checkable
class ToolCapableProviderInterface(ProviderInterface, Protocol):
    """Interface for providers that support tools/functions."""
    
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
        ...

    def build_tool_result_messages(self, 
                                  tool_calls: List['ToolCall'], 
                                  tool_results: List['ToolResult']) -> List[Dict[str, Any]]:
        """
        Builds the list of message dictionaries representing tool results, 
        formatted correctly for the specific provider's API.

        This might return multiple messages (e.g., one per tool for OpenAI)
        or a single message containing all results (e.g., for Anthropic).

        Args:
            tool_calls: The list of ToolCall objects that the AI requested.
            tool_results: The list of corresponding ToolResult objects.

        Returns:
            A list of message dictionaries to be added to the conversation history.
        """
        ...


@runtime_checkable
class MultimediaProviderInterface(Protocol):
    """Interface for providers that support multimedia processing capabilities."""
    
    def transcribe_audio(self, 
                         audio_file: Union[str, BinaryIO], 
                         **options) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio to text.
        
        Args:
            audio_file: Path to audio file or file-like object
            options: Additional options (language, format, etc.)
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        ...
    
    def text_to_speech(self, 
                      text: str, 
                      **options) -> Union[bytes, str]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            options: Additional options (voice, format, etc.)
            
        Returns:
            Audio data as bytes or path to saved audio file
        """
        ...
    
    def analyze_image(self, 
                     image_file: Union[str, BinaryIO], 
                     **options) -> Dict[str, Any]:
        """
        Analyze image content.
        
        Args:
            image_file: Path to image file or file-like object
            options: Additional options
            
        Returns:
            Analysis results as dictionary
        """
        ... 