"""
Core interfaces for AI and provider implementations.
"""
from typing import Protocol, List, Dict, Any, Optional, Union, Tuple, BinaryIO
import asyncio # Added asyncio
from typing_extensions import runtime_checkable
# Import necessary types for tool handling
from src.tools.models import ToolCall
from ..tools.models import ToolResult
from .models import ProviderResponse # Added ProviderResponse import


@runtime_checkable
class AIInterface(Protocol):
    """Interface for AI interactions (now potentially asynchronous)."""
    
    async def request(self, prompt: str, **options) -> str: # Changed to async def
        """
        Make an asynchronous request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response (after any tool calls have been resolved)
        """
        ...
    
    async def stream(self, prompt: str, **options) -> str: # Changed to async def
        """
        Stream a response asynchronously from the AI model.
        
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
    """Interface for AI providers (now potentially asynchronous)."""
    
    async def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> ProviderResponse: # Changed to async def and return type ProviderResponse
        """
        Make an asynchronous request to the AI model provider.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            A ProviderResponse object containing the response content, potential tool calls,
            and other metadata.
        """
        ...
    
    async def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str: # Changed to async def
        """
        Stream a response asynchronously from the AI model provider.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        ...


@runtime_checkable
class ToolCapableProviderInterface(ProviderInterface, Protocol):
    """Interface for providers that support tools/functions (async methods)."""
    
    async def add_tool_message(self, messages: List[Dict[str, Any]], # Changed to async def
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history asynchronously.
        (Method signature might need adjustment based on provider needs)
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        ...

    async def build_tool_result_messages(self, # Changed to async def
                                  tool_calls: List['ToolCall'], 
                                  tool_results: List['ToolResult']) -> List[Dict[str, Any]]:
        """
        Builds the list of message dictionaries representing tool results asynchronously,
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