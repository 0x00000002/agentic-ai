"""
Core module for AI framework.
Contains base implementations and interfaces.
"""
from .interfaces import (
    AIInterface,
    ProviderInterface,
    ToolCapableProviderInterface,
    MultimediaProviderInterface
)
from ..utils.logger import LoggerInterface
from .base_ai import AIBase
from .tool_enabled_ai import ToolEnabledAI
from .providers.base_provider import BaseProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider
from .providers.gemini_provider import GeminiProvider
from .model_selector import ModelSelector

__all__ = [
    # Interfaces
    'AIInterface',
    'ProviderInterface',
    'LoggerInterface',
    'ToolCapableProviderInterface',
    'MultimediaProviderInterface',
    
    # Base Classes
    'AIBase',
    'ToolEnabledAI',
    'BaseProvider',
    'ModelSelector',
    
    # Providers
    'OpenAIProvider',
    'OllamaProvider',
    'GeminiProvider',
    'AnthropicProvider'
] 