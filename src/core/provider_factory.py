"""
Factory for creating provider instances.
"""
from typing import Optional, Dict, Any, Type, Union

from ..config.unified_config import UnifiedConfig
from .providers.base_provider import BaseProvider
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from ..utils.logger import LoggerFactory, LoggerInterface


class ProviderFactory:
    """
    Factory for creating provider instances.
    
    This class provides a centralized way to create provider instances
    based on the provider type and model configuration.
    """
    
    _providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'gemini': GeminiProvider,
        'ollama': OllamaProvider,
    }
    
    @classmethod
    def create(
        cls,
        provider_type: str,
        model_id: str,
        provider_config: Dict[str, Any],
        model_config: Dict[str, Any],
        logger: Optional[LoggerInterface] = None
    ) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            provider_type: Type of provider (e.g., 'openai', 'anthropic')
            model_id: Model identifier
            provider_config: Configuration dictionary for the provider
            model_config: Configuration dictionary for the specific model
            logger: Logger instance
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If the provider type is not supported
        """
        # --- Add EXTREME Factory Logging --- 
        temp_logger = logger or LoggerFactory.create(name="temp_provider_factory")
        temp_logger.info(f"ProviderFactory: ENTERING CREATE. Requested type: '{provider_type}', Model ID: '{model_id}'")
        temp_logger.info(f"ProviderFactory: _providers dict BEFORE get: {{key: val.__name__ for key, val in cls._providers.items()}}") # Log class names
        # --- End Factory Logging ---

        # Get provider class
        provider_class = cls._providers.get(provider_type)
        
        # --- Log Retrieved Class --- 
        retrieved_class_name = provider_class.__name__ if provider_class else 'None'
        temp_logger.info(f"ProviderFactory: _providers.get('{provider_type}') returned class: {retrieved_class_name}")
        # --- End Log ---
        
        if not provider_class:
            temp_logger.error(f"ProviderFactory: Unsupported provider type '{provider_type}'")
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        # Create logger if not provided (use the correct logger now)
        logger = temp_logger 
        
        # Create the provider instance
        logger.info(f"ProviderFactory: About to instantiate class: {provider_class.__name__} using model_id: {model_id}")
        instance = provider_class(
            model_id=model_id,
            provider_config=provider_config,
            model_config=model_config,
            logger=logger
        )
        logger.info(f"ProviderFactory: Instantiation returned instance of type: {type(instance)}")
        logger.info(f"ProviderFactory: EXITING CREATE for type '{provider_type}'.")
        return instance
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a new provider type.
        
        Args:
            provider_type: Type identifier for the provider
            provider_class: Provider class to register
        """
        cls._providers[provider_type] = provider_class 