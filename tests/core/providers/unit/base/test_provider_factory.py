"""
Unit tests for the provider factory implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
from typing import Dict, Any, Type
import os

from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.provider_factory import ProviderFactory
from src.exceptions import AIProviderError, AICredentialsError
from src.utils.logger import LoggerInterface
from src.config.unified_config import UnifiedConfig
from src.core.models import ProviderResponse

class TestProviderFactory:
    """Test suite for ProviderFactory class."""

    @pytest.fixture
    def factory(self) -> ProviderFactory:
        """Create a provider factory instance for testing."""
        return ProviderFactory()

    @pytest.fixture
    def mock_provider_config(self) -> Dict[str, Any]:
        """Create a mock provider configuration for testing."""
        return {
            "api_key": "test_key"
        }

    @pytest.fixture
    def mock_model_config(self) -> Dict[str, Any]:
        """Create a mock model configuration for testing."""
        return {
            "temperature": 0.7,
            "max_tokens": 100
        }

    @pytest.fixture
    def mock_logger(self) -> Mock:
        """Create a mock logger for testing."""
        return Mock()

    def test_initialization(self, factory: ProviderFactory):
        """Test that the provider factory is initialized with expected providers."""
        assert "openai" in factory._providers
        assert "anthropic" in factory._providers
        assert "gemini" in factory._providers
        assert "ollama" in factory._providers

    def test_register_provider(self, factory: ProviderFactory):
        """Test registering a new provider."""
        class TestProvider:
            pass
        
        factory.register_provider("test", TestProvider)
        assert factory._providers["test"] == TestProvider

    def test_register_duplicate_provider(self, factory: ProviderFactory):
        """Test registering a duplicate provider."""
        class TestProvider:
            pass
        
        factory.register_provider("test", TestProvider)
        factory.register_provider("test", TestProvider)  # Should not raise
        assert factory._providers["test"] == TestProvider

    def test_create_openai_provider(self, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating an OpenAI provider by mocking the class in the factory map."""
        mock_openai_class = MagicMock(spec=OpenAIProvider)
        mock_instance = Mock(spec=OpenAIProvider)
        mock_openai_class.return_value = mock_instance
        mock_openai_class.__name__ = OpenAIProvider.__name__

        original_provider = factory._providers.get("openai")
        factory._providers["openai"] = mock_openai_class
        try:
            provider = factory.create("openai", "test_model", mock_provider_config, mock_model_config)
            mock_openai_class.assert_called_once_with(
                model_id="test_model",
                provider_config=mock_provider_config,
                model_config=mock_model_config,
                logger=ANY
            )
            assert provider is mock_instance
        finally:
            if original_provider:
                 factory._providers["openai"] = original_provider
            else:
                 del factory._providers["openai"]

    def test_create_anthropic_provider(self, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating an Anthropic provider by mocking the class in the factory map."""
        mock_anthropic_class = MagicMock(spec=AnthropicProvider)
        mock_instance = Mock(spec=AnthropicProvider)
        mock_anthropic_class.return_value = mock_instance
        mock_anthropic_class.__name__ = AnthropicProvider.__name__

        original_provider = factory._providers.get("anthropic")
        factory._providers["anthropic"] = mock_anthropic_class
        try:
            provider = factory.create("anthropic", "test_model", mock_provider_config, mock_model_config)
            mock_anthropic_class.assert_called_once_with(
                model_id="test_model",
                provider_config=mock_provider_config,
                model_config=mock_model_config,
                logger=ANY
            )
            assert provider is mock_instance
        finally:
            if original_provider:
                 factory._providers["anthropic"] = original_provider
            else:
                 del factory._providers["anthropic"]

    @patch("src.core.providers.gemini_provider.genai")
    def test_create_gemini_provider(self, mock_genai, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating a Gemini provider."""
        with patch('src.core.providers.base_provider.UnifiedConfig.get_api_key', return_value="dummy_global_key"):
            mock_model_instance = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model_instance
            mock_genai.configure.reset_mock()
            mock_genai.GenerativeModel.reset_mock()

            provider = factory.create("gemini", "test_model", mock_provider_config, mock_model_config)
            assert isinstance(provider, GeminiProvider)
            # Assert genai.configure was called AT LEAST once with the correct key
            mock_genai.configure.assert_any_call(api_key=mock_provider_config["api_key"])
            # Assert GenerativeModel was called once (during __init__)
            mock_genai.GenerativeModel.assert_called_once()

    @patch("src.core.providers.ollama_provider.AsyncClient")
    def test_create_ollama_provider(self, mock_async_client, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating an Ollama provider using AsyncClient."""
        # The factory will instantiate OllamaProvider, which in turn instantiates AsyncClient
        provider = factory.create("ollama", "test_model", mock_provider_config, mock_model_config)
        
        assert isinstance(provider, OllamaProvider)
        # Check that AsyncClient was instantiated (called)
        mock_async_client.assert_called_once()

    def test_create_provider_with_invalid_config(self, factory: ProviderFactory, mock_logger):
        """Test creating a provider with invalid configuration (missing API key)."""
        # Patch LoggerFactory to return our mock logger
        # Patch config lookups AND os.environ.get to ensure no key is found
        with patch('src.core.provider_factory.LoggerFactory.create', return_value=mock_logger) as mock_logger_factory, \
             patch('src.core.providers.base_provider.UnifiedConfig.get_api_key', return_value=None) as mock_get_key, \
             patch('src.core.providers.openai_provider.os.environ.get', return_value=None) as mock_env_get: # Re-add env var patch
            
             # Now that all key sources are patched to return None, the provider should 
             # attempt openai.OpenAI(api_key=None), raising the wrapped error.
             expected_error_match = "Failed to initialize OpenAI credentials: The api_key client option must be set"
             with pytest.raises(AICredentialsError, match=expected_error_match):
                 factory.create("openai", "test_model", {}, {})

             # Assertions on the patches
             mock_get_key.assert_called_once_with("openai")
             mock_env_get.assert_called_with("OPENAI_API_KEY") # Re-add assertion
             
             # Assert our mock logger was called with exc_info=True
             error_call = None
             for call in mock_logger.error.call_args_list:
                  if "Failed to initialize OpenAI credentials" in call.args[0]: 
                      error_call = call
                      break
             assert error_call is not None, "Logger.error was not called with the credential failure message"
             assert error_call.kwargs.get('exc_info') is True, "Logger.error was not called with exc_info=True"

    def test_create_invalid_provider(self, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating an invalid provider."""
        with pytest.raises(ValueError):
            factory.create("invalid", "test_model", mock_provider_config, mock_model_config)

    def test_create_provider_with_logger(self, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating a provider with a logger."""
        logger = Mock(spec=LoggerInterface)
        provider = factory.create("openai", "test_model", mock_provider_config, mock_model_config, logger=logger)
        assert provider.logger == logger

    def test_create_custom_provider(self, factory: ProviderFactory, mock_provider_config: Dict[str, Any], mock_model_config: Dict[str, Any]):
        """Test creating a custom provider."""
        class CustomProvider:
            def __init__(self, model_id, provider_config, model_config, logger=None):
                self.model_id = model_id
                self.provider_config = provider_config
                self.model_config = model_config
                self.logger = logger

        factory.register_provider("custom", CustomProvider)
        provider = factory.create("custom", "test_model", mock_provider_config, mock_model_config)
        assert isinstance(provider, CustomProvider)
        assert provider.model_id == "test_model"
        assert provider.provider_config == mock_provider_config
        assert provider.model_config == mock_model_config 