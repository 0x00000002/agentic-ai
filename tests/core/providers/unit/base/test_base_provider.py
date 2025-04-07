"""
Unit tests for the base provider implementation.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.core.providers.base_provider import BaseProvider
from src.core.models import ProviderResponse
from src.tools.models import ToolCall
from src.exceptions import AIProviderError
from src.utils.logger import LoggerInterface

class TestBaseProvider:
    """Test suite for BaseProvider class."""

    @pytest.fixture
    def mock_logger(self):
        logger = Mock(spec=LoggerInterface)
        return logger

    @pytest.fixture
    def mock_provider_config(self):
        return {
            "api_key": "test_key"
        }

    @pytest.fixture
    def mock_model_config(self):
        return {
            "temperature": 0.7,
            "max_tokens": 100
        }

    @pytest.fixture
    def provider(self, mock_logger, mock_provider_config, mock_model_config):
        """Create a BaseProvider instance for testing."""
        return BaseProvider(
            model_id="test_model",
            provider_config=mock_provider_config,
            model_config=mock_model_config,
            logger=mock_logger
        )

    @pytest.fixture
    def mock_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

    @pytest.fixture
    def mock_tool_call(self) -> ToolCall:
        """Create a mock tool call for testing."""
        return ToolCall(
            id="test_tool",
            name="test_function",
            arguments={"arg1": "value1"}
        )

    def test_initialization(self, provider: BaseProvider, mock_provider_config, mock_model_config):
        """Test provider initialization."""
        assert provider.model_id == "test_model"
        assert provider.provider_config == mock_provider_config
        assert provider.model_config == mock_model_config

    def test_request_not_implemented(self, provider: BaseProvider, mock_messages: List[Dict[str, str]], mock_logger):
        """Test that request method returns a response with error for not implemented _make_api_request."""
        response = provider.request(mock_messages)
        assert response.error is not None
        assert "Subclasses must implement _make_api_request" in response.error
        mock_logger.error.assert_called_once()

    def test_stream_not_implemented(self, provider: BaseProvider, mock_messages: List[Dict[str, str]], mock_logger):
        """Test that stream method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subclasses must implement stream"):
            provider.stream(mock_messages)
        mock_logger.error.assert_not_called()

    def test_map_role(self, provider: BaseProvider):
        """Test role mapping."""
        assert provider._map_role("system") == "system"
        assert provider._map_role("user") == "user"
        assert provider._map_role("assistant") == "assistant"
        assert provider._map_role("unknown") == "unknown"

    def test_format_messages(self, provider: BaseProvider, mock_messages: List[Dict[str, str]]):
        """Test message formatting."""
        formatted = provider._format_messages(mock_messages)
        assert formatted == mock_messages

    def test_post_process_formatted_message(self, provider: BaseProvider):
        """Test post-processing of formatted messages."""
        message = {"role": "user", "content": "test"}
        processed = provider._post_process_formatted_message(message, message)
        assert processed == message

    def test_prepare_request_payload(self, provider: BaseProvider, mock_messages: List[Dict[str, str]]):
        """Test request payload preparation."""
        payload = provider._prepare_request_payload(mock_messages, {})
        assert payload == {
            "messages": mock_messages,
            "model": "test_model"
        }

    def test_convert_messages(self, provider: BaseProvider):
        """Test message conversion."""
        messages = "Hello!"
        converted = provider._convert_messages(messages)
        assert converted == messages

    def test_convert_response_not_implemented(self, provider: BaseProvider):
        """Test that convert_response method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider._convert_response({})

    def test_initialize_credentials(self, provider: BaseProvider):
        """Test credentials initialization."""
        provider._initialize_credentials()  # Should not raise any exception 