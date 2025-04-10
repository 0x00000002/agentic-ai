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

    @pytest.mark.asyncio
    async def test_request_not_implemented(self, provider: BaseProvider, mock_messages: List[Dict[str, str]], mock_logger):
        """Test that async request method raises AIProviderError if _make_api_request not implemented."""
        # Patch the internal _make_api_request to simulate non-implementation
        with patch.object(provider, '_make_api_request', side_effect=NotImplementedError("Subclasses must implement _make_api_request")):
            # Expect AIProviderError because request catches and wraps internal errors
            # Update match pattern to include provider prefix
            expected_error_msg = "PROVIDER_BASEPROVIDER: Provider request failed: Subclasses must implement _make_api_request"
            with pytest.raises(AIProviderError, match=expected_error_msg):
                await provider.request(mock_messages)
        # Verify error was logged
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self, provider: BaseProvider, mock_messages: List[Dict[str, str]], mock_logger):
        """Test that async stream method raises NotImplementedError by default."""
        # stream method directly raises NotImplementedError if not overridden
        # Update the match string to the actual error message
        with pytest.raises(NotImplementedError, match="BaseProvider does not support streaming."):
            # Await the coroutine directly instead of using async for
            await provider.stream(mock_messages)
        mock_logger.error.assert_not_called()

    def test_map_role(self, provider: BaseProvider):
        """Test role mapping."""
        assert provider.message_formatter.map_role("system") == "system"
        assert provider.message_formatter.map_role("user") == "user"
        assert provider.message_formatter.map_role("assistant") == "assistant"
        assert provider.message_formatter.map_role("unknown") == "user"

    def test_format_messages(self, provider: BaseProvider, mock_messages: List[Dict[str, str]]):
        """Test message formatting."""
        formatted = provider._format_messages(mock_messages)
        assert formatted == mock_messages

    def test_prepare_request_payload(self, provider: BaseProvider, mock_messages: List[Dict[str, str]]):
        """Test request payload preparation."""
        payload = provider._prepare_request_payload(mock_messages, {})
        assert payload == {
            "messages": mock_messages,
            "model": "test_model"
        }

    def test_convert_response_not_implemented(self, provider: BaseProvider):
        """Test that convert_response method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            provider._convert_response({})

    def test_initialize_credentials(self, provider: BaseProvider):
        """Test credentials initialization."""
        provider._initialize_credentials()  # Should not raise any exception 