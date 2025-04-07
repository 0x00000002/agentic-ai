"""
Shared fixtures for provider tests.
"""
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock
from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.models import ProviderResponse
from src.tools.models import ToolCall, ToolResult
from src.utils.logger import LoggerInterface

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "api_key": "test_key",
        "model_id": "test_model",
        "temperature": 0.7,
        "max_tokens": 1000
    }

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock(spec=LoggerInterface)

@pytest.fixture
def mock_provider_response():
    """Create a mock provider response."""
    return ProviderResponse(
        content="Test response",
        model="test_model",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )

@pytest.fixture
def mock_tool_call():
    """Create a mock tool call."""
    return ToolCall(
        id="test_tool_call",
        name="test_tool",
        arguments={"arg1": "value1"}
    )

@pytest.fixture
def mock_tool_result():
    """Create a mock tool result."""
    return ToolResult(
        tool_call_id="test_tool_call",
        content="Tool execution result"
    )

@pytest.fixture
def mock_messages():
    """Create mock conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]

@pytest.fixture
def mock_response():
    """Create a mock API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response",
                    "role": "assistant"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_tool_response() -> Dict[str, Any]:
    """Create a mock tool response for testing."""
    return {
        "choices": [{
            "message": {
                "content": None,
                "role": "assistant",
                "tool_calls": [{
                    "id": "test_tool",
                    "name": "test_function",
                    "arguments": {"arg1": "value1"}
                }]
            }
        }]
    }

@pytest.fixture
def mock_stream_response() -> List[Dict[str, Any]]:
    """Create a mock streaming response for testing."""
    return [
        {
            "choices": [{
                "delta": {
                    "content": "Test"
                }
            }]
        },
        {
            "choices": [{
                "delta": {
                    "content": " response"
                }
            }]
        }
    ]

@pytest.fixture
def mock_error_response() -> Dict[str, Any]:
    """Create a mock error response for testing."""
    return {
        "error": {
            "message": "Test error",
            "type": "invalid_request_error",
            "code": "invalid_parameter"
        }
    }

@pytest.fixture
def mock_rate_limit_response() -> Dict[str, Any]:
    """Create a mock rate limit response for testing."""
    return {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    }

@pytest.fixture
def mock_auth_error_response() -> Dict[str, Any]:
    """Create a mock authentication error response for testing."""
    return {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    }

@pytest.fixture
def mock_timeout_error_response() -> Dict[str, Any]:
    """Create a mock timeout error response for testing."""
    return {
        "error": {
            "message": "Request timed out",
            "type": "timeout_error",
            "code": "request_timeout"
        }
    }

@pytest.fixture
def mock_network_error_response() -> Dict[str, Any]:
    """Create a mock network error response for testing."""
    return {
        "error": {
            "message": "Network error",
            "type": "network_error",
            "code": "connection_error"
        }
    }

@pytest.fixture
def mock_provider_response_with_tools():
    """Create a mock provider response with tool calls."""
    return ProviderResponse(
        content="I'll use a tool to help with that.",
        tool_calls=[
            ToolCall(
                id="test_tool_call",
                name="test_tool",
                arguments={"arg1": "value1"}
            )
        ],
        model="test_model",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )

@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Create a mock async client for testing."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    return client

@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock session for testing."""
    session = AsyncMock()
    session.post = AsyncMock()
    session.get = AsyncMock()
    session.close = AsyncMock()
    return session 