"""
Shared fixtures for AI provider tests.
"""
import pytest
from unittest.mock import MagicMock, patch
import json

from src.core.provider_factory import ProviderFactory
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.utils.logger import LoggerFactory
from src.config.unified_config import UnifiedConfig
from src.prompts.prompt_template import PromptTemplate
from src.tools.models import ToolDefinition, ToolCall, ToolResult

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()

@pytest.fixture
def unified_config():
    """Create a unified config for testing."""
    config = UnifiedConfig()
    config.providers = {
        "openai": {
            "api_key": "test-openai-key",
            "timeout": 30,
            "models": {
                "gpt-4": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "anthropic": {
            "api_key": "test-anthropic-key",
            "timeout": 30,
            "models": {
                "claude-3-opus": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "gemini": {
            "api_key": "test-gemini-key",
            "timeout": 30,
            "models": {
                "gemini-pro": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": 30,
            "models": {
                "llama3": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        }
    }
    return config

@pytest.fixture
def openai_provider(unified_config, mock_logger):
    """Create an OpenAI provider for testing."""
    provider = OpenAIProvider(
        model_id="gpt-4",
        provider_config=unified_config.providers["openai"],
        model_config=unified_config.providers["openai"]["models"]["gpt-4"],
        logger=mock_logger
    )
    return provider

@pytest.fixture
def anthropic_provider(unified_config, mock_logger):
    """Create an Anthropic provider for testing."""
    provider = AnthropicProvider(
        model_id="claude-3-opus",
        provider_config=unified_config.providers["anthropic"],
        model_config=unified_config.providers["anthropic"]["models"]["claude-3-opus"],
        logger=mock_logger
    )
    return provider

@pytest.fixture
def gemini_provider(unified_config, mock_logger):
    """Create a Gemini provider for testing."""
    provider = GeminiProvider(
        model_id="gemini-pro",
        provider_config=unified_config.providers["gemini"],
        model_config=unified_config.providers["gemini"]["models"]["gemini-pro"],
        logger=mock_logger
    )
    return provider

@pytest.fixture
def ollama_provider(unified_config, mock_logger):
    """Create an Ollama provider for testing."""
    provider = OllamaProvider(
        model_id="llama3",
        provider_config=unified_config.providers["ollama"],
        model_config=unified_config.providers["ollama"]["models"]["llama3"],
        logger=mock_logger
    )
    return provider

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response for testing."""
    return {
        "id": "test-response-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response for testing."""
    return {
        "id": "test-response-id",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a test response from Anthropic."
            }
        ],
        "model": "claude-3-opus",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini response for testing."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a test response from Gemini."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        },
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30
        }
    }

@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama response for testing."""
    return {
        "model": "llama3",
        "created_at": "2023-01-01T00:00:00.000Z",
        "response": "This is a test response from Ollama.",
        "done": True,
        "context": [],
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_duration": 100000000,
        "eval_duration": 400000000,
        "eval_count": 20
    }

@pytest.fixture
def mock_openai_tool_response():
    """Create a mock OpenAI response with tool calls for testing."""
    return {
        "id": "test-response-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "test_tool",
                        "arguments": json.dumps({"param1": "value1"})
                    }
                },
                "finish_reason": "function_call"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool calls for testing."""
    return {
        "id": "test-response-id",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "test_tool",
                "input": {"param1": "value1"}
            }
        ],
        "model": "claude-3-opus",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_gemini_tool_response():
    """Create a mock Gemini response with tool calls for testing."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "test_tool",
                                "args": {"param1": "value1"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "FUNCTION_CALL",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "promptFeedback": {
            "safetyRatings": []
        },
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30
        }
    }

@pytest.fixture
def test_tool_definition():
    """Create a test tool definition for testing."""
    return ToolDefinition(
        name="test_tool",
        description="A test tool for testing",
        parameters={
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "A test parameter"
                }
            },
            "required": ["param1"]
        }
    )

@pytest.fixture
def test_tool_call():
    """Create a test tool call for testing."""
    return ToolCall(
        id="test-call-id",
        name="test_tool",
        arguments={"param1": "value1"}
    )

@pytest.fixture
def test_tool_result():
    """Create a test tool result for testing."""
    return ToolResult(
        tool_call_id="test-call-id",
        name="test_tool",
        content="Tool execution result"
    )

@pytest.fixture
def test_prompt_template():
    """Create a test prompt template for testing."""
    return PromptTemplate(
        name="test_template",
        description="A test prompt template",
        template="This is a test prompt for {user_name}. The task is to {task_description}.",
        variables=["user_name", "task_description"],
        examples=[
            {
                "user_name": "Alice",
                "task_description": "write a poem",
                "output": "Here's a poem for Alice..."
            },
            {
                "user_name": "Bob",
                "task_description": "explain quantum computing",
                "output": "Quantum computing explanation for Bob..."
            }
        ]
    )

@pytest.fixture
def mock_provider_factory(unified_config):
    """Create a mock provider factory for testing."""
    factory = ProviderFactory(unified_config)
    return factory 