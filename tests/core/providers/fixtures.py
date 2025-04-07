"""
Shared fixtures for provider tests.

This module contains common fixtures and helper functions used across
provider test files to reduce code duplication and ensure consistency.
"""
from unittest.mock import MagicMock, patch
import json
from typing import Dict, Any, Optional

from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.prompt_templates.base_template import BaseTemplate
from src.core.prompt_templates.template_registry import TemplateRegistry
from src.core.prompt_templates.template_validator import TemplateValidator
from src.core.prompt_templates.template_cache import TemplateCache
from src.core.prompt_templates.template_metrics import TemplateMetrics
from src.utils.logger import LoggerFactory


def create_mock_unified_config() -> MagicMock:
    """
    Create a mock unified configuration with provider settings.
    
    Returns:
        MagicMock: A mock unified configuration object
    """
    unified_config = MagicMock()
    unified_config.providers = {
        "openai": {
            "api_key": "test-openai-key",
            "timeout": 30,
            "models": {
                "gpt-4": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                },
                "gpt-4o": {
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
    return unified_config


def create_mock_logger() -> MagicMock:
    """
    Create a mock logger for testing.
    
    Returns:
        MagicMock: A mock logger object
    """
    return MagicMock()


def create_provider_instance(
    provider_type: str,
    model_id: str,
    unified_config: MagicMock,
    logger: Optional[MagicMock] = None
) -> BaseProvider:
    """
    Create a provider instance based on the provider type.
    
    Args:
        provider_type: The type of provider to create
        model_id: The model ID to use
        unified_config: The unified configuration
        logger: Optional logger to use
        
    Returns:
        BaseProvider: A provider instance
        
    Raises:
        ValueError: If the provider type is not supported
    """
    if logger is None:
        logger = create_mock_logger()
        
    provider_config = unified_config.providers.get(provider_type, {})
    model_config = provider_config.get("models", {}).get(model_id, {})
    
    if provider_type == "openai":
        return OpenAIProvider(
            model_id=model_id,
            provider_config=provider_config,
            model_config=model_config,
            logger=logger
        )
    elif provider_type == "anthropic":
        return AnthropicProvider(
            model_id=model_id,
            provider_config=provider_config,
            model_config=model_config,
            logger=logger
        )
    elif provider_type == "gemini":
        return GeminiProvider(
            model_id=model_id,
            provider_config=provider_config,
            model_config=model_config,
            logger=logger
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model_id=model_id,
            provider_config=provider_config,
            model_config=model_config,
            logger=logger
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def create_mock_response(content: str, role: str = "assistant") -> Dict[str, Any]:
    """
    Create a mock response for testing.
    
    Args:
        content: The content of the response
        role: The role of the response
        
    Returns:
        Dict[str, Any]: A mock response dictionary
    """
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "role": role
                },
                "finish_reason": "stop"
            }
        ]
    }


def create_mock_tool_response(tool_name: str, result: str) -> Dict[str, Any]:
    """
    Create a mock tool response for testing.
    
    Args:
        tool_name: The name of the tool
        result: The result of the tool call
        
    Returns:
        Dict[str, Any]: A mock tool response dictionary
    """
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"query": "test query"})
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    }


def create_mock_tool_result(tool_name: str, result: str) -> Dict[str, Any]:
    """
    Create a mock tool result for testing.
    
    Args:
        tool_name: The name of the tool
        result: The result of the tool call
        
    Returns:
        Dict[str, Any]: A mock tool result dictionary
    """
    return {
        "tool_call_id": "call_123",
        "role": "tool",
        "name": tool_name,
        "content": result
    }


def create_test_template(
    name: str,
    content: str,
    provider: str = "openai",
    version: str = "1.0",
    parent: Optional[str] = None,
    language: Optional[str] = None
) -> BaseTemplate:
    """
    Create a test template for testing.
    
    Args:
        name: The name of the template
        content: The content of the template
        provider: The provider for the template
        version: The version of the template
        parent: The parent template name
        language: The language of the template
        
    Returns:
        BaseTemplate: A test template instance
    """
    class TestTemplate(BaseTemplate):
        def __init__(self):
            super().__init__(
                name=name,
                content=content,
                provider=provider,
                version=version,
                parent=parent,
                language=language
            )

        def render(self, **kwargs):
            return self.content.format(**kwargs)
            
    return TestTemplate()


def setup_template_registry() -> TemplateRegistry:
    """
    Set up a template registry for testing.
    
    Returns:
        TemplateRegistry: A template registry instance
    """
    return TemplateRegistry()


def setup_template_validator() -> TemplateValidator:
    """
    Set up a template validator for testing.
    
    Returns:
        TemplateValidator: A template validator instance
    """
    return TemplateValidator()


def setup_template_cache() -> TemplateCache:
    """
    Set up a template cache for testing.
    
    Returns:
        TemplateCache: A template cache instance
    """
    return TemplateCache()


def setup_template_metrics() -> TemplateMetrics:
    """
    Set up template metrics for testing.
    
    Returns:
        TemplateMetrics: A template metrics instance
    """
    return TemplateMetrics() 