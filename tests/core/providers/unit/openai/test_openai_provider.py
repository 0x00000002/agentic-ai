"""
Unit tests for OpenAIProvider with helper classes.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
import json

from src.core.providers.openai_provider import OpenAIProvider
from src.utils.logger import LoggerInterface
from src.core.models import ProviderResponse


class TestOpenAIProvider:
    """Test suite for the OpenAI provider with helper classes."""
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture for a mock logger."""
        return MagicMock(spec=LoggerInterface)
    
    @pytest.fixture
    def mock_openai_client(self):
        """Fixture for a mock OpenAI client."""
        mock_client = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create = MagicMock()
        mock_client.chat.completions = mock_completions
        return mock_client
    
    @pytest.fixture
    def provider_config(self):
        """Fixture for provider configuration."""
        return {
            "api_key": "test-api-key",
            "base_url": "https://api.openai.test",
            "organization": "test-org"
        }
    
    @pytest.fixture
    def model_config(self):
        """Fixture for model configuration."""
        return {
            "model_id": "gpt-4",
            "provider": "openai",
            "parameters": {
                "temperature": 0.8,
                "max_tokens": 1000
            }
        }
    
    @pytest.fixture
    def openai_provider(self, mock_logger, mock_openai_client, provider_config, model_config):
        """Fixture for an OpenAI provider."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            provider = OpenAIProvider(
                model_id="gpt-4",
                provider_config=provider_config,
                model_config=model_config,
                logger=mock_logger
            )
            provider.client = mock_openai_client
            return provider
    
    def test_init(self, openai_provider, mock_logger, provider_config, model_config):
        """Test initialization with helper classes."""
        # Check that helper components are created
        assert hasattr(openai_provider, "message_formatter")
        assert hasattr(openai_provider, "parameter_manager")
        assert hasattr(openai_provider, "credential_manager")
        assert hasattr(openai_provider, "tool_manager")
        
        # Check that parameter manager uses runtime parameters from model config
        # Since the mock model_config doesn't have 'runtime_parameters', it defaults to {}
        expected_model_params = model_config.get("runtime_parameters", {})
        assert openai_provider.parameter_manager.get_model_parameters() == expected_model_params
        
        # Check that credential manager has the provider config
        assert openai_provider.credential_manager.provider_config == provider_config
    
    def test_make_api_request(self, openai_provider, mock_openai_client):
        """Test making API requests with prepared payload."""
        # Set up mock response
        mock_response = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create test payload
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "unsupported_key": "value"  # This should be removed
        }
        
        # Call the method
        result = openai_provider._make_api_request(payload)
        
        # Check the result
        assert result == mock_response
        
        # Check that unsupported_key was removed
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert "model" in call_args
        assert "messages" in call_args
        assert "temperature" in call_args
        assert "max_tokens" in call_args
        assert "unsupported_key" not in call_args
    
    def test_convert_response(self, openai_provider):
        """Test converting OpenAI response to ProviderResponse."""
        # Create mock OpenAI response
        choice = MagicMock()
        choice.message.content = "Hello, world!"
        choice.message.tool_calls = []
        choice.finish_reason = "stop"
        
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 20
        usage.total_tokens = 30
        
        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        
        # Call the method
        result = openai_provider._convert_response(response)
        
        # Check the result
        assert isinstance(result, ProviderResponse)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert result.stop_reason == "stop"
        assert result.model == "gpt-4"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
    
    def test_request(self, openai_provider, mock_openai_client):
        """Test the request method with string input."""
        # Mock the API response
        choice = MagicMock()
        choice.message.content = "Hello, I'm an AI assistant."
        choice.message.tool_calls = []
        choice.finish_reason = "stop"
        
        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = None
        
        mock_openai_client.chat.completions.create.return_value = response
        
        # Call the request method with a string
        result = openai_provider.request("Hello, AI!")
        
        # Check the result
        assert isinstance(result, ProviderResponse)
        assert result.content == "Hello, I'm an AI assistant."
        
        # Check that the API was called with correct arguments
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Hello, AI!"
    
    def test_request_with_tool_calls(self, openai_provider, mock_openai_client):
        """Test the request method with tool calls in the response."""
        # Mock the API response with tool calls
        choice = MagicMock()
        choice.message.content = None  # Content can be None when there are tool calls
        
        # Create mock tool calls
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "New York", "unit": "celsius"}'
        
        # Add tool calls to the message
        choice.message.tool_calls = [tool_call]
        choice.finish_reason = "tool_calls"
        
        # Create the response
        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = MagicMock(prompt_tokens=20, completion_tokens=30, total_tokens=50)
        
        # Set the mock response
        mock_openai_client.chat.completions.create.return_value = response
        
        # Call the request method
        result = openai_provider.request("What's the weather in New York?")
        
        # Check the result
        assert isinstance(result, ProviderResponse)
        assert result.content == ""  # Empty content with tool calls
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert isinstance(result.tool_calls[0].arguments, dict)
        assert result.tool_calls[0].arguments["location"] == "New York"
        assert result.tool_calls[0].arguments["unit"] == "celsius"
        assert result.stop_reason == "tool_calls" 