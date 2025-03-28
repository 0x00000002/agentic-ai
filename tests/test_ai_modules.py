import pytest
from unittest.mock import Mock, patch
from src.ai.modules.Anthropic import ClaudeAI, AI_API_Key_Error as AnthropicAPIError
from src.ai.modules.OpenAI import ChatGPT, AI_API_Key_Error as OpenAIAPIError
from src.ai.modules.Google import Gemini, AI_API_Key_Error as GoogleAPIError
from src.ai.ai_config import Model
from src.Logger import Logger
from src.ai.tools.models import ToolCallRequest
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)

@pytest.fixture
def mock_api_keys(monkeypatch):
    # Get API keys from environment variables
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # Set up API keys in test environment
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("GOOGLE_API_KEY", google_key)
    
    yield
    
    # Clean up
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

class TestAIModules:
    @patch('anthropic.Anthropic')
    def test_claude_initialization(self, mock_anthropic_class, mock_api_keys, mock_logger):
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        claude = ClaudeAI(Model.CLAUDE_SONNET_3_5, "test prompt", mock_logger)
        assert claude.client == mock_client
        assert claude.model == Model.CLAUDE_SONNET_3_5
        assert claude.system_prompt == "test prompt"

    @patch('src.ai.modules.OpenAI.OpenAI')
    def test_chatgpt_initialization(self, mock_openai_class, mock_api_keys, mock_logger):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        chatgpt = ChatGPT(Model.CHATGPT_4O_MINI, "test prompt", mock_logger)
        mock_openai_class.assert_called_once_with(api_key=Model.CHATGPT_4O_MINI.api_key)
        assert chatgpt.model == Model.CHATGPT_4O_MINI
        assert chatgpt.client == mock_client
        assert chatgpt.system_prompt == "test prompt"

    @patch('src.ai.modules.OpenAI.OpenAI')
    def test_chatgpt_request_with_tool_calls(self, mock_openai_class, mock_api_keys, mock_logger):
        # Create mock response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_ticket_price"
        mock_tool_call.function.arguments = '{"destination_city": "New York"}'
        
        mock_message = Mock()
        mock_message.content = "Let me check the ticket price for you."
        mock_message.tool_calls = [mock_tool_call]
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.choices[0].finish_reason = "tool_calls"
        
        # Create mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        chatgpt = ChatGPT(Model.CHATGPT_4O_MINI, "test prompt", mock_logger)
        response = chatgpt.request([{"role": "user", "content": "How much is a ticket to New York?"}])
        
        assert isinstance(response, ToolCallRequest)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_ticket_price"
        assert response.content == "Let me check the ticket price for you."
        assert response.finish_reason == "tool_calls"

    @patch('google.genai.Client')
    def test_gemini_request(self, mock_genai_class, mock_api_keys, mock_logger):
        mock_client = Mock()
        mock_response = Mock()
        mock_candidate = Mock()
        mock_candidate.content = Mock(parts=[Mock(text="Test response")])
        mock_candidate.tool_calls = []
        mock_candidate.finish_reason = None
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_class.return_value = mock_client
        
        gemini = Gemini(Model.GEMINI_1_5_PRO, "test prompt", mock_logger)
        response = gemini.request([{"role": "user", "parts": [{"text": "Test"}]}])
        assert response.content == "Test response"
        assert response.finish_reason is None

    def test_ollama_initialization(self):
        try:
            from ollama import Client
            assert True
        except ImportError:
            pytest.skip("Ollama not installed")

    def test_gemini_initialization(self):
        try:
            from google import genai
            assert True
        except ImportError:
            pytest.skip("Google AI not installed")

    @patch('anthropic.Anthropic')
    def test_claude_request(self, mock_anthropic_class, mock_api_keys, mock_logger):
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Test response"
        mock_content.tool_calls = []
        mock_response.content = [mock_content]
        mock_response.stop_reason = None
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        claude = ClaudeAI(Model.CLAUDE_SONNET_3_5, "test prompt", mock_logger)
        response = claude.request([{"role": "user", "content": "Test"}])
        assert isinstance(response, ToolCallRequest)
        assert response.content == "Test response"
        assert response.finish_reason is None

    @patch('src.ai.modules.OpenAI.OpenAI')
    def test_chatgpt_request(self, mock_openai_class, mock_api_keys, mock_logger):
        # Create mock response
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = []
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.choices[0].finish_reason = "stop"
        
        # Create mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create ChatGPT instance and make request
        chatgpt = ChatGPT(Model.CHATGPT_4O_MINI, "test prompt", mock_logger)
        
        # Verify mock was used with correct API key
        mock_openai_class.assert_called_once_with(api_key=Model.CHATGPT_4O_MINI.api_key)
        assert chatgpt.client == mock_client
        
        # Make request and verify response
        response = chatgpt.request([{"role": "user", "content": "Test"}])
        assert isinstance(response, ToolCallRequest)
        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert len(response.tool_calls) == 0

    def test_error_handling(self, monkeypatch):
        # Remove API keys to test error handling
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Test each provider's API key validation
        with pytest.raises(AnthropicAPIError, match="No valid Anthropic API key found"):
            ClaudeAI(Model.CLAUDE_SONNET_3_5)
        with pytest.raises(OpenAIAPIError, match="No valid OpenAI API key found"):
            ChatGPT(Model.CHATGPT_4O_MINI)
        with pytest.raises(GoogleAPIError, match="No valid Google API key found"):
            # Mock dotenv.load_dotenv to do nothing
            with patch('src.ai.modules.Google.load_dotenv'):
                Gemini(Model.GEMINI_1_5_PRO) 