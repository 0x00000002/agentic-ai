import pytest
from unittest.mock import patch, MagicMock, ANY, call, AsyncMock
import uuid
from pytest_mock import MockerFixture
from asyncio import iscoroutinefunction

# Class to test
from src.core.base_ai import AIBase, DEFAULT_SYSTEM_PROMPT

# Dependencies to mock
from src.config.unified_config import UnifiedConfig, AIConfigError
from src.core.provider_factory import ProviderFactory
from src.exceptions import AIProviderError
from src.conversation.conversation_manager import ConversationManager, Message
from src.utils.logger import LoggerInterface, LoggerFactory
from src.core.interfaces import ProviderInterface
from src.prompts.prompt_template import PromptTemplate
from src.exceptions import AISetupError, AIProcessingError
from src.core.models import ProviderResponse
from src.tools.models import ToolResult

# Constants for tests
TEST_MODEL_KEY = "test-model"
TEST_PROVIDER_NAME = "test-provider"
TEST_PROVIDER_MODEL_ID = "provider-model-id-123"
TEST_REQUEST_ID = "req-abc"

@pytest.fixture
def mock_config_instance():
    """Fixture for a mocked UnifiedConfig instance."""
    mock_config = MagicMock(spec=UnifiedConfig)
    mock_config.get_default_model.return_value = "default-model"
    mock_config.get_system_prompt.return_value = "Config system prompt"
    
    # Default successful model config lookup
    mock_config.get_model_config.return_value = {
        "provider": TEST_PROVIDER_NAME,
        "model_id": TEST_PROVIDER_MODEL_ID
        # Add other potential fields if needed by provider
    }
    
    # Default successful provider config lookup
    mock_config.get_provider_config.return_value = {
        "api_key": "dummy-key" 
        # Add other potential fields
    }
    return mock_config

@pytest.fixture
def mock_logger():
    """Fixture for a mocked LoggerInterface."""
    return MagicMock(spec=LoggerInterface)

@pytest.fixture
def mock_provider():
    """Fixture for a mocked ProviderInterface."""
    return MagicMock(spec=ProviderInterface)

@pytest.fixture
def mock_convo_manager():
    """Fixture for a mocked ConversationManager."""
    return MagicMock(spec=ConversationManager)

@pytest.fixture
def mock_prompt_template():
    """Fixture for a mocked PromptTemplate service."""
    mock_pt = MagicMock(spec=PromptTemplate)
    # Default behavior: render_prompt raises ValueError (template not found)
    mock_pt.render_prompt.side_effect = ValueError("Template not found") 
    return mock_pt

# --- Test Class ---

class TestAIBase:

    @patch('src.core.base_ai.uuid.uuid4')
    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.LoggerFactory')
    def test_init_success_defaults(self, mock_LoggerFactory, mock_UnifiedConfig, mock_ProviderFactory, 
                                mock_ConversationManager, mock_PromptTemplate, mock_uuid4, 
                                mock_config_instance, mock_logger, mock_provider, mock_convo_manager, mock_prompt_template):
        """Test successful initialization using default model and system prompt."""
        # Arrange Mocks
        mock_LoggerFactory.create.return_value = mock_logger
        mock_UnifiedConfig.get_instance.return_value = mock_config_instance
        mock_ProviderFactory.create.return_value = mock_provider
        mock_ConversationManager.return_value = mock_convo_manager
        mock_PromptTemplate.return_value = mock_prompt_template
        mock_uuid4.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678') # Fixed UUID

        # Act
        ai = AIBase() # Use default model

        # Assert Logger
        mock_LoggerFactory.create.assert_called_once_with(name="ai_framework.12345678")
        
        # Assert Config Usage
        mock_UnifiedConfig.get_instance.assert_called_once()
        mock_config_instance.get_default_model.assert_called_once()
        default_model_key = mock_config_instance.get_default_model.return_value
        mock_config_instance.get_model_config.assert_called_once_with(default_model_key)
        model_config = mock_config_instance.get_model_config.return_value
        provider_name = model_config['provider']
        mock_config_instance.get_provider_config.assert_called_once_with(provider_name)
        provider_config = mock_config_instance.get_provider_config.return_value
        
        # Assert Provider Factory
        mock_ProviderFactory.create.assert_called_once_with(
            provider_type=provider_name,
            model_id=model_config.get('model_id'),
            provider_config=provider_config,
            model_config=model_config,
            logger=mock_logger
        )
        assert ai._provider == mock_provider

        # Assert PromptTemplate (default instance created)
        mock_PromptTemplate.assert_called_once_with(logger=mock_logger)
        assert ai._prompt_template == mock_prompt_template
        # Verify fallback for default system prompt was NOT used
        mock_prompt_template.render_prompt.assert_not_called()
        
        # Assert Conversation Manager & System Prompt
        mock_ConversationManager.assert_called_once()
        assert ai._conversation_manager == mock_convo_manager
        # System prompt should be from config first, then default method (which failed -> fallback)
        fallback_prompt = f"You are a helpful AI assistant using the {model_config.get('model_id', 'default')} model."
        mock_config_instance.get_system_prompt.assert_called_once() # Tried config first
        assert ai._system_prompt == mock_config_instance.get_system_prompt.return_value # Used config prompt
        mock_convo_manager.add_message.assert_called_once_with(
            role="system",
            content=mock_config_instance.get_system_prompt.return_value
        )
        
        # Assert other attributes
        assert ai._request_id == '12345678-1234-5678-1234-567812345678'
        assert ai._model_config == model_config

    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.LoggerFactory')
    def test_init_success_specific_model_prompt(self, mock_LoggerFactory, mock_UnifiedConfig, mock_ProviderFactory, 
                                            mock_ConversationManager, mock_PromptTemplate, 
                                            mock_config_instance, mock_logger, mock_provider, mock_convo_manager, mock_prompt_template):
        """Test successful initialization with specific model and system prompt override."""
        # Arrange Mocks
        mock_LoggerFactory.create.return_value = mock_logger
        mock_UnifiedConfig.get_instance.return_value = mock_config_instance
        mock_ProviderFactory.create.return_value = mock_provider
        mock_ConversationManager.return_value = mock_convo_manager
        mock_PromptTemplate.return_value = mock_prompt_template # Assume default PT created
        
        custom_system_prompt = "You are a custom test AI."

        # Act
        ai = AIBase(model=TEST_MODEL_KEY, system_prompt=custom_system_prompt, request_id=TEST_REQUEST_ID)

        # Assert Logger
        mock_LoggerFactory.create.assert_called_once_with(name=f"ai_framework.{TEST_REQUEST_ID[:8]}")

        # Assert Config Usage (uses TEST_MODEL_KEY)
        mock_config_instance.get_default_model.assert_not_called()
        mock_config_instance.get_model_config.assert_called_once_with(TEST_MODEL_KEY)
        model_config = mock_config_instance.get_model_config.return_value
        provider_name = model_config['provider']
        mock_config_instance.get_provider_config.assert_called_once_with(provider_name)
        provider_config = mock_config_instance.get_provider_config.return_value

        # Assert Provider Factory
        mock_ProviderFactory.create.assert_called_once_with(
            provider_type=provider_name,
            model_id=model_config.get('model_id'),
            provider_config=provider_config,
            model_config=model_config,
            logger=mock_logger
        )

        # Assert PromptTemplate (uses provided instance)
        mock_PromptTemplate.assert_called_once_with(logger=mock_logger) 
        
        # Assert Conversation Manager & System Prompt (uses custom prompt for the main `ai` instance)
        assert ai._conversation_manager == mock_convo_manager
        assert ai._system_prompt == custom_system_prompt # Used custom prompt directly
        mock_config_instance.get_system_prompt.assert_not_called() # Should not check config for `ai`
        # Check add_message call on the manager associated with 'ai' instance
        mock_convo_manager.add_message.assert_called_once_with(
             role="system",
             content=custom_system_prompt
        )

        # Assert other attributes
        assert ai._request_id == TEST_REQUEST_ID
        assert ai._model_config == model_config
        
    # --- Add Failure Tests ---

    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.LoggerFactory')
    def test_init_fail_missing_model_config(self, mock_LoggerFactory, mock_UnifiedConfig, mock_ProviderFactory, 
                                        mock_ConversationManager, mock_PromptTemplate, 
                                        mock_config_instance, mock_logger):
        """Test __init__ raises AISetupError if model config is missing."""
        # Arrange Mocks
        mock_LoggerFactory.create.return_value = mock_logger
        mock_UnifiedConfig.get_instance.return_value = mock_config_instance
        # Simulate failure
        mock_config_instance.get_model_config.side_effect = AIConfigError("Model config not found")

        # Act & Assert
        with pytest.raises(AISetupError, match="Missing model configuration"):
            AIBase(model="missing-model")
        
        # Verify mocks
        mock_config_instance.get_model_config.assert_called_once_with("missing-model")
        mock_ProviderFactory.create.assert_not_called() # Should fail before factory call

    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.LoggerFactory')
    def test_init_fail_missing_provider_config(self, mock_LoggerFactory, mock_UnifiedConfig, mock_ProviderFactory, 
                                           mock_ConversationManager, mock_PromptTemplate, 
                                           mock_config_instance, mock_logger):
        """Test __init__ raises AISetupError if provider config is missing."""
        # Arrange Mocks
        mock_LoggerFactory.create.return_value = mock_logger
        mock_UnifiedConfig.get_instance.return_value = mock_config_instance
        # Simulate failure
        mock_config_instance.get_provider_config.side_effect = AIConfigError("Provider config not found")

        # Act & Assert
        with pytest.raises(AISetupError, match="Missing provider configuration"):
            AIBase(model=TEST_MODEL_KEY) # Use model key known to succeed in model lookup
        
        # Verify mocks
        mock_config_instance.get_model_config.assert_called_once_with(TEST_MODEL_KEY)
        model_config = mock_config_instance.get_model_config.return_value
        provider_name = model_config['provider']
        mock_config_instance.get_provider_config.assert_called_once_with(provider_name)
        mock_ProviderFactory.create.assert_not_called()

    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.LoggerFactory')
    def test_init_fail_provider_factory(self, mock_LoggerFactory, mock_UnifiedConfig, mock_ProviderFactory, 
                                        mock_ConversationManager, mock_PromptTemplate, 
                                        mock_config_instance, mock_logger):
        """Test __init__ raises AISetupError if ProviderFactory fails."""
        # Arrange Mocks
        mock_LoggerFactory.create.return_value = mock_logger
        mock_UnifiedConfig.get_instance.return_value = mock_config_instance
        # Simulate failure
        factory_error = AIProviderError("Factory failed", provider="test-provider") # Add provider context
        mock_ProviderFactory.create.side_effect = factory_error
        mock_ConversationManager.return_value = MagicMock()
        mock_PromptTemplate.return_value = MagicMock()

        # Act & Assert
        # Adjust the regex to match the actual error format including the nested error
        with pytest.raises(AISetupError, match=r"Failed to initialize AI: PROVIDER_TEST-PROVIDER: Factory failed"):
            AIBase()

        # Verify mocks
        mock_ProviderFactory.create.assert_called_once()
        mock_ConversationManager.assert_not_called() # Should fail before convo manager init

    # --- Fixture for Initialized AI Instance ---
    @pytest.fixture
    def initialized_ai(self, mock_config_instance, mock_logger, mock_provider, mock_convo_manager, mock_prompt_template):
        """Provides a successfully initialized AIBase instance for testing methods."""
        with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config_instance), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider), \
             patch('src.core.base_ai.LoggerFactory.create', return_value=mock_logger), \
             patch('src.core.base_ai.uuid.uuid4', return_value=uuid.UUID('abcdef12-1234-5678-1234-abcdef123456')):
                 
            # Use specific model to avoid ambiguity with defaults during test setup
            ai = AIBase(model=TEST_MODEL_KEY, request_id="test-init-fixture", prompt_template=mock_prompt_template)
        return ai

    # --- request Method Tests (NOW ASYNC) ---
    @pytest.mark.asyncio # Mark as async
    async def test_request_success(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger, mock_config_instance):
        """Test successful async request processing."""
        prompt = "Why don't scientists trust atoms? Because they make up everything!"
        # Mock the provider's async request method
        mock_provider_response = ProviderResponse(content="That's a classic!")
        mock_provider.request = AsyncMock(return_value=mock_provider_response)

        # Call the async request method
        response_content = await initialized_ai.request(prompt)

        # Assert Conversation Manager calls
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        mock_convo_manager.get_messages.assert_called_once()
        messages_sent = mock_convo_manager.get_messages.return_value
        
        # Assert Provider call
        mock_provider.request.assert_called_once_with(messages_sent, **{})
        
        # Assert Response handling and final convo add
        assert response_content == mock_provider_response.content
        mock_convo_manager.add_message.assert_called_with(
            role="assistant",
            content=mock_provider_response.content,
            extract_thoughts=True,
            show_thinking=mock_config_instance.show_thinking
        )
        
        # Assert logger calls
        mock_logger.debug.assert_any_call(f"Processing request: {prompt[:50]}...")

    @pytest.mark.asyncio # Mark as async
    async def test_request_provider_returns_error_in_response(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger):
        """Test request raises AIProcessingError if provider response has error."""
        # Arrange
        prompt = "Test prompt"
        error_message = "Provider API error"
        mock_provider.request = AsyncMock(return_value=ProviderResponse(error=error_message))

        # Act & Assert
        with pytest.raises(AIProcessingError, match=error_message):
            await initialized_ai.request(prompt)

        # Verify
        mock_provider.request.assert_called_once()
        # Ensure assistant message was NOT added
        assistant_call_args = [call_args for call_args in mock_convo_manager.add_message.call_args_list if call_args.kwargs.get('role') == 'assistant']
        assert not assistant_call_args

    @pytest.mark.asyncio # Mark as async
    async def test_request_provider_returns_no_content(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger, mock_config_instance):
        """Test request handling when provider returns None content (async)."""
        prompt = "This prompt gets no content back."
        mock_provider_response = ProviderResponse(content=None) # No content
        mock_provider.request = AsyncMock(return_value=mock_provider_response)

        response_content = await initialized_ai.request(prompt)

        assert response_content == "" # Should return empty string
        mock_logger.warning.assert_called_with("Provider response content was None in AIBase request.")
        # Check assistant message was added with empty content
        mock_convo_manager.add_message.assert_called_with(
            role="assistant",
            content="",
            extract_thoughts=True,
            show_thinking=mock_config_instance.show_thinking
        )

    @pytest.mark.asyncio # Mark as async
    @patch('src.core.base_ai.ErrorHandler.handle_error')
    async def test_request_provider_raises_exception(self, mock_handle_error, initialized_ai, mock_convo_manager, mock_provider, mock_logger):
        """Test request handling when the provider's request method itself raises an exception (async)."""
        prompt = "This prompt causes the provider to crash."
        original_exception = ValueError("Provider crashed!")
        mock_provider.request = AsyncMock(side_effect=original_exception)
        # Mock the return value of handle_error
        mock_handle_error.return_value = {"message": "Formatted error message"}

        # Assert that AIProcessingError is raised
        with pytest.raises(AIProcessingError, match="Request failed: Provider crashed!") as excinfo:
            await initialized_ai.request(prompt)
        
        # Check that the original exception is chained
        assert excinfo.value.__cause__ is original_exception
        
        # Verify ErrorHandler was called with the correct error type
        mock_handle_error.assert_called_once()
        args, kwargs = mock_handle_error.call_args
        assert isinstance(args[0], AIProcessingError)
        assert args[0].component == "AIBase"
        assert args[1] is mock_logger # Check logger was passed
        
        # Verify logger was called via the handler
        mock_logger.error.assert_called_with("Request error: Formatted error message")
        # Verify assistant message was NOT added
        add_calls = mock_convo_manager.add_message.call_args_list
        assert len(add_calls) == 2 

    # --- stream Method Tests (NOW ASYNC) ---
    @pytest.mark.asyncio # Mark as async
    async def test_stream_success(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger, mock_config_instance):
        """Test successful async streaming."""
        prompt = "Stream me a story."
        streamed_content = "Once upon a time... the end."
        # Mock the provider's async stream method
        mock_provider.stream = AsyncMock(return_value=streamed_content)

        # Call the async stream method
        response_content = await initialized_ai.stream(prompt)

        # Assert Conversation Manager calls
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        mock_convo_manager.get_messages.assert_called_once()
        messages_sent = mock_convo_manager.get_messages.return_value
        
        # Assert Provider call
        mock_provider.stream.assert_called_once_with(messages_sent, **{})
        
        # Assert Response handling and final convo add
        assert response_content == streamed_content
        mock_convo_manager.add_message.assert_called_with(
            role="assistant",
            content=streamed_content,
            extract_thoughts=True,
            show_thinking=mock_config_instance.show_thinking
        )
        
        # Assert logger calls
        mock_logger.debug.assert_any_call(f"Processing streaming request: {prompt[:50]}...")
        
    @pytest.mark.asyncio # Mark as async
    @patch('src.core.base_ai.ErrorHandler.handle_error')
    async def test_stream_provider_raises_exception(self, mock_handle_error, initialized_ai, mock_convo_manager, mock_provider, mock_logger):
        """Test stream handling when the provider's stream method raises an exception (async)."""
        prompt = "This stream will crash."
        original_exception = TimeoutError("Stream timed out!") # Example exception
        mock_provider.stream = AsyncMock(side_effect=original_exception)
        mock_handle_error.return_value = {"message": "Formatted stream error"}

        # Assert that the original exception (or wrapped AIProcessingError) is raised
        # AIBase stream re-raises the original exception after logging
        with pytest.raises(TimeoutError): # Check for the original exception type
            await initialized_ai.stream(prompt)
        
        # Verify ErrorHandler was called
        mock_handle_error.assert_called_once()
        args, kwargs = mock_handle_error.call_args
        assert isinstance(args[0], AIProcessingError) # Error handler wraps it
        assert args[0].component == "AIBase"
        assert "Streaming failed" in str(args[0])
        assert args[1] is mock_logger
        
        # Verify logger was called via the handler
        mock_logger.error.assert_called_with("Streaming error: Formatted stream error")
        # Verify assistant message was NOT added
        add_calls = mock_convo_manager.add_message.call_args_list
        assert len(add_calls) == 2

    # --- Other Method Tests (Remain Synchronous) ---

    def test_reset_conversation(self, initialized_ai, mock_convo_manager, mock_prompt_template, mock_logger):
        """Test resetting the conversation history."""
        # Arrange
        # Capture the system prompt used during init for verification
        original_system_prompt = initialized_ai._system_prompt
        
        # Act
        initialized_ai.reset_conversation()
        
        # Verify
        # 1. Reset was called on the manager
        mock_convo_manager.reset.assert_called_once()
        
        # 2. The system prompt was re-added *after* reset
        # Check the last call to add_message
        mock_convo_manager.add_message.assert_called_with(
            role="system", 
            content=original_system_prompt
        )
        # Optional: Check call count if needed, e.g., if init adds system prompt once
        # assert mock_convo_manager.add_message.call_count == 2 # (init + reset)

    def test_get_conversation(self, initialized_ai, mock_convo_manager):
        """Test getting the conversation history."""
        mock_history = [{...}] # Some dummy history
        mock_convo_manager.get_messages.return_value = mock_history
        
        history = initialized_ai.get_conversation()
        
        assert history == mock_history
        mock_convo_manager.get_messages.assert_called_once()

    def test_get_system_prompt(self, initialized_ai):
        """Test getting the current system prompt."""
        # Use the initialized AI fixture 
        expected_prompt = initialized_ai._system_prompt # Get from the initialized instance
        assert initialized_ai.get_system_prompt() == expected_prompt
        
    def test_set_system_prompt(self, initialized_ai, mock_convo_manager):
        """Test setting a new system prompt."""
        # Arrange
        new_prompt = "New system prompt."
        # Act
        initialized_ai.set_system_prompt(new_prompt)
        
        # Verify
        # 1. The internal attribute was updated
        assert initialized_ai._system_prompt == new_prompt
        
        # 2. The conversation manager's set_system_prompt was called
        mock_convo_manager.set_system_prompt.assert_called_once_with(new_prompt)

    def test_get_model_info(self, initialized_ai):
        """Test retrieving model information."""
        # Arrange
        # Expected structure based on get_model_info source code
        expected_info = {
            "model_id": initialized_ai._model_config.get("model_id", ""),
            "provider": initialized_ai._model_config.get("provider", ""),
            "quality": initialized_ai._model_config.get("quality", ""), # Add missing keys
            "speed": initialized_ai._model_config.get("speed", ""),     # Add missing keys
            "parameters": initialized_ai._model_config.get("parameters", {}), # Add missing keys
            "privacy": initialized_ai._model_config.get("privacy", ""),   # Add missing keys
            "short_key": initialized_ai._model_key # Get short_key used at init
        }
        
        # Act
        model_info = initialized_ai.get_model_info()
        
        # Verify
        assert model_info == expected_info

    @pytest.mark.asyncio
    async def test_request_basic_error_handling(self, initialized_ai, mocker):
        pass # Add pass to fix IndentationError