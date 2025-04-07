import pytest
from unittest.mock import patch, MagicMock, ANY, call
import uuid

# Class to test
from src.core.base_ai import AIBase, DEFAULT_SYSTEM_PROMPT

# Dependencies to mock
from src.config.unified_config import UnifiedConfig, AIConfigError
from src.core.provider_factory import ProviderFactory
from src.conversation.conversation_manager import ConversationManager, Message
from src.utils.logger import LoggerInterface, LoggerFactory
from src.core.interfaces import ProviderInterface
from src.prompts.prompt_template import PromptTemplate
from src.exceptions import AISetupError, AIProcessingError

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
        mock_ProviderFactory.create.side_effect = ValueError("Factory failed")

        # Act & Assert
        with pytest.raises(AISetupError, match="Failed to initialize AI: Factory failed"):
            AIBase(model=TEST_MODEL_KEY)
        
        # Verify mocks
        mock_ProviderFactory.create.assert_called_once() # Factory was called
        mock_ConversationManager.assert_not_called() # Should fail before convo manager setup 

    # --- Tests for request method ---

    @pytest.fixture
    def initialized_ai(self, mock_config_instance, mock_logger, mock_provider, mock_convo_manager, mock_prompt_template):
        """Fixture to provide a fully initialized AIBase instance for testing methods.
           Handles necessary patching internally for setup.
        """
        # Use patches *within* the fixture setup
        with patch('src.core.base_ai.LoggerFactory') as mock_LoggerFactory, \
             patch('src.core.base_ai.UnifiedConfig') as mock_UnifiedConfig, \
             patch('src.core.base_ai.ProviderFactory') as mock_ProviderFactory, \
             patch('src.core.base_ai.ConversationManager') as mock_ConversationManager, \
             patch('src.core.base_ai.PromptTemplate') as mock_PromptTemplatePatch, \
             patch('src.core.base_ai.uuid.uuid4') as mock_uuid4:
            
            # Configure mocks for successful init
            mock_LoggerFactory.create.return_value = mock_logger
            mock_UnifiedConfig.get_instance.return_value = mock_config_instance
            mock_ProviderFactory.create.return_value = mock_provider
            mock_ConversationManager.return_value = mock_convo_manager
            mock_PromptTemplatePatch.return_value = mock_prompt_template
            mock_uuid4.return_value = uuid.uuid4()

            # Instantiate AIBase
            ai = AIBase(logger=mock_logger)
            
            # Reset mocks potentially called during init
            mock_convo_manager.reset_mock()
            mock_logger.reset_mock()
            mock_provider.reset_mock()
            mock_prompt_template.reset_mock()
            
            # Return the initialized instance
            yield ai # Use yield to ensure cleanup if needed, though not critical here

    def test_request_success(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger, mock_config_instance):
        """Test successful request flow."""
        # Arrange
        # Need to access show_thinking config potentially
        with patch('src.core.base_ai.UnifiedConfig') as mock_UC:
            mock_UC.get_instance.return_value = mock_config_instance
            mock_config_instance.show_thinking = False

            ai = initialized_ai
            user_prompt = "Tell me a joke."
            ai_response_content = "Why don't scientists trust atoms? Because they make up everything!"
            provider_response = {'content': ai_response_content, 'tool_calls': []}
            dummy_messages = [Message(role="system", content="..."), Message(role="user", content=user_prompt)]
            
            mock_convo_manager.get_messages.return_value = dummy_messages
            mock_provider.request.return_value = provider_response
            
            # Act
            result = ai.request(user_prompt, temperature=0.5)

        # Assert
        assert result == ai_response_content
        add_message_calls = [
            call(role="user", content=user_prompt),
            call(role="assistant", content=ai_response_content, extract_thoughts=True, show_thinking=False)
        ]
        mock_convo_manager.add_message.assert_has_calls(add_message_calls)
        mock_convo_manager.get_messages.assert_called_once()
        mock_provider.request.assert_called_once_with(dummy_messages, temperature=0.5)
        mock_logger.info.assert_any_call(f"Processing request: {user_prompt[:50]}...")

    def test_request_provider_returns_string(self, initialized_ai, mock_convo_manager, mock_provider, mock_logger, mock_config_instance):
        """Test request flow when provider returns a raw string."""
        # Arrange
        with patch('src.core.base_ai.UnifiedConfig') as mock_UC:
            mock_UC.get_instance.return_value = mock_config_instance
            mock_config_instance.show_thinking = False
            
            ai = initialized_ai
            user_prompt = "Hello AI."
            ai_response_string = "Hello user!"
            dummy_messages = [Message(role="system", content="..."), Message(role="user", content=user_prompt)]

            mock_convo_manager.get_messages.return_value = dummy_messages
            mock_provider.request.return_value = ai_response_string
            
            # Act
            result = ai.request(user_prompt)

        # Assert
        assert result == ai_response_string
        add_message_calls = [
            call(role="user", content=user_prompt),
            call(role="assistant", content=ai_response_string, extract_thoughts=True, show_thinking=False) 
        ]
        mock_convo_manager.add_message.assert_has_calls(add_message_calls)
        mock_convo_manager.get_messages.assert_called_once()
        mock_provider.request.assert_called_once_with(dummy_messages)
        mock_logger.info.assert_any_call(f"Processing request: {user_prompt[:50]}...")

    @patch('src.core.base_ai.ErrorHandler.handle_error')
    def test_request_provider_raises_exception(self, mock_handle_error, initialized_ai, mock_convo_manager, mock_provider, mock_logger):
        """Test request raises AIProcessingError if provider fails."""
        # Arrange
        ai = initialized_ai
        user_prompt = "Cause an error."
        dummy_messages = [Message(role="system", content="..."), Message(role="user", content=user_prompt)]
        provider_error = ValueError("Provider exploded")
        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_provider.request.side_effect = provider_error
        # Mock error handler to avoid complex checks, just ensure it's called
        mock_handle_error.return_value = {'error_code': 500, 'message': 'Formatted error'}

        # Act & Assert
        with pytest.raises(AIProcessingError, match="Request failed: Provider exploded"):
            ai.request(user_prompt)

        # Verify mocks
        mock_convo_manager.add_message.assert_called_once_with(role="user", content=user_prompt) # User msg added before fail
        mock_convo_manager.get_messages.assert_called_once()
        mock_provider.request.assert_called_once_with(dummy_messages)
        
        # Verify error handling
        mock_handle_error.assert_called_once()
        # Check that the original exception was passed to the handler
        raised_exception = mock_handle_error.call_args[0][0]
        assert isinstance(raised_exception, AIProcessingError)
        assert raised_exception.__cause__ is provider_error
        mock_logger.error.assert_called_with("Request error: Formatted error") 

    # --- Tests for Conversation Methods ---

    def test_reset_conversation(self, initialized_ai, mock_convo_manager, mock_config_instance):
        """Test reset_conversation calls ConversationManager.reset and adds system prompt."""
        # Arrange
        ai = initialized_ai
        system_prompt = ai._system_prompt # Get the system prompt set during init

        # Act
        ai.reset_conversation()

        # Assert
        mock_convo_manager.reset.assert_called_once()
        # Ensure system prompt is re-added after reset
        mock_convo_manager.add_message.assert_called_once_with(role="system", content=system_prompt)

    def test_get_conversation(self, initialized_ai, mock_convo_manager):
        """Test get_conversation returns messages from ConversationManager."""
        # Arrange
        ai = initialized_ai
        expected_messages = [Message(role="user", content="Hi")]
        mock_convo_manager.get_messages.return_value = expected_messages

        # Act
        result = ai.get_conversation()

        # Assert
        assert result == expected_messages
        mock_convo_manager.get_messages.assert_called_once()

    def test_get_system_prompt(self, initialized_ai):
        """Test get_system_prompt returns the stored system prompt."""
        # Arrange
        ai = initialized_ai
        expected_prompt = ai._system_prompt

        # Act
        result = ai.get_system_prompt()

        # Assert
        assert result == expected_prompt

    def test_set_system_prompt(self, initialized_ai, mock_convo_manager):
        """Test set_system_prompt updates the prompt and ConversationManager."""
        # Arrange
        ai = initialized_ai
        new_prompt = "You are now a pirate AI."

        # Act
        ai.set_system_prompt(new_prompt)

        # Assert
        assert ai._system_prompt == new_prompt
        assert ai.get_system_prompt() == new_prompt
        mock_convo_manager.set_system_prompt.assert_called_once_with(new_prompt)

    # --- Test for get_model_info ---

    def test_get_model_info(self, initialized_ai):
        """Test get_model_info returns the stored model configuration."""
        # Arrange
        ai = initialized_ai
        model_config = ai._model_config # Get the config stored during init
        # Construct the expected dictionary explicitly matching the method's output
        expected_info = {
            "model_id": model_config.get("model_id", ""),
            "provider": model_config.get("provider", ""),
            "quality": model_config.get("quality", ""), # Add default
            "speed": model_config.get("speed", ""),       # Add default
            "parameters": model_config.get("parameters", {}), # Add default
            "privacy": model_config.get("privacy", "")    # Add default
        }
        # Ensure the base mock config has provider/model_id for a meaningful test
        assert "provider" in model_config
        assert "model_id" in model_config

        # Act
        result = ai.get_model_info()

        # Assert
        assert result == expected_info 