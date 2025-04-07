import pytest
from unittest.mock import patch, MagicMock, ANY, call
import uuid
import json
import dataclasses # Import dataclasses

# Class to test
from src.core.tool_enabled_ai import ToolEnabledAI

# Base class and dependencies
from src.core.base_ai import AIBase
from src.config.unified_config import UnifiedConfig, AIConfigError
from src.core.provider_factory import ProviderFactory
from src.conversation.conversation_manager import ConversationManager, Message
from src.utils.logger import LoggerInterface, LoggerFactory
from src.core.interfaces import ProviderInterface, ToolCapableProviderInterface
from src.prompts.prompt_template import PromptTemplate
from src.tools.tool_manager import ToolManager
from src.tools.models import ToolCall, ToolDefinition, ToolResult
from src.exceptions import AIProcessingError, AIToolError

# Constants for tests
TEST_MODEL_KEY = "test-tool-model"
TEST_PROVIDER_NAME = "test-tool-provider"
TEST_PROVIDER_MODEL_ID = "tool-provider-model-id-456"

# --- Fixtures (can potentially reuse/adapt from test_base_ai) ---

@pytest.fixture
def mock_tool_manager():
    """Fixture for a mocked ToolManager."""
    return MagicMock(spec=ToolManager)

@pytest.fixture
def mock_tool_capable_provider():
    """Fixture for a provider that IS tool capable (via interface)."""
    # Remove spec to allow adding format_tool_response in tests
    # return MagicMock(spec=ToolCapableProviderInterface)
    return MagicMock() # Allow arbitrary attributes like format_tool_response

@pytest.fixture
def mock_basic_provider():
    """Fixture for a provider that is NOT tool capable."""
    # Use spec instead of inheritance
    # return MagicMock(spec=ProviderInterface)
    return MagicMock(spec=ProviderInterface)

@pytest.fixture
def mock_convo_manager():
    """Fixture for a mocked ConversationManager."""
    return MagicMock(spec=ConversationManager)

# Reusing logger fixture from base_ai tests implicitly if run together, 
# or define explicitly if needed.
@pytest.fixture
def mock_logger():
    return MagicMock(spec=LoggerInterface)

@pytest.fixture
def mock_prompt_template():
    """Fixture for a mocked PromptTemplate service."""
    mock_pt = MagicMock(spec=PromptTemplate)
    # Default behavior: render_prompt raises ValueError (template not found)
    mock_pt.render_prompt.side_effect = ValueError("Template not found") 
    return mock_pt

# --- Test Class ---

class TestToolEnabledAI:

    # REMOVE @patch.object(AIBase, '__init__') 
    # Add patches for ALL dependencies called during full init chain
    @patch('src.core.base_ai.uuid.uuid4') # Called by AIBase
    @patch('src.core.base_ai.LoggerFactory') # Called by AIBase
    @patch('src.core.base_ai.UnifiedConfig') # Called by AIBase
    @patch('src.core.base_ai.ProviderFactory') # Called by AIBase
    @patch('src.core.base_ai.ConversationManager') # Called by AIBase
    @patch('src.core.base_ai.PromptTemplate') # Called by AIBase
    @patch('src.core.tool_enabled_ai.ToolManager') # Called by ToolEnabledAI (default path)
    @patch('src.core.tool_enabled_ai.UnifiedConfig') # Called by ToolEnabledAI
    def test_init_tool_capable_provider(self, mock_TE_UnifiedConfig, mock_ToolManager_Class, 
                                        mock_BA_PromptTemplate_Class, mock_BA_ConversationManager_Class,
                                        mock_BA_ProviderFactory, mock_BA_UnifiedConfig, 
                                        mock_BA_LoggerFactory, mock_BA_uuid4,
                                        mock_tool_manager, mock_tool_capable_provider, mock_logger,
                                        mock_convo_manager, mock_prompt_template):
        """Test init with a tool-capable provider, using default ToolManager."""
        # Arrange
        # --- Configure Mocks for AIBase dependencies ---
        mock_BA_uuid4.return_value = uuid.uuid4()
        mock_BA_LoggerFactory.create.return_value = mock_logger
        mock_config_instance = MagicMock(spec=UnifiedConfig)
        mock_BA_UnifiedConfig.get_instance.return_value = mock_config_instance
        mock_config_instance.get_default_model.return_value = TEST_MODEL_KEY
        model_config = {'provider': TEST_PROVIDER_NAME, 'model_id': TEST_PROVIDER_MODEL_ID}
        provider_config = {'api_key': 'dummy'}
        mock_config_instance.get_model_config.return_value = model_config
        mock_config_instance.get_provider_config.return_value = provider_config
        mock_config_instance.get_system_prompt.return_value = None # Force use of default prompt logic
        mock_BA_ProviderFactory.create.return_value = mock_tool_capable_provider
        mock_BA_ConversationManager_Class.return_value = mock_convo_manager
        mock_BA_PromptTemplate_Class.return_value = mock_prompt_template
        mock_prompt_template.render_prompt.side_effect = ValueError # Simulate fallback prompt

        # --- Configure Mocks for ToolEnabledAI dependencies ---
        # Ensure UnifiedConfig returns the same instance
        mock_TE_UnifiedConfig.get_instance.return_value = mock_config_instance 
        mock_ToolManager_Class.return_value = mock_tool_manager # Mock default creation

        # Act
        # Real init chain runs, calling mocks
        ai = ToolEnabledAI(logger=mock_logger, model=TEST_MODEL_KEY)

        # Assert
        # Verify base init dependencies were called
        mock_BA_ProviderFactory.create.assert_called_once_with(
            provider_type=TEST_PROVIDER_NAME,
            model_id=TEST_PROVIDER_MODEL_ID,
            provider_config=provider_config,
            model_config=model_config,
            logger=mock_logger
        )
        mock_BA_ConversationManager_Class.assert_called_once()
        mock_BA_PromptTemplate_Class.assert_called_once()
        mock_convo_manager.add_message.assert_called() # Check system prompt was added
        
        # Verify ToolEnabledAI specific logic
        mock_TE_UnifiedConfig.get_instance.assert_called()
        mock_ToolManager_Class.assert_called_once_with(unified_config=mock_config_instance, logger=mock_logger)
        assert ai._tool_manager == mock_tool_manager
        assert ai._supports_tools is True # Determined from mock_tool_capable_provider
        assert hasattr(ai, '_tool_history') and ai._tool_history == []

    @patch('src.core.base_ai.uuid.uuid4')
    @patch('src.core.base_ai.LoggerFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.tool_enabled_ai.ToolManager')
    @patch('src.core.tool_enabled_ai.UnifiedConfig')
    def test_init_basic_provider(self, mock_TE_UnifiedConfig, mock_ToolManager_Class, 
                                 mock_BA_PromptTemplate_Class, mock_BA_ConversationManager_Class,
                                 mock_BA_ProviderFactory, mock_BA_UnifiedConfig, 
                                 mock_BA_LoggerFactory, mock_BA_uuid4,
                                 mock_tool_manager, mock_basic_provider, mock_logger,
                                 mock_convo_manager, mock_prompt_template):
        """Test init with a basic provider, using default ToolManager."""
        # Arrange
        mock_BA_uuid4.return_value = uuid.uuid4()
        mock_BA_LoggerFactory.create.return_value = mock_logger
        mock_config_instance = MagicMock(spec=UnifiedConfig)
        mock_BA_UnifiedConfig.get_instance.return_value = mock_config_instance
        mock_config_instance.get_default_model.return_value = TEST_MODEL_KEY
        model_config = {'provider': 'basic-provider', 'model_id': 'basic-model'}
        provider_config = {}
        mock_config_instance.get_model_config.return_value = model_config
        mock_config_instance.get_provider_config.return_value = provider_config
        mock_config_instance.get_system_prompt.return_value = "Basic system prompt"
        # IMPORTANT: Factory returns the basic provider mock here
        mock_BA_ProviderFactory.create.return_value = mock_basic_provider 
        mock_BA_ConversationManager_Class.return_value = mock_convo_manager
        mock_BA_PromptTemplate_Class.return_value = mock_prompt_template

        mock_TE_UnifiedConfig.get_instance.return_value = mock_config_instance 
        mock_ToolManager_Class.return_value = mock_tool_manager

        # Act
        ai = ToolEnabledAI(logger=mock_logger, model=TEST_MODEL_KEY)

        # Assert
        mock_BA_ProviderFactory.create.assert_called_once()
        # ToolManager still created by default
        mock_ToolManager_Class.assert_called_once_with(unified_config=mock_config_instance, logger=mock_logger)
        assert ai._tool_manager == mock_tool_manager
        # Support check uses the actual mock_basic_provider type
        assert ai._supports_tools is False 
        assert hasattr(ai, '_tool_history') and ai._tool_history == []
        mock_logger.warning.assert_called() # Check warning was logged

    @patch('src.core.base_ai.uuid.uuid4')
    @patch('src.core.base_ai.LoggerFactory')
    @patch('src.core.base_ai.UnifiedConfig')
    @patch('src.core.base_ai.ProviderFactory')
    @patch('src.core.base_ai.ConversationManager')
    @patch('src.core.base_ai.PromptTemplate')
    @patch('src.core.tool_enabled_ai.ToolManager') # Patch class even if not called
    @patch('src.core.tool_enabled_ai.UnifiedConfig')
    def test_init_with_provided_tool_manager(self, mock_TE_UnifiedConfig, mock_ToolManager_Class, 
                                            mock_BA_PromptTemplate_Class, mock_BA_ConversationManager_Class,
                                            mock_BA_ProviderFactory, mock_BA_UnifiedConfig, 
                                            mock_BA_LoggerFactory, mock_BA_uuid4,
                                            mock_tool_manager, mock_tool_capable_provider, mock_logger,
                                            mock_convo_manager, mock_prompt_template):
        """Test init uses the provided ToolManager instance."""
        # Arrange
        mock_BA_uuid4.return_value = uuid.uuid4()
        mock_BA_LoggerFactory.create.return_value = mock_logger
        mock_config_instance = MagicMock(spec=UnifiedConfig)
        mock_BA_UnifiedConfig.get_instance.return_value = mock_config_instance
        model_config = {'provider': TEST_PROVIDER_NAME, 'model_id': TEST_PROVIDER_MODEL_ID}
        provider_config = {'api_key': 'dummy'}
        mock_config_instance.get_model_config.return_value = model_config
        mock_config_instance.get_provider_config.return_value = provider_config
        mock_config_instance.get_system_prompt.return_value = "System prompt"
        mock_BA_ProviderFactory.create.return_value = mock_tool_capable_provider
        mock_BA_ConversationManager_Class.return_value = mock_convo_manager
        mock_BA_PromptTemplate_Class.return_value = mock_prompt_template

        mock_TE_UnifiedConfig.get_instance.return_value = mock_config_instance 
        # ToolManager class is patched, but won't be called
        provided_tool_manager = mock_tool_manager # Use the fixture instance

        # Act
        # Provide the tool manager instance directly
        ai = ToolEnabledAI(logger=mock_logger, model=TEST_MODEL_KEY, tool_manager=provided_tool_manager)

        # Assert
        mock_BA_ProviderFactory.create.assert_called_once()
        # ToolManager class should NOT have been called
        mock_ToolManager_Class.assert_not_called() 
        assert ai._tool_manager == provided_tool_manager # Instance used
        assert ai._supports_tools is True
        assert hasattr(ai, '_tool_history') and ai._tool_history == []

    # --- Tests for request_basic ---

    @pytest.fixture
    def tool_ai_for_methods(self, mock_logger, mock_tool_capable_provider, mock_convo_manager, mock_tool_manager, mock_prompt_template):
        # Patch ALL dependencies needed for the *real* AIBase init
        with patch('src.core.base_ai.uuid.uuid4') as mock_uuid4, \
             patch('src.core.base_ai.LoggerFactory') as mock_LoggerFactory, \
             patch('src.core.base_ai.UnifiedConfig') as mock_UnifiedConfig, \
             patch('src.core.base_ai.ProviderFactory') as mock_ProviderFactory, \
             patch('src.core.base_ai.ConversationManager') as mock_ConversationManager_Class, \
             patch('src.core.base_ai.PromptTemplate') as mock_PromptTemplate_Class, \
             patch('src.core.tool_enabled_ai.ToolManager') as mock_ToolManager_Class, \
             patch('src.core.tool_enabled_ai.UnifiedConfig') as mock_TE_UnifiedConfig:

            # --- Configure mocks --- 
            mock_uuid4.return_value = uuid.uuid4()
            mock_LoggerFactory.create.return_value = mock_logger
            mock_config_instance = MagicMock(spec=UnifiedConfig)
            mock_UnifiedConfig.get_instance.return_value = mock_config_instance
            mock_TE_UnifiedConfig.get_instance.return_value = mock_config_instance # Ensure same instance
            
            # Configure for a successful init with tool capable provider
            mock_config_instance.get_default_model.return_value = 'test-model'
            model_config = {'provider': 'test-provider', 'model_id': 'test-id'}
            provider_config = {}
            mock_config_instance.get_model_config.return_value = model_config
            mock_config_instance.get_provider_config.return_value = provider_config
            mock_config_instance.get_system_prompt.return_value = "Fixture system prompt"
            
            mock_ProviderFactory.create.return_value = mock_tool_capable_provider
            mock_ConversationManager_Class.return_value = mock_convo_manager
            mock_PromptTemplate_Class.return_value = mock_prompt_template
            # ToolManager class mock is only relevant if default is created
            # We provide mock_tool_manager, so ToolManager class mock is not used here.
            
            # --- Instantiate --- 
            # Real init runs using all the mocks configured above
            ai = ToolEnabledAI(logger=mock_logger, tool_manager=mock_tool_manager) 
            
            # --- Verify Init Calls (Optional but good practice) ---
            mock_ProviderFactory.create.assert_called()
            mock_ConversationManager_Class.assert_called()
            mock_ToolManager_Class.assert_not_called() # Because we provided one
            mock_convo_manager.add_message.assert_called() # System prompt added

            # --- Reset mocks for method testing --- 
            mock_convo_manager.reset_mock()
            mock_logger.reset_mock()
            mock_tool_capable_provider.reset_mock()
            mock_ProviderFactory.reset_mock() # Reset factory too
            # ... reset other mocks if necessary ...

            yield ai # Provide the initialized instance

    @patch('src.core.models.ProviderResponse') # Easier to mock the whole object
    def test_request_basic_content_only(self, MockProviderResponse, tool_ai_for_methods, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test request_basic when provider returns only content."""
        # Arrange
        ai = tool_ai_for_methods
        user_prompt = "Explain quantum physics."
        ai_content = "It's complicated."
        dummy_messages = [Message(role="user", content=user_prompt)]
        
        mock_convo_manager.get_messages.return_value = dummy_messages
        # Mock ProviderResponse attributes
        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = ai_content
        mock_response_obj.tool_calls = None
        mock_response_obj.error = None
        mock_tool_capable_provider.request.return_value = mock_response_obj

        # Act
        result_response = ai.request_basic(user_prompt, max_tokens=100)

        # Assert
        assert result_response == mock_response_obj # Should return the response object
        mock_convo_manager.add_message.assert_any_call(role="user", content=user_prompt)
        mock_tool_capable_provider.request.assert_called_once_with(messages=dummy_messages, max_tokens=100)
        # Check assistant message added to history - WITHOUT tool_calls kwarg
        # Find the assistant call
        assistant_call = None
        for c in mock_convo_manager.add_message.call_args_list:
            if c.kwargs.get('role') == 'assistant':
                assistant_call = c
                break
        assert assistant_call is not None, "Assistant message call not found"
        assert assistant_call.kwargs == {'role': 'assistant', 'content': ai_content}
        assert mock_convo_manager.add_message.call_count == 2 # user + assistant
        mock_logger.debug.assert_any_call(f"ToolEnabledAI.request_basic: Received ProviderResponse. Content: True, Tool Calls: 0")

    @patch('src.core.models.ProviderResponse')
    def test_request_basic_tools_only(self, MockProviderResponse, tool_ai_for_methods, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test request_basic when provider returns only tool calls."""
        # Arrange
        ai = tool_ai_for_methods
        user_prompt = "Find example.com IP."
        dummy_messages = [Message(role="user", content=user_prompt)]
        tool_calls_data = [MagicMock(spec=ToolCall)] # Mock ToolCall objects
        
        mock_convo_manager.get_messages.return_value = dummy_messages
        # Mock ProviderResponse attributes
        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = None # No direct content
        mock_response_obj.tool_calls = tool_calls_data
        mock_response_obj.error = None
        mock_tool_capable_provider.request.return_value = mock_response_obj

        # Act
        result_response = ai.request_basic(user_prompt)

        # Assert
        assert result_response == mock_response_obj
        mock_convo_manager.add_message.assert_any_call(role="user", content=user_prompt)
        mock_tool_capable_provider.request.assert_called_once_with(messages=dummy_messages)
        # Check assistant message added with tool calls
        mock_convo_manager.add_message.assert_any_call(
            role="assistant", 
            content=None, 
            tool_calls=tool_calls_data
        )
        assert mock_convo_manager.add_message.call_count == 2
        mock_logger.debug.assert_any_call(f"ToolEnabledAI.request_basic: Received ProviderResponse. Content: False, Tool Calls: 1")

    @patch('src.core.models.ProviderResponse')
    def test_request_basic_content_and_tools(self, MockProviderResponse, tool_ai_for_methods, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test request_basic when provider returns both content and tool calls."""
        # Arrange
        ai = tool_ai_for_methods
        user_prompt = "Summarize and find IP."
        ai_content = "Summary..."
        tool_calls_data = [MagicMock(spec=ToolCall)]
        dummy_messages = [Message(role="user", content=user_prompt)]
        
        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = ai_content
        mock_response_obj.tool_calls = tool_calls_data
        mock_response_obj.error = None
        mock_tool_capable_provider.request.return_value = mock_response_obj

        # Act
        result_response = ai.request_basic(user_prompt)

        # Assert
        assert result_response == mock_response_obj
        mock_convo_manager.add_message.assert_any_call(role="user", content=user_prompt)
        mock_tool_capable_provider.request.assert_called_once_with(messages=dummy_messages)
        # Check assistant message added with both content and tool calls
        mock_convo_manager.add_message.assert_any_call(
            role="assistant", 
            content=ai_content, 
            tool_calls=tool_calls_data
        )
        assert mock_convo_manager.add_message.call_count == 2
        mock_logger.debug.assert_any_call(f"ToolEnabledAI.request_basic: Received ProviderResponse. Content: True, Tool Calls: 1")

    @patch('src.core.models.ProviderResponse')
    def test_request_basic_provider_response_error(self, MockProviderResponse, tool_ai_for_methods, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test request_basic when ProviderResponse contains an error."""
        # Arrange
        ai = tool_ai_for_methods
        user_prompt = "Trigger error."
        error_message = "Provider quota exceeded"
        dummy_messages = [Message(role="user", content=user_prompt)]
        
        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = None
        mock_response_obj.tool_calls = None
        mock_response_obj.error = error_message # Error present
        mock_tool_capable_provider.request.return_value = mock_response_obj

        # Act & Assert
        with pytest.raises(AIProcessingError, match=f"Provider error: {error_message}"):
            ai.request_basic(user_prompt)
            
        # Verify provider was called, user message added, but assistant message was NOT
        mock_convo_manager.add_message.assert_called_once_with(role="user", content=user_prompt)
        mock_tool_capable_provider.request.assert_called_once_with(messages=dummy_messages)
        # Check for the *second* log message from the outer except block
        # Need to construct the expected string representation of the caught error
        expected_inner_error_str = f"PROCESSING: Provider error: {error_message}"
        mock_logger.error.assert_called_with(f"Error during ToolEnabledAI basic request: {expected_inner_error_str}", exc_info=True)

    # Inherit patches needed for the fixture
    @patch('src.core.tool_enabled_ai.ErrorHandler.handle_error') # Mock error handler if needed, though now raising directly
    def test_request_basic_provider_direct_exception(self, mock_handle_error, tool_ai_for_methods, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test request_basic when provider.request raises an exception directly."""
        # Arrange
        ai = tool_ai_for_methods
        user_prompt = "Break it."
        provider_error = ConnectionError("Network failed")
        dummy_messages = [Message(role="user", content=user_prompt)]

        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_tool_capable_provider.request.side_effect = provider_error
        # mock_handle_error.return_value = {'error_code': 500, 'message': 'Formatted direct error'}

        # Act & Assert
        with pytest.raises(AIProcessingError, match="Failed processing basic request: Network failed") as exc_info:
            ai.request_basic(user_prompt)
        
        # Check that the original error is chained
        assert exc_info.value.__cause__ is provider_error
            
        # Verify provider was called, user message added, but assistant message was NOT
        mock_convo_manager.add_message.assert_called_once_with(role="user", content=user_prompt)
        mock_tool_capable_provider.request.assert_called_once_with(messages=dummy_messages)
        mock_logger.error.assert_called_with(f"Error during ToolEnabledAI basic request: {provider_error}", exc_info=True)
        # Ensure error handler wasn't called as the specific catch block doesn't use it
        # mock_handle_error.assert_not_called()

    # --- Tests for process_prompt ---

    @patch.object(AIBase, 'request') # Patch the method from the superclass
    def test_process_prompt_no_tool_support(self, mock_super_request, tool_ai_for_methods, mock_logger):
        """Test process_prompt falls back to super().request if provider lacks tool support."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = False # Manually override for this test
        user_prompt = "A simple question."
        expected_response = "A simple answer."
        mock_super_request.return_value = expected_response
        options = {"temperature": 0.1}

        # Act
        result = ai.process_prompt(user_prompt, **options)

        # Assert
        assert result == expected_response
        mock_logger.warning.assert_called_with("Provider does not support tools. Performing basic request using AIBase.request.")
        # Verify super().request was called correctly
        mock_super_request.assert_called_once_with(user_prompt, **options)
        # Ensure the tool-specific provider request wasn't called
        ai._provider.request.assert_not_called()

    @patch('src.core.models.ProviderResponse')
    def test_process_prompt_no_tools_available(self, MockProviderResponse, tool_ai_for_methods, mock_tool_manager, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test process_prompt when ToolManager has no tools registered."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = True # Ensure tool support is enabled
        user_prompt = "What time is it?"
        ai_content = "It is currently time." 
        dummy_messages = [Message(role="user", content=user_prompt)]
        # Provider returns direct content response
        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = ai_content
        mock_response_obj.tool_calls = None
        mock_response_obj.error = None
        mock_tool_capable_provider.request.return_value = mock_response_obj

        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_tool_manager.get_all_tools.return_value = [] # No tools registered

        # Act
        result = ai.process_prompt(user_prompt, max_tool_iterations=3)

        # Assert
        assert result == ai_content
        mock_tool_manager.get_all_tools.assert_called_once()
        # Provider should be called WITHOUT tools/tool_choice args
        mock_tool_capable_provider.request.assert_called_once()
        call_args, call_kwargs = mock_tool_capable_provider.request.call_args
        assert call_kwargs.get('messages') == dummy_messages
        assert 'tools' not in call_kwargs
        assert 'tool_choice' not in call_kwargs
        mock_logger.info.assert_any_call("No tools registered in ToolManager. Proceeding without tools for this turn.")
        # Verify conversation history (user prompt + final assistant content)
        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_any_call(role="user", content=user_prompt)
        mock_convo_manager.add_message.assert_any_call(role="assistant", content=ai_content) # Final content added

    @patch('src.core.models.ProviderResponse')
    def test_process_prompt_direct_content_response(self, MockProviderResponse, tool_ai_for_methods, mock_tool_manager, mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test process_prompt when tools are available but provider returns content directly."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = True
        user_prompt = "Summarize this text."
        ai_content = "This is the summary."
        dummy_messages = [Message(role="user", content=user_prompt)]
        mock_tool_defs = [MagicMock(spec=ToolDefinition)] # Tools are available

        mock_response_obj = MockProviderResponse()
        mock_response_obj.content = ai_content
        mock_response_obj.tool_calls = None # No tool calls requested
        mock_response_obj.error = None
        mock_tool_capable_provider.request.return_value = mock_response_obj

        mock_convo_manager.get_messages.return_value = dummy_messages
        mock_tool_manager.get_all_tools.return_value = mock_tool_defs

        # Act
        result = ai.process_prompt(user_prompt)

        # Assert
        assert result == ai_content
        mock_tool_manager.get_all_tools.assert_called_once()
        # Provider *should* be called WITH tools/tool_choice args
        mock_tool_capable_provider.request.assert_called_once()
        call_args, call_kwargs = mock_tool_capable_provider.request.call_args
        assert call_kwargs.get('messages') == dummy_messages
        assert call_kwargs.get('tools') == mock_tool_defs
        assert call_kwargs.get('tool_choice') == 'auto' # Default
        
        # Verify conversation history (user prompt + final assistant content)
        assert mock_convo_manager.add_message.call_count == 2
        mock_convo_manager.add_message.assert_any_call(role="user", content=user_prompt)
        mock_convo_manager.add_message.assert_any_call(role="assistant", content=ai_content)

    @patch('src.core.tool_enabled_ai.ToolEnabledAI._execute_tool_call')
    def test_process_prompt_single_tool_call(self, mock_execute_tool,
                                            tool_ai_for_methods, mock_tool_manager, 
                                            mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test the loop for a single successful tool call."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = True
        user_prompt = "What is the weather in London?"
        tool_id = "tool_abc"
        tool_name = "get_weather"
        tool_args = {"location": "London"}
        tool_result_content = "The weather in London is sunny."
        final_ai_content = f"Based on the tool: {tool_result_content}"
        
        mock_tool_defs = [MagicMock(spec=ToolDefinition)]
        mock_tool_manager.get_all_tools.return_value = mock_tool_defs

        # --- Mock Sequence ---
        # 1. First provider call -> requests tool call
        initial_user_message = Message(role="user", content=user_prompt)
        # Mock the *initial* history for the first get_messages call
        # mock_convo_manager.get_messages.return_value = [initial_user_message]
        response1 = MagicMock()
        response1.content = None
        assistant_msg_content = [ToolCall(id=tool_id, name=tool_name, arguments=tool_args)]
        response1.tool_calls = assistant_msg_content
        response1.error = None
        response1.stop_reason = 'tool_calls'
        
        # 2. Tool execution -> returns success result
        tool_result = ToolResult(success=True, tool_call_id=tool_id, result=tool_result_content)
        mock_execute_tool.return_value = tool_result
        
        # 3. Mock provider adding tool result message dict
        tool_result_message_dict = dataclasses.asdict(Message(role="tool", content=tool_result_content, tool_call_id=tool_id))
        mock_tool_capable_provider._add_tool_message.return_value = [tool_result_message_dict]
        
        # 4. Second provider call setup
        # Define the EXACT history expected BEFORE the second provider call
        # Code adds user, then assistant(tool_calls), then tool_result
        expected_history_for_call_2 = [
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=assistant_msg_content), # Tool calls stored in content by tracker
            Message(**tool_result_message_dict) # Recreate from dict added via mock
        ]
        response2 = MagicMock()
        response2.content = final_ai_content
        response2.tool_calls = None
        response2.error = None
        response2.stop_reason = 'stop'
        
        # Configure mocks for the sequence
        mock_tool_capable_provider.request.side_effect = [response1, response2]
        
        # Refined get_messages mock: Use a tracker like in max_iterations test
        message_history = []
        def add_message_tracker(**kwargs):
            nonlocal message_history
            # Handle assistant message with tool calls
            if kwargs.get('role') == 'assistant' and 'tool_calls' in kwargs:
                kwargs['content'] = kwargs.pop('tool_calls')
            message_history.append(Message(**kwargs))
        mock_convo_manager.add_message.side_effect = add_message_tracker
        # Let get_messages return the tracked history
        mock_convo_manager.get_messages.side_effect = lambda: message_history.copy()

        # Act
        # Code internally adds user message first
        result = ai.process_prompt(user_prompt)

        # Assert
        assert result == final_ai_content
        assert mock_tool_manager.get_all_tools.call_count == 2 
        assert mock_tool_capable_provider.request.call_count == 2
        
        # Check provider calls
        first_call_args, first_call_kwargs = mock_tool_capable_provider.request.call_args_list[0]
        # History for first call only contains user msg added by process_prompt
        assert first_call_kwargs.get('messages') == [Message(role="user", content=user_prompt)] 
        assert first_call_kwargs.get('tools') == mock_tool_defs
        
        second_call_args, second_call_kwargs = mock_tool_capable_provider.request.call_args_list[1]
        # History for second call should reflect adds from tracker
        assert second_call_kwargs.get('messages') == expected_history_for_call_2
        assert second_call_kwargs.get('tools') == mock_tool_defs
        
        # Check tool execution and provider formatting method
        mock_execute_tool.assert_called_once_with(response1.tool_calls[0]) # Use original response1
        mock_tool_capable_provider._add_tool_message.assert_called_once()
        add_tool_args_call = mock_tool_capable_provider._add_tool_message.call_args[1]
        assert add_tool_args_call['tool_call_id'] == tool_id
        assert add_tool_args_call['tool_name'] == tool_name
        assert add_tool_args_call['content'] == tool_result_content

        # Check conversation history additions via tracker
        # Expected: user, assistant(tool_calls), tool_result, assistant(final_content)
        assert len(message_history) == 4
        assert message_history == [
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=assistant_msg_content),
            Message(**tool_result_message_dict),
            Message(role="assistant", content=final_ai_content)
        ]
        
    @patch('src.core.tool_enabled_ai.ToolEnabledAI._execute_tool_call')
    def test_process_prompt_tool_execution_error(self, mock_execute_tool,
                                                 tool_ai_for_methods, mock_tool_manager, 
                                                 mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test the loop when tool execution raises AIToolError."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = True
        user_prompt = "Do something risky."
        tool_id = "tool_err"
        tool_name = "risky_op"
        tool_args = {}
        error_message = "Failed because it was risky!"
        final_ai_content = "The tool failed to execute."
        
        mock_tool_defs = [MagicMock(spec=ToolDefinition)]
        mock_tool_manager.get_all_tools.return_value = mock_tool_defs

        # --- Mock Sequence --- 
        # 1. Provider requests tool call
        initial_user_message = Message(role="user", content=user_prompt)
        #mock_convo_manager.get_messages.return_value = [initial_user_message]
        response1 = MagicMock()
        response1.content = None
        assistant_msg_content = [ToolCall(id=tool_id, name=tool_name, arguments=tool_args)]
        response1.tool_calls = assistant_msg_content
        response1.error = None
        response1.stop_reason = 'tool_calls'

        # 2. Tool execution -> returns error result
        tool_error = AIToolError(error_message)
        error_result = ToolResult(success=False, tool_call_id=tool_id, error=f"Error executing tool: {error_message}")
        mock_execute_tool.return_value = error_result

        # 3. Mock provider adding tool result message dict (error case)
        tool_error_message_dict = dataclasses.asdict(Message(role="tool", content=error_result.error, tool_call_id=tool_id))
        mock_tool_capable_provider._add_tool_message.return_value = [tool_error_message_dict]
        
        # 4. Second provider call setup
        expected_history_for_call_2 = [
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=assistant_msg_content), 
            Message(**tool_error_message_dict)
        ]
        response2 = MagicMock()
        response2.content = final_ai_content
        response2.tool_calls = None
        response2.error = None
        response2.stop_reason = 'stop'

        mock_tool_capable_provider.request.side_effect = [response1, response2]
        
        # Use the same tracker approach for get_messages
        message_history = []
        def add_message_tracker(**kwargs):
            nonlocal message_history
            if kwargs.get('role') == 'assistant' and 'tool_calls' in kwargs:
                kwargs['content'] = kwargs.pop('tool_calls')
            message_history.append(Message(**kwargs))
        mock_convo_manager.add_message.side_effect = add_message_tracker
        mock_convo_manager.get_messages.side_effect = lambda: message_history.copy()

        # Act
        result = ai.process_prompt(user_prompt)

        # Assert
        assert result == final_ai_content
        assert mock_tool_manager.get_all_tools.call_count == 2 
        assert mock_tool_capable_provider.request.call_count == 2
        
        # Check provider calls
        first_call_args, first_call_kwargs = mock_tool_capable_provider.request.call_args_list[0]
        assert first_call_kwargs.get('messages') == [Message(role="user", content=user_prompt)]
        
        second_call_args, second_call_kwargs = mock_tool_capable_provider.request.call_args_list[1]
        assert second_call_kwargs.get('messages') == expected_history_for_call_2
        
        # Check tool execution and provider formatting method
        mock_execute_tool.assert_called_once_with(response1.tool_calls[0])
        mock_tool_capable_provider._add_tool_message.assert_called_once()
        add_tool_args_call = mock_tool_capable_provider._add_tool_message.call_args[1]
        assert add_tool_args_call['tool_call_id'] == tool_id
        assert add_tool_args_call['tool_name'] == tool_name
        assert add_tool_args_call['content'] == error_result.error

        # Check conversation history additions via tracker
        # Expected: user, assistant(tool_calls), tool_error_result, assistant(final_content)
        assert len(message_history) == 4
        assert message_history == [
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=assistant_msg_content),
            Message(**tool_error_message_dict),
            Message(role="assistant", content=final_ai_content)
        ]

    @patch('src.core.tool_enabled_ai.ToolEnabledAI._execute_tool_call')
    @patch('src.core.models.ProviderResponse')
    def test_process_prompt_max_iterations_reached(self, MockProviderResponse, mock_execute_tool,
                                                   tool_ai_for_methods, mock_tool_manager,
                                                   mock_convo_manager, mock_tool_capable_provider, mock_logger):
        """Test the loop terminates after max_tool_iterations."""
        # Arrange
        ai = tool_ai_for_methods
        ai._supports_tools = True
        user_prompt = "Run tools forever."
        max_iterations = 2 # Set a low max for testing
        last_ai_content = "Okay, calling tool again..." # Content from last assistant message
        
        mock_tool_defs = [MagicMock(spec=ToolDefinition)]
        mock_tool_manager.get_all_tools.return_value = mock_tool_defs

        # --- Mock Sequence --- 
        # Always return a tool call request from provider
        # Always return a tool result successfully
        def mock_provider_request(*args, **kwargs):
            # Use correct ToolCall structure with dict arguments
            tool_call = ToolCall(id=f"call_{mock_tool_capable_provider.request.call_count}", 
                                 name="fake_tool", 
                                 arguments={}) # Use dict for arguments
            response = MockProviderResponse()
            response.content = last_ai_content # Simulate some intermediate content
            response.tool_calls = [tool_call]
            response.error = None
            return response
        mock_tool_capable_provider.request.side_effect = mock_provider_request
        
        def mock_tool_execution(tool_call: ToolCall):
            # Use correct ToolResult model fields: success=True, result=...
            return ToolResult(success=True, tool_call_id=tool_call.id, result="Tool success")
        mock_execute_tool.side_effect = mock_tool_execution
        
        # Mock provider's _add_tool_message dynamically
        def mock_add_tool_message(**kwargs):
            tool_call_id = kwargs.get('tool_call_id')
            tool_content = kwargs.get('content', 'Tool success') # Get content passed by code
            if tool_call_id:
                msg_dict = {'role': 'tool', 'content': tool_content, 'tool_call_id': tool_call_id}
                return [msg_dict]
            return [] # Should not happen if ID is present
        mock_tool_capable_provider._add_tool_message.side_effect = mock_add_tool_message
        
        # Keep track of messages added for get_messages mock
        message_history = [] # Start empty, first user message added inside process_prompt
        initial_user_message = Message(role="user", content=user_prompt)
        
        def add_message_tracker(**kwargs):
            nonlocal message_history
            # Handle assistant message with tool calls
            if kwargs.get('role') == 'assistant' and 'tool_calls' in kwargs:
                kwargs['content'] = kwargs.pop('tool_calls') # Move tool_calls to content
            message_history.append(Message(**kwargs))
            
        mock_convo_manager.add_message.side_effect = add_message_tracker
        # get_messages now needs to return the *current* history for each provider call
        mock_convo_manager.get_messages.side_effect = lambda: message_history.copy()

        # Act: Add the initial user message *before* calling process_prompt
        #      because the tracker starts *after* this internal add happens.
        # Actually, NO. The code adds the user message internally. The tracker will catch it.
        # Let's start message_history empty and let the code populate it. 

        result = ai.process_prompt(user_prompt, max_tool_iterations=max_iterations)

        # Assert
        assert result == last_ai_content 
        assert mock_tool_capable_provider.request.call_count == max_iterations
        assert mock_execute_tool.call_count == max_iterations
        mock_logger.warning.assert_any_call(f"Exceeded maximum tool iterations ({max_iterations}). Returning last assistant content.")
        
        # Check provider's _add_tool_message calls
        assert mock_tool_capable_provider._add_tool_message.call_count == max_iterations
        
        # History should contain: user + (assistant_tool_call + tool_result) * max_iterations
        # Example max_iterations=2: user, assistant1(tool), tool1_result, assistant2(tool), tool2_result = 5 messages
        expected_len = 1 + (max_iterations * 2)
        assert len(message_history) == expected_len, f"History length mismatch. Expected {expected_len}, Got {len(message_history)}. History: {message_history}"
        
        # Check message types and order (optional but good)
        assert message_history[0].role == "user"
        idx = 1
        for i in range(max_iterations):
            assert message_history[idx].role == "assistant"
            assert isinstance(message_history[idx].content, list) # Should contain ToolCall list
            assert message_history[idx+1].role == "tool"
            idx += 2

# Remove tag below 