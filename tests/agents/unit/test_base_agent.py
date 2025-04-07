"""
Unit tests for the BaseAgent class.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY, call

# Class to test
from src.agents.base_agent import BaseAgent

# Dependencies to mock
from src.core.base_ai import AIBase
from src.tools.tool_manager import ToolManager
from src.config.unified_config import UnifiedConfig
from src.utils.logger import LoggerInterface, LoggerFactory


# --- Test Suite ---

class TestBaseAgent:
    """Test suite for BaseAgent functionality."""

    # Mock dependencies available to all tests in the class
    @pytest.fixture
    def mock_ai_instance(self) -> MagicMock:
        return MagicMock(spec=AIBase)

    @pytest.fixture
    def mock_tool_manager(self) -> MagicMock:
        return MagicMock(spec=ToolManager)

    @pytest.fixture
    def mock_unified_config(self) -> MagicMock:
        mock_config = MagicMock(spec=UnifiedConfig)
        # Default behavior for config calls needed by init
        mock_config.get_agent_config.return_value = {"some_setting": "value"} 
        return mock_config

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        return MagicMock(spec=LoggerInterface)

    # Patch LoggerFactory globally for the class
    @pytest.fixture(autouse=True)
    def patch_logger_factory(self, mock_logger):
        with patch('src.agents.base_agent.LoggerFactory.create', return_value=mock_logger) as mock_create:
            yield mock_create

    # Patch UnifiedConfig.get_instance globally for the class
    @pytest.fixture(autouse=True)
    def patch_config_get_instance(self, mock_unified_config):
         # Only patch get_instance if it's called (i.e., config not passed directly)
         with patch('src.agents.base_agent.UnifiedConfig.get_instance', return_value=mock_unified_config) as mock_get:
             yield mock_get

    # --- __init__ Tests ---

    def test_init_all_args_provided(self, mock_ai_instance, mock_tool_manager, mock_unified_config, mock_logger, 
                                  patch_logger_factory, patch_config_get_instance):
        """Test initialization when all dependencies are explicitly provided."""
        agent_id = "test_agent_all_args"
        
        agent = BaseAgent(
            ai_instance=mock_ai_instance,
            tool_manager=mock_tool_manager,
            unified_config=mock_unified_config,
            logger=mock_logger,
            agent_id=agent_id
        )

        assert agent.ai_instance is mock_ai_instance
        assert agent.tool_manager is mock_tool_manager
        assert agent.config is mock_unified_config
        assert agent.logger is mock_logger
        assert agent.agent_id == agent_id
        mock_unified_config.get_agent_config.assert_called_once_with(agent_id)
        assert agent.agent_config == {"some_setting": "value"}
        # Use the fixture arguments directly
        patch_logger_factory.assert_not_called() 
        patch_config_get_instance.assert_not_called()
        mock_logger.info.assert_called_with(f"Initialized {agent_id} agent")

    def test_init_defaults(self, mock_ai_instance, mock_tool_manager, mock_unified_config, mock_logger, patch_logger_factory, patch_config_get_instance):
        """Test initialization using default dependencies (config, logger, agent_id)."""
        # Only provide AI and ToolManager, others should use defaults
        agent = BaseAgent(
            ai_instance=mock_ai_instance,
            tool_manager=mock_tool_manager
        )

        assert agent.ai_instance is mock_ai_instance
        assert agent.tool_manager is mock_tool_manager
        # Check defaults were used
        assert agent.config is mock_unified_config # Should have called get_instance
        assert agent.logger is mock_logger         # Should have called LoggerFactory.create
        assert agent.agent_id == "base_agent"      # Default agent_id
        
        # Verify get_instance was called
        patch_config_get_instance.assert_called_once()
        # Verify LoggerFactory.create was called with default agent_id
        patch_logger_factory.assert_called_once_with(name="agent.base_agent")
        # Verify config was fetched with default agent_id
        mock_unified_config.get_agent_config.assert_called_once_with("base_agent")
        assert agent.agent_config == {"some_setting": "value"}
        # Verify logger was used
        mock_logger.info.assert_called_with("Initialized base_agent agent")
        
    def test_init_agent_config_not_found(self, mock_unified_config, patch_config_get_instance):
         """Test initialization when agent config is not found for the ID."""
         mock_unified_config.get_agent_config.return_value = None # Simulate not found
         agent_id = "no_config_agent"
         
         agent = BaseAgent(agent_id=agent_id, unified_config=mock_unified_config)
         
         mock_unified_config.get_agent_config.assert_called_once_with(agent_id)
         assert agent.agent_config == {} # Should default to empty dict 

    # --- process_request Tests (Core Logic) ---

    def test_process_request_success_dict_input(self, mock_ai_instance, mock_logger):
        """Test basic successful request processing with a dict input."""
        agent = BaseAgent(ai_instance=mock_ai_instance, logger=mock_logger)
        prompt = "Test prompt content"
        request_dict = {"prompt": prompt}
        ai_response_content = "Successful AI response"
        mock_ai_instance.request.return_value = ai_response_content
        
        response = agent.process_request(request_dict)
        
        mock_logger.info.assert_any_call(f"Processing request with {agent.agent_id} agent")
        mock_ai_instance.request.assert_called_once_with(prompt)
        assert response["content"] == ai_response_content
        assert response["agent_id"] == agent.agent_id
        assert response["status"] == "success"

    def test_process_request_success_str_input(self, mock_ai_instance, mock_logger):
        """Test basic successful request processing with a string input."""
        agent = BaseAgent(ai_instance=mock_ai_instance, logger=mock_logger)
        prompt = "Test prompt as string"
        ai_response_content = "Successful AI response from string"
        mock_ai_instance.request.return_value = ai_response_content
        
        response = agent.process_request(prompt) # Pass string directly
        
        mock_logger.info.assert_any_call(f"Processing request with {agent.agent_id} agent")
        mock_ai_instance.request.assert_called_once_with(prompt)
        assert response["content"] == ai_response_content
        assert response["agent_id"] == agent.agent_id
        assert response["status"] == "success"

    def test_process_request_no_ai_instance(self, mock_logger):
        """Test processing when ai_instance is None."""
        agent = BaseAgent(ai_instance=None, logger=mock_logger)
        request = {"prompt": "Test prompt"}
        
        response = agent.process_request(request)
        
        mock_logger.info.assert_any_call(f"Processing request with {agent.agent_id} agent")
        mock_logger.warning.assert_called_once_with("No AI instance available for processing")
        assert "Error: No AI instance available" in response["content"]
        assert response["agent_id"] == agent.agent_id
        assert response["status"] == "error"

    def test_process_request_ai_error(self, mock_ai_instance, mock_logger):
        """Test processing when ai_instance.request raises an exception."""
        agent = BaseAgent(ai_instance=mock_ai_instance, logger=mock_logger)
        request = {"prompt": "Test prompt"}
        error_message = "AI failed!"
        mock_ai_instance.request.side_effect = Exception(error_message)
        
        response = agent.process_request(request)
        
        mock_logger.info.assert_any_call(f"Processing request with {agent.agent_id} agent")
        mock_logger.error.assert_called_once_with(f"Error processing request: {error_message}")
        assert f"Error: {error_message}" in response["content"]
        assert response["agent_id"] == agent.agent_id
        assert response["status"] == "error"
        assert response["error"] == error_message

    # --- process_request Tests (Overrides) ---

    def test_process_request_model_override_success(self, mock_ai_instance, mock_unified_config, mock_logger):
        """Test successful model override when request specifies a different model key."""
        agent_id = "override_agent"
        original_model_key = "original-model"
        original_model_api_id = "original-model-api-id"
        override_model_key = "override-model"
        prompt = "Test override"
        request = {"prompt": prompt, "model": override_model_key}
        ai_response = "Response from overridden model"

        # Configure original AI instance mock
        mock_ai_instance.get_model_info.return_value = {"model_id": original_model_api_id}
        mock_ai_instance.get_system_prompt.return_value = "Original System Prompt"
        mock_ai_instance._logger = MagicMock() # Give it a mock logger
        mock_ai_instance._request_id = None
        mock_ai_instance._prompt_template = None
        
        # Configure config mock to map API ID back to original key
        mock_unified_config.get_all_models.return_value = {
            original_model_key: {"model_id": original_model_api_id, "provider": "p1"},
            override_model_key: {"model_id": "override-api-id", "provider": "p2"}
        }
        
        # Mock the AI class itself to capture instantiation
        MockAIClass = MagicMock(spec=AIBase)
        mock_overridden_instance = MagicMock(spec=AIBase)
        mock_overridden_instance.request.return_value = ai_response
        MockAIClass.return_value = mock_overridden_instance # Instantiation returns this mock
        mock_ai_instance.__class__ = MockAIClass # Make original instance's class the mock

        # Initialize agent with the original mock instance
        agent = BaseAgent(ai_instance=mock_ai_instance, unified_config=mock_unified_config, logger=mock_logger, agent_id=agent_id)

        # --- Act ---
        response = agent.process_request(request)

        # --- Assert ---
        # Check config was queried to find the original model key
        mock_unified_config.get_all_models.assert_called_once()
        # Check a new AI instance was created with the override key
        MockAIClass.assert_called_once_with(
            model=override_model_key,
            system_prompt="Original System Prompt",
            logger=ANY, # Check logger was passed
            request_id=None,
            prompt_template=None
        )
        # Check the overridden instance's request method was called
        mock_overridden_instance.request.assert_called_once_with(prompt)
        # Check the response came from the overridden instance
        assert response["content"] == ai_response
        assert response["status"] == "success"
        # Check the original AI instance was restored
        assert agent.ai_instance is mock_ai_instance
        mock_logger.info.assert_any_call("Restoring original AI instance.")
        
    def test_process_request_model_override_skipped_same_key(self, mock_ai_instance, mock_unified_config, mock_logger):
        """Test model override is skipped if request specifies the same model key."""
        agent_id = "override_agent_same"
        model_key = "same-model"
        model_api_id = "same-model-api-id"
        prompt = "Test no override"
        request = {"prompt": prompt, "model": model_key} # Same key as original
        ai_response = "Response from original model"
        
        mock_ai_instance.get_model_info.return_value = {"model_id": model_api_id}
        mock_ai_instance.request.return_value = ai_response
        # Mock the AI class but expect it NOT to be called
        MockAIClass = MagicMock(spec=AIBase)
        mock_ai_instance.__class__ = MockAIClass
        
        mock_unified_config.get_all_models.return_value = {
            model_key: {"model_id": model_api_id, "provider": "p1"}
        }
        
        agent = BaseAgent(ai_instance=mock_ai_instance, unified_config=mock_unified_config, logger=mock_logger, agent_id=agent_id)
        response = agent.process_request(request)
        
        # Assert new instance was NOT created
        MockAIClass.assert_not_called()
        # Assert original instance's request was called
        mock_ai_instance.request.assert_called_once_with(prompt)
        assert response["content"] == ai_response
        assert response["status"] == "success"
        assert agent.ai_instance is mock_ai_instance # Should not have changed

    def test_process_request_model_override_skipped_id_not_found(self, mock_ai_instance, mock_unified_config, mock_logger):
        """Test model override is skipped if original model ID isn't in config."""
        agent_id = "override_agent_not_found"
        original_model_api_id = "original-model-api-id-not-in-config"
        override_model_key = "override-model"
        prompt = "Test override not found"
        request = {"prompt": prompt, "model": override_model_key}
        ai_response = "Response from original model"

        mock_ai_instance.get_model_info.return_value = {"model_id": original_model_api_id}
        mock_ai_instance.request.return_value = ai_response
        MockAIClass = MagicMock(spec=AIBase)
        mock_ai_instance.__class__ = MockAIClass
        
        # Config mock doesn't contain the original_model_api_id
        mock_unified_config.get_all_models.return_value = {
            override_model_key: {"model_id": "override-api-id", "provider": "p2"}
        }
        
        agent = BaseAgent(ai_instance=mock_ai_instance, unified_config=mock_unified_config, logger=mock_logger, agent_id=agent_id)
        response = agent.process_request(request)
        
        MockAIClass.assert_not_called() # No override should happen
        mock_ai_instance.request.assert_called_once_with(prompt)
        assert response["content"] == ai_response
        assert response["status"] == "success"
        mock_logger.error.assert_called_once_with(f"Could not find matching short key for original model API ID '{original_model_api_id}'. Cannot safely determine if override is needed. Skipping override.")
        assert agent.ai_instance is mock_ai_instance

    def test_process_request_system_prompt_override_success(self, mock_ai_instance, mock_logger):
        """Test successful system prompt override."""
        agent_id = "sys_prompt_agent"
        original_prompt = "Original System Prompt"
        override_prompt = "Override System Prompt"
        request = {"prompt": "Test prompt", "system_prompt": override_prompt}
        ai_response = "Response with overridden system prompt"

        # Side effect to allow restoration logic to trigger
        prompt_return_sequence = [
            original_prompt,  # Call 1 (in try)
            override_prompt,  # Call 2 (current value in finally)
            original_prompt   # Call 3 (original value in finally)
        ]
        mock_ai_instance.get_system_prompt.side_effect = prompt_return_sequence
        
        mock_ai_instance.request.return_value = ai_response
        mock_ai_instance.set_system_prompt.reset_mock()
        
        agent = BaseAgent(ai_instance=mock_ai_instance, logger=mock_logger, agent_id=agent_id)
        response = agent.process_request(request)

        # Basic assertions
        mock_ai_instance.request.assert_called_once_with("Test prompt")
        assert response["content"] == ai_response
        assert response["status"] == "success"
        assert mock_ai_instance.get_system_prompt.call_count == 3 
        
        # Check set_system_prompt call count 
        assert mock_ai_instance.set_system_prompt.call_count == 2 
        # Check that the override call was made
        mock_ai_instance.set_system_prompt.assert_any_call(override_prompt)
        
        # Verify restoration occurred by checking:
        # 1. Total set_system_prompt calls = 2 (already checked above)
        # 2. One of those calls was with override_prompt (already checked above)
        # 3. A log message about restoration exists
        
        # Find any log message containing "Restoring original system prompt"
        restoration_logs = [
            call_args[0][0] for call_args in mock_logger.info.call_args_list 
            if isinstance(call_args[0][0], str) and "Restoring original system prompt" in call_args[0][0]
        ]
        assert restoration_logs, "Expected a log message about restoring the system prompt"

    def test_process_request_system_prompt_override_skipped_same(self, mock_ai_instance, mock_logger):
        """Test system prompt override is skipped if prompt is the same."""
        agent_id = "sys_prompt_agent_same"
        original_prompt = "Same System Prompt"
        request = {"prompt": "Test prompt", "system_prompt": original_prompt} # Same prompt
        ai_response = "Response with original system prompt"

        mock_ai_instance.get_system_prompt.return_value = original_prompt
        mock_ai_instance.request.return_value = ai_response
        
        agent = BaseAgent(ai_instance=mock_ai_instance, logger=mock_logger, agent_id=agent_id)
        response = agent.process_request(request)

        mock_ai_instance.set_system_prompt.assert_not_called()
        # Expect 3 calls now
        assert mock_ai_instance.get_system_prompt.call_count == 3 
        mock_ai_instance.request.assert_called_once_with("Test prompt")
        assert response["content"] == ai_response
        assert response["status"] == "success"
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Restoring original system prompt" not in log_calls

    # --- can_handle Tests ---
    def test_can_handle_default(self, mock_ai_instance):
        """Test the default can_handle implementation returns low confidence."""
        agent = BaseAgent(ai_instance=mock_ai_instance)
        request = {"prompt": "Any request"}
        confidence = agent.can_handle(request)
        assert confidence == 0.1

    # --- _enrich_response Tests ---
    def test_enrich_response_adds_defaults(self, mock_ai_instance):
        """Test _enrich_response adds default agent_id and status if missing."""
        agent = BaseAgent(ai_instance=mock_ai_instance, agent_id="enrich_test_agent")
        base_response = {"content": "Some content"}
        enriched = agent._enrich_response(base_response)
        
        assert enriched["content"] == "Some content"
        assert enriched["agent_id"] == "enrich_test_agent"
        assert enriched["status"] == "success"

    def test_enrich_response_preserves_existing(self, mock_ai_instance):
        """Test _enrich_response does not overwrite existing agent_id or status."""
        agent = BaseAgent(ai_instance=mock_ai_instance, agent_id="enrich_test_agent")
        base_response = {
            "content": "Some content",
            "agent_id": "original_agent",
            "status": "error"
        }
        enriched = agent._enrich_response(base_response.copy()) # Pass copy
        
        assert enriched["content"] == "Some content"
        assert enriched["agent_id"] == "original_agent" # Preserved
        assert enriched["status"] == "error"         # Preserved