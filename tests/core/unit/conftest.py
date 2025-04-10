import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import logging

from src.core.base_ai import AIBase
from src.core.tool_enabled_ai import ToolEnabledAI


# -----------------------------------------------------------------------------
# Shared fixtures for unit tests
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    """Mock the logger to avoid actual logging during tests."""
    logger = MagicMock()
    # Add all commonly used logger methods
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    logger.exception = MagicMock()
    return logger

@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = MagicMock()
    config.get_conversation_config.return_value = {}
    config.get_default_model.return_value = "test_model"
    config.get_model_config.return_value = {
        "model_id": "test_model",
        "provider": "test_provider"
    }
    config.get_provider_config.return_value = {
        "api_key": "test_key"
    }
    config.get_system_prompt.return_value = "You are a test assistant"
    config.show_thinking = False
    return config

@pytest.fixture
def mock_provider():
    """Mock AI provider with complete tool capabilities."""
    from src.core.models import ProviderResponse
    from src.tools.models import ToolCall
    from src.core.interfaces import ToolCapableProviderInterface
    
    # Create a provider mock that will fulfill the ToolCapableProviderInterface check
    provider = AsyncMock(spec=ToolCapableProviderInterface)
    
    # Basic provider methods
    provider.supports_tools = MagicMock(return_value=True)  # Default to supporting tools
    provider.get_completion = AsyncMock(return_value="Regular response")
    
    # Tool-capable provider methods
    provider.get_completion_with_tools = AsyncMock(return_value={
        "content": "Response with tool calls",
        "tool_calls": []
    })
    
    # Standard provider interface
    provider.request = AsyncMock(return_value=ProviderResponse(
        content="Response content",
        tool_calls=[]
    ))
    
    # Streaming methods
    provider.get_streaming_completion = AsyncMock()
    provider.get_streaming_completion_with_tools = AsyncMock()
    provider.stream = AsyncMock(return_value="Streamed response")
    
    # Tool interface methods
    provider.build_tool_result_messages = AsyncMock(return_value=[
        {"role": "tool", "content": "Tool result", "name": "test_tool"}
    ])
    provider.add_tool_message = AsyncMock(return_value=[])
    
    return provider

@pytest.fixture
def mock_tool_manager():
    """Mock tool manager."""
    manager = MagicMock()
    manager.execute_tool = AsyncMock()
    manager.get_available_tools = MagicMock(return_value={})
    manager.get_all_tools = MagicMock(return_value={})
    return manager

@pytest.fixture
def mock_convo_manager():
    """Mock conversation manager."""
    manager = MagicMock()
    manager.reset = MagicMock()
    manager.get_messages = MagicMock(return_value=[])
    manager.add_user_message = MagicMock()
    manager.add_assistant_message = MagicMock()
    manager.add_system_message = MagicMock()
    manager.add_tool_message = MagicMock()
    return manager

@pytest.fixture
def mock_prompt_template():
    """Mock prompt template."""
    template = MagicMock()
    template.render = MagicMock(return_value="Rendered prompt")
    return template

@pytest.fixture
def basic_ai(mock_config, mock_provider, mock_logger, mock_convo_manager, mock_prompt_template):
    """Create a basic AIBase instance with mocked dependencies."""
    with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager):
        with patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config):
            with patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider):
                ai = AIBase(
                    model="test_model",
                    logger=mock_logger,
                    prompt_template=mock_prompt_template
                )
                yield ai

@pytest.fixture
def tool_enabled_ai(mock_config, mock_provider, mock_tool_manager, mock_logger, 
                   mock_convo_manager, mock_prompt_template):
    """Create a ToolEnabledAI instance with mocked dependencies."""
    with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager):
        with patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config):
            with patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider):
                with patch('src.core.tool_enabled_ai.ToolManager', return_value=mock_tool_manager):
                    with patch('src.core.tool_enabled_ai.UnifiedConfig.get_instance', return_value=mock_config):
                        ai = ToolEnabledAI(
                            model="test_model",
                            logger=mock_logger,
                            tool_manager=mock_tool_manager,
                            prompt_template=mock_prompt_template
                        )
                        # Set up the fixtures for tool-specific tests
                        ai._build_tool_call_message = AsyncMock()
                        ai._build_tool_result_message = AsyncMock()
                        # ai._execute_tool_call = AsyncMock() # REMOVED - We need to test the actual method
                        yield ai 