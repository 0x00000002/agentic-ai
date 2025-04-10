import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
from enum import Enum

from src.core.tool_enabled_ai import ToolEnabledAI

# Define local Role enum for this test module
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# -----------------------------------------------------------------------------
# Tests for ToolEnabledAI initialization
# -----------------------------------------------------------------------------

class TestToolEnabledAIInit:
    """Tests for the initialization of ToolEnabledAI."""
    
    @pytest.mark.asyncio
    async def test_init_with_defaults(self, tool_enabled_ai, mock_provider):
        """Test initialization with default values."""
        # tool_enabled_ai fixture handles initialization
        ai = tool_enabled_ai
        assert ai._provider == mock_provider
        assert ai._conversation_manager is not None
        assert ai._prompt_template is not None
        assert ai._tool_manager is not None # ToolManager is created by default
        assert ai._tool_history == []
        assert ai._supports_tools is True # Default based on mock provider
        
    def test_init_with_custom_values(self, mock_convo_manager, mock_provider, mock_config, mock_logger, mock_prompt_template):
        """Test that ToolEnabledAI initializes correctly with custom values."""
        # Execute - Initialize directly, mocking dependencies
        with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider), \
             patch('src.core.tool_enabled_ai.ToolManager') as MockToolManager: # Patch ToolManager class
            
            custom_tool_manager = MagicMock() # Create a mock instance to pass
            ai = ToolEnabledAI(
                model="custom-model", # Pass valid AIBase args
                logger=mock_logger,
                tool_manager=custom_tool_manager, # Provide custom manager instance
                prompt_template=mock_prompt_template
                # Removed role and max_tool_iterations (not init params)
                # Removed config and provider (handled by patches)
            )
        
        # Verify
        assert ai._provider == mock_provider
        # assert ai.config == mock_config # Internal AIBase detail
        # assert ai.role == Role.USER # Not an attribute
        assert ai._tool_manager == custom_tool_manager # Check custom manager was used
        assert ai._tool_history == []
        MockToolManager.assert_not_called() # Ensure default wasn't created because we provided one
    
    def test_init_with_tool_manager(self, mock_convo_manager, mock_provider, mock_config, mock_logger, mock_prompt_template):
        """Test that ToolEnabledAI initializes correctly with provided tool manager."""
        # Setup
        mock_tool_manager_instance = MagicMock()
        
        # Execute - Initialize directly, mocking dependencies
        with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider), \
             patch('src.core.tool_enabled_ai.ToolManager') as MockToolManagerClass: # Patch the class
            
            ai = ToolEnabledAI(
                model="test-model", # Basic AIBase args
                logger=mock_logger,
                tool_manager=mock_tool_manager_instance, # Pass the instance
                prompt_template=mock_prompt_template
                # Removed config and provider (handled by patches)
            )
        
        # Verify
        assert ai._tool_manager == mock_tool_manager_instance
        MockToolManagerClass.assert_not_called() # Verify default TM wasn't created

    @pytest.mark.asyncio
    async def test_provider_supports_tools_property(self, tool_enabled_ai, mock_provider, mock_config):
        """Test the _supports_tools attribute reflects provider state during init."""
        # 1. Test TRUE case using the fixture instance
        # Assuming the fixture's mock_provider defaults to supports_tools=True
        assert tool_enabled_ai._supports_tools is True

        # 2. Test FALSE case by setting mock provider BEFORE initialization
        # Create a simple dummy class that is NOT a ToolCapableProviderInterface
        # but has supports_tools = False and a mock request method.
        class DummyProviderNoTools:
            supports_tools = False
            request = AsyncMock()
            # Add other methods/attributes if AIBase init requires them
            
        provider_instance_no_tools = DummyProviderNoTools()

        # Initialize manually with the modified provider
        with patch('src.core.base_ai.ConversationManager'), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=provider_instance_no_tools), \
             patch('src.core.tool_enabled_ai.ToolManager'):
            
            ai_no_tools = ToolEnabledAI(model="test-model-no-tools")
            
        # Assert on the newly created instance
        assert ai_no_tools._supports_tools is False 