import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.models import ProviderResponse, ToolCall

# -----------------------------------------------------------------------------
# Tests for basic request processing (no tool calls)
# -----------------------------------------------------------------------------

class TestToolEnabledAIBasicRequest:
    """Tests for basic request processing in ToolEnabledAI (no tool calls)."""
    
    @pytest.mark.asyncio
    async def test_process_prompt_provider_doesnt_support_tools(
        self, tool_enabled_ai, mock_provider, mock_convo_manager, mock_prompt_template, mock_logger, mocker
    ):
        """Test that process_prompt calls super().request when tools are not supported."""
        # Setup
        prompt = "Hello, world!"
        tool_enabled_ai._supports_tools = False # Explicitly set for clarity
        expected_response = "Basic response string"
        # Mock the base class request method
        mock_super_request = mocker.patch('src.core.base_ai.AIBase.request', new_callable=AsyncMock)
        mock_super_request.return_value = expected_response

        # Execute
        result = await tool_enabled_ai.process_prompt(prompt)
        
        # Verify
        assert result == expected_response
        mock_super_request.assert_called_once_with(prompt)
        mock_provider.request.assert_not_called() # Ensure the ToolEnabledAI provider.request wasn't called
    
    @pytest.mark.asyncio
    async def test_process_prompt_provider_no_tool_manager(
        self, mock_provider, mock_convo_manager, mock_config, mock_logger, mock_prompt_template
    ):
        """Test that process_prompt works with no tool manager (should still function)."""
        # Setup
        prompt = "Tell me a joke"
        expected_content = "Why did the chicken cross the road?"
        provider_response = ProviderResponse(content=expected_content)

        # Mock dependencies for direct init
        with patch('src.core.base_ai.ConversationManager', return_value=mock_convo_manager), \
             patch('src.core.base_ai.UnifiedConfig.get_instance', return_value=mock_config), \
             patch('src.core.base_ai.ProviderFactory.create', return_value=mock_provider), \
             patch('src.core.tool_enabled_ai.ToolManager') as MockToolManager: # Patch ToolManager

            # Create AI with tool_manager=None
            ai = ToolEnabledAI(
                model="test-model", # Basic AIBase args
                logger=mock_logger,
                tool_manager=None, # Explicitly None
                prompt_template=mock_prompt_template
            )
            # Ensure it thinks tools are supported (otherwise it calls super().request)
            ai._supports_tools = True 
            mock_provider.request.return_value = provider_response # Set mock response
            mock_convo_manager.get_messages.return_value = []
        
        # Execute
        result = await ai.process_prompt(prompt)
        
        # Verify
        assert result == expected_content
        mock_provider.request.assert_called_once()
        # Verify tools WERE passed (from default ToolManager)
        call_args = mock_provider.request.call_args
        assert "tools" in call_args[1] # Assert key is present
        # Optionally check value is not None or is a MagicMock
        assert call_args[1]["tools"] is not None 
    
    @pytest.mark.asyncio
    async def test_process_prompt_provider_supports_tools_but_none_returned(
        self, tool_enabled_ai, mock_provider, mock_convo_manager, mock_prompt_template, 
        mock_tool_manager, mock_logger
    ):
        """Test process_prompt when provider supports tools but returns none."""
        # Setup
        prompt = "Tell me about yourself."
        expected_content = "I am an AI assistant."
        provider_response = ProviderResponse(content=expected_content)
        
        tool_enabled_ai._supports_tools = True
        mock_provider.request.return_value = provider_response
        mock_convo_manager.get_messages.return_value = []
        tool_defs = {"tool1": {}, "tool2": {}} # Dummy tool defs
        mock_tool_manager.get_all_tools.return_value = tool_defs
        
        # Execute
        result = await tool_enabled_ai.process_prompt(prompt)
        
        # Verify
        assert result == expected_content
        mock_provider.request.assert_called_once()
        call_args = mock_provider.request.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tool_defs
        assert "tool_choice" in call_args[1]
        mock_tool_manager.execute_tool.assert_not_called() # Check ToolManager wasn't called
    
    @pytest.mark.asyncio
    async def test_process_prompt_with_messages(
        self, tool_enabled_ai, mock_provider, mock_convo_manager, mock_prompt_template, mock_logger, mock_tool_manager
    ):
        """Test process_prompt correctly passes messages from conversation manager."""
        # Setup
        prompt = "Continue our discussion"
        expected_content = "Sure, let's continue"
        provider_response = ProviderResponse(content=expected_content)
        
        tool_enabled_ai._supports_tools = True # Assume tools supported for this flow
        mock_provider.request.return_value = provider_response
        mock_tool_manager.get_all_tools.return_value = {} # Assume no tools for simplicity here
        
        # Execute
        result = await tool_enabled_ai.process_prompt(prompt)
        
        # Verify
        assert result == expected_content
        mock_provider.request.assert_called_once()
        # Verify user prompt was added before provider call
        mock_convo_manager.add_message.assert_any_call(role="user", content=prompt)
        # Verify provider was called with some messages
        call_args, call_kwargs = mock_provider.request.call_args
        assert "messages" in call_kwargs
    
    @pytest.mark.asyncio
    async def test_process_prompt_conversation_update(
        self, tool_enabled_ai, mock_provider, mock_convo_manager, mock_prompt_template, mock_logger, mock_tool_manager
    ):
        """Test process_prompt updates conversation history correctly (no tools called)."""
        # Setup
        prompt = "Hello"
        expected_content = "Hi there"
        provider_response = ProviderResponse(content=expected_content)
        
        tool_enabled_ai._supports_tools = True
        mock_provider.request.return_value = provider_response
        mock_tool_manager.get_all_tools.return_value = {}
        mock_convo_manager.get_messages.return_value = [] # Start fresh
        
        # Execute
        result = await tool_enabled_ai.process_prompt(prompt)
        
        # Verify
        assert result == expected_content
        # Check calls to add_message (should include system, user, assistant)
        # Assuming system prompt is added during init by the fixture
        mock_convo_manager.add_message.assert_has_calls([
            # call(role="system", content=ANY), # Added during AIBase init
            call(role="user", content=prompt),
            call(role="assistant", content=expected_content)
        ]) 