"""
Unit tests for the agent registrar functions.
"""

import pytest
from unittest.mock import MagicMock, patch, call

# Functions to test
from src.agents.agent_registrar import register_core_agents, register_extension_agents

# Import Interface and Concrete Class
from src.agents.interfaces import AgentRegistryInterface
from src.agents.agent_registry import AgentRegistry # Import concrete class

# --- Test Suite ---

class TestAgentRegistrar:
    """Tests for agent registration functions."""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Provides a mock AgentRegistry instance."""
        # Mock the CONCRETE class instead of the interface
        mock = MagicMock(spec=AgentRegistry) 
        # Now has_agent_type should exist on the spec
        mock.has_agent_type.return_value = False
        return mock

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Provides a mock LoggerInterface."""
        return MagicMock()

    def test_register_core_agents(self, mock_registry: MagicMock, mock_logger: MagicMock):
        """Verify register_core_agents calls registry.register correctly."""
        # Patch the agent classes where they are defined
        with patch('src.agents.base_agent.BaseAgent') as MockBaseAgent, \
             patch('src.agents.coordinator.Coordinator') as MockCoordinator, \
             patch('src.agents.listener_agent.ListenerAgent') as MockListenerAgent, \
             patch('src.agents.chat_agent.ChatAgent') as MockChatAgent:
            
            # Assume has_agent_type always returns False so all agents are registered
            mock_registry.has_agent_type.return_value = False
            
            # Ensure the mocks passed to register are the ones from the patch context
            register_core_agents(mock_registry, mock_logger)

            # Check expected calls to registry.register
            expected_calls = [
                call("base", MockBaseAgent),        # Assuming base isn't registered yet
                call("coordinator", MockCoordinator),
                call("listener", MockListenerAgent),
                call("chat_agent", MockChatAgent)   # Add expected call for chat_agent
            ]
            
            # Check if all expected calls were made, regardless of order
            mock_registry.register.assert_has_calls(expected_calls, any_order=True)
            assert mock_registry.register.call_count == len(expected_calls)

            # Check has_agent_type calls (should be called for each before registering)
            mock_registry.has_agent_type.assert_any_call("base")
            mock_registry.has_agent_type.assert_any_call("coordinator")
            mock_registry.has_agent_type.assert_any_call("listener")
            mock_registry.has_agent_type.assert_any_call("chat_agent")

            # Check logger calls (optional)
            mock_logger.info.assert_any_call("Registering core agents")
            mock_logger.info.assert_any_call("Registered Coordinator")
            mock_logger.info.assert_any_call("Registered ListenerAgent")
            mock_logger.info.assert_any_call("Registered ChatAgent")

    def test_register_core_agents_already_registered(self, mock_registry: MagicMock, mock_logger: MagicMock):
        """Verify register_core_agents skips registration if agent type already exists."""
        # Simulate all agents being pre-registered
        mock_registry.has_agent_type.return_value = True 
        
        # Patch the original locations
        with patch('src.agents.base_agent.BaseAgent'), \
             patch('src.agents.coordinator.Coordinator'), \
             patch('src.agents.listener_agent.ListenerAgent'):
            
            register_core_agents(mock_registry, mock_logger)

            # Assert that register was NOT called because has_agent_type returned True
            mock_registry.register.assert_not_called()

            # Check has_agent_type was still called
            mock_registry.has_agent_type.assert_any_call("base")
            mock_registry.has_agent_type.assert_any_call("coordinator")
            mock_registry.has_agent_type.assert_any_call("listener")
            
    def test_register_extension_agents(self, mock_registry: MagicMock, mock_logger: MagicMock):
        """Verify register_extension_agents calls registry.register correctly."""
        # Patch the original location
        with patch('src.agents.coding_assistant_agent.CodingAssistantAgent') as MockCodingAgent:
            
            register_extension_agents(mock_registry, mock_logger)

            # Check expected call, using the mock from the patch context
            mock_registry.register.assert_called_once_with("coding_assistant", MockCodingAgent)
            
            # Check logger call (optional)
            mock_logger.info.assert_any_call("Registering extension agents") 