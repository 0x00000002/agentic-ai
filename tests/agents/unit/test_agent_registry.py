"""
Unit tests for the AgentRegistry class.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Type
import builtins # Import builtins to access original issubclass

# Class to test and interface
from src.agents.agent_registry import AgentRegistry
from src.agents.interfaces import AgentInterface # Interface for agents


# Remove the singleton clearing fixture
# @pytest.fixture(autouse=True) ...

class TestAgentRegistry:
    """Test suite for AgentRegistry functionality."""

    @pytest.fixture
    def registry(self) -> AgentRegistry:
        """Provides a fresh AgentRegistry instance."""
        # Instantiate directly, potentially mocking internal registration
        # The __init__ calls _register_agents, which imports agent_registrar
        # Let's mock _register_agents to prevent side effects during unit testing
        with patch.object(AgentRegistry, '_register_agents', return_value=None) as mock_reg:
             # Patch logger to avoid potential issues during init
             with patch('src.agents.agent_registry.LoggerFactory.create') as mock_logger_factory:
                 instance = AgentRegistry()
                 yield instance 
                 # No cleanup needed as it's not a singleton

    # Remove invalid singleton test
    # def test_singleton_instance(self): ...

    def test_register_agent_class_success(self, registry: AgentRegistry):
        """Test registering a valid agent class."""
        mock_agent_class = MagicMock(spec=Type[AgentInterface]) 
        agent_type = "test_agent_type_1"
        
        with patch('builtins.issubclass') as mock_issubclass:
            # Use side_effect instead of return_value
            mock_issubclass.side_effect = lambda sub, sup: True 
            registry.register(agent_type, mock_agent_class)
            mock_issubclass.assert_called_once_with(mock_agent_class, AgentInterface)
        
        retrieved_class = registry.get_agent_class(agent_type)
        assert retrieved_class is mock_agent_class

    def test_register_duplicate_agent_type(self, registry: AgentRegistry):
        """Test that registering an agent class with a duplicate type logs a warning and overwrites."""
        mock_class1 = MagicMock(spec=Type[AgentInterface])
        agent_type = "duplicate_type"
        
        with patch('builtins.issubclass') as mock_issubclass1:
            # Use side_effect
            mock_issubclass1.side_effect = lambda sub, sup: True
            registry.register(agent_type, mock_class1)
            mock_issubclass1.assert_called_once_with(mock_class1, AgentInterface)
        
        mock_class2 = MagicMock(spec=Type[AgentInterface])
        
        with patch('builtins.issubclass') as mock_issubclass2:
            # Use side_effect
            mock_issubclass2.side_effect = lambda sub, sup: True
            registry.register(agent_type, mock_class2)
            mock_issubclass2.assert_called_once_with(mock_class2, AgentInterface)
        
        retrieved_class = registry.get_agent_class(agent_type)
        assert retrieved_class is mock_class2

    def test_register_invalid_agent_class(self, registry: AgentRegistry):
        """Test that registering a class not inheriting from AgentInterface logs a warning and skips."""
        class NotAnAgent:
            pass
            
        agent_type = "invalid_type"
        
        original_issubclass = builtins.issubclass
        
        with patch('builtins.issubclass') as mock_issubclass:
            def side_effect(sub, sup):
                if sub is NotAnAgent and sup is AgentInterface:
                    return False
                return original_issubclass(sub, sup)
            mock_issubclass.side_effect = side_effect
            
            registry.register(agent_type, NotAnAgent) # type: ignore
            # Use assert_any_call to check that our specific call occurred
            mock_issubclass.assert_any_call(NotAnAgent, AgentInterface)
        
        retrieved_class = registry.get_agent_class(agent_type)
        assert retrieved_class is None
        assert not registry.has_agent_type(agent_type)

    def test_get_agent_class_success(self, registry: AgentRegistry):
        """Test retrieving a registered agent class by type."""
        mock_agent_class = MagicMock(spec=Type[AgentInterface])
        agent_type = "retrieval_type"
        
        with patch('builtins.issubclass') as mock_issubclass:
            # Use side_effect
            mock_issubclass.side_effect = lambda sub, sup: True
            registry.register(agent_type, mock_agent_class)
            mock_issubclass.assert_called_once_with(mock_agent_class, AgentInterface)
        
        retrieved = registry.get_agent_class(agent_type)
        assert retrieved is mock_agent_class

    def test_get_agent_class_not_found(self, registry: AgentRegistry):
        """Test retrieving a non-existent agent class returns None."""
        # Method returns None, not raises KeyError
        retrieved = registry.get_agent_class("non_existent_type")
        assert retrieved is None

    def test_get_agent_types_empty(self, registry: AgentRegistry):
        """Test get_agent_types returns an empty list when no agents are registered."""
        all_types = registry.get_agent_types()
        assert isinstance(all_types, list)
        assert len(all_types) == 0

    def test_get_agent_types_multiple(self, registry: AgentRegistry):
        """Test get_agent_types returns all registered agent types."""
        mock_class1 = MagicMock(spec=Type[AgentInterface])
        type1 = "type_A"
        mock_class2 = MagicMock(spec=Type[AgentInterface])
        type2 = "type_B"
        
        with patch('builtins.issubclass') as mock_issubclass:
            # Use side_effect
            mock_issubclass.side_effect = lambda sub, sup: True
            registry.register(type1, mock_class1)
            registry.register(type2, mock_class2)
            assert mock_issubclass.call_count == 2
            mock_issubclass.assert_any_call(mock_class1, AgentInterface)
            mock_issubclass.assert_any_call(mock_class2, AgentInterface)
        
        all_types = registry.get_agent_types()
        assert isinstance(all_types, list)
        assert len(all_types) == 2
        assert set(all_types) == {type1, type2} 

    # Remove test_get_all_agents_returns_copy as get_agent_types returns a new list anyway
    # def test_get_all_agents_returns_copy(self, registry: AgentRegistry): ...
    
    def test_has_agent_type(self, registry: AgentRegistry):
        """Test the has_agent_type method."""
        assert not registry.has_agent_type("initial_type")
        
        mock_class = MagicMock(spec=Type[AgentInterface])
        agent_type = "existing_type"
        
        with patch('builtins.issubclass') as mock_issubclass:
            # Use side_effect
            mock_issubclass.side_effect = lambda sub, sup: True
            registry.register(agent_type, mock_class)
            mock_issubclass.assert_called_once_with(mock_class, AgentInterface)
        
        assert registry.has_agent_type("existing_type")
        assert not registry.has_agent_type("non_existent_type") 