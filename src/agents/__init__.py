"""
Agents Package

This package provides the multi-agent architecture for the Agentic-AI framework.
"""

# Base interfaces and abstract classes
from .interfaces import AgentInterface, AgentRegistryInterface, AgentFactoryInterface
from .base_agent import BaseAgent

# Core components
from .agent_registry import AgentRegistry
from .agent_factory import AgentFactory

# Agent implementations - imported last to avoid circular imports
# from .tool_finder_agent import ToolFinderAgent # Removed - ToolFinderAgent deleted
from .coordinator import Coordinator
from .listener_agent import ListenerAgent
from .coding_assistant_agent import CodingAssistantAgent

__all__ = [
    'AgentInterface',
    'AgentRegistryInterface',
    'AgentFactoryInterface',
    'BaseAgent',
    'AgentRegistry',
    'AgentFactory',
    'Coordinator',
    # 'ToolFinderAgent', # Removed - ToolFinderAgent deleted
    'ListenerAgent',
    'CodingAssistantAgent'
]