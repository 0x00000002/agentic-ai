"""
Agent Registrar Module

This module provides functions to register all available agents in the framework.
The separation helps avoid circular imports while ensuring all agents are registered.
"""
from typing import Optional
from src.utils.logger import LoggerInterface


def register_core_agents(registry, logger: Optional[LoggerInterface] = None):
    """
    Register all core agents with the registry.
    
    Args:
        registry: The agent registry to register agents with
        logger: Optional logger for debugging
    """
    if logger:
        logger.info("Registering core agents")
    
    # BaseAgent is imported in registry already
    from .base_agent import BaseAgent
    
    # Register the BaseAgent (if not already registered)
    if not registry.has_agent_type("base"):
        registry.register("base", BaseAgent)
        
    # Register specific agents
    # _register_tool_finder_agent(registry, logger) # Removed - ToolFinderAgent deleted
    _register_coordinator_agent(registry, logger) # Renamed from orchestrator
    _register_listener_agent(registry, logger)
    _register_chat_agent(registry, logger) # Add registration for ChatAgent
    
    if logger:
        agent_types = registry.get_agent_types()
        logger.info(f"Registered agents: {', '.join(agent_types)}")


def _register_coordinator_agent(registry, logger=None):
    """Register the Coordinator agent."""
    try:
        # Import the correct Coordinator class
        from .coordinator import Coordinator
        # Register with type "coordinator"
        if not registry.has_agent_type("coordinator"):
            registry.register("coordinator", Coordinator)
            if logger:
                logger.info("Registered Coordinator")
    except Exception as e:
        if logger:
            logger.error(f"Failed to register Coordinator: {str(e)}")


def _register_listener_agent(registry, logger=None):
    """Register the ListenerAgent."""
    try:
        from .listener_agent import ListenerAgent
        if not registry.has_agent_type("listener"):
            registry.register("listener", ListenerAgent)
            if logger:
                logger.info("Registered ListenerAgent")
    except Exception as e:
        if logger:
            logger.error(f"Failed to register ListenerAgent: {str(e)}")


def _register_chat_agent(registry, logger=None):
    """Register the ChatAgent."""
    try:
        from .chat_agent import ChatAgent  # Assuming the file is chat_agent.py
        if not registry.has_agent_type("chat_agent"):
            registry.register("chat_agent", ChatAgent)
            if logger:
                logger.info("Registered ChatAgent")
    except ImportError:
        if logger:
            logger.warning("ChatAgent not found, skipping registration.") # Log if file doesn't exist
    except Exception as e:
        if logger:
            logger.error(f"Failed to register ChatAgent: {str(e)}")


def register_extension_agents(registry, logger=None):
    """
    Register any extension agents.
    
    Args:
        registry: The agent registry to register agents with
        logger: Optional logger for debugging
    """
    if logger:
        logger.info("Registering extension agents")
    
    # Register the CodingAssistantAgent
    from .coding_assistant_agent import CodingAssistantAgent
    registry.register("coding_assistant", CodingAssistantAgent)
    
    if logger:
        agent_types = registry.get_agent_types()
        logger.info(f"Registered extension agents: {', '.join(agent_types)}") 