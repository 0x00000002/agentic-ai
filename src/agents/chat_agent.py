"""
Simple Chat Agent Implementation
"""
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface
from ..config.unified_config import UnifiedConfig
from ..core.base_ai import AIBase

class ChatAgent(BaseAgent):
    """
    A simple agent that handles basic chat interactions using its AI instance.
    It primarily relies on the process_request method inherited from BaseAgent.
    """
    def __init__(self, 
                 ai_instance: Optional[AIBase] = None,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 **kwargs):
        """
        Initialize the ChatAgent.
        
        Args:
            ai_instance: AI instance for processing.
            unified_config: UnifiedConfig instance.
            logger: Logger instance.
            **kwargs: Additional arguments passed to BaseAgent.
        """
        # Ensure agent_id is set correctly for BaseAgent's initialization
        super().__init__(ai_instance=ai_instance, 
                         unified_config=unified_config, 
                         logger=logger, 
                         agent_id="chat_agent",  # Explicitly set agent_id
                         **kwargs)
        
        self.logger.info(f"ChatAgent initialized with ID: {self.agent_id}")

    # No need to override process_request unless specific chat logic is required.
    # The BaseAgent.process_request will handle calling the ai_instance. 