"""
Base agent implementation for the multi-agent architecture.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig
# Import ToolEnabledAI for type checking and history retrieval
from ..core.tool_enabled_ai import ToolEnabledAI 

if TYPE_CHECKING:
    from ..core.base_ai import AIBase
    from ..tools.tool_manager import ToolManager


class BaseAgent:
    """Base implementation for all agents."""
    
    def __init__(self, 
                 ai_instance: Optional['AIBase'] = None,
                 tool_manager: Optional['ToolManager'] = None,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 agent_id: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            ai_instance: AI instance for processing
            tool_manager: Tool manager for handling tools
            unified_config: UnifiedConfig instance
            logger: Logger instance
            agent_id: Agent identifier
        """
        self.ai_instance = ai_instance
        self.tool_manager = tool_manager
        self.config = unified_config or UnifiedConfig.get_instance()
        self.agent_id = agent_id or "base_agent"
        self.logger = logger or LoggerFactory.create(name=f"agent.{self.agent_id}")
        
        # Get agent-specific configuration
        self.agent_config = self.config.get_agent_config(self.agent_id) or {}
        
        self.logger.debug(f"Initialized {self.agent_id} agent")
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request asynchronously with this agent.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        self.logger.info(f"Processing request with {self.agent_id} agent (async)")
        
        # Handle string input first
        if isinstance(request, str):
            prompt = request
            model_override = None
            system_prompt_override = None
        elif isinstance(request, dict):
            prompt = request.get("prompt", str(request)) # Default to str(dict) if no prompt
            # Check for model override in the request
            model_override = request.get("model")
            system_prompt_override = request.get("system_prompt")
        else:
             # Handle other unexpected types gracefully
             prompt = str(request)
             model_override = None
             system_prompt_override = None
             self.logger.warning(f"Received request of unexpected type: {type(request)}. Treating as string.")

        original_ai_instance = self.ai_instance # Store original instance
        restored = False

        try:
            # Apply model override if specified
            if model_override and self.ai_instance:
                original_model_info = self.ai_instance.get_model_info()
                original_model_api_id = original_model_info.get("model_id")
                original_model_short_key = original_model_info.get("short_key")
                
                # Simplified override logic - direct comparison of keys
                if original_model_short_key:
                    # Short key is available, use direct comparison
                    if model_override != original_model_short_key:
                        # Override is needed because short keys differ
                        self.logger.debug(f"Overriding model: {original_model_short_key} -> {model_override}")
                        self.ai_instance = self.ai_instance.__class__(
                            model=model_override,
                            system_prompt=original_ai_instance.get_system_prompt(), 
                            logger=original_ai_instance._logger, 
                            request_id=original_ai_instance._request_id, 
                            prompt_template=original_ai_instance._prompt_template 
                        )
                    # else: short keys match, no override needed
                else:
                    # Short key is not available, this indicates a potential config issue
                    self.logger.warning(f"Model info missing short key for API ID '{original_model_api_id}'. Cannot safely determine if override is needed. Skipping override.")
            
            # Apply system prompt override if specified
            if system_prompt_override is not None and self.ai_instance:
                # Check if it differs from the current system prompt
                if system_prompt_override != self.ai_instance.get_system_prompt(): 
                    self.logger.debug("Applying system prompt override from request")
                    self.ai_instance.set_system_prompt(system_prompt_override)
                
            # Extract prompt from request
            if isinstance(request, dict) and "prompt" in request:
                prompt = request["prompt"]
            else:
                prompt = str(request)
                
            # Process with AI instance
            response_content = ""
            tool_history = None
            
            if self.ai_instance:
                # Check if the AI instance supports the tool loop
                if isinstance(self.ai_instance, ToolEnabledAI):
                    self.logger.debug(f"Using ToolEnabledAI process_prompt for agent {self.agent_id}")
                    # Use process_prompt for tool loop handling
                    response_content = await self.ai_instance.process_prompt(prompt)
                    # Get tool history after processing
                    tool_history = self.ai_instance.get_tool_history()
                    if tool_history:
                        self.logger.debug(f"Retrieved {len(tool_history)} tool history entries.")
                else:
                    # Fallback to basic request for non-tool-enabled AI
                    self.logger.debug(f"Using basic AI request for agent {self.agent_id}")
                    response_content = await self.ai_instance.request(prompt)
                    
                response = {
                        "content": response_content,
                        "agent_id": self.agent_id,
                        "status": "success",
                        # Add raw tool history to metadata if available
                        "metadata": {"tool_history": tool_history} if tool_history else {}
                    }
            else:
                self.logger.warning("No AI instance available for processing")
                response = {
                    "content": "Error: No AI instance available for processing",
                    "agent_id": self.agent_id,
                    "status": "error"
                }

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            response = {
                "content": f"Error: {str(e)}",
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e)
            }
        finally:
            # Restore original AI instance if it was overridden
            if self.ai_instance is not original_ai_instance:
                self.logger.debug("Restoring original AI instance.")
                self.ai_instance = original_ai_instance
                restored = True
            # Ensure system prompt is reset if it was overridden AND we didn't restore the whole instance    
            elif system_prompt_override is not None and not restored:
                 original_system_prompt = original_ai_instance.get_system_prompt()
                 current_prompt = self.ai_instance.get_system_prompt()
                 if current_prompt != original_system_prompt:
                    self.logger.debug(f"Restoring original system prompt (from '{current_prompt}' to '{original_system_prompt}')")
                    self.ai_instance.set_system_prompt(original_system_prompt)

        return self._enrich_response(response)
    
    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            request: The request object
            
        Returns:
            Confidence score (0.0-1.0) indicating ability to handle
        """
        # Base implementation returns low confidence
        # Specialized agents should override this
        return 0.1
    
    def _enrich_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a response with additional metadata.
        
        Args:
            response: The response to enrich
            
        Returns:
            Enriched response with metadata
        """
        # Add agent metadata if not present
        if "agent_id" not in response:
            response["agent_id"] = self.agent_id
        
        # Add status if not present
        if "status" not in response:
            response["status"] = "success"
        
        return response