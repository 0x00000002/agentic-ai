"""
Coding Assistant Agent.

This agent is specialized in software development, providing expertise in:
- Code review and best practices
- Performance optimization
- Modern language features
- Testing and deployment strategies
- Documentation and maintainability
"""
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig


class CodingAssistantAgent(BaseAgent):
    """
    An agent specialized in software development assistance.
    
    This agent provides expertise in:
    - Code review and best practices
    - Performance optimization
    - Modern language features
    - Testing and deployment strategies
    - Documentation and maintainability
    """
    
    def __init__(self, 
                 ai_instance=None,
                 unified_config=None,
                 logger=None,
                 **kwargs):
        """
        Initialize the coding assistant agent.
        
        Args:
            ai_instance: AI instance for processing
            unified_config: UnifiedConfig instance
            logger: Logger instance
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(
            ai_instance=ai_instance,
            unified_config=unified_config,
            logger=logger,
            **kwargs
        )
        
        # Set agent-specific configuration
        self.system_prompt = """
        You are an expert software developer with deep knowledge of:
        - Software architecture and design patterns
        - Code review and best practices
        - Performance optimization techniques
        - Modern programming language features
        - Testing and deployment strategies
        - Documentation and maintainability
        
        When responding to queries:
        1. Focus on code quality and maintainability
        2. Consider performance implications
        3. Use modern language features when appropriate
        4. Follow best practices for the specific technology
        5. Provide testing and deployment guidance
        6. Include relevant documentation examples
        """
        
        # Update AI instance with coding-specific system prompt
        if self.ai_instance:
            self.ai_instance.system_prompt = self.system_prompt
        
        self.logger.info("Initialized coding assistant agent")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request asynchronously with coding-specific context.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        # Extract the prompt from the request
        prompt = request.get("prompt", "")
        
        # Add coding-specific context to the prompt
        enhanced_prompt = f"""
        As a coding assistant, please help with the following:
        
        {prompt}
        
        Please provide a detailed response that includes:
        1. Code quality and maintainability considerations
        2. Performance optimization techniques
        3. Modern language features used
        4. Testing and deployment guidance
        5. Documentation examples where relevant
        """
        
        # Create a new request with the enhanced prompt
        enhanced_request = {
            "prompt": enhanced_prompt,
            "conversation_history": request.get("conversation_history", [])
        }
        
        # Process the enhanced request using the base agent's async method
        return await super().process_request(enhanced_request) 