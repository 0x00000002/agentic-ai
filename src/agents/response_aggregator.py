"""
Response Aggregator component.
Aggregates responses from multiple agents into a coherent final response.
"""
from typing import Dict, Any, List, Optional
from ..core.tool_enabled_ai import AI
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate


class ResponseAggregator:
    """
    Component responsible for aggregating responses from multiple agents
    into a coherent final response for the user.
    """
    
    def __init__(self, 
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 model: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the ResponseAggregator.
        
        Args:
            unified_config: UnifiedConfig instance
            logger: Logger instance
            model: The model to use for aggregation (or None for default)
            prompt_template: PromptTemplate instance (or None to create default)
        """
        self._config = unified_config or UnifiedConfig.get_instance()
        self._logger = logger or LoggerFactory.create(name="response_aggregator")
        
        # Get configuration
        self._agent_config = self._config.get_agent_config("response_aggregator")
        
        # Set up prompt template service (loads YAML automatically)
        self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
        
        # Get system prompt string using PromptTemplate
        system_prompt_str = None
        try:
            # Render the 'response_aggregator' template
            system_prompt_str, _ = self._prompt_template.render_prompt(
                 template_id="response_aggregator"
            ) 
        except ValueError:
            error_msg = "Response Aggregator system prompt template ('response_aggregator') is missing."
            self._logger.error(error_msg)
            system_prompt_str = "You are a Response Aggregator responsible for combining multiple agent responses." # Basic fallback
        except Exception as e:
             error_msg = f"Error loading Response Aggregator system prompt: {e}"
             self._logger.error(error_msg)
             system_prompt_str = "You are a Response Aggregator responsible for combining multiple agent responses." # Basic fallback

        # Set up AI instance for aggregation
        self._ai = AI(
            model=model or self._agent_config.get("default_model"),
            system_prompt=system_prompt_str, # Use loaded/fallback prompt
            logger=self._logger,
            prompt_template=self._prompt_template # Pass template service to AI base
        )
    
    def aggregate_responses(self, 
                           agent_responses: List[Dict[str, Any]], 
                           original_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents.
        
        Args:
            agent_responses: List of agent responses
            original_request: The original request
            
        Returns:
            Aggregated response
            
        Raises:
            AIAgentError: If aggregation fails
        """
        if not agent_responses:
            return {
                "content": "No agents were able to process your request.",
                "agent_id": "response_aggregator",
                "status": "error"
            }
        
        # If only one response, just return it with minimal processing
        if len(agent_responses) == 1:
            response = agent_responses[0]["response"]
            return self._enrich_response(response, [agent_responses[0]["agent_id"]])
        
        # Multiple responses need aggregation
        try:
            # Format responses for the prompt
            responses_text = self._format_responses_for_prompt(agent_responses)
            
            # Prepare template variables
            variables = {
                "prompt": original_request.get("prompt", ""),
                "responses_text": responses_text
            }
            
            # Use prompt template service directly
            prompt, usage_id = self._prompt_template.render_prompt(
                template_id="aggregate_responses",
                variables=variables
            )
            # Let render_prompt raise error if template is missing

            # Get aggregated response
            self._logger.info(f"Aggregating {len(agent_responses)} responses...")
            content = self._ai.request(prompt)
            
            # Create final response
            contributing_agents = [resp.get("agent_id") for resp in agent_responses]
            return {
                "content": content,
                "agent_id": "response_aggregator",
                "contributing_agents": contributing_agents,
                "status": "success"
            }
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to aggregate responses: {str(e)}", agent_id="response_aggregator"),
                self._logger
            )
            self._logger.error(f"Aggregation error: {error_response['message']}")
            
            # Fallback to highest confidence response
            self._logger.info("Falling back to highest confidence response")
            sorted_responses = sorted(agent_responses, key=lambda r: r.get("confidence", 0), reverse=True)
            best_response = sorted_responses[0]["response"]
            return self._enrich_response(best_response, [sorted_responses[0]["agent_id"]])
    
    def _format_responses_for_prompt(self, agent_responses: List[Dict[str, Any]]) -> str:
        """
        Format agent responses for inclusion in the prompt.
        
        Args:
            agent_responses: List of agent responses
            
        Returns:
            Formatted string of responses
        """
        formatted = ""
        for i, resp in enumerate(agent_responses, 1):
            agent_id = resp.get("agent_id", "unknown")
            confidence = resp.get("confidence", 0.0)
            content = resp.get("response", {}).get("content", "No content")
            status = resp.get("status", "unknown")
            
            formatted += f"--- Response {i} (Agent: {agent_id}, Confidence: {confidence:.2f}, Status: {status}) ---\n"
            formatted += f"{content}\n\n"
        
        return formatted
    
    def _enrich_response(self, response: Dict[str, Any], contributing_agents: List[str]) -> Dict[str, Any]:
        """
        Enrich a single response with metadata.
        
        Args:
            response: Original response
            contributing_agents: List of contributing agent IDs
            
        Returns:
            Enriched response
        """
        return {
            "content": response.get("content", ""),
            "agent_id": "response_aggregator",
            "contributing_agents": contributing_agents,
            "status": response.get("status", "success"),
            "metadata": response.get("metadata", {})
        } 