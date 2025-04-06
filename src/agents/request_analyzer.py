"""
Request Analyzer component.
Analyzes user requests to determine appropriate agents and tools.
"""
from typing import Dict, Any, List, Tuple, Optional, Literal
import json
import re
from ..core.tool_enabled_ai import AI
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate


class RequestAnalyzer:
    """
    Component responsible for analyzing user requests and determining
    which agents and tools should handle them.
    """
    
    def __init__(self, 
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 model: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the RequestAnalyzer.
        
        Args:
            unified_config: UnifiedConfig instance
            logger: Logger instance
            model: The model to use for analysis (or None for default)
            prompt_template: PromptTemplate instance (or None to create default)
        """
        self._config = unified_config or UnifiedConfig.get_instance()
        self._logger = logger or LoggerFactory.create(name="request_analyzer")
        
        # Get configuration
        self._agent_config = self._config.get_agent_config("request_analyzer")
        self._confidence_threshold = self._agent_config.get("confidence_threshold", 0.6)
        
        # Set up prompt template service (loads YAML automatically)
        self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
        
        # Get system prompt string using PromptTemplate
        system_prompt_str = None
        try:
            # Render the 'request_analyzer' template (assumes no variables needed for system prompt)
            system_prompt_str, _ = self._prompt_template.render_prompt(
                 template_id="request_analyzer"
            ) 
        except ValueError:
             # Handle case where system prompt template is missing
            error_msg = "Request Analyzer system prompt template ('request_analyzer') is missing."
            self._logger.error(error_msg)
            # Let's allow it to continue with a fallback for now, but ideally raise error
            # raise ValueError(error_msg) 
            system_prompt_str = "You are a Request Analyzer responsible for analyzing user requests." # Basic fallback
        except Exception as e:
             error_msg = f"Error loading Request Analyzer system prompt: {e}"
             self._logger.error(error_msg)
             system_prompt_str = "You are a Request Analyzer responsible for analyzing user requests." # Basic fallback

        # Set up AI instance for analysis
        self._ai = AI(
            model=model or self._agent_config.get("default_model"),
            system_prompt=system_prompt_str, # Use loaded/fallback prompt
            logger=self._logger,
            prompt_template=self._prompt_template # Pass template service to AI base
        )
    
    def classify_request_intent(self, request: Dict[str, Any]) -> Literal["META", "TASK", "QUESTION", "UNKNOWN"]:
        """
        Classify the user's request intent.

        Args:
            request: The request object containing the prompt.

        Returns:
            The classified intent: "META", "TASK", "QUESTION", or "UNKNOWN" on error.
        """
        prompt_text = request.get("prompt", "")
        if not prompt_text:
            return "UNKNOWN"

        try:
            # Prepare template variables
            variables = {
                "user_prompt": prompt_text
            }
            
            # Use prompt template service directly
            prompt, usage_id = self._prompt_template.render_prompt(
                template_id="classify_intent",
                variables=variables
            )
            # If render_prompt raises an error for missing template, let it propagate

            # Get response from AI
            self._logger.debug(f"Classifying intent for request: {prompt_text[:50]}...")
            response = self._ai.request(prompt).strip().upper()
            self._logger.debug(f"Intent classification response: {response}")

            # Validate response
            if response in ["META", "TASK", "QUESTION"]:
                return response
            else:
                self._logger.warning(f"Unexpected intent classification response: {response}")
                # Attempt a fallback interpretation (e.g., if it contains the word)
                if "META" in response: return "META"
                if "TASK" in response: return "TASK"
                if "QUESTION" in response: return "QUESTION"
                return "UNKNOWN" # Could not reliably classify

        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to classify request intent: {str(e)}", agent_id="request_analyzer"),
                self._logger
            )
            self._logger.error(f"Intent classification error: {error_response['message']}")
            return "UNKNOWN"

    def get_agent_assignments(self, 
                              request: Dict[str, Any], 
                              available_agents: List[str], 
                              agent_descriptions: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        Analyze a request (assumed to be a TASK) to determine appropriate agents.
        
        Args:
            request: The request object
            available_agents: List of available agent IDs
            agent_descriptions: Map of agent IDs to descriptions
            
        Returns:
            List of (agent_id, confidence) tuples sorted by confidence (descending)
            
        Raises:
            AIAgentError: If the analysis fails
        """
        try:
            # Format agent list for template
            agent_list = self._format_agent_list(available_agents, agent_descriptions)
            
            # Prepare template variables
            variables = {
                "prompt": request.get("prompt", ""),
                "agent_list": agent_list,
                "confidence_threshold": self._confidence_threshold
            }
            
            # Use prompt template service directly
            prompt, usage_id = self._prompt_template.render_prompt(
                template_id="analyze_request",
                variables=variables
            )
            # If render_prompt raises an error for missing template, let it propagate

            # Get response from AI
            self._logger.info(f"Analyzing request: {request.get('prompt', '')[:50]}...")
            response = self._ai.request(prompt)
            
            # Parse response
            agents = self._parse_agent_assignments(response)
            
            # Filter by confidence threshold
            filtered_agents = [(agent_id, confidence) 
                              for agent_id, confidence in agents 
                              if confidence >= self._confidence_threshold]
            
            # Sort by confidence (descending)
            sorted_agents = sorted(filtered_agents, key=lambda x: x[1], reverse=True)
            
            self._logger.info(f"Analysis result: {sorted_agents}")
            return sorted_agents
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to analyze request: {str(e)}", agent_id="request_analyzer"),
                self._logger
            )
            self._logger.error(f"Analysis error: {error_response['message']}")
            return []
    
    def analyze_tools(self, 
                     request: Dict[str, Any], 
                     available_tools: List[str], 
                     tool_descriptions: Dict[str, str]) -> List[str]:
        """
        Analyze a request to determine appropriate tools.
        
        Args:
            request: The request object
            available_tools: List of available tool IDs
            tool_descriptions: Map of tool IDs to descriptions
            
        Returns:
            List of tool IDs that should be used
            
        Raises:
            AIAgentError: If the analysis fails
        """
        try:
            # Format tool list for template
            tool_list = self._format_tool_list(available_tools, tool_descriptions)
            
            # Prepare template variables
            variables = {
                "prompt": request.get("prompt", ""),
                "tool_list": tool_list
            }
            
            # Use prompt template service directly
            prompt, usage_id = self._prompt_template.render_prompt(
                template_id="analyze_tools",
                variables=variables
            )
            # If render_prompt raises an error for missing template, let it propagate
            
            # Get response from AI
            self._logger.info(f"Analyzing tools for request: {request.get('prompt', '')[:50]}...")
            response = self._ai.request(prompt)
            
            # Parse response
            tools = self._parse_tool_assignments(response)
            
            self._logger.info(f"Tool analysis result: {tools}")
            return tools
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to analyze tools: {str(e)}", agent_id="request_analyzer"),
                self._logger
            )
            self._logger.error(f"Tool analysis error: {error_response['message']}")
            return []
    
    def _format_agent_list(self, available_agents: List[str], agent_descriptions: Dict[str, str]) -> str:
        """
        Format the agent list for inclusion in a prompt.
        
        Args:
            available_agents: List of available agent IDs
            agent_descriptions: Map of agent IDs to descriptions
            
        Returns:
            Formatted string of agents
        """
        agent_list = ""
        for agent_id in available_agents:
            description = agent_descriptions.get(agent_id, f"Agent type: {agent_id}")
            agent_list += f"- {agent_id}: {description}\n"
        return agent_list
    
    def _format_tool_list(self, available_tools: List[str], tool_descriptions: Dict[str, str]) -> str:
        """
        Format the tool list for inclusion in a prompt.
        
        Args:
            available_tools: List of available tool IDs
            tool_descriptions: Map of tool IDs to descriptions
            
        Returns:
            Formatted string of tools
        """
        tool_list = ""
        for tool_id in available_tools:
            description = tool_descriptions.get(tool_id, f"Tool: {tool_id}")
            tool_list += f"- {tool_id}: {description}\n"
        return tool_list
    
    def _parse_agent_assignments(self, response: str) -> List[Tuple[str, float]]:
        """
        Parse the agent assignment response from the AI.
        
        Args:
            response: AI response string
            
        Returns:
            List of (agent_id, confidence) tuples
        """
        try:
            # Try to parse as JSON list of [agent_id, confidence] pairs
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [(str(item[0]), float(item[1])) for item in parsed 
                       if len(item) == 2 and isinstance(item[1], (int, float))]
            else:
                self._logger.warning(f"Invalid response format, expected list: {response}")
                return []
                
        except json.JSONDecodeError:
            self._logger.warning(f"Failed to parse JSON response: {response}")
            
            # Fallback: Try to extract agent assignments using regex
            pattern = r'["\']([\w_]+)["\'],\s*(0\.\d+)'
            matches = re.findall(pattern, response)
            
            if matches:
                return [(match[0], float(match[1])) for match in matches]
            else:
                return []
        except Exception as e:
            self._logger.error(f"Error parsing agent assignments: {str(e)}")
            return []
    
    def _parse_tool_assignments(self, response: str) -> List[str]:
        """
        Parse the tool assignment response from the AI.
        
        Args:
            response: AI response string
            
        Returns:
            List of tool IDs
        """
        try:
            # Try to parse as JSON list of tool_ids
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, (str, int))]
            else:
                self._logger.warning(f"Invalid response format, expected list: {response}")
                return []
                
        except json.JSONDecodeError:
            self._logger.warning(f"Failed to parse JSON response: {response}")
            
            # Fallback: Try to extract tool IDs using regex
            pattern = r'["\']([\w_]+)["\']'
            matches = re.findall(pattern, response)
            
            if matches:
                return matches
            else:
                return []
        except Exception as e:
            self._logger.error(f"Error parsing tool assignments: {str(e)}")
            return [] 