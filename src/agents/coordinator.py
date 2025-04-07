from typing import Dict, Any, Optional, List, Tuple
import json
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig
from .request_analyzer import RequestAnalyzer # To classify intent/use_case/agents
from .response_aggregator import ResponseAggregator # To combine results
from ..tools.tool_manager import ToolManager # To get tool info for context
# Import ToolDefinition for type checking in _get_available_tools_info
from ..tools.models import ToolDefinition
# Ensure AgentFactory is correctly imported (adjust path if needed)
from .agent_factory import AgentFactory # Crucial for creating agents to delegate to
from ..exceptions import AIAgentError, ErrorHandler # Standard error handling
from .agent_registry import AgentRegistry # To get agent registry
class Coordinator(BaseAgent):
    """
    Coordinates the workflow between specialized agents based on request analysis.
    Routes requests, delegates execution, and manages aggregation.
    Version: 2.0
    """
    def __init__(self, 
                 agent_factory: Optional[AgentFactory] = None,
                 request_analyzer: Optional[RequestAnalyzer] = None,
                 response_aggregator: Optional[ResponseAggregator] = None,
                 tool_manager: Optional[ToolManager] = None,
                 agent_registry: Optional[AgentRegistry] = None,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 **kwargs):

        # Use 'coordinator' as agent_id for config/logging - agent_id is passed via kwargs
        # The agent_id used here will be whatever was passed in kwargs by the factory
        super().__init__(unified_config=unified_config, logger=logger, **kwargs)

        # --- Dependencies (Inject or Create Defaults) ---
        # ALWAYS create a standard logger for dependencies to ensure exc_info support
        dep_logger = LoggerFactory.create(f"coordinator_deps")
        dep_logger.info(f"Coordinator '{self.agent_id}' initializing dependencies with standard logger.")

        # Create Registry first if not provided
        self.agent_registry = agent_registry or AgentRegistry()
        dep_logger.info(f"Coordinator using AgentRegistry: {self.agent_registry}")
        
        # Now create Factory using the Registry and the standard dependency logger
        self.agent_factory = agent_factory or AgentFactory(registry=self.agent_registry, unified_config=self.config, logger=dep_logger)
        dep_logger.info(f"Coordinator using AgentFactory: {self.agent_factory}")

        # Other dependencies, ensuring they use the standard dependency logger
        # Use the refactored RequestAnalyzer V2 (simpler __init__)
        self.request_analyzer = request_analyzer or RequestAnalyzer(unified_config=self.config, logger=dep_logger)
        self.response_aggregator = response_aggregator or ResponseAggregator(unified_config=self.config, logger=dep_logger)
        self.tool_manager = tool_manager or ToolManager(unified_config=self.config, logger=dep_logger)
        
        # --- Configuration ---
        # Agent config comes from BaseAgent __init__ calling self.config.get_agent_config(self.agent_id)
        self.max_parallel_agents = self.agent_config.get("max_parallel_agents", 1)
        # Agent ID to use for handling simple Q&A or fallbacks
        self.default_handler_agent_id = self.agent_config.get("default_handler_agent", "chat_agent") # e.g., chat_agent
        
        self.logger.info(f"Coordinator initialized. Default handler agent: '{self.default_handler_agent_id}'")

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing user requests.
        Routes based on classified intent.
        """
        self.logger.info(f"--- Coordinator process_request starting ---")
        self.logger.debug(f"Received request: {request}")

        try:
            # --- 1. Classify Intent ---
            # Use a fresh call to RequestAnalyzer - avoid shared state issues
            # Pass the *original* request for classification
            # Re-create analyzer to ensure it uses its own config/state if needed
            analyzer = RequestAnalyzer(unified_config=self.config, logger=self.logger)
            intent = analyzer.classify_request_intent(request)
            self.logger.info(f"Request Analyzer classified intent as: '{intent}'")

            # --- 2. Route based on Intent ---
            if intent == "META":
                # Handle directly, no LLM needed for this
                return self._handle_meta_request(request)
            elif intent == "QUESTION":
                # Delegate simple questions to a default agent
                return self._handle_delegated_request(request, self.default_handler_agent_id, "QUESTION")
            elif intent == "TASK":
                # Handle complex tasks involving multiple potential agents
                return self._handle_task_request(request)
            else: # Includes "UNKNOWN"
                self.logger.warning(f"Unknown or failed intent classification ('{intent}'). Handling via default agent.")
                # Fallback to default handler agent
                return self._handle_delegated_request(request, self.default_handler_agent_id, "UNKNOWN_FALLBACK")

        except Exception as e:
            # Log with traceback
            err_logger = LoggerFactory.create("coordinator_process_error")
            err_logger.error(f"Critical error during Coordinator process_request: {e}", exc_info=True) # RESTORED exc_info
            return self._create_error_response(f"Coordination failed due to unexpected error: {e}")

    # --- Intent Handlers ---

    def _handle_meta_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles META requests directly by formatting available config info."""
        self.logger.info("Handling META request directly...")
        try:
            agents_info = self._get_available_agents_info()
            tools_info = self._get_available_tools_info()

            # Simple, direct formatting
            response_content = f"System Configuration Information:\n\nAvailable Agents:\n{agents_info}\n\nAvailable Tools:\n{tools_info}"

            return {
                "content": response_content,
                "agent_id": self.agent_id, # Report as coordinator
                "status": "success",
                "metadata": {"intent": "META", "handler": "direct_format"}
            }
        except Exception as e:
            # Log with traceback
            err_logger = LoggerFactory.create("coordinator_meta_error")
            err_logger.error(f"Error retrieving info for META request: {e}", exc_info=True) # RESTORED exc_info
            return self._create_error_response(f"Failed to retrieve system info for META request: {e}")

    def _handle_delegated_request(self, request: Dict[str, Any], agent_id: str, handling_reason: str) -> Dict[str, Any]:
        """Handles simple QUESTION or fallback requests by delegating to a specified agent."""
        self.logger.info(f"Delegating request to single agent '{agent_id}' (Reason: {handling_reason})")
        if not self.agent_factory:
            return self._create_error_response("Agent factory not configured, cannot delegate request.")

        try:
            agent = self.agent_factory.create(agent_id)
            if not agent:
                 return self._create_error_response(f"Could not create agent '{agent_id}' for delegation.")

            # Process request using the chosen agent
            agent_response = agent.process_request(request.copy()) # Pass a copy

            # Ensure response is a dict (basic normalization)
            if not isinstance(agent_response, dict):
                 agent_response = {"content": str(agent_response), "status": "success"}
            
            # Add coordinator metadata
            if "metadata" not in agent_response: agent_response["metadata"] = {}
            agent_response["metadata"]["delegated_to"] = agent_id
            agent_response["metadata"]["delegation_reason"] = handling_reason
            agent_response["metadata"]["coordinator_id"] = self.agent_id # Add coordinator id

            return agent_response

        except Exception as e:
            # Log with traceback
            err_logger = LoggerFactory.create("coordinator_delegate_error")
            err_logger.error(f"Error delegating request to agent '{agent_id}': {e}", exc_info=True) # RESTORED exc_info
            return self._create_error_response(f"Failed during delegated handling by '{agent_id}': {e}")

    def _handle_task_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles TASK requests potentially involving multiple agents and aggregation."""
        self.logger.info("Handling TASK request...")
        if not self.agent_factory or not self.request_analyzer or not self.response_aggregator:
             return self._create_error_response("Required components (AgentFactory, RequestAnalyzer, ResponseAggregator) not available.")

        agent_responses = []
        try:
            # --- 1. Analyze Task for Agent Assignment ---
            # TODO: Re-implement agent assignment logic if needed (maybe using LLM call)
            # For now, assume TASK always goes to default handler or needs specific logic here.
            # Let's simply delegate to default for now to test the flow.
            self.logger.warning("Agent assignment logic for TASK not yet implemented in Coordinator V2. Delegating to default handler.")
            return self._handle_delegated_request(request, self.default_handler_agent_id, "TASK_DEFAULT_DELEGATION")

            # --- OLD V1 Logic Below (Requires Re-evaluation/Implementation) ---
            # # Use a fresh analyzer instance if concerned about state, or reuse self.request_analyzer
            # analyzer = RequestAnalyzer(unified_config=self.config, logger=self.logger)
            # available_agents = self.agent_factory.registry.get_all_agents() if hasattr(self.agent_factory, 'registry') else []
            # agent_descriptions = self.config.get_agent_descriptions() if self.config else {}
            # agent_assignments = analyzer.get_agent_assignments(
            #     request=request,
            #     available_agents=available_agents,
            #     agent_descriptions=agent_descriptions
            # )
            # self.logger.info(f"Task assigned to agents: {agent_assignments}")

            # if not agent_assignments:
            #     self.logger.warning("No specific agents assigned for TASK. Falling back to default handler.")
            #     # Delegate to default agent if no specific agents are found
            #     return self._handle_delegated_request(request, self.default_handler_agent_id, "TASK_NO_AGENTS_FOUND")

            # # Limit parallel execution
            # selected_assignments = agent_assignments[:self.max_parallel_agents]

            # # --- 2. Execute Assigned Agents ---
            # # (Consider parallel execution here for performance - ThreadPoolExecutor, asyncio etc.)
            # for agent_id, confidence in selected_assignments:
            #     self.logger.info(f"Executing agent: {agent_id} (Confidence: {confidence:.2f})")
            #     try:
            #         agent = self.agent_factory.create(agent_id)
            #         if not agent:
            #             raise AIAgentError(f"Could not create agent instance for '{agent_id}'")

            #         # Pass a copy of the request to avoid cross-contamination if needed
            #         # TODO: Enrich request with context? (e.g., relevant tools, use case) - Needs ToolFinder integration maybe
            #         response = agent.process_request(request.copy())

            #         # Basic normalization before adding
            #         normalized_response = self._normalize_response(response)
            #         agent_responses.append({"agent_id": agent_id, "response": normalized_response, "status": normalized_response.get("status", "unknown")})

            #     except Exception as agent_err:
            #         # Log with traceback
            #         err_logger = LoggerFactory.create("coordinator_task_agent_error")
            #         err_logger.error(f"Error executing agent '{agent_id}': {agent_err}", exc_info=True) # RESTORED exc_info
            #         agent_responses.append({
            #             "agent_id": agent_id,
            #             "response": self._create_error_response(f"Agent failed execution: {agent_err}", agent_id),
            #             "status": "error"
            #         })

            # # --- 3. Aggregate Responses ---
            # if not agent_responses:
            #     self.logger.warning("No agent responses received for TASK request.")
            #     return self._create_error_response("No agents executed or returned responses for the task.")
            
            # # Check if all responses were errors
            # if all(resp.get("status") == "error" for resp in agent_responses):
            #     self.logger.warning("All agents failed during TASK execution.")
            #     # Aggregate the error messages or return a generic error
            #     aggregated_error = "; ".join([resp["response"].get("content", "Unknown error") 
            #                                 for resp in agent_responses if resp.get("status") == "error"])
            #     return self._create_error_response(f"All assigned agents failed: {aggregated_error}")

            # # Use ResponseAggregator
            # # Pass only successful/relevant responses? Or let aggregator decide?
            # # Assuming aggregator handles mixed success/error states
            # final_response = self.response_aggregator.aggregate_responses(
            #     original_request=request,
            #     agent_responses=agent_responses
            # )

            # # Add coordinator metadata
            # if "metadata" not in final_response: final_response["metadata"] = {}
            # final_response["metadata"]["assigned_agents"] = agent_assignments
            # final_response["metadata"]["executed_agents"] = [resp["agent_id"] for resp in agent_responses]
            # final_response["metadata"]["coordinator_id"] = self.agent_id

            # return final_response

        except Exception as e:
            # Log with traceback
            err_logger = LoggerFactory.create("coordinator_task_error")
            err_logger.error(f"Critical error handling TASK request: {e}", exc_info=True) # RESTORED exc_info
            return self._create_error_response(f"Failed during task handling pipeline: {e}")

    # --- Helper Methods (Copied from original, verify imports/dependencies) ---

    def _get_available_agents_info(self) -> str:
        self.logger.debug("Retrieving available agents info...")
        # Check agent_registry directly, as it's created in Coordinator.__init__
        # Corrected method name: get_agent_types()
        if self.agent_registry:
            try:
                # Get list of registered agent type strings
                available_agent_types = self.agent_registry.get_agent_types() 
                agent_descriptions = self.config.get_agent_descriptions() if self.config else {}
                if not available_agent_types: return "No specialized agents are currently registered."
                # Use agent_types list to format the output
                info_lines = [f"- {agent_type}: {agent_descriptions.get(agent_type, 'No description available.')}" 
                              for agent_type in available_agent_types]
                return "\n".join(info_lines)
            except Exception as e:
                self.logger.error(f"Failed to get agent info from registry: {e}", exc_info=True)
                return "Could not retrieve agent information."
        self.logger.warning("AgentRegistry not available in Coordinator.")
        return "AgentRegistry not available."

    def _get_available_tools_info(self) -> str:
        self.logger.debug("Retrieving available tools info...")
        if self.tool_manager:
            try:
                # Call the correct method: get_all_tools()
                all_tools_info = self.tool_manager.get_all_tools() 
                if not all_tools_info:
                    return "No tools registered with ToolManager."
                
                info_lines = []
                # Iterate through the dictionary returned by get_all_tools
                # Corrected: tool_manager.get_all_tools() now returns Dict[str, ToolDefinition]
                for tool_name, tool_definition in all_tools_info.items():
                     # Check if the value is a ToolDefinition object
                     if isinstance(tool_definition, ToolDefinition):
                         # Access the description attribute directly
                         description = tool_definition.description
                         info_lines.append(f"- {tool_name}: {description}")
                     else:
                         # Log warning if the format is unexpected (shouldn't happen ideally)
                         self.logger.warning(f"Skipping tool '{tool_name}' due to unexpected data format: {type(tool_definition)}")
                         
                return "\n".join(info_lines) if info_lines else "No valid tool information found."
            except AttributeError as ae:
                 # Catch if get_all_tools is somehow missing (shouldn't happen now)
                 self.logger.error(f"ToolManager missing expected method: {ae}")
                 return "Could not retrieve tool information (ToolManager method missing/error)."
            except Exception as e:
                self.logger.error(f"Failed to get tool info from ToolManager: {e}")
                return "Could not retrieve tool information (Error)."
        self.logger.warning("ToolManager not available in Coordinator.")
        return "ToolManager not available."

    def _normalize_response(self, response: Any) -> Dict[str, Any]:
         self.logger.debug(f"Normalizing response of type: {type(response)}")
         # Default agent_id
         agent_id = "unknown"
         
         # Extract agent_id if possible
         if isinstance(response, dict):
             agent_id = response.get("agent_id", agent_id)
         elif hasattr(response, "agent_id"):
             agent_id = response.agent_id
             
         # Handle AgentResponse objects
         if hasattr(response, "status") and hasattr(response, "content"):
             return {
                 "content": response.content,
                 "status": response.status.value if hasattr(response.status, "value") else str(response.status),
                 "metadata": getattr(response, "metadata", {}),
                 "agent_id": agent_id # Use extracted or default
             }
         
         # Handle dictionaries
         if isinstance(response, dict):
             if "content" in response:
                 response.setdefault("status", "success")
                 response.setdefault("agent_id", agent_id)
                 return response
             else:
                 # If dict has no content, serialize it as content
                 return {"content": json.dumps(response), "status": "success", "agent_id": agent_id}
         
         # Handle strings and other types
         return {"content": str(response), "status": "success", "agent_id": agent_id}

    def _create_error_response(self, message: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a standardized error response dictionary."""
        return {
            "content": f"Error: {message}",
            "agent_id": agent_id or self.agent_id, # Use specific agent ID if error came from delegation
            "status": "error",
            "error": message # Include error message in a dedicated field
        }

# --- End Coordinator --- 