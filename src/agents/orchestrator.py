"""
Orchestrator for the multi-agent architecture.
Coordinates the workflow between specialized agents.
"""
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface
from ..config.unified_config import UnifiedConfig
from .tool_finder_agent import ToolFinderAgent
from .request_analyzer import RequestAnalyzer
from .response_aggregator import ResponseAggregator
from ..tools.tool_registry import ToolRegistry
from .interfaces import AgentResponse, AgentResponseStatus
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate
from ..core.model_selector import ModelSelector, UseCase
from ..core.tool_enabled_ai import AI
from ..metrics.request_metrics import RequestMetricsService
import time
import json
import re


class Orchestrator(BaseAgent):
    """
    Orchestrator agent responsible for coordinating the workflow between specialized agents.
    Serves as the entry point for all user interactions in the multi-agent system.
    """
    
    def __init__(self, 
                 agent_factory=None,
                 tool_finder_agent=None,
                 request_analyzer=None,
                 response_aggregator=None,
                 unified_config=None,
                 logger=None,
                 prompt_template: Optional[PromptTemplate] = None,
                 model_selector=None,
                 **kwargs):
        """
        Initialize the orchestrator.
        Ensures essential prompt templates ('use_case_classifier', 'plan_generation') 
        are created if they don't exist.
        
        Args:
            agent_factory: Factory for creating agent instances
            tool_finder_agent: Tool finder agent instance
            request_analyzer: Request analyzer component
            response_aggregator: Response aggregator component
            unified_config: UnifiedConfig instance
            logger: Logger instance
            prompt_template: PromptTemplate instance for generating prompts
            model_selector: ModelSelector for intelligent model selection
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id="orchestrator", unified_config=unified_config, logger=logger, **kwargs)
        
        # Core components
        self.agent_factory = agent_factory
        self.tool_finder_agent = tool_finder_agent
        self.request_analyzer = request_analyzer
        self.response_aggregator = response_aggregator
        
        # Set up prompt template service (loads YAML automatically)
        self._prompt_template = prompt_template or PromptTemplate(logger=self.logger)
        
        # Set up model selector
        self.model_selector = model_selector or ModelSelector()
        
        # Get configuration
        self.max_parallel_agents = self.agent_config.get("max_parallel_agents", 3)
        
        # Validate dependencies
        if self.agent_factory is None:
            self.logger.warning("No agent factory provided, will be unable to route requests")
            
        # Create missing components if needed
        if self.request_analyzer is None:
            self.logger.info("Creating default RequestAnalyzer")
            self.request_analyzer = RequestAnalyzer(
                unified_config=self.config,
                logger=self.logger,
                prompt_template=self._prompt_template
            )
            
        if self.response_aggregator is None:
            self.logger.info("Creating default ResponseAggregator")
            self.response_aggregator = ResponseAggregator(
                unified_config=self.config,
                logger=self.logger,
                prompt_template=self._prompt_template
            )
            
        # Get system prompt via PromptTemplate
        system_prompt_str = None
        try:
             system_prompt_str, _ = self._prompt_template.render_prompt(template_id="orchestrator")
        except ValueError:
             error_msg = "Orchestrator system prompt template ('orchestrator') is missing."
             self.logger.error(error_msg)
             system_prompt_str = "You are the Orchestrator agent." # Basic fallback
        except Exception as e:
             error_msg = f"Error loading Orchestrator system prompt: {e}"
             self.logger.error(error_msg)
             system_prompt_str = "You are the Orchestrator agent." # Basic fallback
        
        # Ensure we have an AI instance for meta-query handling, plan generation, etc.
        default_model = self.agent_config.get("default_model", self.model_selector.select_model(UseCase.CHAT).value if self.model_selector else None)
        if not default_model:
             self.logger.warning("No default model found for Orchestrator AI instance, some features might be limited.")

        self.ai_instance = AI(
            model=default_model,
            system_prompt=system_prompt_str,
            logger=self.logger,
            prompt_template=self._prompt_template
        )
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by orchestrating specialized agents or handling directly.

        This method follows these steps:
        1. Check for Resumption: If the request metadata contains 'resumption_context',
           it means the user has approved a previously generated plan. Extract the
           context and jump directly to agent processing (Step 6).
        2. Format Request & Classify Intent: Ensure the request is valid and use
           RequestAnalyzer to classify the intent (META, QUESTION, TASK).
        3. Handle META/QUESTION: Handle simple meta-queries or direct questions.
        4. Determine Use Case (for TASK): Use an LLM (_determine_use_case with
           'use_case_classifier' prompt) to find the appropriate UseCase.
        5. Find Tools & Assign Agents (for TASK): Use ToolFinderAgent and
           RequestAnalyzer to identify relevant tools and agent assignments.
        6. Generate Plan (for TASK with agents): If agents are assigned, generate an
           execution plan using an LLM ('plan_generation' prompt).
        7. Request Approval (if plan generated): Return a response with status
           'awaiting_approval', the plan, and the 'resumption_context' needed
           to continue after user approval.
        8. Process with Agents (Step 6 - if no approval needed or resuming):
           Dispatch the request (potentially enriched with the plan) to assigned agents.
        9. Aggregate Responses (Step 7 - if no approval needed or resuming):
           Combine responses from agents using ResponseAggregator.
        
        Args:
            request: The request object containing prompt and metadata. 
                     If resuming, metadata should contain 'resumption_context'.
            
        Returns:
            Response object with content and metadata. Possible statuses include:
            - 'success': Request completed successfully.
            - 'error': An error occurred.
            - 'awaiting_approval': A plan was generated and needs user approval.
              The metadata will contain 'resumption_context'.
            - 'partial': Aggregation failed, returning a partial result.
        """
        # Import request metrics service
        metrics_service = RequestMetricsService()
        
        # Generate request ID if not present
        if "request_id" not in request:
            import uuid
            request["request_id"] = str(uuid.uuid4())
            
        # Start tracking the request
        metrics_service.start_request_tracking(
            request_id=request["request_id"],
            prompt=request.get("prompt", ""),
            metadata={"user_id": request.get("user_id")}
        )
        
        try:
            # --- Check for Resumption Context --- 
            if isinstance(request, dict) and "resumption_context" in request.get("metadata", {}):
                self.logger.info("Resuming execution after plan approval.")
                context = request["metadata"]["resumption_context"]
                
                # Ensure context is valid
                if not all(k in context for k in ["original_request", "agent_assignments", "relevant_tools", "use_case", "generated_plan"]):
                    self.logger.error("Invalid or incomplete resumption context received.")
                    return {
                        "content": "Error: Could not resume execution due to invalid context.",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": "Invalid resumption context"
                    }
                    
                # Extract data from context
                original_request = context["original_request"]
                agent_assignments = context["agent_assignments"]
                relevant_tools = context["relevant_tools"]
                use_case = context["use_case"] # Note: UseCase might need deserialization if not stored as string
                generated_plan = context["generated_plan"]
                request_id = original_request.get("request_id", "unknown_resumed")
                
                # Directly proceed to agent processing (Step 6)
                self.logger.info(f"Resuming Step 6: Processing with agents for request {request_id}")
                try:
                    agent_responses = self._process_with_agents(
                        request=original_request, # Use original request data
                        agent_assignments=agent_assignments,
                        relevant_tools=relevant_tools,
                        use_case=use_case,
                        generated_plan=generated_plan
                    )
                    self.logger.info(f"Resumed processing with {len(agent_responses)} agents")
                except Exception as e:
                    # ... (handle agent processing error, identical to non-resumed case)
                    self.logger.error(f"Error processing with agents during resumption: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    metrics_service.end_request_tracking(request_id=request_id, success=False, error=f"Error processing with agents: {str(e)}")
                    return {
                        "content": f"An error occurred while processing your request with specialized agents after approval: {str(e)}",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": str(e)
                    }
                
                # Proceed to aggregation (Step 7)
                self.logger.info(f"Resuming Step 7: Aggregating responses for request {request_id}")
                start_time = time.time()
                try:
                    aggregated_response = self._aggregate_responses(agent_responses, original_request)
                    self.logger.info("Successfully aggregated responses after resumption")
                    # ... (Track aggregator usage, end request tracking, add metadata - identical to non-resumed case)
                    metrics_service.track_agent_usage(request_id=request_id, agent_id="response_aggregator", duration_ms=int((time.time() - start_time) * 1000), success=True)
                    metrics_service.end_request_tracking(request_id=request_id, success=True if "error" not in aggregated_response else False, error=aggregated_response.get("error"))
                    if "metadata" not in aggregated_response: aggregated_response["metadata"] = {}
                    aggregated_response["metadata"]["request_id"] = request_id
                    aggregated_response["metadata"]["agents_used"] = [a["agent_id"] for a in agent_responses]
                    aggregated_response["metadata"]["tools_used"] = relevant_tools
                    if generated_plan: aggregated_response["metadata"]["generated_plan"] = generated_plan
                    aggregated_response["metadata"]["resumed_from_approval"] = True # Indicate resumption
                    return aggregated_response
                except Exception as e:
                    # ... (Handle aggregation error, identical to non-resumed case)
                    self.logger.error(f"Error aggregating responses during resumption: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    metrics_service.track_agent_usage(request_id=request_id, agent_id="response_aggregator", duration_ms=int((time.time() - start_time) * 1000), success=False, metadata={"error": str(e)})
                    metrics_service.end_request_tracking(request_id=request_id, success=False, error=f"Error aggregating responses: {str(e)}")
                    return {
                        "content": f"An error occurred while aggregating responses after approval: {str(e)}",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": str(e)
                    }
            # --- End Check for Resumption Context ---
            
            # Ensure request is properly formatted (if not resuming)
            if not isinstance(request, dict):
                self.logger.warning(f"Request is not a dictionary: {type(request)}")
                request = {"prompt": str(request), "request_id": request["request_id"]}
            # ... rest of initial checks ...
                
            self.logger.info(f"Processing request: {request.get('prompt', '')[:50]}...")
            
            # Step 1: Classify the request intent using the RequestAnalyzer
            request_intent = "UNKNOWN"
            try:
                self.logger.info("Classifying request intent...")
                request_intent = self.request_analyzer.classify_request_intent(request)
                self.logger.info(f"Request classified as: {request_intent}")
                
                # Track intent classification
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id="request_analyzer",
                    metadata={"intent_classification": request_intent}
                )
            except Exception as e:
                self.logger.error(f"Error classifying request intent: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Default to TASK if we can't classify
                request_intent = "TASK"
            
            # Step 2: Handle the request based on its intent
            if request_intent == "META":
                self.logger.info("Handling META query about the system...")
                result = self._handle_meta_query(request)
                
                # Track meta-query handling
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id=self.agent_id,
                    metadata={"meta_query_handled": True}
                )
                
                # End request tracking
                metrics_service.end_request_tracking(
                    request_id=request["request_id"],
                    success=True if result.get("status") != "error" else False,
                    error=result.get("error")
                )
                return result
                
            elif request_intent == "QUESTION":
                self.logger.info("Handling QUESTION directly with LLM...")
                
                # Determine the use case for model selection
                use_case = self._determine_use_case(request)
                if use_case:
                    self.logger.info(f"Determined use case for question: {use_case.name}")
                    
                    # Add use case to request for proper handling
                    request["use_case"] = use_case.name
                    
                    # Get system prompt for this use case
                    try:
                        # Select appropriate model and system prompt
                        system_prompt = self.model_selector.get_system_prompt(use_case)
                        request["system_prompt"] = system_prompt
                        
                        model = self.model_selector.select_model(use_case)
                        if model:
                            self.logger.info(f"Selected model for question: {model.value}")
                            request["model"] = model.value
                            
                            # Track model selection
                            metrics_service.track_model_usage(
                                request_id=request["request_id"],
                                model_id=model.value,
                                metadata={"use_case": use_case.name}
                            )
                            
                            # Update our AI instance with the selected model/prompt
                            self.ai_instance = AI(
                                model=model.value,
                                system_prompt=system_prompt,
                                logger=self.logger
                            )
                    except Exception as e:
                        self.logger.warning(f"Error configuring model for question: {str(e)}")
                
                # Handle the question directly with our AI instance
                try:
                    # For direct questions, let's use a simpler, more focused response structure
                    direct_result = super().process_request(request)
                    
                    # Track direct handling
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id=self.agent_id,
                        metadata={"direct_question_handled": True}
                    )
                    
                    # End request tracking
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=True if "error" not in direct_result else False,
                        error=direct_result.get("error")
                    )
                    
                    return direct_result
                except Exception as e:
                    self.logger.error(f"Error handling question directly: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    # End request tracking with error
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=False,
                        error=f"Error handling question: {str(e)}"
                    )
                    
                    return {
                        "content": f"An error occurred while answering your question: {str(e)}",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": str(e)
                    }
            
            else:  # TASK or UNKNOWN (default to TASK)
                self.logger.info("Handling as a TASK requiring specialized processing...")
                generated_plan = None # Initialize plan variable

                # Step 3: Determine use case for specialized task handling
                use_case = self._determine_use_case(request)
                if use_case:
                    self.logger.info(f"Determined use case: {use_case.name}")
                    # Add use case to request for agents
                    request["use_case"] = use_case.name
                    
                    # Select appropriate model based on use case
                    try:
                        # Get system prompt for this use case
                        system_prompt = self.model_selector.get_system_prompt(use_case)
                        request["system_prompt"] = system_prompt
                        
                        # Get the best model for this use case
                        model = self.model_selector.select_model(use_case)
                        if model:
                            self.logger.info(f"Selected model for request: {model.value}")
                            request["model"] = model.value
                            
                            # Track model selection
                            metrics_service.track_model_usage(
                                request_id=request["request_id"],
                                model_id=model.value,
                                metadata={"use_case": use_case.name}
                            )
                            
                            # If we have an AI instance and no specialized agents will handle this,
                            # update the model directly on our AI instance
                            if not self.ai_instance:
                                self.logger.info("Creating new AI instance with selected model")
                                self.ai_instance = AI(
                                    model=model.value,
                                    system_prompt=system_prompt,
                                    logger=self.logger
                                )
                    except Exception as e:
                        self.logger.warning(f"Error selecting model for use case {use_case}: {str(e)}")
                
                # Step 4: Find relevant tools for the request
                start_time = time.time()
                try:
                    relevant_tools = self._find_relevant_tools(request)
                    self.logger.info(f"Found {len(relevant_tools)} relevant tools")
                    
                    # Track tool finder agent usage
                    if self.tool_finder_agent:
                        metrics_service.track_agent_usage(
                            request_id=request["request_id"],
                            agent_id="tool_finder",
                            duration_ms=int((time.time() - start_time) * 1000),
                            success=True
                        )
                except Exception as e:
                    self.logger.error(f"Error finding relevant tools: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    relevant_tools = []
                    
                    # Track tool finder failure
                    if self.tool_finder_agent:
                        metrics_service.track_agent_usage(
                            request_id=request["request_id"],
                            agent_id="tool_finder",
                            duration_ms=int((time.time() - start_time) * 1000),
                            success=False,
                            metadata={"error": str(e)}
                        )
                
                # Step 5: Get agent assignments for the task
                start_time = time.time()
                try:
                    agent_assignments = self.request_analyzer.get_agent_assignments(
                        request=request,
                        available_agents=self.agent_factory.registry.get_all_agents() if self.agent_factory else [],
                        agent_descriptions=self.config.get_agent_descriptions() if self.config else {}
                    )
                    self.logger.info(f"Analyzed request, found {len(agent_assignments)} agent assignments")
                    
                    # Track request analyzer usage
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="request_analyzer",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=True,
                        metadata={"num_agents_found": len(agent_assignments)}
                    )
                except Exception as e:
                    self.logger.error(f"Error analyzing request: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    agent_assignments = []
                    
                    # Track request analyzer failure
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="request_analyzer",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        metadata={"error": str(e)}
                    )
                
                # If no agents identified, handle directly
                if not agent_assignments:
                    self.logger.info("No specialized agents identified for task, handling directly")
                    
                    # Track direct handling by orchestrator
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id=self.agent_id,
                        confidence=1.0,
                        metadata={"direct_handling": True}
                    )
                    
                    # Use base agent's process_request for direct handling
                    result = super().process_request(request)
                    
                    # End request tracking
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=True if "error" not in result else False,
                        error=result.get("error")
                    )
                    
                    return result
                
                # --- Plan Generation Step ---
                self.logger.info("Complex task detected, generating plan...")
                plan_generation_successful = False
                try:
                    if self.ai_instance and self._prompt_template:
                        # Prepare context for plan generation prompt
                        assigned_agents_str = ", ".join([f"{agent_id} (conf: {conf:.2f})" for agent_id, conf in agent_assignments])
                        relevant_tools_str = ", ".join(relevant_tools) if relevant_tools else "None"
                        use_case_name = use_case.name if use_case else "UNKNOWN"

                        # Assumes a 'plan_generation' template exists
                        plan_prompt, _ = self._prompt_template.render_prompt(
                            template_id="plan_generation",
                            variables={
                                "user_prompt": request.get("prompt", ""),
                                "determined_use_case": use_case_name,
                                "assigned_agents_str": assigned_agents_str,
                                "relevant_tools_str": relevant_tools_str
                            }
                        )

                        self.logger.debug("Requesting plan generation from LLM...")
                        generated_plan = self.ai_instance.request(plan_prompt)
                        self.logger.info(f"Generated Plan:\n{generated_plan}")
                        # Optionally: Add plan generation to metrics
                        metrics_service.track_agent_usage(
                             request_id=request["request_id"],
                             agent_id=self.agent_id, # Or a specific "planner" ID
                             metadata={"plan_generated": True, "plan_content": generated_plan[:200]} # Truncate plan for metadata
                        )

                        if generated_plan: # Check if plan was actually generated
                           plan_generation_successful = True 
                           self.logger.info(f"Generated Plan:\n{generated_plan}")
                           # ... (metrics tracking for plan generation) ...
                        else:
                           self.logger.warning("Plan generation attempt did not produce a plan.")

                    else:
                        self.logger.warning("AI instance or PromptTemplate not available for plan generation.")

                except ValueError as e: # Handles missing template
                    self.logger.error(f"Failed to render plan generation prompt: {e}")
                except Exception as e:
                    self.logger.error(f"Error during plan generation: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                # --- End Plan Generation Step ---
                
                # --- Return for Approval Step ---
                if plan_generation_successful:
                    self.logger.info("Plan generated, returning for user approval.")
                    
                    # Store context needed for resumption
                    # Note: UseCase might need to be stored as string (use_case.name)
                    resumption_context = {
                        "original_request": request, # Store the initial request
                        "agent_assignments": agent_assignments,
                        "relevant_tools": relevant_tools,
                        "use_case": use_case.name if use_case else None, # Store UseCase name
                        "generated_plan": generated_plan
                    }
                    
                    # Stop request tracking here for now, will be resumed later
                    # Or adjust tracking logic to handle pending states
                    metrics_service.track_request_status(
                        request_id=request["request_id"],
                        status="awaiting_approval"
                    ) # Assuming metrics service has such a method

                    return {
                        "content": f"Please review the following plan:\n\n{generated_plan}",
                        "agent_id": self.agent_id,
                        "status": "awaiting_approval", # Special status
                        "metadata": {
                            "request_id": request["request_id"],
                            "resumption_context": resumption_context # Pass context back
                        }
                    }
                else:
                     # If plan generation failed or wasn't attempted, proceed directly
                     self.logger.warning("Plan generation failed or skipped, proceeding without approval.")
                     # Fall through to Step 6 directly
                # --- End Return for Approval Step ---

                # Step 6: Process with identified agents (Now only runs if plan approval wasn't needed/failed)
                try:
                    agent_responses = self._process_with_agents(
                        request=request,
                        agent_assignments=agent_assignments,
                        relevant_tools=relevant_tools,
                        use_case=use_case,
                        generated_plan=generated_plan # Plan might be None here
                    )
                    self.logger.info(f"Processed with {len(agent_responses)} agents (no approval required)")
                except Exception as e:
                    # Restore original error handling for agent processing when no approval needed
                    self.logger.error(f"Error processing with agents (no approval): {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=False,
                        error=f"Error processing with agents: {str(e)}"
                    )
                    return {
                        "content": f"An error occurred while processing your request with specialized agents: {str(e)}",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": str(e)
                    }

                # Step 7: Aggregate responses (Now only runs if plan approval wasn't needed/failed)
                start_time = time.time()
                try:
                    aggregated_response = self._aggregate_responses(agent_responses, request)
                    self.logger.info("Successfully aggregated responses (no approval required)")
                    # Track aggregator usage
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="response_aggregator",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=True
                    )
                    # End request tracking
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=True if "error" not in aggregated_response else False,
                        error=aggregated_response.get("error")
                    )
                    # Add metrics data to the response
                    if "metadata" not in aggregated_response:
                        aggregated_response["metadata"] = {}
                    aggregated_response["metadata"]["request_id"] = request["request_id"]
                    aggregated_response["metadata"]["agents_used"] = [a["agent_id"] for a in agent_responses]
                    aggregated_response["metadata"]["tools_used"] = relevant_tools
                    if generated_plan: # Include plan if it existed but approval failed/skipped
                        aggregated_response["metadata"]["generated_plan"] = generated_plan
                    return aggregated_response
                except Exception as e:
                    # Restore original error handling for aggregation when no approval needed
                    self.logger.error(f"Error aggregating responses (no approval): {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="response_aggregator",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        metadata={"error": str(e)}
                    )
                    metrics_service.end_request_tracking(
                        request_id=request["request_id"],
                        success=False,
                        error=f"Error aggregating responses: {str(e)}"
                    )
                    return {
                        "content": f"An error occurred while aggregating responses: {str(e)}",
                        "agent_id": self.agent_id,
                        "status": "error",
                        "error": str(e)
                    }

        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Error orchestrating request: {str(e)}", agent_id="orchestrator"),
                self.logger
            )
            self.logger.error(f"Orchestration error: {error_response['message']}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # End request tracking with error
            try:
                metrics_service.end_request_tracking(
                    request_id=request.get("request_id", "unknown"),
                    success=False,
                    error=f"Orchestration error: {str(e)}"
                )
            except:
                pass
            
            return {
                "content": f"An error occurred while processing your request: {str(e)}",
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e)
            }
    
    def _handle_meta_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request identified as a meta-query about the system.
        
        Args:
            request: The request object
            
        Returns:
            Response dictionary
        """
        if not self.ai_instance:
             return {
                "content": "I cannot process meta-queries at the moment as my internal AI instance is not available.",
                "agent_id": self.agent_id,
                "status": "error",
                "error": "AI instance not available in Orchestrator"
            }
            
        user_query = request.get("prompt", "")
        
        try:
            # 1. Gather relevant system information (Agents, Tools, etc.)
            agents_info = self._get_available_agents_info()
            tools_info = self._get_available_tools_info()
            
            # Prepare context for the AI
            system_context = f"Available Agents:\n{agents_info}\n\nAvailable Tools:\n{tools_info}\n\nGeneral Process: The system analyzes requests, identifies relevant agents/tools, processes the request (potentially using multiple specialized agents), and aggregates the results."
            # Add more context as needed (e.g., config details, process explanation)

            # 2. Use AI to generate the response based on the query and context
            #    Use a template if available
            template_vars = {
                "user_query": user_query,
                "system_context": system_context
            }
            
            # Use PromptTemplate directly, remove fallback
            response_prompt, _ = self._prompt_template.render_prompt(
                template_id="answer_meta_query",
                variables=template_vars
            )
            # Let render_prompt raise error if template is missing

            self.logger.debug("Sending meta-query answer generation request to AI.")
            response_content = self.ai_instance.request(response_prompt)
            
            return {
                "content": response_content,
                "agent_id": self.agent_id,
                "status": "success",
                "metadata": {
                    "meta_query_handled": True,
                    "context_provided": { # Optionally include what context was given
                         "agents": agents_info != "No agent information available.",
                         "tools": tools_info != "No tool information available."
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Error handling meta-query: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "content": f"An error occurred while trying to answer your question about the system: {str(e)}",
                "agent_id": self.agent_id,
                "status": "error",
                "error": f"Error handling meta-query: {str(e)}"
            }
            
    def _get_available_agents_info(self) -> str:
        """Retrieve a formatted string of available agents and descriptions."""
        if self.agent_factory and hasattr(self.agent_factory, 'registry'):
            try:
                available_agents = self.agent_factory.registry.get_all_agents()
                agent_descriptions = self.config.get_agent_descriptions() if self.config else {}
                
                if not available_agents: return "No specialized agents are currently registered."
                
                info_lines = []
                for agent_id in available_agents:
                    # Exclude orchestrator, request_analyzer, etc. from user-facing list? Or keep them? Let's keep for now.
                    description = agent_descriptions.get(agent_id, "No description available.")
                    info_lines.append(f"- {agent_id}: {description}")
                return "\n".join(info_lines)
            except Exception as e:
                self.logger.error(f"Failed to get agent info: {e}")
                return "Could not retrieve agent information."
        return "No agent information available."

    def _get_available_tools_info(self) -> str:
        """Retrieve a formatted string of available tools and descriptions."""
        # This depends heavily on how tools are registered and described.
        # Assuming ToolFinderAgent might have access or a central ToolRegistry exists.
        if self.tool_finder_agent and hasattr(self.tool_finder_agent, 'tool_registry'):
             try:
                 tool_data = self.tool_finder_agent.tool_registry.get_tool_schemas() # Assuming this method exists
                 if not tool_data: return "No tools are currently registered."
                 
                 info_lines = []
                 for tool_name, schema in tool_data.items():
                      description = schema.get('description', 'No description available.')
                      info_lines.append(f"- {tool_name}: {description}")
                 return "\n".join(info_lines)
             except Exception as e:
                 self.logger.error(f"Failed to get tool info: {e}")
                 return "Could not retrieve tool information."
        elif hasattr(self, 'tool_registry'): # Maybe orchestrator has direct access?
             # Similar logic using self.tool_registry
             pass 
        return "No tool information available." # Fallback

    def _determine_use_case(self, request: Dict[str, Any]) -> Optional[UseCase]:
        """
        Determine the use case for this request using an LLM classifier.
        Relies on the 'use_case_classifier' prompt template in PromptTemplate.
        
        Args:
            request: The request object
            
        Returns:
            UseCase enum or None if unable to determine
        """
        prompt = request.get("prompt", "")
        
        # 1. Check for explicitly requested use case first
        if "use_case" in request and isinstance(request["use_case"], str):
            try:
                explicit_use_case = UseCase.from_string(request["use_case"])
                self.logger.info(f"Using explicitly provided use case: {explicit_use_case.name}")
                return explicit_use_case
            except ValueError:
                self.logger.warning(f"Invalid use case specified in request: {request['use_case']}, attempting LLM classification.")

        # 2. Use LLM for classification if not explicitly provided
        if not self.ai_instance:
            self.logger.warning("No AI instance available for use case classification. Defaulting to CHAT.")
            return UseCase.CHAT
            
        if not self._prompt_template:
            self.logger.warning("No PromptTemplate available for use case classification. Defaulting to CHAT.")
            return UseCase.CHAT

        try:
            # Prepare the classification prompt
            # Assumes a template 'use_case_classifier' exists.
            # This template should list valid UseCases and ask the LLM to choose one.
            available_use_cases = [uc.name for uc in UseCase] # Get list of valid UseCase names
            
            classification_prompt, _ = self._prompt_template.render_prompt(
                template_id="use_case_classifier",
                variables={
                    "user_prompt": prompt,
                    "available_use_cases": ", ".join(available_use_cases) 
                }
            )
            
            self.logger.info("Attempting LLM-based use case classification...")
            llm_response = self.ai_instance.request(classification_prompt)
            
            # Parse the response - expect just the UseCase name string
            # Add cleaning/stripping as needed based on LLM behavior
            classified_use_case_str = llm_response.strip().upper()
            
            try:
                determined_use_case = UseCase.from_string(classified_use_case_str)
                self.logger.info(f"LLM classified use case as: {determined_use_case.name}")
                return determined_use_case
            except ValueError:
                self.logger.warning(f"LLM returned an invalid UseCase name: '{classified_use_case_str}'. Defaulting to CHAT.")
                return UseCase.CHAT

        except ValueError as e: # Handles missing template from render_prompt
            self.logger.error(f"Failed to render use case classification prompt: {e}. Defaulting to CHAT.")
            return UseCase.CHAT
        except Exception as e:
            self.logger.error(f"Error during LLM use case classification: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Default to CHAT on error
            return UseCase.CHAT
    
    def _find_relevant_tools(self, request: Dict[str, Any]) -> List[str]:
        """
        Find relevant tools for the request using the tool finder agent.
        
        Args:
            request: The request object
            
        Returns:
            List of relevant tool IDs
        """
        if not self.tool_finder_agent:
            self.logger.warning("No tool finder agent available")
            return []
            
        try:
            self.logger.info("Finding relevant tools...")
            
            # Ensure request is properly formatted
            if not isinstance(request, dict):
                self.logger.warning(f"Request is not a dictionary: {type(request)}")
                request = {"prompt": str(request)}
                
            # Make sure prompt exists in request
            if "prompt" not in request:
                self.logger.warning("No prompt in request, using empty string")
                request["prompt"] = ""
                
            response = self.tool_finder_agent.process_request(request)
            
            # Handle different response formats
            if isinstance(response, dict) and "selected_tools" in response:
                tools = response["selected_tools"]
                self.logger.info(f"Found {len(tools)} tools from dict response")
                return tools
            elif hasattr(response, "selected_tools"):
                tools = response.selected_tools
                self.logger.info(f"Found {len(tools)} tools from object response")
                return tools
            elif hasattr(response, "content") and isinstance(response.content, list):
                self.logger.info(f"Using content as tools list: {response.content}")
                return response.content
            else:
                self.logger.warning(f"Unexpected response format from tool finder agent: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error finding relevant tools: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _process_with_agents(self,
                            request: Dict[str, Any],
                            agent_assignments: List[Tuple[str, float]],
                            relevant_tools: List[str],
                            use_case: Optional[UseCase] = None,
                            generated_plan: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process the request with assigned agents.

        Args:
            request: The request object
            agent_assignments: List of (agent_id, confidence) tuples
            relevant_tools: List of relevant tool IDs
            use_case: Optional UseCase identified for the request
            generated_plan: Optional plan generated by the orchestrator
        Returns:
            List of agent responses
        """
        # Import metrics service
        from ..metrics.request_metrics import RequestMetricsService
        metrics_service = RequestMetricsService()

        agent_responses = []

        # Limit number of agents to prevent excessive processing
        agent_assignments = agent_assignments[:self.max_parallel_agents]

        # Process with each identified agent
        for agent_id, confidence in agent_assignments:
            agent_start_time = time.time()
            success = False
            error_message = None

            try:
                self.logger.info(f"Processing with agent {agent_id} (confidence: {confidence})")

                # Create a new request enriched with tools AND the plan
                enriched_request = self._enrich_request(
                    request=request,
                    relevant_tools=relevant_tools,
                    generated_plan=generated_plan # Pass plan here
                )

                # Select appropriate model for this agent and use case
                if use_case:
                    try:
                        # Add system prompt based on use case if possible
                        enriched_request["system_prompt"] = self.model_selector.get_system_prompt(use_case)
                        
                        # Add model selection to the request
                        model = self.model_selector.select_model(use_case)
                        if model:
                            self.logger.info(f"Selected model for {agent_id}: {model.value}")
                            enriched_request["model"] = model.value
                            
                            # Track model usage for this agent
                            metrics_service.track_model_usage(
                                request_id=request["request_id"],
                                model_id=model.value,
                                metadata={
                                    "use_case": use_case.name,
                                    "agent_id": agent_id
                                }
                            )
                    except Exception as e:
                        self.logger.warning(f"Error selecting model for use case {use_case}: {str(e)}")
                
                # Get agent instance
                if not self.agent_factory:
                    self.logger.error("Agent factory not available, cannot create agent")
                    continue
                    
                agent = self.agent_factory.create(agent_id)
                
                if not agent:
                    self.logger.error(f"Failed to create agent {agent_id}")
                    continue
                
                # Pass request_id to agent if needed
                if "request_id" not in enriched_request and "request_id" in request:
                    enriched_request["request_id"] = request["request_id"]
                
                # Process the request
                try:
                    response = agent.process_request(enriched_request)
                    success = True
                except Exception as e:
                    self.logger.error(f"Error in agent {agent_id} process_request: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    response = {
                        "content": f"Error: {str(e)}",
                        "status": "error"
                    }
                    error_message = str(e)
                
                # Normalize the response format
                try:
                    normalized_response = self._normalize_response(response)
                except Exception as e:
                    self.logger.error(f"Error normalizing response from agent {agent_id}: {str(e)}")
                    normalized_response = {
                        "content": f"Error normalizing response: {str(e)}",
                        "status": "error"
                    }
                    success = False
                    error_message = str(e)
                
                # Add to responses
                agent_responses.append({
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "response": normalized_response,
                    "status": normalized_response.get("status", "success")
                })
                
                # Track successful agent usage
                if success:
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id=agent_id,
                        confidence=confidence,
                        duration_ms=int((time.time() - agent_start_time) * 1000),
                        success=True
                    )
                
            except Exception as e:
                error_response = ErrorHandler.handle_error(
                    AIAgentError(f"Error with agent {agent_id}: {str(e)}", agent_id=agent_id),
                    self.logger
                )
                self.logger.error(f"Agent processing error: {error_response['message']}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                agent_responses.append({
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "response": {"content": f"Error: {str(e)}", "status": "error"},
                    "status": "error",
                    "error": str(e)
                })
                
                error_message = str(e)
                
            # Track failed agent usage if not already tracked
            if not success:
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id=agent_id,
                    confidence=confidence,
                    duration_ms=int((time.time() - agent_start_time) * 1000),
                    success=False,
                    metadata={"error": error_message}
                )
        
        return agent_responses

    def _enrich_request(self,
                        request: Dict[str, Any],
                        relevant_tools: List[str],
                        generated_plan: Optional[str] = None) -> Dict[str, Any]:
        """
        Enrich the request with additional context, including the generated plan.

        Args:
            request: The original request
            relevant_tools: List of relevant tool IDs
            generated_plan: Optional plan generated by the orchestrator

        Returns:
            Enriched request
        """
        # Create a copy to avoid modifying the original
        enriched = dict(request)

        # Add relevant tools
        if relevant_tools:
            enriched["relevant_tools"] = relevant_tools

        # Add orchestrator context
        if "context" not in enriched:
            enriched["context"] = {}

        enriched["context"]["orchestrator_id"] = self.agent_id

        # Add the generated plan to the context if it exists
        if generated_plan:
            enriched["context"]["generated_plan"] = generated_plan
            self.logger.debug(f"Added generated plan to context for agent request.")

        return enriched
    
    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        """
        Normalize agent responses to a consistent format.
        
        Args:
            response: Agent response (various formats)
            
        Returns:
            Normalized response dictionary
        """
        # Handle AgentResponse objects
        if hasattr(response, "status") and hasattr(response, "content"):
            return {
                "content": response.content,
                "status": response.status.value if hasattr(response.status, "value") else str(response.status),
                "metadata": getattr(response, "metadata", {})
            }
        
        # Handle dictionaries
        if isinstance(response, dict):
            if "content" in response:
                return response
            else:
                return {"content": str(response), "status": "success"}
        
        # Handle strings
        if isinstance(response, str):
            return {"content": response, "status": "success"}
        
        # Handle other types
        return {"content": str(response), "status": "success"}
    
    def _aggregate_responses(self, 
                           agent_responses: List[Dict[str, Any]], 
                           original_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents using the ResponseAggregator.
        
        Args:
            agent_responses: List of agent responses
            original_request: The original request
            
        Returns:
            Aggregated response
        """
        if not agent_responses:
            self.logger.warning("No agent responses to aggregate")
            return {
                "content": "No agents were able to process your request.",
                "agent_id": self.agent_id,
                "status": "error"
            }
        
        # Check if response aggregator is available
        if not self.response_aggregator:
            self.logger.error("Response aggregator not available")
            return {
                "content": "Error: Response aggregator not available",
                "agent_id": self.agent_id,
                "status": "error"
            }
        
        try:
            # Use the ResponseAggregator to aggregate responses
            aggregated_response = self.response_aggregator.aggregate_responses(
                agent_responses=agent_responses,
                original_request=original_request
            )
            
            # Ensure the response has the required fields
            if not isinstance(aggregated_response, dict):
                self.logger.warning(f"Aggregated response is not a dictionary: {type(aggregated_response)}")
                return {
                    "content": str(aggregated_response),
                    "agent_id": self.agent_id,
                    "status": "success"
                }
                
            if "content" not in aggregated_response:
                self.logger.warning("Aggregated response missing 'content' field")
                aggregated_response["content"] = "No content available"
                
            if "agent_id" not in aggregated_response:
                aggregated_response["agent_id"] = self.agent_id
                
            if "status" not in aggregated_response:
                aggregated_response["status"] = "success"
                
            return aggregated_response
            
        except Exception as e:
            self.logger.error(f"Error aggregating responses: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return a fallback response with the first agent's content
            if agent_responses and "response" in agent_responses[0] and "content" in agent_responses[0]["response"]:
                return {
                    "content": agent_responses[0]["response"]["content"],
                    "agent_id": self.agent_id,
                    "status": "partial",
                    "note": "Using first agent response due to aggregation error"
                }
            else:
                return {
                    "content": "An error occurred while aggregating responses from multiple agents.",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e)
                }
    
    def set_tool_finder_agent(self, tool_finder_agent: ToolFinderAgent) -> None:
        """
        Set the tool finder agent.
        
        Args:
            tool_finder_agent: Tool finder agent instance
        """
        self.tool_finder_agent = tool_finder_agent
        self.logger.info("Tool finder agent set")
    
    def set_request_analyzer(self, request_analyzer: RequestAnalyzer) -> None:
        """
        Set the request analyzer.
        
        Args:
            request_analyzer: Request analyzer instance
        """
        self.request_analyzer = request_analyzer
        self.logger.info("Request analyzer set")
    
    def set_response_aggregator(self, response_aggregator: ResponseAggregator) -> None:
        """
        Set the response aggregator.
        
        Args:
            response_aggregator: Response aggregator instance
        """
        self.response_aggregator = response_aggregator
        self.logger.info("Response aggregator set")
    
    def set_model_selector(self, model_selector: ModelSelector) -> None:
        """
        Set the model selector.
        
        Args:
            model_selector: ModelSelector instance
        """
        self.model_selector = model_selector
        self.logger.info("Model selector set")