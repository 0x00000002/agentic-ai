#!/usr/bin/env python
"""
Test for Orchestrator component.

This test verifies that:
1. The Orchestrator correctly routes requests based on intent classification (META, QUESTION, TASK)
2. META queries are handled directly by the Orchestrator
3. QUESTION queries are handled using the Orchestrator's AI model
4. TASK queries are routed to appropriate specialized agents
5. Fallback handling works correctly when no specialized agents are found
"""
import os
import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_orchestrator")


class TestOrchestrator(unittest.TestCase):
    """Test cases for Orchestrator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        from src.agents.orchestrator import Orchestrator
        from src.agents.request_analyzer import RequestAnalyzer
        from src.agents.tool_finder_agent import ToolFinderAgent
        from src.agents.response_aggregator import ResponseAggregator
        from src.agents.agent_registry import AgentRegistry
        from src.agents.agent_factory import AgentFactory
        from src.config.unified_config import UnifiedConfig
        from src.prompts.prompt_template import PromptTemplate
        from src.core.model_selector import ModelSelector, UseCase
        from src.core.tool_enabled_ai import AI
        from src.core.providers.base_provider import BaseProvider
        
        # Create mocks for dependencies
        self.mock_ai = MagicMock(spec=AI)
        self.mock_ai.request.return_value = "Mocked AI response"
        self.mock_ai.get_model_info.return_value = {"model_id": "test-model"}
        
        # Mock UnifiedConfig - SETUP THE MOCK FIRST
        self.mock_config = MagicMock(spec=UnifiedConfig)
        agent_config = {"max_parallel_agents": 3, "default_model": "test-model"}
        # Define the internal config structure (useful for reference, but direct mocks below are key)
        test_model_data = {
            'name': 'Test Model',
            'provider': 'mock_provider', # Ensure this is a valid string
            'model_id': 'test-model',
            'api_key_env': 'MOCK_API_KEY',
            'quality': 'medium',
            'speed': 'standard',
            'temperature': 0.7
        }
        self.mock_config._config = { 
            "models": {"models": {"test-model": test_model_data}},
            "agents": {"agent_descriptions": {
                "orchestrator": "Coordinates workflows",
                "coding_assistant": "Helps with coding",
                "content_creator": "Creates content"
            }}
            # Add other sections like 'providers', 'use_cases' if needed
        }
        self.mock_config._default_model = "test-model"
        
        # --- ADD BACK Explicit Mocks for methods called on the config instance ---
        self.mock_config.get_model_config = MagicMock(return_value=test_model_data)
        self.mock_config.get_default_model = MagicMock(return_value="test-model") 
        self.mock_config.get_agent_config = MagicMock(return_value=agent_config) # Needed by Orchestrator itself
        # Mock get_agent_descriptions if used by _handle_meta_query or others
        self.mock_config.get_agent_descriptions = MagicMock(return_value=self.mock_config._config["agents"]["agent_descriptions"])
        # --- END Explicit Mocks ---
        
        # --- PATCH UnifiedConfig.get_instance START ---
        self.config_patcher = patch('src.config.unified_config.UnifiedConfig.get_instance', return_value=self.mock_config)
        # Start the patcher - subsequent calls to UnifiedConfig.get_instance() will return mock_config
        self.mock_get_instance = self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop) # Ensure patch stops after test
        # --- PATCH UnifiedConfig.get_instance END ---
        
        # --- PATCH ProviderFactory.create START ---
        # Create a mock provider instance to be returned by the factory
        self.mock_provider_instance = MagicMock(spec=BaseProvider)
        self.mock_provider_instance.supports_tools = True # Assume tool support for tests
        self.provider_patcher = patch('src.core.provider_factory.ProviderFactory.create', return_value=self.mock_provider_instance)
        self.mock_provider_create = self.provider_patcher.start()
        self.addCleanup(self.provider_patcher.stop)
        # --- PATCH ProviderFactory.create END ---
        
        # Mock PromptTemplate
        self.mock_prompt_template = MagicMock(spec=PromptTemplate)
        
        def mock_render_prompt(template_id, variables=None):
            if template_id == 'orchestrator':
                return ("System prompt for Orchestrator", None)
            elif template_id == 'answer_meta_query':
                return ("Prompt for answering meta query", "usage-meta")
            elif template_id == 'use_case_classifier':
                 return ("Prompt for use case classification", "usage-uc")
            elif template_id == 'plan_generation':
                 return ("Prompt for plan generation", "usage-plan")
            else:
                # Simulate template loading behavior - raise error if not found
                raise ValueError(f"Template {template_id} not found")
                
        self.mock_prompt_template.render_prompt = MagicMock(side_effect=mock_render_prompt)
        self.mock_prompt_template.record_prompt_performance = MagicMock()
        
        # Mock ModelSelector
        self.mock_model_selector = MagicMock(spec=ModelSelector)
        # Return a mock object that mimics the Model enum member behavior
        mock_model_enum_member = MagicMock()
        mock_model_enum_member.value = "test-model" 
        self.mock_model_selector.select_model.return_value = mock_model_enum_member
        self.mock_model_selector.get_system_prompt.return_value = "System prompt for test"
        
        # Mock RequestAnalyzer
        self.mock_request_analyzer = MagicMock(spec=RequestAnalyzer)
        
        # Configure intent classification
        def classify_intent(request):
            prompt = request.get("prompt", "").lower()
            if "what tools" in prompt or "agent used" in prompt:
                return "META"
            elif "capital of france" in prompt or "write a poem" in prompt:
                return "QUESTION"
            else:
                return "TASK"
                
        self.mock_request_analyzer.classify_request_intent.side_effect = classify_intent
        
        # Configure agent assignments
        def get_agent_assignments(request, available_agents, agent_descriptions):
            prompt = request.get("prompt", "").lower()
            if "python" in prompt or "code" in prompt:
                return [("coding_assistant", 0.9)]
            elif "write article" in prompt or "blog post" in prompt:
                return [("content_creator", 0.8)]
            else:
                return []
                
        self.mock_request_analyzer.get_agent_assignments.side_effect = get_agent_assignments
        
        # Mock ToolFinderAgent
        self.mock_tool_finder = MagicMock(spec=ToolFinderAgent)
        self.mock_tool_finder.process_request.return_value = {"selected_tools": ["code_search", "code_execution"]}
        
        # Mock ResponseAggregator
        self.mock_response_aggregator = MagicMock(spec=ResponseAggregator)
        self.mock_response_aggregator.aggregate_responses.return_value = {
            "content": "Aggregated response",
            "agent_id": "orchestrator",
            "status": "success"
        }
        
        # Mock agents
        self.mock_coding_assistant = MagicMock()
        self.mock_coding_assistant.process_request.return_value = {
            "content": "Coding solution",
            "agent_id": "coding_assistant",
            "status": "success"
        }
        
        self.mock_content_creator = MagicMock()
        self.mock_content_creator.process_request.return_value = {
            "content": "Created content",
            "agent_id": "content_creator",
            "status": "success"
        }
        
        # Mock AgentRegistry
        self.mock_registry = MagicMock(spec=AgentRegistry)
        self.mock_registry.get_all_agents.return_value = ["orchestrator", "coding_assistant", "content_creator"]
        
        # Mock AgentFactory
        self.mock_agent_factory = MagicMock(spec=AgentFactory)
        self.mock_agent_factory.registry = self.mock_registry
        
        def create_agent(agent_id, **kwargs):
            if agent_id == "coding_assistant":
                return self.mock_coding_assistant
            elif agent_id == "content_creator":
                return self.mock_content_creator
            else:
                return None
                
        self.mock_agent_factory.create.side_effect = create_agent
        
        # Patch metrics service
        self.metrics_service_patch = patch('src.agents.orchestrator.RequestMetricsService')
        self.mock_metrics_service = self.metrics_service_patch.start()
        self.addCleanup(self.metrics_service_patch.stop) # Use addCleanup for metrics patch too
        
        # Create the Orchestrator instance - NOW it will get the MOCKED config via the patch
        self.orchestrator = Orchestrator(
            agent_factory=self.mock_agent_factory,
            tool_finder_agent=self.mock_tool_finder,
            request_analyzer=self.mock_request_analyzer,
            response_aggregator=self.mock_response_aggregator,
            unified_config=self.mock_config,
            logger=logger,
            prompt_template=self.mock_prompt_template,
            model_selector=self.mock_model_selector,
            ai_instance=self.mock_ai
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.metrics_service_patch.stop()
    
    def test_meta_query_handling(self):
        """Test that META queries are handled directly by Orchestrator."""
        # Create a request about the system
        request = {
            "prompt": "What tools do you have available?",
            "request_id": "test-meta-1"
        }
        
        # Mock _handle_meta_query to return a pre-defined response
        with patch.object(self.orchestrator, '_handle_meta_query') as mock_handle_meta:
            mock_handle_meta.return_value = {
                "content": "Here are the tools: code_search, code_execution",
                "agent_id": "orchestrator",
                "status": "success"
            }
            
            # Process the request
            response = self.orchestrator.process_request(request)
            
            # Verify request analyzer was called to classify intent
            self.mock_request_analyzer.classify_request_intent.assert_called_once_with(request)
            
            # Verify meta handler was called
            mock_handle_meta.assert_called_once_with(request)
            
            # Verify response structure
            self.assertIsNotNone(response, "No response received")
            self.assertEqual(response["agent_id"], "orchestrator", "Response should be from orchestrator")
            self.assertEqual(response["status"], "success", "Response status should be success")
            self.assertIn("tools", response["content"], "Response should mention tools")
            
            # Verify specialized agents were not used
            self.mock_agent_factory.create.assert_not_called()
    
    def test_question_handling(self):
        """Test that QUESTION queries are handled directly with LLM."""
        from src.core.tool_enabled_ai import AI # Import AI for patching target
        
        # Create a general knowledge question
        request = {
            "prompt": "What is the capital of France?",
            "request_id": "test-question-1"
        }
        
        # Patch the AI constructor where it's used in orchestrator.py to return our mock AI.
        # Also patch BaseAgent.process_request as it's likely called for the actual response generation.
        with patch('src.agents.orchestrator.AI', return_value=self.mock_ai) as mock_ai_constructor, \
             patch('src.agents.base_agent.BaseAgent.process_request') as mock_super_process:
            
            # Configure the mock BaseAgent.process_request to return the expected answer
            mock_super_process.return_value = {
                "content": "The capital of France is Paris.",
                "agent_id": "orchestrator",
                "status": "success"
            }
            
            # Process the request
            response = self.orchestrator.process_request(request)
            
            # Verify request analyzer was called to classify intent
            self.mock_request_analyzer.classify_request_intent.assert_called_once_with(request)

            # Verify AI constructor was called (or attempted) with the correct model
            mock_ai_constructor.assert_called_once()
            args, kwargs = mock_ai_constructor.call_args
            self.assertEqual(kwargs.get('model'), 'test-model')
            
            # Verify BaseAgent.process_request was called (this handles the AI call via self.ai_instance)
            mock_super_process.assert_called_once()
            
            # Verify response structure comes from the patched BaseAgent call
            self.assertIsNotNone(response, "No response received")
            self.assertEqual(response["agent_id"], "orchestrator", "Response should be from orchestrator")
            self.assertEqual(response["status"], "success", "Response status should be success")
            self.assertIn("Paris", response["content"], "Response should contain the answer")
            
            # Verify specialized agents were not used
            self.mock_agent_factory.create.assert_not_called()
            self.mock_request_analyzer.get_agent_assignments.assert_not_called()
    
    def test_task_with_specialized_agent(self):
        """Test that TASK queries are routed to specialized agents."""
        # Create a coding request
        request = {
            "prompt": "Write a Python function to calculate factorial",
            "request_id": "test-task-1"
        }
        
        # Process the request
        response = self.orchestrator.process_request(request)
        
        # Verify request analyzer was called to classify intent and find agents
        self.mock_request_analyzer.classify_request_intent.assert_called_once_with(request)
        self.mock_request_analyzer.get_agent_assignments.assert_called_once()
        
        # Verify tool finder was called
        self.mock_tool_finder.process_request.assert_called_once()
        
        # Verify agent was created and called
        self.mock_agent_factory.create.assert_called_with("coding_assistant")
        self.mock_coding_assistant.process_request.assert_called_once()
        
        # Verify response aggregator was called
        self.mock_response_aggregator.aggregate_responses.assert_called_once()
        
        # Verify response structure
        self.assertIsNotNone(response, "No response received")
        self.assertEqual(response["content"], "Aggregated response", "Response should be the aggregated content")
    
    def test_task_with_fallback(self):
        """Test fallback to Orchestrator when no specialized agents are found for a task."""
        # Create a request that doesn't match any specialized agent
        request = {
            "prompt": "Tell me about quantum physics",
            "request_id": "test-task-fallback"
        }
        
        # Configure get_agent_assignments to return empty
        self.mock_request_analyzer.get_agent_assignments.return_value = []
        
        # Configure BaseAgent.process_request (super) to return a simple answer
        with patch('src.agents.base_agent.BaseAgent.process_request') as mock_super_process:
            mock_super_process.return_value = {
                "content": "Quantum physics is a branch of physics...",
                "agent_id": "orchestrator",
                "status": "success"
            }
            
            # Process the request
            response = self.orchestrator.process_request(request)
            
            # Verify request analyzer was called to classify intent and find agents
            self.mock_request_analyzer.classify_request_intent.assert_called_once_with(request)
            self.mock_request_analyzer.get_agent_assignments.assert_called_once()
            
            # Verify tool finder was called
            self.mock_tool_finder.process_request.assert_called_once()
            
            # Verify BaseAgent.process_request was called for fallback
            mock_super_process.assert_called_once()
            
            # Verify response structure
            self.assertIsNotNone(response, "No response received")
            self.assertEqual(response["agent_id"], "orchestrator", "Response should be from orchestrator")
            self.assertEqual(response["status"], "success", "Response status should be success")
            self.assertIn("Quantum physics", response["content"], "Response should contain relevant content")
            
            # Verify specialized agents were not used
            self.mock_coding_assistant.process_request.assert_not_called()
            self.mock_content_creator.process_request.assert_not_called()
    
    def test_meta_data_gathering(self):
        """Test the internal methods for gathering META query information."""
        # Test _get_available_agents_info
        with patch.object(self.mock_registry, 'get_all_agents', return_value=["agent1", "agent2"]):
            agent_info = self.orchestrator._get_available_agents_info()
            self.assertIsInstance(agent_info, str, "Should return a string")
            self.assertIn("agent1", agent_info, "Should include agent1")
            self.assertIn("agent2", agent_info, "Should include agent2")
        
        # Test _get_available_tools_info when tool_finder has no tool_registry
        # First remove the attribute to simulate missing tool_registry
        delattr(self.mock_tool_finder, 'tool_registry')
        tool_info = self.orchestrator._get_available_tools_info()
        self.assertEqual(tool_info, "No tool information available.", "Should handle missing tool_registry")
        
        # Restore tool_registry with a mock
        self.mock_tool_finder.tool_registry = MagicMock()
        self.mock_tool_finder.tool_registry.get_tool_schemas.return_value = {
            "tool1": {"description": "Tool 1 description"},
            "tool2": {"description": "Tool 2 description"}
        }
        
        tool_info = self.orchestrator._get_available_tools_info()
        self.assertIn("tool1", tool_info, "Should include tool1")
        self.assertIn("Tool 1 description", tool_info, "Should include tool1 description")


if __name__ == "__main__":
    unittest.main() 