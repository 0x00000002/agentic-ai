#!/usr/bin/env python
"""
Test for RequestAnalyzer component.

This test verifies that:
1. The RequestAnalyzer can correctly classify request intents (META, QUESTION, TASK)
2. The agent assignment functionality works correctly
3. Error handling is robust
"""
import os
import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_request_analyzer")


class TestRequestAnalyzer(unittest.TestCase):
    """Test cases for RequestAnalyzer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        from src.agents.request_analyzer import RequestAnalyzer
        from src.config.unified_config import UnifiedConfig
        from src.prompts.prompt_template import PromptTemplate
        from src.core.tool_enabled_ai import AI
        
        # Create mock components
        self.mock_ai = MagicMock(spec=AI)
        self.mock_config = MagicMock(spec=UnifiedConfig)
        self.mock_prompt_template = MagicMock(spec=PromptTemplate)
        
        # Configure mock AI responses for different intent types
        def mock_ai_request(prompt):
            if "classify" in prompt.lower() and "intent" in prompt.lower():
                if "system capabilities" in prompt.lower() or "what tools" in prompt.lower():
                    return "META"
                elif "capital of france" in prompt.lower() or "write a poem" in prompt.lower():
                    return "QUESTION"
                else:
                    return "TASK"
            else:
                return '[["coding_assistant", 0.9], ["solidity_expert", 0.8]]'
        
        self.mock_ai.request = MagicMock(side_effect=mock_ai_request)
        
        # Configure mock prompt template service
        def mock_render_prompt(template_id, variables=None):
            if template_id == 'request_analyzer':
                # Return a dummy system prompt string and None usage_id
                return ("System prompt for Request Analyzer", None)
            elif template_id == 'classify_intent':
                # Return a prompt string containing the user prompt for the mock AI
                user_prompt = variables.get('user_prompt', '')
                return (f"classify intent for: {user_prompt}", "usage-123")
            elif template_id == 'analyze_request':
                # Return a dummy analysis prompt string
                return ("Analyze request prompt", "usage-456")
            elif template_id == 'analyze_tools':
                 # Return a dummy tool analysis prompt string
                 return ("Analyze tools prompt", "usage-789")
            else:
                # Raise error for unmocked templates, matching PromptTemplate behavior
                raise ValueError(f"Template {template_id} not found")
                
        self.mock_prompt_template.render_prompt = MagicMock(side_effect=mock_render_prompt)
        self.mock_prompt_template.record_prompt_performance = MagicMock()

        # Configure mock config
        agent_config = {"confidence_threshold": 0.7, "default_model": "test-model"}
        self.mock_config.get_agent_config = MagicMock(return_value=agent_config)
        
        # Create the RequestAnalyzer instance with mocks
        with patch('src.agents.request_analyzer.AI', return_value=self.mock_ai) as mock_ai_constructor:
            self.analyzer = RequestAnalyzer(
                unified_config=self.mock_config,
                logger=logger,
                prompt_template=self.mock_prompt_template
            )
            # Ensure the AI instance within the analyzer uses the correct system prompt from the mock manager
            mock_ai_constructor.assert_called_once()
            call_args, call_kwargs = mock_ai_constructor.call_args
            self.assertEqual(call_kwargs.get('system_prompt'), "System prompt for Request Analyzer")
            # Keep the mock AI instance for direct manipulation if needed
            self.analyzer._ai = self.mock_ai
    
    def test_classify_meta_intent(self):
        """Test that META queries are correctly classified."""
        request = {
            "prompt": "What tools do you have available?",
            "conversation_history": []
        }
        
        intent = self.analyzer.classify_request_intent(request)
        self.assertEqual(intent, "META", "Should classify as META query")
    
    def test_classify_question_intent(self):
        """Test that QUESTION queries are correctly classified."""
        request = {
            "prompt": "What is the capital of France?",
            "conversation_history": []
        }
        
        intent = self.analyzer.classify_request_intent(request)
        self.assertEqual(intent, "QUESTION", "Should classify as QUESTION query")
    
    def test_classify_task_intent(self):
        """Test that TASK queries are correctly classified."""
        request = {
            "prompt": "Write a Python function to sort a list of dictionaries by a specific key",
            "conversation_history": []
        }
        
        intent = self.analyzer.classify_request_intent(request)
        self.assertEqual(intent, "TASK", "Should classify as TASK query")
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        request = {
            "prompt": "",
            "conversation_history": []
        }
        
        intent = self.analyzer.classify_request_intent(request)
        self.assertEqual(intent, "UNKNOWN", "Empty prompt should be classified as UNKNOWN")
    
    def test_ai_error_handling(self):
        """Test error handling when AI request fails."""
        request = {
            "prompt": "Regular request",
            "conversation_history": []
        }
        
        # Temporarily modify AI to raise an exception
        original_request = self.mock_ai.request
        self.mock_ai.request = MagicMock(side_effect=Exception("AI service failure"))
        
        try:
            intent = self.analyzer.classify_request_intent(request)
            self.assertEqual(intent, "UNKNOWN", "AI failure should result in UNKNOWN classification")
        finally:
            # Restore original mock behavior
            self.mock_ai.request = original_request
    
    def test_get_agent_assignments(self):
        """Test agent assignment functionality."""
        request = {
            "prompt": "Help me debug this Python code",
            "conversation_history": []
        }
        
        available_agents = ["orchestrator", "coding_assistant", "solidity_expert", "content_creator"]
        agent_descriptions = {
            "orchestrator": "Coordinates the system",
            "coding_assistant": "Helps with coding tasks",
            "solidity_expert": "Specializes in Solidity and blockchain",
            "content_creator": "Creates various content"
        }
        
        assignments = self.analyzer.get_agent_assignments(request, available_agents, agent_descriptions)
        
        self.assertIsInstance(assignments, list, "Should return a list of assignments")
        self.assertTrue(all(isinstance(a, tuple) and len(a) == 2 for a in assignments),
                       "Each assignment should be a tuple of (agent_id, confidence)")
        
        # Check if coding_assistant is in the assignments with high confidence
        found_coding_assistant = False
        for agent_id, confidence in assignments:
            if agent_id == "coding_assistant":
                found_coding_assistant = True
                self.assertGreaterEqual(confidence, 0.7, "Coding assistant should have high confidence")
                
        self.assertTrue(found_coding_assistant, "Coding assistant should be in the assignments")
        
    def test_invalid_ai_response_handling(self):
        """Test handling of invalid AI responses in agent assignments."""
        request = {
            "prompt": "Help with something",
            "conversation_history": []
        }
        
        available_agents = ["orchestrator", "coding_assistant"]
        agent_descriptions = {
            "orchestrator": "Coordinates the system",
            "coding_assistant": "Helps with coding tasks"
        }
        
        # Temporarily modify AI to return invalid response
        original_request = self.mock_ai.request
        self.mock_ai.request = MagicMock(return_value="Not valid JSON")
        
        try:
            assignments = self.analyzer.get_agent_assignments(request, available_agents, agent_descriptions)
            self.assertEqual(assignments, [], "Invalid response should result in empty assignments")
        finally:
            # Restore original mock behavior
            self.mock_ai.request = original_request


if __name__ == "__main__":
    unittest.main() 