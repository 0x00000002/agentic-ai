# Multi-Agent Implementation Plan

This document outlines the detailed implementation plan for the multi-agent architecture in the Agentic-AI framework.

## Implementation Phases

### Phase 1: Foundation

#### 1.1 Agent Interface Design

```python
# src/agents/interfaces.py
from typing import Protocol, Dict, List, Any, Optional, Union, AsyncIterator
from typing_extensions import runtime_checkable
from ..core.interfaces import AIInterface

@runtime_checkable
class AgentInterface(Protocol):
    """Interface for agent implementation."""

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and return a response.

        Args:
            request: The request object containing prompt and metadata

        Returns:
            Response object with content and metadata
        """
        ...

    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.

        Args:
            request: The request object

        Returns:
            Confidence score (0.0-1.0) indicating ability to handle
        """
        ...
```

#### 1.2 Base Agent Implementation

```python
# src/agents/base_agent.py
from typing import Dict, Any, Optional, List
from ..core.base_ai import AIBase
from ..core.tool_enabled_ai import AI
from ..tools.tool_manager import ToolManager
from ..utils.logger import LoggerInterface
from .interfaces import AgentInterface

class BaseAgent(AgentInterface):
    """Base implementation for all agents."""

    def __init__(self,
                 ai_instance: Optional[AIBase] = None,
                 tool_manager: Optional[ToolManager] = None,
                 logger: Optional[LoggerInterface] = None):
        """Initialize the base agent."""
        self.ai = ai_instance or AI()
        self.tool_manager = tool_manager
        self.logger = logger

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation of request processing."""
        response = self.ai.request(request.get("prompt", ""))
        return {"content": response}

    def can_handle(self, request: Dict[str, Any]) -> float:
        """Default implementation returns low confidence."""
        return 0.1
```

#### 1.3 Agent Registry

```python
# src/agents/agent_registry.py
from typing import Dict, Type, List, Optional
from .interfaces import AgentInterface
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface, LoggerFactory

class AgentRegistry:
    """Registry for all available agents."""

    def __init__(self, logger: Optional[LoggerInterface] = None):
        """Initialize the agent registry."""
        self.agents: Dict[str, Type[AgentInterface]] = {}
        self.logger = logger or LoggerFactory.create(name="agent_registry")

    def register(self, name: str, agent_class: Type[AgentInterface]) -> None:
        """Register an agent class."""
        self.agents[name] = agent_class
        self.logger.info(f"Registered agent: {name}")

    def get_agent_class(self, name: str) -> Optional[Type[AgentInterface]]:
        """Get an agent class by name."""
        return self.agents.get(name)

    def get_all_agents(self) -> List[str]:
        """Get names of all registered agents."""
        return list(self.agents.keys())
```

#### 1.4 Agent Factory

```python
# src/agents/agent_factory.py
from typing import Dict, Any, Optional, Type
from .interfaces import AgentInterface
from .agent_registry import AgentRegistry
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.config_manager import ConfigManager

class AgentFactory:
    """Factory for creating agent instances."""

    def __init__(self,
                 registry: Optional[AgentRegistry] = None,
                 config_manager: Optional[ConfigManager] = None,
                 logger: Optional[LoggerInterface] = None):
        """Initialize the agent factory."""
        self.registry = registry or AgentRegistry()
        self.config_manager = config_manager or ConfigManager()
        self.logger = logger or LoggerFactory.create(name="agent_factory")

    def create(self,
              agent_type: str,
              **kwargs) -> AgentInterface:
        """
        Create an agent instance.

        Args:
            agent_type: Type of agent to create
            kwargs: Additional arguments for agent initialization

        Returns:
            Initialized agent instance
        """
        agent_class = self.registry.get_agent_class(agent_type)
        if not agent_class:
            self.logger.warning(f"Agent type not found: {agent_type}, using BaseAgent")
            return BaseAgent(**kwargs)

        # Get agent-specific configuration if available
        agent_config = self.config_manager.get_agent_config(agent_type)
        if agent_config:
            # Merge configuration with provided kwargs (kwargs take precedence)
            agent_kwargs = {**agent_config, **kwargs}
        else:
            agent_kwargs = kwargs

        try:
            return agent_class(**agent_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_type}: {str(e)}")
            return BaseAgent(**kwargs)
```

### Phase 2: Core Agents Implementation (3-4 weeks)

#### 2.1 Request Rooter Agent

```python
# src/agents/request_rooter.py
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from .agent_factory import AgentFactory
from ..utils.logger import LoggerInterface

class RequestRooterAgent(BaseAgent):
    """
    Agent responsible for routing requests to appropriate specialized agents.
    """

    def __init__(self,
                 agent_factory: Optional[AgentFactory] = None,
                 **kwargs):
        """Initialize the request rooter agent."""
        super().__init__(**kwargs)
        self.agent_factory = agent_factory or AgentFactory()

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by routing it to appropriate agents.

        Args:
            request: The request object

        Returns:
            Aggregated response from all involved agents
        """
        # Analyze the request to determine appropriate agents
        agent_assignments = self._analyze_request(request)

        # If no specialized agents needed, handle directly
        if not agent_assignments:
            return super().process_request(request)

        # Process with each identified agent
        responses = []
        for agent_type, confidence in agent_assignments:
            agent = self.agent_factory.create(agent_type)
            response = agent.process_request(request)
            responses.append({
                "agent": agent_type,
                "confidence": confidence,
                "response": response
            })

        # Aggregate responses (simple concatenation for now)
        return self._aggregate_responses(responses)

    def _analyze_request(self, request: Dict[str, Any]) -> List[tuple]:
        """
        Analyze the request to determine appropriate agents.

        Args:
            request: The request object

        Returns:
            List of (agent_type, confidence) tuples
        """
        # Use AI to analyze the request
        prompt = f"""Analyze this user request and determine which specialized agents should handle it:
        Request: {request.get('prompt', '')}

        Available agents:
        - listener: Handles audio processing and speech recognition
        - translator: Translates between languages
        - website_parser: Searches websites for information
        - content_generator: Creates multimedia content
        - action_planner: Breaks complex tasks into subtasks
        - mcp_searcher: Finds relevant Model-Centric Processes
        - paralleliser: Executes tasks in parallel

        Return a JSON list of [agent_name, confidence] pairs, where confidence is 0.0-1.0.
        Only include agents with confidence > 0.5. If no agents are appropriate, return [].
        """

        try:
            response = self.ai.request(prompt)
            # Parse response as JSON list of [agent_name, confidence] pairs
            import json
            agents = json.loads(response)
            return [(agent[0], agent[1]) for agent in agents if len(agent) == 2]
        except Exception as e:
            self.logger.error(f"Failed to analyze request: {str(e)}")
            return []

    def _aggregate_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents.

        Args:
            responses: List of agent responses

        Returns:
            Aggregated response
        """
        if not responses:
            return {"content": "No agents were able to process your request."}

        # Sort by confidence
        sorted_responses = sorted(responses, key=lambda r: r.get("confidence", 0), reverse=True)

        # For now, just return the highest confidence response
        # TODO: Implement more sophisticated aggregation
        best_response = sorted_responses[0]["response"]

        return best_response
```

#### 2.2 Action Planner Agent

```python
# src/agents/action_planner.py
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent

class ActionPlannerAgent(BaseAgent):
    """
    Agent responsible for planning complex actions by breaking them down into subtasks.
    """

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by creating an action plan.

        Args:
            request: The request object

        Returns:
            Response with action plan
        """
        prompt = f"""
        Create a detailed action plan for the following request:
        {request.get('prompt', '')}

        Break it down into subtasks with the following information for each:
        1. Task description
        2. Tools or agents needed
        3. Dependencies on other tasks
        4. Expected output

        Return the plan as a structured JSON object.
        """

        response = self.ai.request(prompt)

        try:
            import json
            action_plan = json.loads(response)
            return {
                "content": "I've created an action plan for your request.",
                "action_plan": action_plan
            }
        except json.JSONDecodeError:
            return {
                "content": "I've analyzed your request and created a plan:\n\n" + response
            }

    def can_handle(self, request: Dict[str, Any]) -> float:
        """Determine if this agent can handle the request."""
        # Check if request is complex and needs planning
        prompt = f"""
        On a scale of 0.0 to 1.0, how complex is this request and would it benefit from being broken down into subtasks?
        Request: {request.get('prompt', '')}
        Return only a number between 0.0 and 1.0.
        """

        try:
            response = self.ai.request(prompt)
            confidence = float(response.strip())
            return min(max(confidence, 0.0), 1.0)  # Clamp to 0.0-1.0
        except:
            return 0.3  # Default moderate confidence
```

### Phase 3: Specialized Agents (4-6 weeks)

For each specialized agent, we'll need to:

1. Create the agent class
2. Implement specific functionality
3. Develop integration with external services
4. Write tests for the agent

### Phase 4: Integration & Testing (2-3 weeks)

Implement full integration testing across agents, including:

1. E2E request flow
2. Error handling and recovery
3. Performance benchmarking
4. Load testing

## Configuration Updates

Add agent configuration to `config.yml`:

```yaml
agents:
  request_rooter:
    default_model: "gpt-4"
    system_prompt: "You are an expert at analyzing user requests and routing them to specialized agents."

  action_planner:
    default_model: "gpt-4"
    system_prompt: "You are an expert at breaking complex tasks into well-defined subtasks."

  website_parser:
    default_model: "claude-3-opus"
    system_prompt: "You are an expert at finding and extracting relevant information from websites."
    max_pages_per_request: 5
    relevance_threshold: 0.7

  content_generator:
    default_model: "claude-3-sonnet"
    system_prompt: "You are an expert at creating high-quality content based on user specifications."
    supported_formats: ["image", "audio", "video", "diagram"]
    default_service_providers:
      image: "dalle3"
      audio: "elevenlabs"
      video: "runway"
      diagram: "mermaid"
```

## Testing Strategy

### Unit Tests

Create unit tests for each agent:

```python
# tests/agents/test_request_rooter.py
import unittest
from unittest.mock import MagicMock, patch
from src.agents.request_rooter import RequestRooterAgent

class TestRequestRooterAgent(unittest.TestCase):

    def setUp(self):
        # Create mock dependencies
        self.mock_ai = MagicMock()
        self.mock_agent_factory = MagicMock()

        # Create agent with mocks
        self.agent = RequestRooterAgent(
            ai_instance=self.mock_ai,
            agent_factory=self.mock_agent_factory
        )

    def test_analyze_request_speech(self):
        # Test request analysis for speech-related request
        self.mock_ai.request.return_value = '[["listener", 0.9]]'

        request = {"prompt": "Transcribe this audio file"}
        result = self.agent._analyze_request(request)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "listener")
        self.assertAlmostEqual(result[0][1], 0.9)

    def test_process_request_routing(self):
        # Test routing functionality
        self.mock_ai.request.return_value = '[["listener", 0.9]]'

        mock_listener = MagicMock()
        mock_listener.process_request.return_value = {"content": "Transcribed content"}
        self.mock_agent_factory.create.return_value = mock_listener

        request = {"prompt": "Transcribe this audio file"}
        result = self.agent.process_request(request)

        self.mock_agent_factory.create.assert_called_once_with("listener")
        mock_listener.process_request.assert_called_once()
        self.assertEqual(result["content"], "Transcribed content")
```

### Integration Tests

Create integration tests for multi-agent scenarios:

```python
# tests/agents/test_integration.py
import unittest
from src.agents.request_rooter import RequestRooterAgent
from src.agents.action_planner import ActionPlannerAgent
from src.agents.agent_factory import AgentFactory
from src.agents.agent_registry import AgentRegistry

class TestAgentIntegration(unittest.TestCase):

    def setUp(self):
        # Set up registry with actual agent implementations
        self.registry = AgentRegistry()
        self.registry.register("request_rooter", RequestRooterAgent)
        self.registry.register("action_planner", ActionPlannerAgent)

        # Create factory with registry
        self.factory = AgentFactory(registry=self.registry)

    def test_complex_request_flow(self):
        # Test flow from rooter to planner
        rooter = self.factory.create("request_rooter", agent_factory=self.factory)

        request = {
            "prompt": "Create a report on renewable energy, with charts and data from the top 3 research sites"
        }

        response = rooter.process_request(request)

        # Verify response has action plan
        self.assertIn("action_plan", response)
        # Verify plan has appropriate steps
        self.assertTrue(any("chart" in str(step).lower() for step in response["action_plan"]))
```

## Deployment Plan

1. Deploy first to staging environment
2. Implement metrics collection for each agent
3. Start with low traffic allocation (5-10%)
4. Monitor performance and error rates
5. Gradually increase traffic allocation
