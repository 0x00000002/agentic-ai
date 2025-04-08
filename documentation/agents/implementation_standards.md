# Agent Implementation Standards

This document outlines the standard practices for implementing agents in the Agentic-AI framework. Following these guidelines ensures consistency, maintainability, and proper integration with the framework's architecture.

## Agent Inheritance and Structure

All agents should:

1. Inherit from `BaseAgent` class
2. Implement the required methods from the `AgentInterface`
3. Follow consistent initialization patterns

```python
from .base_agent import BaseAgent

class YourAgent(BaseAgent):
    """
    Agent for performing specific functionality.

    Each agent should have a clear docstring describing its purpose and capabilities.
    """

    def __init__(self, agent_id="your_agent", **kwargs):
        """
        Initialize the agent.

        Args:
            agent_id: ID of the agent (optional, defaults to class name)
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)

        # Initialize agent-specific attributes
        # ...
```

## Configuration Management

### Use the ConfigFactory/ConfigManager

Always use the configuration system rather than hardcoding values or using environment variables directly:

```python
# CORRECT:
def __init__(self, **kwargs):
    super().__init__(agent_id="your_agent", **kwargs)

    # Get agent-specific configuration
    self.some_option = self.agent_config.get("some_option", "default_value")

    # Use ConfigFactory for framework-wide configuration
    config_factory = kwargs.get("config_factory", ConfigFactory.get_instance())
    # Use the configuration factory for any additional config needs
```

### Avoid Direct Environment Variable Access

```python
# INCORRECT:
api_key = os.getenv("SOME_API_KEY")  # Direct environment access

# CORRECT:
config_factory = kwargs.get("config_factory", ConfigFactory.get_instance())
provider = ProviderFactory.create(
    provider_type="some_provider",
    model_id=self.model_id,
    config_factory=config_factory,
    logger=self.logger
)
```

## AI and Provider Usage

### Use the AI Instance

Always use the `self.ai` instance provided by `BaseAgent` for AI interactions:

```python
# CORRECT:
response = self.ai.request(prompt)

# INCORRECT:
ai_instance = AI()
response = ai_instance.generate(prompt)
```

### Use ProviderFactory for Provider Access

Never instantiate provider clients directly:

```python
# INCORRECT:
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORRECT:
provider = ProviderFactory.create(
    provider_type="openai",
    model_id=self.model_id,
    config_factory=self.config_factory,
    logger=self.logger
)
```

### Interface with Providers Through Standardized Interfaces

If you need specialized provider capabilities (like multimedia), use the appropriate interface:

```python
# Check for special capabilities
if isinstance(provider, MultimediaProviderInterface):
    # Use the interface methods
    transcription, metadata = provider.transcribe_audio(audio_file)
```

## Error Handling

### Use Standardized Exceptions

Always use the framework's exception hierarchy:

```python
from ..exceptions import AIAgentError, AIProviderError, ErrorHandler

# Raise appropriate exceptions
raise AIAgentError("Failed to process request", agent_id=self.agent_id)
```

### Use ErrorHandler for Consistent Error Handling

```python
try:
    # Agent operations
except Exception as e:
    error_response = ErrorHandler.handle_error(
        AIAgentError(f"Operation failed: {str(e)}", agent_id=self.agent_id),
        self.logger
    )
    self.logger.error(f"Error: {error_response['message']}")

    return {
        "error": error_response['message'],
        "agent_id": self.agent_id,
        "status": "error"
    }
```

### Proper Error Propagation

```python
def _internal_method(self):
    try:
        # Operations
    except Exception as e:
        # Wrap in appropriate exception
        raise AIAgentError(f"Internal operation failed: {str(e)}", agent_id=self.agent_id)
```

## Logging

### Use the Logger from BaseAgent

```python
# CORRECT:
self.logger.info("Processing request")
self.logger.warning("Potential issue: {issue}")
self.logger.error(f"Error occurred: {str(error)}")

# INCORRECT:
print("Processing request")  # Avoid print statements
```

### Log at Appropriate Levels

- `debug`: Detailed information, typically useful only for diagnostics
- `info`: Confirmation that things are working as expected
- `warning`: Indication that something unexpected happened, but the application still works
- `error`: Due to a more serious problem, some functionality couldn't be performed
- `critical`: A serious error indicating that the program itself may be unable to continue running

## Response Format

### Standard Response Structure

Ensure all responses follow the standard format:

```python
return {
    "content": result,  # Main content/result
    "metadata": {       # Optional metadata
        "key": "value",
        # Additional metadata
    },
    "agent_id": self.agent_id,
    "status": "success"  # or "error"
}
```

### Error Response Structure

```python
return {
    "error": error_message,
    "agent_id": self.agent_id,
    "status": "error"
}
```

## Method Implementation

### Essential Methods

Every agent should implement:

1. `process_request(self, request)`: Main entry point for processing
2. `can_handle(self, request)`: Returns confidence score (0.0-1.0)

### Method Documentation

Always include clear docstrings:

```python
def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a request and return a response.

    Args:
        request: Request object containing prompt and metadata

    Returns:
        Response object with content and metadata

    Raises:
        AIAgentError: If processing fails
    """
```

## Testing

### Test All Error Paths

Ensure your agent handles failures gracefully:

```python
def test_agent_handles_failure(self):
    agent = YourAgent()
    response = agent.process_request({"prompt": "test", "will_fail": True})
    assert response["status"] == "error"
    assert "error" in response
```

### Test Configuration Loading

```python
def test_agent_loads_config(self):
    # Create mock config
    config = {"some_option": "test_value"}
    config_manager = MockConfigManager(agent_configs={"your_agent": config})

    agent = YourAgent(config_manager=config_manager)
    assert agent.some_option == "test_value"
```

## Performance Considerations

### Lazy Loading

Use lazy loading for heavy resources:

```python
def _load_model(self):
    if not hasattr(self, "_model"):
        self.logger.info("Loading model...")
        self._model = SomeHeavyModel()
    return self._model
```

### Resource Cleanup

Ensure proper cleanup of resources:

```python
def __del__(self):
    # Clean up resources
    if hasattr(self, "_connection") and self._connection:
        self.logger.info("Closing connection...")
        self._connection.close()
```

## Example: Complete Agent Implementation

```python
"""
Example agent implementation demonstrating best practices.
"""
from typing import Dict, Any
from .base_agent import BaseAgent
from ..config.config_factory import ConfigFactory
from ..exceptions import AIAgentError, ErrorHandler

class ExampleAgent(BaseAgent):
    """
    Example agent demonstrating implementation standards.
    """

    def __init__(self, agent_id="example", **kwargs):
        """
        Initialize the example agent.

        Args:
            agent_id: ID of the agent (defaults to "example")
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)

        # Load configuration
        self.option1 = self.agent_config.get("option1", "default_value")
        self.option2 = self.agent_config.get("option2", 100)

        self.logger.info(f"Initialized {self.agent_id} agent with options: {self.option1}, {self.option2}")

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and return a response.

        Args:
            request: Request object containing prompt and metadata

        Returns:
            Response object with content and metadata
        """
        try:
            self.logger.info(f"Processing request: {request.get('prompt', '')[:50]}...")

            # Process the request
            if not request.get("prompt"):
                raise AIAgentError("Request must include a prompt", agent_id=self.agent_id)

            # Use the AI to process
            prompt = request["prompt"]
            response = self.ai.request(prompt)

            # Return standardized response
            return {
                "content": response,
                "metadata": {
                    "option1": self.option1,
                    "length": len(response)
                },
                "agent_id": self.agent_id,
                "status": "success"
            }

        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Error processing request: {str(e)}", agent_id=self.agent_id),
                self.logger
            )
            self.logger.error(f"Request error: {error_response['message']}")

            return {
                "error": error_response['message'],
                "agent_id": self.agent_id,
                "status": "error"
            }

    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.

        Args:
            request: The request object

        Returns:
            Confidence score (0.0-1.0)
        """
        # Check for keywords indicating this agent should handle the request
        if not isinstance(request.get("prompt"), str):
            return 0.0

        prompt = request["prompt"].lower()
        keywords = ["example", "demo", "test"]

        for keyword in keywords:
            if keyword in prompt:
                return 0.8

        # Default confidence
        return 0.2
```
