# Getting Started

## Basic Usage

Here's a simple example to get started with Agentic-AI:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.config import configure, get_config

# Configure the framework (optional - uses defaults otherwise)
# Example: Use a specific model
# configure(model="claude-3-haiku")

# Create AI instance (uses the configured model or default)
ai = ToolEnabledAI()

# Make a simple request (uses the underlying provider)
# Note: For tool usage, use process_prompt() instead of request()
response = ai.request("What is the capital of France?")
print(response)
```

## Using Tools

Tools allow the AI to perform actions. They are typically defined in `src/config/tools.yml` and loaded automatically by the `ToolManager`.

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.config import configure, UseCasePreset

# Configure the framework, maybe selecting a model good at tool use
configure(
    model="claude-3-5-sonnet",
    use_case=UseCasePreset.CHAT # Or another relevant use case
)

# Create AI instance. It automatically gets a ToolManager
# which loads tools defined in tools.yml.
ai = ToolEnabledAI()

# The AI will use tools loaded by ToolManager when appropriate.
# Use process_prompt() to enable the tool-calling loop.
response = ai.process_prompt("What's the weather like in Tokyo today?")
print(f"Weather Response: {response}")

response = ai.process_prompt("What time is it now?")
print(f"Time Response: {response}")
```

See the [Tool Integration](tools/overview.md) section for more details on defining and managing tools.

## Using Agents (Coordinator)

For more complex interactions involving specific workflows or routing, you often use the `Coordinator` agent:

```python
from src.agents import Coordinator
from src.config import configure, UseCasePreset

# Configure the framework (e.g., model, use case)
# configure(model="claude-3-haiku", use_case=UseCasePreset.CHAT)

# Create a Coordinator instance (uses default dependencies & config)
coordinator = Coordinator()

# Prepare a request for the coordinator
request_data = {
    "prompt": "Tell me a joke and then tell me the weather in London.",
}

# Process the request through the coordinator
response_coord = coordinator.process_request(request_data)

# Print the final content from the coordinator's response
print(f"Coordinator Response: {response_coord.get('content')}")
```

Refer to the [Agent System](agents.md) documentation for more on agents.
