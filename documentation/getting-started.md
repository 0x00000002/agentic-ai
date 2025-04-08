# Getting Started

## Basic Usage

Here's a simple example to get started with Agentic-AI:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.unified_config import UnifiedConfig

# Initialize configuration (if needed, often handled internally)
# config = UnifiedConfig.get_instance()

# Create AI instance
ai = ToolEnabledAI(
    model="claude-3-5-haiku" # Example: Use a specific model
)

# Make a simple request
response = ai.request("What is the capital of France?")
print(response)
```

## Adding Tool Capabilities

Register tools to allow the AI to perform actions:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
import requests
import datetime

def get_weather(location: str):
    # ... (function remains same) ...

def get_current_time():
    # ... (function remains same) ...

# Create tool-enabled AI
ai = ToolEnabledAI(
    model="claude-3-7-sonnet" # Example: A model good at tool use
)

# Register tools
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get the current weather for a specific location."
)
ai.register_tool(
    tool_name="get_current_time",
    tool_function=get_current_time,
    description="Get the current date and time."
)

# The AI will now use the tool when appropriate
response = ai.request("What's the weather like in Tokyo today?")
print(f"Weather Response: {response}")

response = ai.request("What time is it now?")
print(f"Time Response: {response}")
```

## Automatic Tool Finding

Enable the AI to automatically select relevant tools:

```python
from src.core.tool_enabled_ai import ToolEnabledAI

# Dummy tool functions
def get_weather(location: str):
    return f"It's always sunny in {location}!"

def calculator_function(expression: str):
    try: return str(eval(expression))
    except: return "Invalid expression"

# Create AI with auto tool finding
ai = ToolEnabledAI(
    model="claude-3-7-sonnet",
    auto_tool_finding=True # Enable auto finding
)

# Register tools (still required for the AI to know about them)
ai.register_tool("get_weather", get_weather, "Get weather for a location")
ai.register_tool("calculate", calculator_function, "Perform calculations")

# The AI will automatically select the appropriate tool
response = ai.request("What's the weather like in Paris today?")
print(f"Auto Weather: {response}")

response = ai.request("Calculate 3 + 5 * 2")
print(f"Auto Calculation: {response}")
```

See the [Tool Integration](tools/overview.md) section for more details.
