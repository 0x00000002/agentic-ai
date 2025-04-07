# Tool Integration

## Overview

Tools allow the AI to perform actions and retrieve information beyond its training data. The Agentic-AI framework includes a sophisticated tool management system that enables:

1. Registering Python functions as tools
2. Automatic discovery of relevant tools for user requests
3. Systematic execution of tool calls with proper error handling

## Tool Components

- **ToolManager**: Central service that coordinates tool registration (via `ToolRegistry`), execution (via `ToolExecutor`), and retrieval of tool definitions.
- **ToolRegistry**: Stores tool definitions (`ToolDefinition`), handles provider-specific formatting, and tracks usage statistics.
- **ToolExecutor**: Executes the actual tool functions safely with timeout and retry logic.
- **Models** (`src/tools/models.py`): Defines core data structures like `ToolDefinition`, `ToolCall`, and `ToolResult` using Pydantic.

## Creating and Registering Tools

Tools are Python functions that can be made available to the AI. The core element is the `ToolDefinition` model, which includes:

- `name`: A unique name for the tool.
- `description`: A clear description of what the tool does (used by the LLM).
- `parameters_schema`: A JSON schema defining the expected input parameters.
- `function`: The actual Python callable that implements the tool's logic.

Tools are registered with the `ToolManager` (often during application setup or via configuration loading), which uses the `ToolRegistry` internally.

```python
from src.tools import ToolManager, ToolDefinition
# Assume get_weather is defined elsewhere
# from my_tools import get_weather

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Implementation...
    return f"Weather data for {location}"

# Create a ToolDefinition
weather_tool_def = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a specific city or location.",
    parameters_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location (e.g., 'Paris, France')"
            }
        },
        "required": ["location"]
    },
    function=get_weather
)

# Get a ToolManager instance (e.g., from dependency injection or create default)
tool_manager = ToolManager()

# Register the tool definition
tool_manager.register_tool(weather_tool_def.name, weather_tool_def)

```

## Tool Execution Flow

When `ToolEnabledAI.process_prompt` is called:

1.  `ToolEnabledAI` retrieves available tool definitions from `ToolManager`.
2.  It passes these definitions (formatted by `ToolRegistry` via the provider) to the underlying AI Provider (e.g., OpenAI, Anthropic) along with the user prompt and conversation history.
3.  The LLM decides if a tool needs to be called. If so, the provider returns a `ProviderResponse` containing one or more `ToolCall` objects (with name, arguments, ID).
4.  `ToolEnabledAI` receives the response.
5.  If `ToolCall` objects are present, `ToolEnabledAI` iterates through them.
6.  For each `ToolCall`, it invokes its internal `_execute_tool_call(tool_call)` method, which in turn uses `ToolManager.execute_tool(tool_name=..., **arguments)`.
7.  `ToolManager` uses `ToolExecutor` to run the actual tool function associated with the `tool_name`.
8.  `ToolExecutor` returns a `ToolResult` (success/failure, result/error) to `ToolManager`, which passes it back to `ToolEnabledAI` via `_execute_tool_call`.
9.  `ToolEnabledAI` formats the `ToolResult` into the appropriate message format(s) for the specific provider (using the provider's `_add_tool_message` method) and adds these messages to the conversation history.
10. `ToolEnabledAI` calls the provider _again_ with the updated history (including tool results).
11. The loop repeats if the provider requests more tool calls (up to a maximum iteration limit).
12. Once the provider returns a final response without tool calls, `ToolEnabledAI` returns the text content.

This loop allows the AI to use tools iteratively to accomplish tasks.

## Provider-Specific Tool Call Handling

Different AI providers handle tool calls in slightly different ways. The framework normalizes these differences to provide a consistent experience:

### OpenAI Provider

The OpenAI provider automatically parses tool call arguments from JSON strings into Python dictionaries:

- Tool call arguments returned by the OpenAI API are JSON strings by default
- The `_convert_response` method automatically parses these strings into Python dictionaries
- If JSON parsing fails, the raw string is preserved in a `_raw_args` field
- This automatic parsing makes it easier to work with tool arguments in your code

```python
# Example of what happens internally when OpenAI returns a tool call
# Original from OpenAI:
# tool_call.function.arguments = '{"location": "New York", "unit": "celsius"}'

# After _convert_response processing:
tool_call.arguments = {
    "location": "New York",
    "unit": "celsius"
}
```

This ensures that tool calls work consistently across different providers while taking advantage of each provider's specific capabilities.
