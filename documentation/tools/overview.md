# Tool Integration

## Overview

Tools allow the AI to perform actions and retrieve information beyond its training data. The Agentic-AI framework includes a tool management system that enables:

1.  **Loading** tool definitions from configuration (`tools.yml`).
2.  **Registering** these tools.
3.  **Formatting** tool definitions for specific AI providers.
4.  **Executing** tool calls requested by the AI asynchronously, with error handling and timeout/retry logic.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Users Of Tools"
        ToolEnabledAI -- Calls Execute (await) --> ToolManager
    end

    subgraph "Tools"
        ToolManager -- Delegates Execution (await) --> ToolExecutor
        ToolManager -- Gets Definition --> ToolRegistry
        ToolManager -- Records Stats --> ToolStatsManager

        ToolExecutor -- Returns (awaitable) --> ToolResult["ToolResult Model"]

        ToolRegistry -- Stores/Provides --> ToolDefinition["ToolDefinition Model"]

        ProviderToolHandler["ProviderToolHandler (src/core/providers)"] -- Gets Definitions --> ToolRegistry

        ToolStatsManager -- Persists --> StatsFile["Tool Stats JSON"]

        UnifiedConfigRef["UnifiedConfig"] -- Provides Config --> ToolRegistry
        UnifiedConfigRef -- Provides Config --> ToolStatsManager
        ToolsYAML["tools.yml"] -- Read By --> UnifiedConfigRef
    end

    subgraph "Dependencies"
       ToolManager -- Uses --> LoggerFactory["LoggerFactory"]
       ToolRegistry -- Uses --> LoggerFactory
       ToolExecutor -- Uses --> LoggerFactory
       ToolStatsManager -- Uses --> LoggerFactory
       ProviderToolHandler -- Uses --> LoggerFactory
       ToolManager -- Uses --> UnifiedConfig["UnifiedConfig"]
       ToolRegistry -- Uses --> UnifiedConfigRef %% Show registry uses config
    end

    ToolEnabledAI -- Gets Formatted Tools --> ProviderToolHandler

    %% AI Interaction Flow (Async)
    ToolEnabledAI -- Calls Request (await) --> AIProvider["AIProvider (e.g., OpenAIProvider)"]
    AIProvider -- Returns (awaitable) --> ProviderResponse["ProviderResponse Model (incl. ToolCall)"]
    ToolEnabledAI -- Parses --> ToolCallData["ToolCall Data"]

    %% Highlighting Async Calls
    style ToolManager fill:#f9f,stroke:#333,stroke-width:2px
    style ToolExecutor fill:#f9f,stroke:#333,stroke-width:2px
    style ToolEnabledAI fill:#ccf,stroke:#333,stroke-width:2px
    style AIProvider fill:#ccf,stroke:#333,stroke-width:2px
```

## Tool Components (Async Aspects Highlighted)

- **ToolManager**: Central service coordinating **asynchronous** tool execution (`async def execute_tool`) using `ToolExecutor` and retrieving tool definitions from `ToolRegistry`. It no longer handles tool _finding_ or direct registration.
- **ToolRegistry**: Loads tool definitions (`ToolDefinition`) from configuration, stores them, and provides them. Handles provider-specific formatting.
- **ToolExecutor**: Executes the actual tool functions safely with **asyncio-based** timeout and retry logic (`async def execute`). Handles both synchronous and asynchronous tool functions.
- **ToolStatsManager**: Tracks and manages tool usage statistics.
- **ProviderToolHandler**: Handles provider-specific formatting of tool definitions and `ToolResult` objects.
- **Models**: Defines core data structures (`ToolDefinition`, `ToolCall`, `ToolResult`).
- **Configuration**: Defines available internal tools and their implementations.
- **Users** (e.g., `ToolEnabledAI`): Components that utilize the tool system. `ToolEnabledAI` now handles the **asynchronous** interaction loop with the AI provider and the `ToolManager`.

## Creating and Registering Tools (Configuration-Based)

Tools are primarily defined in `src/config/tools.yml`. Each entry specifies:

- `name`: Unique name.
- `description`: Clear description for the LLM.
- `module`: Python module path (e.g., `src.tools.core.calculator_tool`).
- `function`: Name of the Python function (sync or async) implementing the tool.
- `parameters_schema`: JSON schema for input parameters.
- `category` (optional): Grouping category.

The `ToolRegistry` automatically loads these during startup.

## Tool Execution Flow (Async)

The execution flow is now inherently asynchronous:

1.  On startup, `ToolRegistry` loads tool definitions.
2.  When `ToolEnabledAI` needs to interact (`async def process_prompt`), it retrieves formatted tools via `ProviderToolHandler`.
3.  `ToolEnabledAI` passes definitions to the AI Provider during its **asynchronous** request (`await self._provider.request(...)`).
4.  The LLM decides if a tool needs to be called, returning `ToolCall` data in the `ProviderResponse`.
5.  `ToolEnabledAI` parses the `ToolCall` data.
6.  For each requested tool call, `ToolEnabledAI` **awaits** `ToolManager.execute_tool(tool_name=..., **arguments)`.
7.  `ToolManager` retrieves the `ToolDefinition` from `ToolRegistry`.
8.  `ToolManager` **awaits** `ToolExecutor.execute(...)`, passing the tool definition and arguments.
9.  `ToolExecutor` determines if the tool function is sync or async.
    - If async, it **awaits** the tool function directly using `asyncio.wait_for`.
    - If sync, it runs the function in a thread pool (`loop.run_in_executor`) within `asyncio.wait_for` to avoid blocking the event loop.
10. `ToolExecutor` returns a `ToolResult` to `ToolManager`.
11. `ToolManager` returns the `ToolResult` to `ToolEnabledAI`.
12. `ToolEnabledAI` uses `ProviderToolHandler` to format the `ToolResult` into messages.
13. `ToolEnabledAI` **awaits** the provider again (`await self._provider.request(...)`) with the updated history.
14. The loop continues or finishes, returning the final content.

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

## Tool Usage Statistics

The framework includes a dedicated component for tracking and analyzing tool usage:

- **ToolStatsManager** (`src/tools/tool_stats_manager.py`): Manages collection, storage, and retrieval of tool usage statistics.

### Features

- **Usage Tracking**: Records each tool invocation, including success/failure status and execution duration.
- **Performance Metrics**: Calculates and maintains average execution durations for successful calls.
- **Persistent Storage**: Saves statistics to a JSON file for analysis and preservation between sessions.
- **Configurable Behavior**: Can be enabled/disabled and configured via the `UnifiedConfig` system.

### Configuration

Tool statistics tracking can be configured in `src/config/tools.yml` under the `stats` section:

```yaml
tools:
  stats:
    track_usage: true # Enable/disable statistics tracking
    storage_path: "data/tool_stats.json" # Path for storing statistics
```

### Usage Statistics Data

For each tool, the following statistics are recorded:

- **Total uses**: Total number of times the tool was called
- **Successes**: Number of successful executions
- **Failures**: Number of failed executions
- **First used**: Timestamp of first recorded use
- **Last used**: Timestamp of most recent use
- **Total duration**: Cumulative execution time (ms) of successful calls
- **Average duration**: Average execution time (ms) of successful calls

### Example

```python
import asyncio

# Assuming async setup to get manager
async def main():
    # Get the ToolManager instance (normally accessed via ToolEnabledAI)
    # Requires async initialization if dependencies are async
    # Simplified example assumes sync creation for stats_manager
    stats_manager = ToolStatsManager()

    # Record a successful tool execution (update_stats is sync)
    stats_manager.update_stats(
        tool_name="weather_tool",
        success=True,
        duration_ms=250
    )

    # Record a failed tool execution
    stats_manager.update_stats(
        tool_name="search_tool",
        success=False
    )

    # Get statistics for a specific tool
    weather_stats = stats_manager.get_stats("weather_tool")
    print(f"Weather tool used {weather_stats['uses']} times with "
          f"{weather_stats['successes']} successes")

    # Get statistics for all tools
    all_stats = stats_manager.get_all_stats()

    # Save current statistics to disk
    stats_manager.save_stats()

# Run the async example
# asyncio.run(main())
```

These statistics can be valuable for:

- Monitoring which tools are most frequently used
- Identifying tools with high failure rates
- Optimizing performance of slow tools
- Creating usage reports for system administrators

A complete example demonstrating ToolStatsManager usage is available at `examples/tool_stats_example.py`.
