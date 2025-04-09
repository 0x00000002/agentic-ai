# Tool Integration

## Overview

Tools allow the AI to perform actions and retrieve information beyond its training data. The Agentic-AI framework includes a tool management system that enables:

1.  **Loading** tool definitions from configuration (`tools.yml`).
2.  **Registering** these tools.
3.  **Formatting** tool definitions for specific AI providers.
4.  **Executing** tool calls requested by the AI with error handling and timeout/retry logic.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Users Of Tools"
        ToolEnabledAI --- ToolManager
    end

    subgraph "Tools"
        %% ToolCall Model - Represents data parsed by ToolEnabledAI, not a direct dependency here
        ToolManager -- Delegates Execution --> ToolExecutor
        ToolManager -- Gets Definition --> ToolRegistry
        ToolManager -- Records Stats --> ToolStatsManager

        ToolExecutor -- Returns --> ToolResult["ToolResult Model"]

        %% ToolRegistry Loads/Stores Tool Definitions
        ToolRegistry -- Stores/Provides --> ToolDefinition["ToolDefinition Model"]

        %% ProviderToolHandler Formats Tools
        ProviderToolHandler["ProviderToolHandler (src/core/providers)"] -- Gets Definitions --> ToolRegistry

        %% ToolStatsManager manages usage statistics
        ToolStatsManager -- Persists --> StatsFile["Tool Stats JSON"]

        %% Config Loading (Implicitly happens in ToolRegistry.__init__)
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

    %% Showing ToolEnabledAI uses ProviderToolHandler for formatting
    ToolEnabledAI -- Gets Formatted Tools --> ProviderToolHandler

    %% Simplified View: ToolEnabledAI triggers execution based on parsed ToolCall data
    ToolEnabledAI -- Calls Execute --> ToolManager
```

## Tool Components

- **ToolManager**: Central service coordinating tool execution (`execute_tool`) using `ToolExecutor` and retrieving tool definitions from `ToolRegistry`. It no longer handles tool _finding_ or direct registration (which is now config-driven via `ToolRegistry`).
- **ToolRegistry**: Loads tool definitions (`ToolDefinition`) from configuration (`tools.yml` via `UnifiedConfig`), stores them, and provides them to other components. It handles provider-specific formatting logic (used by `ProviderToolHandler`).
- **ToolExecutor**: Executes the actual tool functions safely with improved timeout and retry logic.
- **ToolStatsManager**: Tracks and manages tool usage statistics, recording execution counts, success/failure rates, and performance metrics. Provides persistence of statistics through save/load operations to JSON files.
- **ProviderToolHandler** (`src/core/providers/provider_tool_handler.py`): Handles provider-specific formatting of tool definitions for the AI model and formats `ToolResult` objects into messages for the provider.
- **Models** (`src/tools/models.py`): Defines core data structures:
  - `ToolDefinition`: Represents a tool's metadata (name, description, schema) and its implementation (`function`). Loaded from config.
  - `ToolCall`: Represents the AI's request to call a tool (name, arguments). Data parsed from the AI response by `ToolEnabledAI`.
  - `ToolResult`: Represents the outcome of a tool execution (success/failure, result/error).
- **Configuration** (`src/config/tools.yml`): YAML file defining the available internal tools, their implementations (module/function), and schemas.
- **Users** (e.g., `ToolEnabledAI`): Components that utilize the tool system, typically by getting formatted tools for an AI model and requesting execution via `ToolManager` based on the AI's response.

## Creating and Registering Tools (Configuration-Based)

Tools are now primarily defined in `src/config/tools.yml`. Each entry specifies:

- `name`: A unique name for the tool.
- `description`: A clear description of what the tool does (used by the LLM).
- `module`: The Python module path where the tool's function resides (e.g., `src.tools.core.calculator_tool`).
- `function`: The name of the Python function implementing the tool's logic (e.g., `calculate`).
- `parameters_schema`: A JSON schema defining the expected input parameters.
- `category` (optional): A grouping category.

The `ToolRegistry` automatically loads and registers tools defined in this file during application startup. Programmatic registration via `ToolManager.register_tool` is still possible but mainly intended for dynamic scenarios (like potential future MCP integration) rather than standard tool setup.

## Tool Execution Flow

The execution flow is now centered around configuration and explicit execution requests:

1.  On startup, `ToolRegistry` loads tool definitions from `tools.yml` via `UnifiedConfig`.
2.  When `ToolEnabledAI` needs to interact with an AI model, it retrieves available tool definitions formatted for the specific provider via `ProviderToolHandler` (which uses `ToolRegistry`).
3.  `ToolEnabledAI` passes these definitions to the AI Provider.
4.  The LLM decides if a tool needs to be called, returning `ToolCall` data in the response.
5.  `ToolEnabledAI` parses the `ToolCall` data (name, arguments).
6.  For each requested tool call, `ToolEnabledAI` invokes `ToolManager.execute_tool(tool_name=..., **arguments)`.
7.  `ToolManager` retrieves the corresponding `ToolDefinition` (including the actual function callable) from `ToolRegistry` using the `tool_name`.
8.  `ToolManager` delegates the execution to `ToolExecutor`, passing the function and arguments.
9.  `ToolExecutor` runs the function (with timeout/retry) and returns a `ToolResult` to `ToolManager`.
10. `ToolManager` returns the `ToolResult` to `ToolEnabledAI`.
11. `ToolEnabledAI` uses `ProviderToolHandler` to format the `ToolResult` into the appropriate message format(s) for the AI provider.
12. `ToolEnabledAI` calls the provider again with the updated history (including tool results).
13. The loop continues or finishes.

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
# Get the ToolStatsManager instance (normally accessed via ToolManager)
stats_manager = ToolStatsManager()

# Record a successful tool execution
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
```

These statistics can be valuable for:

- Monitoring which tools are most frequently used
- Identifying tools with high failure rates
- Optimizing performance of slow tools
- Creating usage reports for system administrators

A complete example demonstrating ToolStatsManager usage is available at `examples/tool_stats_example.py`.
