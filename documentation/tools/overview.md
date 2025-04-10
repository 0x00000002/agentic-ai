# Tool Integration

## Overview

Tools allow the AI to perform actions and retrieve information beyond its training data. The Agentic-AI framework includes a tool management system that enables:

1.  **Loading** tool definitions from configuration (`tools.yml` for internal tools, `mcp.yml` for external MCP servers and their declared tools).
2.  **Unified Discovery**: Providing a single list of all available tools (both internal and MCP).
3.  **Formatting** tool definitions for specific AI providers.
4.  **Executing** tool calls requested by the AI asynchronously, dispatching to the correct executor (internal Python or MCP client) based on the tool's source, with error handling and timeout/retry logic.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Users Of Tools"
        ToolEnabledAI -- Calls Execute (await) --> ToolManager
        BaseAgent -- Calls Execute (await) --> ToolManager %% Agents can also use tools
    end

    subgraph "Tools Subsystem"
        ToolManager -- Gets All Definitions --> ToolRegistry[\"ToolRegistry (Internal Tools)\"]
        ToolManager -- Gets All Definitions --> MCPClientManager[\"MCPClientManager (MCP Tools)\"]

        ToolManager -- Dispatches --> ToolExecutor[\"ToolExecutor (Internal)\"]
        ToolManager -- Dispatches --> MCPClientManager %% For MCP execution

        ToolManager -- Records Stats --> ToolStatsManager

        ToolExecutor -- Returns (awaitable) --> ToolResult[\"ToolResult Model\"]
        MCPClientManager -- Returns MCP Response --> ToolManager %% ToolManager maps to ToolResult

        ToolRegistry -- Stores/Provides --> InternalToolDef[\"ToolDefinition (source='internal')\"]
        MCPClientManager -- Stores/Provides --> MCPToolDef[\"ToolDefinition (source='mcp')\"]

        ToolStatsManager -- Persists --> StatsFile[\"Tool Stats JSON\"]

        UnifiedConfigRef[\"UnifiedConfig\"] -- Provides Config --> ToolRegistry
        UnifiedConfigRef -- Provides Config --> MCPClientManager
        UnifiedConfigRef -- Provides Config --> ToolExecutor
        UnifiedConfigRef -- Provides Config --> ToolStatsManager

        ToolsYAML[\"tools.yml\"] -- Read By --> UnifiedConfigRef
        MCPYAML[\"mcp.yml\"] -- Read By --> UnifiedConfigRef
    end

    subgraph \"Dependencies\"
       ToolManager -- Uses --> LoggerFactory[\"LoggerFactory\"]
       ToolRegistry -- Uses --> LoggerFactory
       MCPClientManager -- Uses --> LoggerFactory
       ToolExecutor -- Uses --> LoggerFactory
       ToolStatsManager -- Uses --> LoggerFactory
       ToolManager -- Uses --> UnifiedConfigRef
    end

    %% AI Interaction Flow (Simplified)
    ToolEnabledAI -- Gets Formatted Tools --> ToolManager %% Manager handles formatting request

    %% Highlighting Async/Key Components
    style ToolManager fill:#f9f,stroke:#333,stroke-width:2px
    style ToolExecutor fill:#fdf,stroke:#333,stroke-width:1px
    style MCPClientManager fill:#fdf,stroke:#333,stroke-width:1px
    style ToolEnabledAI fill:#ccf,stroke:#333,stroke-width:2px
```

## Tool Components (Async Aspects Highlighted)

- **ToolManager**: Central service coordinating **asynchronous** tool execution (`async def execute_tool`). It loads all tool definitions (internal and MCP) via `ToolRegistry` and `MCPClientManager`, provides unified discovery methods, handles formatting requests, and dispatches execution calls to `ToolExecutor` (for internal tools) or `MCPClientManager` (for MCP tools) based on the tool's `source`.
- **ToolRegistry**: Loads **internal** tool definitions (`ToolDefinition` with `source='internal'`) from `tools.yml` configuration, validates them, stores them, and provides methods to retrieve them.
- **MCPClientManager**: Loads **MCP server configurations** and **declared MCP tool definitions** (`ToolDefinition` with `source='mcp'`) from `mcp.yml`. It manages connections (`ClientSession`) to external MCP servers and handles the `call_tool` request for MCP tools.
- **ToolExecutor**: Executes **only internal** Python tool functions safely with **asyncio-based** timeout and retry logic (`async def execute`). Handles both synchronous and asynchronous tool functions.
- **ToolStatsManager**: Tracks and manages tool usage statistics for all tool types.
- **Models**: Defines core data structures (`ToolDefinition`, `ToolCall`, `ToolResult`).
- **Configuration**: Defines available internal tools (`tools.yml`) and external MCP servers/declared tools (`mcp.yml`).
- **Users** (e.g., `ToolEnabledAI`, `BaseAgent`): Components that utilize the tool system, typically interacting primarily with `ToolManager`. `ToolEnabledAI` handles the **asynchronous** interaction loop with the AI provider and the `ToolManager`.

## Defining Tools (Configuration-Based)

Tools are defined declaratively in YAML configuration files, specifying their source ("internal" or "mcp").

### Internal Tools (`src/config/tools.yml`)

These are Python functions defined within the project.

- `name`: Unique name for the tool.
- `description`: Clear description for the LLM.
- `module`: Python module path (e.g., `src.tools.core.calculator_tool`).
- `function`: Name of the Python function (sync or async) implementing the tool.
- `parameters_schema`: JSON schema for input parameters.
- `category` (optional): Grouping category.
- `speed` (optional): Estimated speed (e.g., "instant", "fast", "medium", "slow"). Defaults to "medium".
- `safety` (optional): Safety level (e.g., "native", "sandboxed", "external"). Defaults to "native".
- `source`: Must be `"internal"`.

**Example (`tools.yml`):**

```yaml
tools:
  - name: "calculator"
    description: "Perform mathematical calculations..."
    module: "src.tools.core.calculator_tool"
    function: "calculate"
    parameters_schema:
      type: "object"
      properties:
        expression:
          type: "string"
          description: "The mathematical expression..."
      required: ["expression"]
    category: "core_utils"
    source: "internal" # Explicitly internal
    speed: "fast"
    safety: "sandboxed"
```

### MCP Servers and Declared Tools (`src/config/mcp.yml`)

This file defines how to connect to external MCP servers and which tools those servers _declare_ they provide. The framework uses this declaration to inform the LLM; the actual tool list might differ when connecting to the server. The MCP server process must be running independently and accessible at the specified network address.

- `mcp_servers`: A dictionary where each key is a unique server name.
  - `description` (optional): Description of the server.
  - `url`: Network endpoint (e.g., `http://localhost:8001`) where the MCP server is listening. **Required**.
  - `auth` (optional): Authentication details for the server.
    - `type`: The authentication type (e.g., `bearer`).
    - `token_env_var`: The name of the environment variable containing the authentication token.
  - `provides_tools`: A list of _declared_ tool definitions for this server.
    - `name`: Unique name for the tool (must be unique across _all_ tools).
    - `description`: Clear description for the LLM.
    - `inputSchema` (or `parameters_schema`): JSON schema for input parameters.
    - `speed` (optional): Estimated speed. Defaults to "medium".
    - `safety` (optional): Safety level. Defaults to "external".
    - `source`: Automatically set to `"mcp"` by `MCPClientManager`.
    - `mcp_server_name`: Automatically set to the server key by `MCPClientManager`.

**Example (`mcp.yml`):**

```yaml
# src/config/mcp.yml

mcp_servers:
  code_execution_server:
    description: "Server for executing Python code snippets securely."
    url: "http://localhost:8001" # Example URL
    auth:
      type: "bearer"
      token_env_var: "CODE_EXEC_AUTH_TOKEN"
    provides_tools:
      - name: "execute_python_code"
        description: "Executes a given Python code snippet in a restricted environment and returns the output."
        inputSchema:
          type: "object"
          properties:
            code:
              type: "string"
              description: "The Python code to execute."
            timeout_seconds:
              type: "integer"
              default: 10
              description: "Maximum execution time in seconds."
          required: ["code"]
        speed: "medium"
        safety: "sandboxed" # Overrides default 'external'

  web_search_server:
    description: "Server for performing web searches."
    url: "http://localhost:8002" # Example URL
    # No auth specified for this example
    provides_tools:
      - name: "perform_web_search"
        description: "Searches the web for a given query."
        inputSchema:
          type: "object"
          properties:
            query: { type: "string" }
          required: ["query"]
        # speed/safety use defaults (medium/external)
```

## Tool Execution Flow (Async)

The execution flow is now inherently asynchronous and handles dispatching based on the tool source:

1.  On startup, `ToolRegistry` loads internal tool definitions from `tools.yml` and `MCPClientManager` loads MCP server configs and declared tools from `mcp.yml`. `ToolManager` aggregates these into a unified list.
2.  When `ToolEnabledAI` needs to interact (`async def process_prompt`), it retrieves formatted tools for the target model via `ToolManager.format_tools_for_model(...)`.
3.  `ToolEnabledAI` passes these definitions to the AI Provider during its **asynchronous** request (`await self._provider.request(...)`).
4.  The LLM decides if a tool needs to be called, returning `ToolCall` data in the `ProviderResponse`.
5.  `ToolEnabledAI` parses the `ToolCall` data into `ToolCall` objects.
6.  For each `ToolCall`, `ToolEnabledAI` **awaits** `ToolManager.execute_tool(tool_call)`.
7.  `ToolManager` retrieves the full `ToolDefinition` using the `tool_call.name`. If not found, it returns an error `ToolResult`.
8.  `ToolManager` checks the `tool_definition.source`:
    - If `source == "internal"`:
      - `ToolManager` **awaits** `ToolExecutor.execute(tool_definition, **tool_call.arguments)`.
      - `ToolExecutor` resolves the function, handles sync/async execution with timeout/retry, and returns a `ToolResult`.
    - If `source == "mcp"`:
      - `ToolManager` retrieves `mcp_server_name` from the definition.
      - `ToolManager` **awaits** `MCPClientManager.get_tool_client(mcp_server_name)` to get or create a `ClientSession`.
      - `ToolManager` **awaits** `client_session.call_tool(tool_call.name, tool_call.arguments)`.
      - `ToolManager` maps the raw MCP response (including potential errors) into a `ToolResult`.
9.  `ToolManager` updates usage statistics via `ToolStatsManager`.
10. `ToolManager` returns the `ToolResult` to `ToolEnabledAI`.
11. `ToolEnabledAI` formats the `ToolResult` into messages suitable for the AI provider (e.g., using helper methods in the provider implementation).
12. `ToolEnabledAI` **awaits** the provider again (`await self._provider.request(...)`) with the updated history including the tool result message.
13. The loop continues or finishes, returning the final content.

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

Tool statistics tracking can be configured in `
