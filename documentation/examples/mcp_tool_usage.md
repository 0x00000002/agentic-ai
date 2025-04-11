# Example: Using Tools from an MCP Server

This example demonstrates how to configure Agentic-AI to connect to an external tool server using the Model Context Protocol (MCP) and utilize the tools it provides.

## 1. Configuration (`config.yml`)

First, you need to define the MCP server and the tools it _declares_ it provides within your main configuration file (e.g., `config.yml`). This tells Agentic-AI how to connect to the server and what tools to expect.

Assume we have an external weather service running as an MCP server accessible at `http://localhost:8005` which requires a bearer token for authentication.

```yaml
# Example snippet within your config.yml

# ... other configurations ...

mcp_servers:
  weather_service_mcp:
    description: "External MCP server providing weather information."
    url: "http://localhost:8005" # URL where the MCP server is running
    auth:
      type: "bearer"
      token_env_var: "WEATHER_MCP_TOKEN" # Environment variable holding the auth token
    provides_tools:
      - name: "get_current_weather_mcp"
        description: "Retrieves the current weather conditions for a specified location from the external MCP server."
        inputSchema: # Use inputSchema or parameters_schema
          type: "object"
          properties:
            location:
              type: "string"
              description: "The city and state/country (e.g., San Francisco, CA)."
            unit:
              type: "string"
              enum: ["celsius", "fahrenheit"]
              default: "celsius"
              description: "Temperature unit."
          required: ["location"]
        # speed/safety will use defaults (medium/external)
# ... other configurations ...
```

Key points:

- `mcp_servers`: The top-level key containing all MCP server definitions.
- `weather_service_mcp`: A unique name chosen for this server configuration.
- `url`: The address where the Agentic-AI framework can reach the MCP server.
- `auth`: Defines authentication. Here, `bearer` type indicates a token is needed, read from the environment variable specified by `token_env_var`.
- `provides_tools`: A list of tools declared by this server. Agentic-AI uses this list to inform the LLM about available tools. The `name`, `description`, and `inputSchema` are crucial for the LLM. `source` and `mcp_server_name` are added internally by the framework.

## 2. Environment Setup

Since the configuration specifies `token_env_var: "WEATHER_MCP_TOKEN"`, you need to set this environment variable before running your Agentic-AI application.

```bash
# Example in bash/zsh
export WEATHER_MCP_TOKEN="your_secret_mcp_api_token"
```

Replace `"your_secret_mcp_api_token"` with the actual token provided by the weather service.

## 3. Code Example

Now, you can use the tools provided by the MCP server just like any other internal tool. The `ToolManager` (or an agent using it) handles the discovery and execution.

```python
# Example Python script (e.g., run_mcp_example.py)

import asyncio
import os
from src.config import UnifiedConfig, get_config
from src.tools import ToolManager
from src.providers.openai import OpenAIProvider # Or any other provider
from src.ai import ToolEnabledAI

# Ensure the environment variable is set (replace with your actual token)
os.environ["WEATHER_MCP_TOKEN"] = "your_secret_mcp_api_token"
# Replace with your OpenAI key if using OpenAI
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

async def main():
    # Load configuration (which includes mcp_servers)
    # Assuming default config path or config object passed appropriately
    config = get_config()

    # Initialize components
    tool_manager = ToolManager(config=config)
    ai_provider = OpenAIProvider(config=config)
    ai = ToolEnabledAI(provider=ai_provider, tool_manager=tool_manager, config=config)

    # List all available tools (internal + MCP)
    print("Available tools:")
    all_tools = tool_manager.list_tool_definitions()
    for tool_def in all_tools:
        print(f"- {tool_def.name} (Source: {tool_def.source}, MCP Server: {tool_def.mcp_server_name or 'N/A'})")

    # Example prompt that should trigger the MCP tool
    prompt = "What's the current weather like in London?"

    print(f"\nSending prompt: '{prompt}'")

    # The AI will potentially call the 'get_current_weather_mcp' tool.
    # The ToolManager will see source='mcp', get the client from MCPClientManager,
    # and execute the call against the configured URL (http://localhost:8005).
    # NOTE: This requires the MCP server to be running at that address.
    #       The actual network call won't happen here unless a live server exists.
    try:
        response = await ai.process_prompt(prompt)
        print("\nFinal AI Response:")
        print(response.content)

        # You can also inspect tool usage stats
        stats = tool_manager.stats_manager.get_stats()
        print("\nTool Usage Stats:")
        print(stats)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Ensure the MCP server is running at the configured URL and the API keys/tokens are correct.")


if __name__ == "__main__":
    asyncio.run(main())

```

## 4. Explanation

When `ai.process_prompt` is called with a prompt like "What's the current weather like in London?":

1. The `ToolEnabledAI` interacts with the AI Provider (e.g., OpenAI), providing the definition of `get_current_weather_mcp` (loaded from the config by `ToolManager` via `MCPClientManager`).
2. The LLM identifies that `get_current_weather_mcp` is suitable and returns a request to call it with arguments like `{"location": "London"}`.
3. The `ToolEnabledAI` passes this request to `ToolManager.execute_tool`.
4. `ToolManager` sees the tool's `source` is `mcp` and its `mcp_server_name` is `weather_service_mcp`.
5. `ToolManager` calls `MCPClientManager.get_tool_client("weather_service_mcp")`.
6. `MCPClientManager` checks its configuration for `weather_service_mcp`, sees it's an HTTP URL with bearer auth, retrieves the token from the `WEATHER_MCP_TOKEN` environment variable, and creates/returns an internal HTTP client wrapper configured with the URL and authentication header.
7. `ToolManager` uses the returned client wrapper to make the actual HTTP POST request to `http://localhost:8005/call_tool` with the tool name and arguments. (If it were a WebSocket URL, `MCPClientManager` would establish a WS connection instead).
8. The external MCP server (if running) processes the request and returns the weather data.
9. `ToolManager` receives the response, wraps it in a `ToolResult`, and returns it to `ToolEnabledAI`.
10. `ToolEnabledAI` sends the tool result back to the LLM.
11. The LLM generates the final natural language response based on the weather data.

This example illustrates how to integrate external functionalities into your AI agent by configuring access to MCP-compliant tool servers.
