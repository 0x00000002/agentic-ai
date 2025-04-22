# Configuration System

## Overview

The Agentic AI configuration system provides a unified and flexible way to manage settings for the entire framework. It handles model selection, provider settings, agent behaviors, tool configurations, and use-case-specific parameters. The system is designed to be user-friendly while allowing for deep customization. It combines default configurations stored in YAML files with multiple override mechanisms.

## Core Components

- **`UnifiedConfig`**: A singleton class (`src/config/unified_config.py`) that acts as the central hub for all configuration data. It loads base configurations from YAML files and merges overrides from various sources.
- **`UserConfig`**: A data class (`src/config/user_config.py`) representing user-provided configuration settings. Used by the `configure` function to pass overrides to `UnifiedConfig`.
- **`configure()` function**: The primary user-facing function (`src/config/__init__.py`) to apply configuration settings and overrides.
- **`get_config()` function**: A function (`src/config/__init__.py`) to retrieve the singleton `UnifiedConfig` instance for accessing detailed configuration data.

## Configuration Files

The base configurations are stored in several YAML files located in the `src/config` directory:

- `models.yml`: Defines AI model parameters, capabilities, costs, and provider information.
- `providers.yml`: Configures API providers (e.g., OpenAI, Anthropic, Ollama) including base URLs and potentially API key environment variable names.
- `agents.yml`: Contains configurations specific to different agent types or roles.
- `use_cases.yml`: Defines settings presets for different task types (e.g., chat, coding, solidity_coding), often specifying default models or parameters.
- `tools.yml`: Configures available **internal** tools (Python functions within the project) that agents can use. See below for structure.
- `mcp.yml`: Configures connections to external **Model-Centric Protocol (MCP)** servers and declares the tools those servers are expected to provide. See below for structure.

These files provide the default settings for the framework.

### `tools.yml` Structure

This file defines tools that are implemented as Python functions within this framework.

```yaml
# src/config/tools.yml
tools:
  - name: "tool_name" # Unique identifier
    description: "Description for LLM" # How the AI understands the tool
    module: "src.path.to.module" # Python module containing the function
    function: "function_name" # Name of the sync or async Python function
    parameters_schema: { ... } # JSON schema for arguments
    category: "grouping_category" # Optional: for organization
    source: "internal" # Must be "internal" for this file
    speed: "fast" # Optional: fast, medium, slow (default: medium)
    safety: "native" # Optional: native, sandboxed, external (default: native)

  # Example: Updated generate_image tool definition
  - name: "generate_image"
    description: "Generates an image from a text description using the latest Stable Diffusion 3.5 Medium model via Replicate API. Use this tool when asked to create, draw, or generate an image."
    module: "src.tools.media.image_generator"
    function: "generate_image"
    parameters_schema:
      type: "object"
      properties:
        prompt:
          type: "string"
          description: "Description of the image to generate"
        negative_prompt:
          type: "string"
          description: "Elements to avoid in the generated image"
        width:
          type: "integer"
          description: "Image width (used to approximate aspect_ratio if aspect_ratio kwarg not provided)"
          default: 1024
        height:
          type: "integer"
          description: "Image height (used to approximate aspect_ratio if aspect_ratio kwarg not provided)"
          default: 1024
        num_inference_steps:
          type: "integer"
          description: "Number of denoising steps (passed as 'steps' to API)"
          default: 40 # Default for SD 3.5 Medium
        # Optional kwargs like aspect_ratio, output_format, output_quality, cfg can be passed via **kwargs
      required: ["prompt"]
    category: "media"
    source: "internal"
    speed: "medium"
    safety: "external"

# Tool execution settings (optional)
execution:
  timeout: 360 # Example increased global timeout
  max_retries: 3

# Tool statistics settings (optional)
stats:
  storage_path: "data/tool_stats.json"
  track_usage: true
```

The `ToolRegistry` component is responsible for loading and managing these internal tool definitions.

### `mcp.yml` Structure

This file defines connections to external MCP servers and _declares_ the tools they provide. This declaration helps the LLM know about the tool, but the actual available tools might differ when connecting to the server.

The MCP server itself must be running independently and accessible at the specified network address.

```yaml
# src/config/mcp.yml
mcp_servers:
  unique_server_name_1: # Identifier for this MCP server connection
    description: "Description of server" # Optional
    url: "http://localhost:8001" # REQUIRED: Network endpoint (http, https, ws, wss)
    auth: # Optional: Authentication details
      type: "bearer" # Currently only "bearer" is supported
      token_env_var: "MCP_SERVER_1_TOKEN" # Env var containing the Bearer token
      # NOTE: Headers (including auth) are NOT currently sent for ws/wss connections
      # due to limitations in the underlying mcp library (v1.6.0).
    provides_tools: # List of tools *declared* by this server
      - name: "mcp_tool_name_a" # Unique name across ALL tools (internal & MCP)
        description: "Description for LLM"
        inputSchema: { ... } # JSON schema for arguments (or parameters_schema)
        speed: "medium" # Optional: (default: medium)
        safety: "external" # Optional: (default: external)
        # 'source' and 'mcp_server_name' are added automatically

  unique_server_name_2:
    description: "Another server"
    url: "wss://secure.example.com:8002"
    # No auth section means no authentication required or sent
    provides_tools:
      - name: "mcp_tool_name_b"
        # ... tool definition ...
```

The `MCPClientManager` component is responsible for loading these configurations, establishing connections to the servers using the appropriate network client (`sse_client` for http/https, `websocket_client` for ws/wss), and providing the declared tool definitions.

## Usage

The most common way to interact with the configuration system is through the `configure()` function, typically called once at the beginning of an application.

### Basic Configuration

Apply basic settings like model, use case, and temperature directly:

```python
from src.config import configure
from src.core.tool_enabled_ai import ToolEnabledAI # Or your main AI class

# Configure the framework
configure(
    model="claude-3-5-sonnet",
    use_case="solidity_coding",
    temperature=0.8,
    show_thinking=True # Example debug/verbosity flag
)

# AI instances created after this will use the applied configuration
# ai = ToolEnabledAI()
# response = ai.request("Write a simple ERC20 token contract")
```

### Using Use Case Presets

Apply predefined configurations suitable for specific tasks using the `UseCasePreset` enum or strings:

```python
from src.config import configure, UseCasePreset

# Using string
configure(use_case="solidity_coding")

# Using enum for better IDE support and type safety
configure(use_case=UseCasePreset.SOLIDITY_CODING)
```

Check `src/config/user_config.py` or `src/config/use_cases.yml` for available presets.

### Overriding Configuration

The system supports multiple ways to override the default configurations:

1.  **`configure()` function arguments**: Settings passed directly to `configure()` take precedence over defaults (as shown in the basic example).
2.  **External Configuration File**: Specify a YAML or JSON file containing overrides.

    ```python
    # In your Python script
    configure(config_file="path/to/your_config.yml")
    ```

    ```yaml
    # path/to/your_config.yml
    model: claude-3-5-sonnet
    temperature: 0.9
    system_prompt: "You are an expert Solidity auditor."
    show_thinking: false
    ```

3.  **Environment Variables**: The system uses `python-dotenv` to load variables from a `.env` file in the project root. Standard environment variables can also be used. Naming conventions might apply (e.g., provider API keys like `OPENAI_API_KEY`). Check `src/config/providers.yml` and `src/config/unified_config.py` for details on environment variable usage.

**Order of Precedence (Highest to Lowest):**

1.  Arguments passed to `configure()`.
2.  Settings loaded from the `config_file` specified in `configure()`.
3.  Environment variables.
4.  Default values from the base YAML files (`models.yml`, etc.).

## Accessing Configuration Details

For advanced use cases or framework development, you can access the detailed configuration values stored in the `UnifiedConfig` instance:

```python
from src.config import get_config

config = get_config() # Retrieve the singleton instance

# Get the configured default model
default_model_id = config.default_model # Property access
# OR
default_model_id = config.get_default_model() # Method access

# Get configuration for a specific model
model_config = config.get_model_config("claude-3-5-sonnet")
print(f"Input limit for Claude 3.5 Sonnet: {model_config.get('input_limit')}")

# Get all available model names
all_model_names = config.get_model_names()

# Get the effective system prompt (after overrides)
system_prompt = config.get_system_prompt()

# Get provider configuration
openai_config = config.get_provider_config("openai")

# Check debugging flags
if config.show_thinking:
    print("Showing AI thinking process...")
```

Refer to the `UnifiedConfig` class (`src/config/unified_config.py`) for all available methods and properties.

## Dynamic Model Enum

For convenience and type safety, a dynamic `Model` enum is generated based on the models defined in `models.yml`.

```python
from src.config import Model, get_available_models, is_valid_model

# Use enum member for comparisons or assignments
if selected_model == Model.CLAUDE_3_5_SONNET:
    print("Using Claude 3.5 Sonnet")

# Check if a model string identifier is valid
if is_valid_model("gpt-4o"):
    print("GPT-4o is a configured model.")

# Get all available model enum members
available_models_enum = get_available_models() # Returns list of Model enum members
```

## Benefits

- **Unified Interface**: Single point of configuration via `configure()` and access via `get_config()`.
- **Modularity**: Base configurations are split into logical YAML files.
- **Flexibility**: Multiple override mechanisms (function args, file, env vars).
- **Clarity**: `UserConfig` provides a clear structure for overrides.
- **Discoverability**: Methods like `get_model_names()`, `get_available_providers()` aid exploration.
- **Testability**: The singleton instance can be reset (`UnifiedConfig.reset_instance()`) for isolated testing.
