# Core AI Components

This section describes the fundamental classes responsible for AI interaction and capabilities.

## `AIBase` (`src/core/base_ai.py`)

`AIBase` serves as the foundational class for all AI interactions. It implements the basic `AIInterface` and handles core responsibilities:

- **Initialization**: Takes a model identifier (`model`), optional `system_prompt`, `logger`, `request_id`, and `PromptTemplate` service.
- **Configuration**: Fetches model and provider configurations from `UnifiedConfig`.
- **Provider Instantiation**: Uses `ProviderFactory` to create the appropriate `ProviderInterface` instance (e.g., `OpenAIProvider`, `AnthropicProvider`) based on the model's configuration.
- **Conversation Management**: Initializes and maintains a `ConversationManager` instance to track message history.
- **System Prompt**: Sets the initial system prompt in the conversation history, using the provided `system_prompt`, a configured default, or a basic fallback.
- **Basic Request Handling**:
  - `request(prompt, **options)`: Adds the user prompt to the conversation, sends the full message history to the provider, receives the response, adds the assistant's response to history, and returns the response content string.
  - `stream(prompt, **options)`: Similar to `request` but uses the provider's streaming capabilities.
- **State Management**: Provides methods like `reset_conversation()`, `get_conversation()`, `get_system_prompt()`, `set_system_prompt()`, and `get_model_info()`.

`AIBase` itself does _not_ handle tool calls.

## `ToolEnabledAI` (`src/core/tool_enabled_ai.py`)

`ToolEnabledAI` inherits from `AIBase` and adds the capability to use tools (functions or external APIs) during request processing.

- **Initialization**: Takes the same arguments as `AIBase`, plus an optional `ToolManager` instance. If no `ToolManager` is provided, it creates a default one.
- **Tool Support Check**: Determines if the underlying provider (obtained via `AIBase` initialization) supports tool calling based on its interface or attributes.
- **Tool Management**: Interacts with the `ToolManager` to get definitions of available tools.
- **Tool Calling Loop (`process_prompt`)**: This is the primary method for handling requests that might involve tools.
  1.  Adds the user prompt to history.
  2.  Enters a loop (limited by `max_tool_iterations`).
  3.  Gets available tool definitions from `ToolManager`.
  4.  Calls the provider with the current messages and tool definitions.
  5.  Receives the provider's response (which might include text content and/or tool call requests).
  6.  Adds the assistant's message (content and requested tool calls) to history.
  7.  **If tool calls are requested**: Executes each tool call using `ToolManager.execute_tool()`. Adds the tool results back into the conversation history using the provider-specific format.
  8.  **If no tool calls are requested**: Exits the loop and returns the final text content from the assistant.
  9.  Continues the loop if tools were called and the iteration limit is not reached.
- **Basic Request (No Tool Loop)**:
  - `request_basic(prompt, **options)`: Sends the prompt and history to the provider but _does not_ automatically execute any tool calls requested in the response. It returns the raw `ProviderResponse` object, allowing the caller to inspect potential tool calls.
- **Tool Information**: Provides `get_tool_history()` and `get_available_tools()`.

For most standard interactions where you want the AI to be able to use tools automatically, you should instantiate and use `ToolEnabledAI` and call its `process_prompt` method.
