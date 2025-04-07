# Provider Test Cases

This document outlines the standard test cases that should be implemented for each AI Provider implementation (e.g., OpenAI, Anthropic, Gemini, Ollama) to ensure consistent functionality, error handling, and adherence to the framework's interfaces.

**Testing Philosophy:**

- Focus on unit testing the provider's specific logic (request formatting, response parsing, error mapping).
- Mock the external SDK client to avoid actual API calls and dependencies during tests.
- Use `pytest` conventions and fixtures for setup and parameterization.
- Keep mocks minimal and focused on the interaction boundary with the SDK.

## General Test Categories for All Providers

The following categories should be covered for each provider. Specific details will vary based on the provider's features and SDK.

### 1. Initialization (`__init__`)

- **Test Case 1.1: Successful Initialization (with API Key)**

  - **Goal:** Verify the provider initializes correctly when valid configuration (including API key) is provided.
  - **Checks:**
    - Provider attributes (`model_id`, `provider_config`, `model_config`) are set correctly.
    - Internal parameters (`self.parameters` like temperature, max_tokens) are parsed correctly from config.
    - The underlying SDK client is instantiated correctly (mocked) with the API key.
    - Logger is initialized.

- **Test Case 1.2: Initialization Fails (Missing API Key)**

  - **Goal:** Verify `AICredentialsError` (or appropriate error) is raised if the API key is missing in the configuration or cannot be resolved.
  - **Checks:**
    - Correct exception type is raised.
    - Exception message is informative.
    - SDK client is _not_ instantiated.

- **Test Case 1.3: Initialization Fails (Missing Configuration)**
  - **Goal:** Verify `AISetupError` or `AIConfigError` is raised if essential provider or model configuration is missing.
  - **Checks:**
    - Correct exception type is raised.
    - Exception message indicates the missing configuration.

### 2. Prepare Request Payload (`_prepare_request_payload`)

_(Note: This method might be implicitly tested via `_make_api_request` or `request` tests, but explicit tests can be useful if the logic is complex)_

- **Test Case 2.1: Basic Payload Formatting**

  - **Goal:** Verify the provider correctly formats the input messages and base parameters into the structure expected by the SDK's API call.
  - **Checks:**
    - `model` ID is included correctly.
    - `messages` are formatted according to the provider's requirements (roles, content structure).
    - Base parameters (`temperature`, `max_tokens`, etc.) from `self.parameters` are included.

- **Test Case 2.2: Payload Formatting with Options Override**
  - **Goal:** Verify that parameters passed via `**options` in the `request` method override the default `self.parameters`.
  - **Checks:**
    - Payload includes the overridden values for parameters like `temperature`.

### 3. Make API Request (`_make_api_request`)

- **Test Case 3.1: Successful API Call**

  - **Goal:** Verify the method correctly calls the mocked SDK client's relevant function with the prepared payload and returns the raw SDK response.
  - **Checks:**
    - Mock SDK client's method (e.g., `chat.completions.create`, `messages.create`) is called once.
    - The call is made with the expected payload dictionary.
    - The raw mock response object from the SDK client is returned.

- **Test Case 3.2: SDK Error Mapping**
  - **Goal:** Verify that various errors raised by the (mocked) SDK are caught and mapped to the framework's specific exceptions (`AIAuthenticationError`, `AIRateLimitError`, `InvalidRequestError`, `ModelNotFoundError`, `ContentModerationError`, `AIProviderError`).
  - **Setup:** Use `pytest.mark.parametrize` to test different SDK error types.
  - **Checks:**
    - For each simulated SDK error, the correct custom framework exception is raised.
    - The exception message is informative.
    - Relevant details (like status code or error code) are potentially preserved in the custom exception.
    - Logger is called appropriately upon error.

### 4. Convert Response (`_convert_response`)

- **Test Case 4.1: Convert Response (Text Only)**

  - **Goal:** Verify the provider correctly parses a simple text response from the (mocked) SDK response object into the standardized `ProviderResponse` model.
  - **Checks:**
    - `ProviderResponse.content` contains the correct text.
    - `ProviderResponse.tool_calls` is `None` or empty.
    - `ProviderResponse.stop_reason` is mapped correctly.
    - `ProviderResponse.usage` (tokens) is extracted correctly.
    - `ProviderResponse.model` ID is extracted correctly.
    - `ProviderResponse.error` is `None`.

- **Test Case 4.2: Convert Response (Content Moderation / Stop Reason)**
  - **Goal:** Verify specific stop reasons (like content filters) are correctly identified and mapped in the `ProviderResponse`.
  - **Checks:**
    - `ProviderResponse.stop_reason` reflects the moderation or specific stop condition.
    - `ProviderResponse.content` might be empty or contain a specific marker.

### 5. Tool Handling (If Provider Supports Tools)

- **Test Case 5.1: Prepare Payload with Tools**

  - **Goal:** Verify `_prepare_request_payload` (or equivalent logic) correctly formats the `tools` and `tool_choice` parameters according to the provider's specification when tools are provided.
  - **Setup:** Provide a list of `ToolDefinition` mocks.
  - **Checks:**
    - The `tools` parameter in the payload matches the SDK's expected format.
    - `tool_choice` is included if specified.

- **Test Case 5.2: Convert Response with Tool Calls**

  - **Goal:** Verify `_convert_response` correctly parses tool call requests from the SDK response into a list of `ToolCall` objects within the `ProviderResponse`.
  - **Setup:** Mock an SDK response indicating tool calls.
  - **Checks:**
    - `ProviderResponse.tool_calls` is a list of `ToolCall` objects.
    - Each `ToolCall` has the correct `id`, `name`, and parsed `arguments` (as a dictionary).
    - `ProviderResponse.content` may or may not be present, depending on the SDK response.
    - `ProviderResponse.stop_reason` indicates tool use (if applicable).

- **Test Case 5.3: Convert Response with Invalid Tool Arguments**

  - **Goal:** Verify how `_convert_response` handles cases where the SDK returns tool call arguments that are not valid JSON (if applicable to the provider).
  - **Setup:** Mock an SDK response with a tool call where `arguments` is an invalid JSON string.
  - **Checks:**
    - `ProviderResponse.tool_calls` contains a `ToolCall` object.
    - The `arguments` field in the `ToolCall` should ideally contain the raw, unparsed string or a representation indicating the parsing failure (e.g., `{"_raw_args": "..."}`). It should not raise an unhandled exception.

- **Test Case 5.4: Add Tool Message Formatting (`_add_tool_message`)**
  - **Goal:** Verify the provider correctly formats a `ToolResult` into the message structure expected by the provider to be sent back in the next turn.
  - **Setup:** Provide a `tool_call_id`, `tool_name`, and `content` (result string).
  - **Checks:**
    - The returned message list/dictionary matches the provider's required format for tool result messages (e.g., role='tool', specific keys for ID/name/content).

## Next Steps

With these general cases defined, the next step is to write the specific `pytest` files for each provider, starting with one (e.g., `OpenAIProvider`), implementing these tests using mocks for the `openai` SDK.
