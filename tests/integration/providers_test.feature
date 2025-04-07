Feature: AI Provider Implementation
  As a developer using the agentic-ai framework
  I want the AI providers to handle various scenarios correctly
  So that I can rely on consistent behavior across different AI services

  Background:
    Given the AI framework is initialized
    And provider configurations are available

  # Provider Initialization Tests
  Scenario: Successfully initialize a provider with valid configuration
    Given a valid provider configuration for "openai"
    And a valid model configuration for "gpt-4o"
    When I create a new provider instance
    Then the provider should be initialized successfully
    And the provider should have the correct model ID "gpt-4o"
    And the provider should be configured with the API key

  Scenario: Fail to initialize a provider with missing API key
    Given a provider configuration for "anthropic" without an API key
    And a valid model configuration for "claude-3-opus"
    When I try to create a new provider instance
    Then an AICredentialsError should be raised
    And the error message should indicate the missing API key

  Scenario: Fail to initialize a provider with missing configuration
    Given a provider type "gemini" without provider configuration
    And a valid model configuration for "gemini-pro"
    When I try to create a new provider instance
    Then an AIConfigError should be raised
    And the error message should indicate the missing provider configuration

  # Request Payload Formatting Tests
  Scenario: Format a basic request payload without tools
    Given an initialized "openai" provider for model "gpt-4o"
    And a list of conversation messages
    When I prepare a request payload
    Then the payload should include the model ID "gpt-4o"
    And the payload should have correctly formatted messages
    And the payload should include default parameters

  Scenario: Format a request payload with options overriding defaults
    Given an initialized "anthropic" provider for model "claude-3-opus"
    And a list of conversation messages
    When I prepare a request payload with temperature 0.2 and max_tokens 500
    Then the payload should include the temperature 0.2
    And the payload should include max_tokens 500
    And the payload should override any default values

  # API Request Tests
  Scenario: Make a successful API request
    Given an initialized "openai" provider with a mocked SDK client
    And a prepared request payload
    When I make an API request
    Then the SDK client should be called once with the correct payload
    And the raw SDK response should be returned

  Scenario Outline: Map SDK errors to framework exceptions
    Given an initialized "<provider>" provider with a mocked SDK client
    And the SDK client is configured to raise a "<sdk_error>"
    When I make an API request
    Then a "<framework_error>" should be raised
    And the error message should include relevant details

    Examples:
      | provider   | sdk_error              | framework_error        |
      | openai     | AuthenticationError    | AIAuthenticationError  |
      | anthropic  | BadRequestError        | InvalidRequestError    |
      | gemini     | RateLimitError         | AIRateLimitError       |
      | openai     | InternalServerError    | AIProviderError        |
      | anthropic  | ContentModerationError | ContentModerationError |

  # Response Conversion Tests
  Scenario: Convert a text-only response
    Given an initialized "openai" provider
    And a raw SDK response with text content "This is a test response"
    When I convert the response
    Then the ProviderResponse.content should be "This is a test response"
    And the ProviderResponse.tool_calls should be empty
    And the ProviderResponse.error should be None

  Scenario: Convert a response with content moderation
    Given an initialized "anthropic" provider
    And a raw SDK response with a content moderation stop reason
    When I convert the response
    Then the ProviderResponse.stop_reason should indicate content moderation
    And the ProviderResponse.content might be empty or contain a warning

  # Tool Handling Tests
  Scenario: Format a request payload with tools
    Given an initialized provider that supports tools
    And a list of tool definitions
    When I prepare a request payload with tools
    Then the payload should include properly formatted tools
    And the tool definitions should match the provider's expected format

  Scenario: Convert a response with tool calls
    Given an initialized provider that supports tools
    And a raw SDK response with tool call requests
    When I convert the response
    Then the ProviderResponse.tool_calls should contain ToolCall objects
    And each ToolCall should have the correct id, name, and arguments

  Scenario: Handle a response with invalid tool arguments
    Given an initialized provider that supports tools
    And a raw SDK response with a tool call containing invalid JSON arguments
    When I convert the response
    Then the ProviderResponse.tool_calls should contain a ToolCall object
    And the ToolCall arguments should be handled gracefully without exceptions

  Scenario: Format a tool result message
    Given an initialized provider that supports tools
    And a tool call ID, name, and result content
    When I format a tool result message
    Then the message should match the provider's required format
    And the message should include the tool call ID, name, and content

  # Streaming Tests
  Scenario: Stream a response from the provider
    Given an initialized provider that supports streaming
    And a list of conversation messages
    When I request a streaming response
    Then the provider should stream chunks of the response
    And the complete response should be assembled correctly
    And the conversation history should be updated with the full response 