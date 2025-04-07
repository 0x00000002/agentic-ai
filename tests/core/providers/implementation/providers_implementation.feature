Feature: Specific AI Provider Implementations
  As a developer using the agentic-ai framework
  I want each provider implementation to handle its unique requirements
  So that I can use different AI services seamlessly

  Background:
    Given the AI framework is initialized
    And provider configurations are available

  # OpenAI Provider Implementation Tests
  Scenario: OpenAI provider correctly handles system messages
    Given an initialized "openai" provider
    And a conversation with a system message
    And user and assistant messages
    When I prepare a request payload
    Then the system message should be the first message in the array
    And all messages should have the correct role mappings

  Scenario: OpenAI provider formats tool calls correctly
    Given an initialized "openai" provider
    And a list of tool definitions with function schemas
    When I prepare a request payload with tools
    Then the tools should be formatted as function objects
    And each function should have a proper name, description and parameters schema

  Scenario: OpenAI provider parses tool calls from response
    Given an initialized "openai" provider
    And a raw response with function call objects
    When I convert the response
    Then the ProviderResponse.tool_calls should contain the function calls
    And the arguments should be parsed from JSON strings to dictionaries

  # Anthropic Provider Implementation Tests
  Scenario: Anthropic provider correctly handles system messages
    Given an initialized "anthropic" provider
    And a conversation with a system message
    And user and assistant messages
    When I prepare a request payload
    Then the system message should be in the system parameter
    And not included in the messages array
    And messages should only contain user and assistant roles

  Scenario: Anthropic provider formats tools correctly
    Given an initialized "anthropic" provider
    And a list of tool definitions
    When I prepare a request payload with tools
    Then the tools should be formatted according to Anthropic's schema
    And each tool should have a proper name, description and input schema

  Scenario: Anthropic provider handles tool results with conversation anchoring
    Given an initialized "anthropic" provider
    And a conversation with a tool call in the last assistant message
    When I format a tool result message
    Then the message should include a reference to the assistant's tool request
    And follow Anthropic's required format for tool responses
    
  # Gemini Provider Implementation Tests
  Scenario: Gemini provider formats roles correctly
    Given an initialized "gemini" provider
    And a conversation with system, user and assistant messages
    When I prepare a request payload
    Then the system message should be formatted as a "model" role
    And other messages should have their roles mapped correctly

  Scenario: Gemini provider handles content parts
    Given an initialized "gemini" provider
    And a message with text and image content parts
    When I prepare a request payload
    Then the content should be formatted as multi-part content
    And each part should have the correct mime type

  Scenario: Gemini provider converts multi-part responses
    Given an initialized "gemini" provider
    And a raw response with multiple content parts
    When I convert the response
    Then the ProviderResponse.content should combine all text parts
    And maintain the correct ordering

  # Ollama Provider Implementation Tests
  Scenario: Ollama provider formats messages correctly
    Given an initialized "ollama" provider
    And a conversation with system, user and assistant messages
    When I prepare a request payload
    Then the payload should have the correct format for Ollama
    And include all messages with proper role mapping

  Scenario: Ollama provider handles local model URLs
    Given a provider configuration for "ollama" with a local URL
    And a model configuration for "llama3"
    When I create a new provider instance
    Then the provider should use the local URL for requests
    And not try to retrieve remote credentials

  Scenario: Ollama provider handles streaming responses
    Given an initialized "ollama" provider
    And a conversation with messages
    When I request a streaming response
    Then the provider should handle the Ollama streaming format
    And correctly combine the streamed chunks 