Feature: Tool-Provider Integration
  As a developer using the agentic-ai framework
  I want tools to work seamlessly with different AI providers
  So that I can implement complex workflows with tool usage

  Background:
    Given the AI framework is initialized
    And provider configurations are available
    And tools are registered in the system

  # Basic Tool Integration Tests
  Scenario: Send tools to OpenAI provider
    Given an initialized ToolEnabledAI with "openai" provider
    And a set of registered tools
    When I process a prompt that requires tool usage
    Then the provider should receive the tools in the correct format
    And the provider should make tool calls in its response
    And the ToolEnabledAI should execute the requested tools

  Scenario: Send tools to Anthropic provider
    Given an initialized ToolEnabledAI with "anthropic" provider
    And a set of registered tools
    When I process a prompt that requires tool usage
    Then the provider should receive the tools in the correct format
    And the provider should make tool calls in its response
    And the ToolEnabledAI should execute the requested tools

  # Tool Execution Flow Tests
  Scenario: Complete multi-turn tool conversation
    Given an initialized ToolEnabledAI with a tool-capable provider
    And a set of registered tools
    When I send a prompt that triggers multiple tool calls
    Then the provider should make the first tool call
    And the tool should be executed
    And the tool result should be sent back to the provider
    And the provider should make additional tool calls if needed
    And the final response should include the results of the tool executions

  Scenario: Handle tool execution errors
    Given an initialized ToolEnabledAI with a tool-capable provider
    And a registered tool that will raise an exception
    When I process a prompt that calls the failing tool
    Then the tool error should be captured
    And the error should be sent back to the provider
    And the provider should handle the error gracefully
    And the final response should acknowledge the tool error

  # Provider-Specific Tool Formatting
  Scenario: Format tools for OpenAI according to function calling spec
    Given an initialized ToolEnabledAI with "openai" provider
    And a collection of tools with different parameter types
    When the provider prepares the request payload with tools
    Then the tools should be formatted as OpenAI functions
    And function parameters should have the correct JSON Schema
    And required parameters should be marked as required

  Scenario: Format tools for Anthropic according to their tool spec
    Given an initialized ToolEnabledAI with "anthropic" provider
    And a collection of tools with different parameter types
    When the provider prepares the request payload with tools
    Then the tools should be formatted as Anthropic tools
    And tool parameters should have the correct schema format
    And required parameters should be marked as required

  # Tool Result Handling
  Scenario: Format tool results for different providers
    Given a tool call from a provider
    And a result from executing the tool
    When I format the tool result for the provider
    Then the result should be formatted according to the provider's requirements
    And the conversation should be updated with the tool result
    And the provider should be able to process the tool result in the next turn

  Scenario: Handle streaming with tools
    Given an initialized ToolEnabledAI with a provider that supports streaming
    And a registered tool that will be called
    When I request a streaming response that uses tools
    Then the initial streaming should stop when a tool call is requested
    And after the tool executes, streaming should resume
    And the final response should combine all streamed content 