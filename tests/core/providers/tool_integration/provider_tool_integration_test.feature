Feature: Provider Tool Integration
  As a developer using the agentic-ai framework
  I want to ensure proper integration between providers and tools
  So that tools can be effectively used through different providers

  Background:
    Given the AI framework is initialized
    And provider configurations are available
    And a tool-enabled provider instance is created
    And tool definitions are available

  Scenario: Execute tool calls through provider
    Given a provider that supports tool calls
    And a list of available tools
    When I make a request with tool usage
    Then the provider should format tool calls correctly
    And the tool calls should be properly executed
    And the results should be returned to the provider
    And the response should include tool results

  Scenario: Format tool results for provider
    Given a provider that supports tool calls
    And tool execution results
    When I format the results for the provider
    Then the results should be properly formatted
    And the format should match provider requirements
    And the results should be properly validated

  Scenario: Validate tool calls
    Given a provider that supports tool calls
    And a tool call request
    When I validate the tool call
    Then the tool call should be checked for required fields
    And the parameters should be validated
    And invalid tool calls should be rejected
    And appropriate error messages should be generated

  Scenario: Manage tool history
    Given a provider that supports tool calls
    And a conversation with tool usage
    When I track tool usage history
    Then the history should be properly maintained
    And tool calls should be associated with messages
    And the history should be accessible for context
    And the history should be properly formatted

  Scenario: Check tool availability
    Given a provider that supports tool calls
    And a list of registered tools
    When I check tool availability
    Then the provider should report available tools
    And tool capabilities should be properly described
    And tool requirements should be validated
    And unavailable tools should be properly indicated

  Scenario: Handle tool execution errors
    Given a provider that supports tool calls
    And a tool that may fail
    When I execute the tool
    Then tool execution errors should be handled
    And appropriate error messages should be generated
    And the error should be properly propagated
    And the system should log tool execution errors

  Scenario: Support tool streaming
    Given a provider that supports streaming
    And a tool that returns streaming results
    When I execute the tool with streaming
    Then the results should be properly streamed
    And the stream should be properly formatted
    And the stream should be properly handled
    And the final result should be properly assembled

  Scenario: Handle tool timeouts
    Given a provider that supports tool calls
    And a tool with a timeout setting
    When I execute the tool
    Then the timeout should be properly enforced
    And timeout errors should be handled
    And appropriate error messages should be generated
    And the system should log timeout events 