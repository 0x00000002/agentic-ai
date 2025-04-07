Feature: AI Provider Factory
  As a developer using the agentic-ai framework
  I want the provider factory to create the correct provider instances
  So that I can get the right implementation for each provider type

  Background:
    Given the AI framework is initialized
    And provider configurations are available in the unified config

  Scenario: Create OpenAI provider via factory
    Given a provider type "openai"
    And a model ID "gpt-4o"
    And valid provider and model configurations
    When I create a provider using the factory
    Then the factory should return an instance of OpenAIProvider
    And the provider should be initialized with the correct model ID

  Scenario: Create Anthropic provider via factory
    Given a provider type "anthropic"
    And a model ID "claude-3-opus"
    And valid provider and model configurations
    When I create a provider using the factory
    Then the factory should return an instance of AnthropicProvider
    And the provider should be initialized with the correct model ID

  Scenario: Create Gemini provider via factory
    Given a provider type "gemini"
    And a model ID "gemini-pro"
    And valid provider and model configurations
    When I create a provider using the factory
    Then the factory should return an instance of GeminiProvider
    And the provider should be initialized with the correct model ID

  Scenario: Create Ollama provider via factory
    Given a provider type "ollama"
    And a model ID "llama3"
    And valid provider and model configurations
    When I create a provider using the factory
    Then the factory should return an instance of OllamaProvider
    And the provider should be initialized with the correct model ID

  Scenario: Handle invalid provider type
    Given a provider type "nonexistent"
    And a model ID "test-model"
    And valid model configuration
    When I try to create a provider using the factory
    Then a ValueError should be raised
    And the error message should indicate the invalid provider type

  Scenario: Register custom provider
    Given a custom provider class "MyCustomProvider"
    When I register the custom provider with type "custom"
    And I create a provider of type "custom"
    Then the factory should return an instance of MyCustomProvider

  Scenario: Provider factory with Model enum
    Given a Model enum value for "gpt_4o"
    And valid provider and model configurations
    When I create a provider using the factory with the Model enum
    Then the factory should return an instance of OpenAIProvider
    And the provider should be initialized with the correct model ID

  Scenario: Factory with provider-specific logger
    Given a provider type "openai"
    And a model ID "gpt-4o"
    And a custom logger instance
    When I create a provider using the factory with the logger
    Then the provider should be initialized with the custom logger
    And log messages should be directed to the custom logger 