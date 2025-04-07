Feature: Provider Configuration Management
  As a developer using the agentic-ai framework
  I want to ensure proper configuration management for providers
  So that providers can be properly initialized and configured

  Background:
    Given the AI framework is initialized
    And provider configurations are available in the unified config

  Scenario: Load provider configuration from UnifiedConfig
    Given a provider type "openai"
    When I load the provider configuration
    Then the configuration should contain required fields
    And the configuration should be properly validated
    And the configuration should be accessible through UnifiedConfig

  Scenario: Validate provider configuration
    Given a provider type "anthropic"
    And a malformed provider configuration
    When I attempt to validate the configuration
    Then a ConfigurationError should be raised
    And the error message should indicate the specific validation failure

  Scenario: Handle environment variable overrides
    Given a provider type "openai"
    And environment variables for provider configuration
    When I load the provider configuration
    Then the environment variables should override default values
    And the overridden values should be properly validated

  Scenario: Reload provider configuration
    Given a provider type "gemini"
    And an existing provider configuration
    When I update the configuration file
    And I reload the configuration
    Then the new configuration should be loaded
    And existing provider instances should use the new configuration

  Scenario: Validate provider-specific settings
    Given a provider type "ollama"
    And provider-specific configuration requirements
    When I validate the configuration
    Then all provider-specific settings should be present
    And the settings should meet provider requirements

  Scenario: Handle missing configuration files
    Given a non-existent configuration file
    When I attempt to load the configuration
    Then a ConfigurationError should be raised
    And the error should indicate the missing file

  Scenario: Handle invalid configuration format
    Given a malformed configuration file
    When I attempt to load the configuration
    Then a ConfigurationError should be raised
    And the error should indicate the format issue

  Scenario: Merge multiple configuration sources
    Given multiple configuration sources
    When I load the provider configuration
    Then the configurations should be properly merged
    And conflicts should be resolved according to priority
    And the final configuration should be consistent 