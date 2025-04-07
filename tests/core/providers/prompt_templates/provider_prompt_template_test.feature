Feature: Provider Prompt Template Integration
  As a developer using the agentic-ai framework
  I want to ensure proper integration between providers and prompt templates
  So that prompts can be effectively managed and customized per provider

  Background:
    Given the AI framework is initialized
    And provider configurations are available
    And a provider instance is created
    And prompt templates are available

  Scenario: Use provider-specific templates
    Given a provider with specific template requirements
    And a provider-specific template
    When I request a prompt using the template
    Then the correct template should be selected
    And the template should match provider requirements
    And the template should be properly rendered
    And the result should be provider-compatible

  Scenario: Substitute template variables
    Given a provider with a template containing variables
    And variable values for substitution
    When I render the template
    Then all variables should be properly substituted
    And the substitution should be provider-aware
    And the result should be properly validated
    And the result should be provider-compatible

  Scenario: Handle template versioning
    Given a provider with multiple template versions
    And a specific version request
    When I request a template version
    Then the correct version should be selected
    And the version should be compatible with the provider
    And the version should be properly rendered
    And the result should be provider-compatible

  Scenario: Track template performance
    Given a provider with template performance tracking
    And multiple template renderings
    When I track template performance
    Then performance metrics should be collected
    And metrics should be provider-specific
    And metrics should be properly stored
    And metrics should be accessible for analysis

  Scenario: Handle template validation
    Given a provider with template validation requirements
    And a template to validate
    When I validate the template
    Then the template should be checked for required fields
    And the template should be checked for provider compatibility
    And invalid templates should be rejected
    And appropriate error messages should be generated

  Scenario: Support template inheritance
    Given a provider with template inheritance
    And a base template and derived template
    When I render the derived template
    Then the base template should be properly extended
    And provider-specific overrides should be applied
    And the result should be properly validated
    And the result should be provider-compatible

  Scenario: Handle template caching
    Given a provider with template caching
    And multiple requests for the same template
    When I request templates
    Then templates should be properly cached
    And cache hits should be properly handled
    And cache invalidation should work correctly
    And performance should be optimized

  Scenario: Support template localization
    Given a provider with template localization
    And templates in different languages
    When I request a localized template
    Then the correct language should be selected
    And the template should be properly localized
    And the result should be provider-compatible
    And the localization should be properly validated 