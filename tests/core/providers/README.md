# Provider Tests

This directory contains comprehensive tests for the AI provider components of the framework. The tests are organized into several categories, each focusing on specific aspects of provider functionality.

## Test Categories

### Factory Tests (`factory/`)

- Provider creation and initialization
- Configuration validation
- Custom provider registration
- Model selection and validation

### Implementation Tests (`implementation/`)

- Provider-specific functionality
- Message handling
- Response formatting
- Streaming support
- Multi-part content handling

### Configuration Tests (`configuration/`)

- Provider configuration validation
- Environment variable handling
- Configuration file management
- Default value handling

### Error Handling Tests (`error_handling/`)

- Rate limiting
- Network errors
- Invalid responses
- Concurrent request limits
- Retry mechanisms
- Authentication errors
- Timeout handling
- Provider-specific errors

### Tool Integration Tests (`tool_integration/`)

- Tool execution
- Result formatting
- Tool validation
- Usage tracking
- Availability checking
- Error handling
- Streaming support
- Timeout handling

### Prompt Template Tests (`prompt_templates/`)

- Template selection
- Variable substitution
- Version management
- Performance tracking
- Template validation
- Template inheritance
- Template caching
- Localization support

## Test Structure

Each test category follows a consistent structure:

- Feature files (`.feature`) defining test scenarios
- Step definitions (`steps/`) implementing the test logic
- Shared fixtures (`fixtures.py`) providing common test components

## Running Tests

To run the tests, use the following commands:

```bash
# Run all provider tests
behave tests/core/providers

# Run specific test category
behave tests/core/providers/factory
behave tests/core/providers/implementation
behave tests/core/providers/configuration
behave tests/core/providers/error_handling
behave tests/core/providers/tool_integration
behave tests/core/providers/prompt_templates
```

## Writing New Tests

When adding new tests:

1. Create a new feature file in the appropriate category directory
2. Define test scenarios using Gherkin syntax
3. Implement step definitions using the shared fixtures
4. Follow the existing patterns for error handling and assertions
5. Ensure proper cleanup in `@then` steps

## Best Practices

1. Use shared fixtures for common setup
2. Follow consistent naming conventions
3. Include proper error handling
4. Add comprehensive assertions
5. Document complex test scenarios
6. Maintain test isolation
7. Use appropriate mocking strategies

## Dependencies

The tests rely on the following components:

- Behave for BDD testing
- Mock for test doubles
- Shared fixtures for common setup
- Provider implementations
- Core framework components

## Contributing

When contributing to the test suite:

1. Follow the existing patterns and structure
2. Use shared fixtures for common setup
3. Add comprehensive documentation
4. Ensure proper error handling
5. Maintain test isolation
6. Follow naming conventions
7. Add appropriate assertions
