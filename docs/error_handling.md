# Error Handling

The Agentic-AI framework uses a custom exception hierarchy to provide detailed error information. All framework-specific exceptions inherit from `AIFrameworkError`.

## Base Exception

- **`AIFrameworkError`**: The base class for all custom exceptions in this framework.

## Configuration Errors

- **`AIConfigError`**: Base class for configuration-related issues.
  - **`AIConfigLoadingError`**: Error during loading of configuration files.
  - **`AIConfigValidationError`**: Invalid configuration values found.

## Provider Errors

- **`AIProviderError`**: Base class for errors originating from AI model providers.
  - **`AIAuthenticationError`**: Authentication issues with the provider (e.g., invalid API key).
  - **`AIRateLimitError`**: Rate limits exceeded for the provider API.
  - **`AIServiceUnavailableError`**: Provider's service is temporarily unavailable.
  - **`AIProviderInvalidRequestError`**: The request sent to the provider was malformed or invalid.
  - **`AIProviderResponseError`**: Error parsing or handling the provider's response.

## Core Errors

- **`AISetupError`**: Errors during the setup or initialization of core components (e.g., model selection failure).
- **`AIProcessingError`**: General errors during AI request processing.

## Tool Errors

- **`AIToolError`**: Base class for errors related to tool handling.
  - **`AIToolNotFoundError`**: Requested tool could not be found.
  - **`AIToolExecutionError`**: An error occurred during the execution of a tool function.
  - **`AIToolTimeoutError`**: Tool execution exceeded the configured timeout.

## Conversation Errors

- **`AIConversationError`**: Errors related to conversation management.

## Usage

When interacting with the framework, you can use `try...except` blocks to catch specific errors or the base `AIFrameworkError` for general handling.

```python
from src.core import ToolEnabledAI
from src.exceptions import AIProviderError, AIToolError, AIFrameworkError

try:
    ai = ToolEnabledAI()
    response = ai.process_prompt("Use the calculator tool to find 5+5")
    print(response)
except AIProviderError as e:
    print(f"AI Provider Issue: {e}")
except AIToolError as e:
    print(f"Tool Execution Issue: {e}")
except AIFrameworkError as e:
    print(f"General Framework Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```
