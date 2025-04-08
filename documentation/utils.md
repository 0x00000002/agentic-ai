# Utilities

This section covers utility components used throughout the Agentic AI framework.

## Logging (`src/utils/logger.py`)

The framework utilizes a flexible logging system based on a `LoggerFactory`.

- **`LoggerFactory`**: A factory class used to create logger instances.
- **Purpose**: Provides a consistent way to obtain configured loggers across different modules.
- **Usage**:

  ```python
  from src.utils.logger import LoggerFactory

  # Get a logger instance (typically within a class __init__)
  logger = LoggerFactory.create(name="my_module_or_class_name")

  # Use the logger
  logger.info("This is an informational message.")
  logger.debug("This is a debug message.")
  logger.warning("This is a warning.")
  logger.error("This is an error message.")
  ```

- **Configuration**: Logging levels and output handlers (e.g., console, file) are typically configured globally, potentially influenced by environment variables or application setup.
