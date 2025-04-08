# Configuration System

## Overview

The configuration system is designed to be modular, extensible, and easy to use. It provides a centralized way to manage configurations for models, agents, and use cases. The system uses a hybrid approach that combines default configurations with the ability to override them with external configurations.

## Architecture

The configuration system consists of the following main components:

- `ConfigFactory`: A singleton class that manages the creation and access to specialized configuration managers.
- `BaseConfigManager`: A base class for all configuration managers.
- Specialized managers:
  - `ModelConfigManager`: Manages AI model configurations.
  - `AgentConfigManager`: Manages agent configurations.
  - `UseCaseConfigManager`: Manages use case configurations.

## Configuration Files

The system uses two main configuration files:

1. `config.yml`: Contains base configurations for models and use cases.
2. `agents.yml`: Contains agent-specific configurations.

These files are located in the `src/config` directory and serve as default configurations. Users can override these defaults by providing their own configuration files.

### Default Configurations

The system comes with default configuration files that provide a good starting point for most use cases. These files are:

- `src/config/config.yml`: Contains default model and use case configurations.
- `src/config/agents.yml`: Contains default agent configurations.

### Overriding Default Configurations

Users can override the default configurations by providing their own configuration files:

```python
from src.config.config_factory import ConfigFactory

# Initialize with custom config files
config_factory = ConfigFactory.get_instance(
    config_path="path/to/custom/config.yml",
    agent_config_path="path/to/custom/agents.yml"
)
```

If no custom configuration files are provided, the system will use the default configurations.

## Specialized Managers

### ModelConfigManager

The `ModelConfigManager` is responsible for managing AI model configurations. It provides methods to:

- Get available models
- Get model information
- Get model configuration

Example usage:

```python
from src.config.config_factory import ConfigFactory

# Get the ConfigFactory instance
config_factory = ConfigFactory.get_instance()

# Get available models
models = config_factory.model_config.get_available_models()

# Get model info
model_info = config_factory.model_config.get_model_info("gpt-4")

# Get model config
model_config = config_factory.model_config.get_model_config("gpt-4")
```

### AgentConfigManager

The `AgentConfigManager` is responsible for managing agent configurations. It provides methods to:

- Get available agents
- Get agent information
- Get agent configuration

Example usage:

```python
from src.config.config_factory import ConfigFactory

# Get the ConfigFactory instance
config_factory = ConfigFactory.get_instance()

# Get available agents
agents = config_factory.agent_config.get_available_agents()

# Get agent info
agent_info = config_factory.agent_config.get_agent_info("default")

# Get agent config
agent_config = config_factory.agent_config.get_agent_config("default")
```

### UseCaseConfigManager

The `UseCaseConfigManager` is responsible for managing use case configurations. It provides methods to:

- Get available use cases
- Get use case information
- Get use case configuration

Example usage:

```python
from src.config.config_factory import ConfigFactory

# Get the ConfigFactory instance
config_factory = ConfigFactory.get_instance()

# Get available use cases
usecases = config_factory.usecase_config.get_available_usecases()

# Get use case info
usecase_info = config_factory.usecase_config.get_usecase_info("default")

# Get use case config
usecase_config = config_factory.usecase_config.get_usecase_config("default")
```

## ConfigFactory

The `ConfigFactory` is a singleton class that manages the creation and access to specialized configuration managers. It provides methods to:

- Get the singleton instance
- Reset the singleton instance
- Get specialized configuration managers

Example usage:

```python
from src.config.config_factory import ConfigFactory

# Get the ConfigFactory instance
config_factory = ConfigFactory.get_instance()

# Get specialized managers
model_config = config_factory.model_config
agent_config = config_factory.agent_config
usecase_config = config_factory.usecase_config

# Reset the singleton instance
ConfigFactory.reset()
```

## Benefits

The configuration system provides several benefits:

1. **Separation of Concerns**: Each manager is responsible for a specific type of configuration.
2. **Improved Testability**: Each manager can be tested independently.
3. **Better Extensibility**: New configuration managers can be added easily.
4. **Clearer API**: Each manager provides a clear and focused API.
5. **Reduced Complexity**: The system is easier to understand and maintain.
6. **Default Configurations**: The system comes with default configurations that can be overridden.
7. **Singleton Pattern**: The `ConfigFactory` uses the singleton pattern to ensure a single instance is used throughout the application.

## Example

See `examples/config_system_example.py` for a complete example of how to use the configuration system.

The configuration system manages settings for various components of the AI framework. Key configuration files include:

- `models.yml`: Defines AI model parameters, capabilities, and costs.
- `agents.yml`: Contains agent-specific configurations.
- `providers.yml`: Configures API providers (e.g., OpenAI, Anthropic).
- `use_cases.yml`: Defines settings for different task types (e.g., chat, coding).

- `src/config/models.yml`: Contains base model definitions.
- `src/config/agents.yml`: Contains default agent configurations.
- `src/config/providers.yml`: Contains API provider settings.
- `src/config/use_cases.yml`: Defines default settings for different use cases.

See `UserConfig` class and `UnifiedConfig._apply_user_config` for details.

```bash
python your_app.py --config-file path/to/user_config.yml
# OR using environment variables:
export APP_MODEL="gpt-4o"
export APP_TEMPERATURE="0.8"
# OR command line arguments:
python your_app.py --model gpt-4o --temperature 0.8
```
