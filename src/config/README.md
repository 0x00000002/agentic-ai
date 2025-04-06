# Agentic AI Configuration System

The configuration system has been completely redesigned to provide a clean, user-facing interface for the AI framework.

## Key Components

1. **UserConfig**: A simple, clean interface for users to configure the AI framework.
2. **UnifiedConfig**: A unified configuration manager that handles all configuration aspects.
3. **Dynamic Models**: A dynamically generated Model enumeration based on the models defined in the configuration files.
4. **Dynamic Providers**: A dynamically generated Provider enumeration based on the providers defined in the configuration files.

## Configuration Files

The system uses these modular YAML files for configuration:

- `models.yml` - Model definitions and parameters
- `providers.yml` - Provider configurations
- `use_cases.yml` - Use case specific configurations
- `agents.yml` - Agent configurations
- `tools.yml` - Tool configurations

## Usage

### Basic Usage

```python
from src.config import configure
from src.core.tool_enabled_ai import ToolEnabledAI

# Configure the framework
configure(model="claude-3-5-sonnet", temperature=0.8)

# Create an AI instance
ai = ToolEnabledAI()
response = ai.request("Hello, world!")
```

### Advanced Usage

See the documentation in `docs/config/README.md` and examples in `docs/examples/configuration_example.md` for more details on advanced usage.

## Implementation Notes

- The `Model` enum is dynamically generated from the configuration files, eliminating the need to manually update the enum when adding or removing models.
- The `Provider` enum is also dynamically generated, making it easy to add new providers without code changes.
- The system supports configuration through YAML or JSON files, environment variables, and programmatic overrides.

# Example: Create AI instance using configuration

def create_ai_instance():
config = UnifiedConfig.get_instance() # AI instance uses the loaded configuration (e.g., default model)
ai = ToolEnabledAI()
response = ai.request("Hello, world!")
print(response)
