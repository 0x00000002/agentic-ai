# Agentic AI Configuration System

This document explains how to use the Agentic AI configuration system to customize your AI experience.

## Overview

The configuration system provides a clean, user-facing interface for configuring the AI framework without exposing implementation details. It handles:

- Model selection and configuration
- Use case presets
- Temperature and other model parameters
- System prompts
- Debugging options

## Basic Usage

```python
from src.config import configure
from src.core.tool_enabled_ai import ToolEnabledAI

# Configure the framework
configure(
    model="claude-3-5-sonnet",
    use_case="solidity_coding",
    temperature=0.8,
    show_thinking=True
)

# Create an AI instance - it will use the configuration applied above
ai = AI()
response = ai.request("Write a simple ERC20 token contract")
```

## Configuration Options

### Model Selection

Select a specific model by name:

```python
configure(model="claude-3-5-sonnet")  # Use Claude 3.5 Sonnet
configure(model="phi4")               # Use local Phi-4 model
```

### Use Case Presets

Apply a predefined configuration suitable for a specific task:

```python
from src.config import configure, UseCasePreset

# Using string
configure(use_case="solidity_coding")

# Using enum for better IDE support
configure(use_case=UseCasePreset.SOLIDITY_CODING)
```

Available use cases include:

- `CHAT` - General conversational AI
- `CODING` - General programming assistance
- `SOLIDITY_CODING` - Ethereum smart contract development
- `TRANSLATION` - Language translation
- `CONTENT_GENERATION` - Creating creative content
- `DATA_ANALYSIS` - Analyzing data
- `WEB_ANALYSIS` - Web content analysis
- `IMAGE_GENERATION` - Generating images (for compatible models)

### Model Parameters

Customize model behavior:

```python
configure(
    temperature=0.8,          # Higher values = more creative responses
    max_tokens=2000,          # Max response length
    system_prompt="You are a helpful assistant specialized in Solidity smart contract development."
)
```

### Debugging Options

Enable advanced options for development:

```python
configure(
    show_thinking=True        # Show AI's reasoning process
)
```

## Advanced Configuration

### External Configuration File

Load configuration from a YAML or JSON file:

```python
configure(config_file="my_config.yml")
```

Example YAML configuration:

```yaml
model: claude-3-5-sonnet
use_case: solidity_coding
temperature: 0.8
show_thinking: true
system_prompt: You are a helpful assistant specialized in Solidity smart contract development.
```

### Custom Parameters

Pass any custom parameters to the configuration system:

```python
configure(
    model="claude-3-5-sonnet",
    my_custom_param="value",
    another_param=123
)
```

These custom parameters will be stored and can be accessed through the configuration API.

## Using the Configuration API

If you need direct access to the configuration system:

```python
from src.config import get_config

config = get_config()

# Get the default model
default_model = config.get_default_model()

# Get configuration for a specific model
model_config = config.get_model_config("claude-3-5-sonnet")

# Get all available models
all_models = config.get_all_models()

# Get system prompt (if configured)
system_prompt = config.get_system_prompt()
```

## Model Enumeration

For type-safe model references, use the dynamically generated `Model` enum:

```python
from src.config import Model, get_available_models, is_valid_model

# Use enum member
if Model.CLAUDE_3_5_SONNET == Model.CLAUDE_3_5_SONNET:
    print("Using Claude 3.5 Sonnet")

# Check if a model is valid
if is_valid_model("claude-3-5-sonnet"):
    print("Valid model")

# Get all available models
available_models = get_available_models()
```

## Configuration Files

The system uses these configuration files:

- `models.yml` - Model definitions and parameters
- `providers.yml` - Provider configurations
- `use_cases.yml` - Use case specific configurations
- `agents.yml` - Agent configurations
- `tools.yml` - Tool configurations

These files are managed by system administrators and should not be modified directly by users.
