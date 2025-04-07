# Agentic-AI

A modular framework for building AI applications with tool integration capabilities.

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://0x00000002.github.io/agentic-ai/)

## Overview

Agentic-AI is a Python library designed to create AI-powered applications that can:

- Use multiple AI model providers (OpenAI, Anthropic, Google, Ollama)
- Dynamically discover and call tools based on user input
- Manage conversations and maintain context
- Template and version prompts with metrics tracking

## Key Features

- **Multiple Provider Support**: Use models from OpenAI, Anthropic, Google, and Ollama seamlessly
- **Tool Integration**: Register Python functions as tools the AI can use
- **Automatic Tool Discovery**: AI-powered selection of relevant tools based on user queries
- **Prompt Management**: Create, version, and track performance of prompt templates
- **Conversation Management**: Maintain context across multiple interactions

## Installation

```bash
# With pip
pip install -r requirements.txt

# For development installation
pip install -e .
```

## Quick Example

```python
from src.core.tool_enabled_ai import ToolEnabledAI

# Create an AI instance
ai = ToolEnabledAI()

# Make a request
response = ai.request("What is the capital of France?")
print(response)
```

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory.

To build and view the documentation:

```bash
# Install MkDocs and the Material theme
pip install mkdocs mkdocs-material

# Serve the documentation locally
mkdocs serve
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-ai.git
cd agentic-ai

# Create and activate conda environment
conda env create -f environment.yml
conda activate agentic-ai

# Install in development mode
pip install -e .
```

### Testing

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific module tests
python -m unittest discover -s tests/tools

# Run tools test suite
python tests/tools/run_tools_tests.py
```

## License

[MIT License](LICENSE)

### Basic Usage

```python
from src.core.tool_enabled_ai import ToolEnabledAI

# Create an AI instance
ai = ToolEnabledAI()

# Make a request
response = ai.request("What is the capital of France?")
print(response)
```

### Using Specific Models

You can specify a model during initialization:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.dynamic_models import Model

# Use a specific model
ai = ToolEnabledAI(model=Model.CLAUDE_3_7_SONNET)
response = ai.request("Write a short poem about AI.")
print(response)
```

### Using Tools

Define a Python function and register it as a tool:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
import requests

def get_weather(location: str):
# ... (function remains same)

# Create AI instance
ai = ToolEnabledAI()

# Register the tool
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get the current weather for a specified location"
)

# Use the AI with tools
response = ai.request("What's the weather like in Paris today?")
print(response)
```
