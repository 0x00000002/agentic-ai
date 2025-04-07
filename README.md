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
- **Audio Input (via UI)**: Provides a simple chat UI (`src/ui/simple_chat.py`) that supports microphone input for transcribing user requests in multiple languages (currently English and Russian) using `openai-whisper`.

## Installation

First, ensure you have the necessary **system dependencies**:

- **Python** (version 3.9+)
- **`ffmpeg`**: Required for audio processing by the `openai-whisper` library.
  - On macOS: `brew install ffmpeg`
  - On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
  - On other systems: Follow instructions at [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

Then, install the Python packages:

```bash
# Install Python dependencies from requirements file (if applicable)
# pip install -r requirements.txt

# Install the whisper library for audio transcription
pip install -U openai-whisper

# For development installation
# pip install -e . # (Run this if you need editable install)
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

## Simple Chat UI

The framework includes a simple Gradio-based chat interface that demonstrates agent interaction, including the audio input feature.

To run it:

```bash
python src/ui/simple_chat.py
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
