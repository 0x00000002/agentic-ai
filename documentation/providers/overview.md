# AI Providers & Models

## Supported Providers

Agentic-AI supports multiple AI model providers:

- **Anthropic**: Claude models (3.5 Haiku, 3.7 Sonnet, etc.)
- **OpenAI**: GPT models (4o, o3-mini, etc.)
- **Google**: Gemini models (2.5 Pro, 1.5 Pro, etc.)
- **Ollama**: Open-source models for local deployment

Providers are supported via official SDKs.

## Models Configuration

Models are configured in the `config.yml` file:

```yaml
models:
  claude-3-7-sonnet:
    name: "Claude 3.7 Sonnet"
    model_id: claude-3-7-sonnet-20250219
    provider: anthropic
    privacy: EXTERNAL
    quality: MAXIMUM
    speed: STANDARD
    parameters: 350000000000 # 350B
    input_limit: 1000000
    output_limit: 200000
    temperature: 0.7
    cost:
      input_tokens: 3 # $3 per 1M input tokens
      output_tokens: 15 # $15 per 1M output tokens
      minimum_cost: 0.0001 # Minimum cost per request
  ollama:
    api_url: http://localhost:11434
    # Ollama-specific settings
```

## Model Selection

Models are defined in the `models.py` file as an enumeration:

```python
from enum import Enum

class Model(Enum):
    # Anthropic models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_35_TURBO = "gpt-3.5-turbo"

    # Google models
    GEMINI_PRO = "gemini-pro"
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # Ollama models
    LLAMA3_8B = "llama3:8b"
    LLAMA3_70B = "llama3:70b"
```

And mapped to providers in the configuration:

```yaml
models:
  claude-3-5-sonnet-20240620:
    provider: anthropic
    model_id: claude-3-5-sonnet-20240620
    ...
  gpt-4o:
    provider: openai
    model_id: gpt-4o
```

## Using Models

Create an AI instance with your chosen model:

```python
from src.config.models import Model
from src.core.tool_enabled_ai import ToolEnabledAI

# Using an Anthropic model
ai = AI(model=Model.CLAUDE_3_7_SONNET)

# Using an OpenAI model
ai = AI(model=Model.GPT_4O)

# Using a Google model
ai = AI(model=Model.GEMINI_2_5_PRO)

# Using an Ollama model
ai = AI(model=Model.GEMMA3-27B)
```

## Model Selection Helper

The ModelSelector helps choose the appropriate model for different use cases:

```python
from src.core.model_selector import ModelSelector, UseCase

# Get a model recommendation
recommended_model = ModelSelector.select_model(
    use_case=UseCase.CREATIVE_WRITING,
    quality_preference=Quality.HIGH,
    speed_preference=Speed.BALANCED
)

# Create AI with the recommended model
ai = AI(model=recommended_model)
```
