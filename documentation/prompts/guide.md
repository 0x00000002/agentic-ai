# Prompt Templates

This document describes the template system in Agentic-AI, which provides a way to create, manage, and reuse prompt templates with variable substitution and performance tracking.

## Overview

The prompt template system allows you to:

1. Define reusable prompt patterns with variable placeholders
2. Version and track prompt performance
3. Substitute variables into templates
4. Collect metrics on template usage

## Template Format

Templates are defined in YAML files in the `src/prompts/templates` directory. Each template has:

- **ID**: Unique identifier for the template
- **Name**: Human-readable name
- **Description**: Description of the template's purpose
- **Versions**: Multiple versions of the template (for A/B testing or iteration)
- **Default Version**: Which version to use by default

Example template file (`analysis.yml`):

```yaml
analyze_request:
  name: "Request Analysis"
  description: "Template for analyzing user requests to determine appropriate agents"
  default_version: "v1"
  versions:
    - version: "v1"
      template: |
        Analyze this user request and determine which specialized agents should handle it:

        Request: {{prompt}}

        Available agents:
        {{agent_list}}

        Return a JSON list of [agent_id, confidence] pairs, where confidence is 0.0-1.0.
        Only include agents with confidence > {{confidence_threshold}}. If no agents are appropriate, return [].
```

## Variable Substitution

Templates use a simple variable substitution syntax with double curly braces:

- `{{variable_name}}` - Will be replaced with the value of `variable_name`

## Using Templates in Code

### Basic Usage

```python
from src.prompts.prompt_template import PromptTemplate

# Create a template service
template_service = PromptTemplate()

# Render a template with variables
prompt, usage_id = template_service.render_prompt(
    template_id="analyze_request",
    variables={
        "prompt": "What's the weather like in Paris?",
        "agent_list": "- weather_agent: Gets weather information\n- translator: Translates text",
        "confidence_threshold": 0.6
    }
)

print(prompt)

# Record performance metrics
template_service.record_prompt_performance(
    usage_id=usage_id,
    metrics={
        "success": True,
        "tokens": 150,
        "latency": 0.75
    }
)
```

### With Specific Version

```python
prompt, usage_id = template_service.render_prompt(
    template_id="analyze_request",
    variables={"prompt": "What's the weather like in Paris?"},
    version="v2"  # Use a specific version
)
```

### With Additional Context

```python
prompt, usage_id = template_service.render_prompt(
    template_id="analyze_request",
    variables={"prompt": "What's the weather like in Paris?"},
    context={"model": "gpt-4", "user_id": "user123"}  # Additional context
)
```

## Creating New Templates

To create a new template:

1. Create or edit a YAML file in `src/prompts/templates/`
2. Define your template with an ID, name, description, and versions
3. Use `{{variable_name}}` syntax for variables
4. Set a default version if you have multiple versions

Or programmatically:

```python
template_service = PromptTemplate()

template_service.add_template(
    template_id="my_template",
    template_data={
        "name": "My Template",
        "description": "A template for...",
        "default_version": "v1",
        "versions": [
            {
                "version": "v1",
                "template": "This is a template with {{variable}}"
            }
        ]
    }
)
```

## Performance Tracking

The template system automatically tracks:

- **Start time**: When the template was rendered
- **Variables used**: What values were substituted
- **Template version**: Which version was used

You can add additional metrics when recording performance:

```python
template_service.record_prompt_performance(
    usage_id=usage_id,
    metrics={
        "success": True,        # Whether the response was successful
        "latency": 1.25,        # Processing time in seconds
        "tokens": 150,          # Tokens used
        "model": "claude-3-5",  # Model used
        "custom_metric": 0.95   # Any custom metrics you want to track
    }
)
```

Metrics are saved to `src/prompts/templates/metrics/` as JSON files, which can be analyzed to improve prompts over time.
