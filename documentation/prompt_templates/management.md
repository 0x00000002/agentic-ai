# Prompt Management using PromptTemplate

## Overview

The primary way to manage and use prompts in Agentic AI is through the `PromptTemplate` service (`src/prompts/prompt_template.py`). This service allows you to:

- Define reusable prompt templates with variables in YAML files.
- Include multiple versions of a template within the same definition.
- Specify a default version for each template.
- Render specific template versions with variable substitution.
- Perform basic usage tracking for rendered prompts.

## Defining Templates in YAML

Prompt templates are defined in YAML files located in a designated directory (default: `src/prompts/templates/`). Each file can contain definitions for one or more templates.

The structure for a template definition is as follows:

````yaml
# Example: src/prompts/templates/coding_prompts.yaml

explain_code:
  description: "Explains a given code snippet."
  default_version: "v1.1"
  versions:
    - version: "v1.0"
      template: |
        Explain the following {{language}} code:
        ```{{language}}
        {{code_snippet}}
        ```
    - version: "v1.1"
      template: |
        Act as an expert {{language}} programmer.
        Provide a clear and concise explanation for this code snippet:
        ```{{language}}
        {{code_snippet}}
        ```
        Focus on the core logic and potential edge cases.

generate_function:
  description: "Generates a function based on a description."
  default_version: "v1.0"
  versions:
    - version: "v1.0"
      template: |
        Write a {{language}} function that does the following:
        {{function_description}}
````

Key elements:

- **Top-level key (`explain_code`, `generate_function`)**: This is the `template_id` used to reference the template.
- **`description`**: A brief explanation of the template's purpose.
- **`default_version`**: (Optional) The version string to use if no specific version is requested during rendering.
- **`versions`**: A list of dictionaries, each representing a specific version.
  - **`version`**: A unique identifier string for this version (e.g., "v1.0", "v1.1", "experimental").
  - **`template`**: The actual prompt text for this version. Variables are enclosed in double curly braces (e.g., `{{language}}`, `{{code_snippet}}`).

## Using the PromptTemplate Service

The `PromptTemplate` service loads these YAML files upon initialization and provides methods to render them.

```python
from src.core.tool_enabled_ai import ToolEnabledAI # Or your base AI class
from src.prompts.prompt_template import PromptTemplate

# Initialize the template service (loads templates from default directory)
# You can optionally specify a different directory: PromptTemplate(templates_dir="path/to/custom/templates")
template_service = PromptTemplate()

# --- Integrate with AI instance ---
# Pass the service instance during AI initialization
ai = ToolEnabledAI(
    # ... other AI config like model, tools etc. ...
    prompt_template=template_service
)

# --- Rendering a Prompt ---

# 1. Define variables required by the template
variables = {
    "language": "Python",
    "code_snippet": "def hello():\n  print(\"Hello, World!\")"
}

# 2. Request using the template ID (from the YAML file)
# This will use the default version ("v1.1" in the example YAML)
response_default = ai.request_with_template(
    template_id="explain_code",
    variables=variables
)
print("--- Default Version Response ---")
print(response_default)

# 3. Request a specific version
response_v1 = ai.request_with_template(
    template_id="explain_code",
    variables=variables,
    version="v1.0" # Specify the desired version string
)
print("\n--- Specific Version (v1.0) Response ---")
print(response_v1)

# --- Direct Rendering (Lower Level) ---
# If you need to render without going through the AI's request method:
try:
    rendered_prompt, usage_id = template_service.render_prompt(
        template_id="explain_code",
        variables=variables,
        version="v1.1"
    )
    print(f"\n--- Directly Rendered Prompt (v1.1, Usage ID: {usage_id}) ---")
    print(rendered_prompt)

    # Basic performance tracking can be done using the usage_id
    # (Example - actual metrics depend on what you measure)
    metrics = {"tokens_used": 150, "latency_ms": 550}
    template_service.record_prompt_performance(usage_id, metrics)

except ValueError as e:
    print(f"Error rendering prompt: {e}")

```

## Key Methods of `PromptTemplate` Service

- **`__init__(templates_dir=None, logger=None)`**: Initializes the service, loading templates from the specified or default directory.
- **`render_prompt(template_id, variables=None, version=None, context=None)`**: Finds the specified template and version (or default), substitutes variables, and returns the rendered prompt string along with a unique `usage_id` for tracking.
- **`record_prompt_performance(usage_id, metrics)`**: Records performance data (like latency, token count, success status) associated with a specific `usage_id`. This data is typically saved to a file.
- **`get_template_ids()`**: Returns a list of loaded template IDs.
- **`get_template_info(template_id)`**: Returns the loaded data (description, versions, etc.) for a specific template ID.
- **`reload_templates()`**: Clears the cache and reloads templates from the YAML files.

## Note on `PromptManager`

There is another class, `PromptManager` (`src/prompts/prompt_manager.py`), which appears to be a more complex system focused on programmatic template/version creation, detailed metrics storage (in `metrics.json`), and A/B testing infrastructure. This system seems to operate independently of the YAML-based templates loaded by the `PromptTemplate` service. For standard usage involving predefined prompts, the `PromptTemplate` service described here is the recommended approach.
