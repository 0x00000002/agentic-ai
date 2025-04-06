# Automatic Tool Finding

Agentic-AI includes a powerful feature called `AIToolFinder` that enables AI to automatically discover and select relevant tools based on user input.

## Overview

The `AIToolFinder` component uses an AI model to analyze:

1. The user's request
2. Available tools and their descriptions
3. Conversation context (optional)

Based on this analysis, it selects the most appropriate tools for handling the request.

## How It Works

### Sequence Diagram

```
User Query
    |
    v
AIToolFinder
    |
    +-----> Available Tools Catalog
    |           |
    |<-----------
    |
    +-----> Tool Selection AI Model
    |           |
    |<-----------
    |
    v
Selected Tools
    |
    v
Tool Executor
    |
    v
Final Response
```

### Behind the Scenes

1. When a user submits a query, the `AIToolFinder` prepares a prompt containing:

   - The user's query
   - A list of all available tools with descriptions
   - Optional conversation history for context

2. This prompt is sent to a configured AI model that specializes in tool selection

3. The AI analyzes the query and tool descriptions, then returns a JSON response listing the relevant tools:

   ```json
   {
     "tools": ["tool_name_1", "tool_name_2"]
   }
   ```

4. `AIToolFinder` verifies the selected tools against the available tools catalog

5. The validated list of tool names is returned to the calling component (usually `ToolManager`)

## Implementation

The core of `AIToolFinder` is in `src/tools/ai_tool_finder.py`:

```python
class AIToolFinder:
    """
    Uses an AI model to find relevant tools based on user prompts and tool descriptions.
    """

    def __init__(self,
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: LoggerInterface):
        """
        Initialize the AIToolFinder.

        Args:
            model_id: The ID of the AI model to use for tool finding.
            config_manager: Configuration manager instance.
            logger: Logger instance.
        """
        # ... initialization ...

    def find_tools(self, user_prompt: str, conversation_history: Optional[List[str]] = None) -> Set[str]:
        """
        Analyzes the user's prompt and returns a set of relevant tool names.
        """
        # ... implementation ...
```

## Usage

### Enabling Auto Tool Finding

Enable automatic tool finding by setting `auto_find_tools=True` when creating an AI instance:

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.config.models import Model

# Create AI with auto tool finding
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    auto_find_tools=True
)

# Register tools
ai.register_tool("tool1", tool1_function, "Tool 1 description")
ai.register_tool("tool2", tool2_function, "Tool 2 description")

# When user requests come in, AIToolFinder will automatically select tools
response = ai.request("Can you tell me the weather in New York?")
```

### Manual Tool Finding via ToolManager

You can also use the `ToolManager` directly:

```python
from src.tools.tool_manager import ToolManager
from src.config.models import Model

# Create tool manager with auto tool finding
tool_manager = ToolManager(
    config_manager=config_manager,
    logger=logger
)

# Enable auto tool finding with a specific model
tool_manager.enable_auto_tool_finding(
    enabled=True,
    tool_finder_model_id=Model.GPT_4O_MINI
)

# Register tools
tool_manager.register_tool("tool1", tool1_function, "Tool 1 description")
tool_manager.register_tool("tool2", tool2_function, "Tool 2 description")

# Find and execute tools for a user prompt
results = tool_manager.process_with_tools("What's the weather like today?")
```

## Customizing the Tool Finding Model

You can specify which AI model should be used for tool finding:

```python
from src.tools.tool_manager import ToolManager
from src.config.models import Model

# Create tool manager
tool_manager = ToolManager()

# Enable auto tool finding with a specific model
tool_manager.enable_auto_tool_finding(
    enabled=True,
    tool_finder_model_id=Model.GPT_4O_MINI  # Choose a smaller/faster model for tool finding
)
```

## Best Practices

1. **Tool Descriptions**: Provide clear, concise descriptions for each tool
2. **Tool Naming**: Use descriptive names that indicate the tool's function
3. **Model Selection**: For tool finding, smaller/faster models are often sufficient
4. **Performance Monitoring**: Track which tools are selected and adjust descriptions if tools are being incorrectly selected
5. **Conversation Context**: Include conversation history when relevant to provide better context for tool selection
