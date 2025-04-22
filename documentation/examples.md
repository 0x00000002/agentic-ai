# Usage Examples

This page presents practical examples of using Agentic-AI in different scenarios.

## Basic AI Interaction

```python
from src.config.models import Model
from src.config.config_manager import ConfigManager
from src.core.tool_enabled_ai import ToolEnabledAI
from src.utils.logger import LoggerFactory

# Set up logger
logger = LoggerFactory.create()

# Initialize ConfigManager
config_manager = ConfigManager()

# Create AI instance
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    config_manager=config_manager,
    logger=logger
)

# Send a request
response = ai.request("What is the capital of France?")
print(response)
```

## Creating a Weather Assistant

```python
from src.config.models import Model
from src.core.tool_enabled_ai import ToolEnabledAI

# Create a weather assistant
ai = AI(
    model=Model.GPT_4O,
    system_prompt="You are a helpful weather assistant. Your goal is to provide weather information."
)

# Define weather tool
def get_weather(location: str) -> str:
    """Get current weather for a location (mocked for example)"""
    # In a real application, this would call a weather API
    weather_data = {
        "New York": "Sunny, 75°F",
        "London": "Rainy, 62°F",
        "Tokyo": "Partly cloudy, 80°F",
        "Paris": "Clear skies, 70°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

# Register weather tool
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get current weather for a specific location",
    parameters_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or location name"
            }
        },
        "required": ["location"]
    }
)

# Use the weather assistant
response = ai.request("What's the weather like in Tokyo today?")
print(response)
```

## AI with Auto Tool Finding

This example demonstrates using `AIToolFinder` to automatically select relevant tools.

```python
from src.config.models import Model
from src.config.config_manager import ConfigManager
from src.core.tool_enabled_ai import ToolEnabledAI
from src.utils.logger import LoggerFactory

# Set up
logger = LoggerFactory.create()
config_manager = ConfigManager()

# Create AI with auto tool finding
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    system_prompt="You are a helpful assistant. Use tools when appropriate to answer user queries.",
    config_manager=config_manager,
    logger=logger,
    auto_find_tools=True  # Enable auto tool finding
)

# Register multiple tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"It's sunny in {location} today!"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Sorry, I couldn't evaluate that expression."

def get_ticket_price(destination: str) -> str:
    """Get ticket price for a destination."""
    return f"A ticket to {destination} costs $1000 USD"

# Register all tools
ai.register_tool("get_weather", get_weather, "Get weather for a location")
ai.register_tool("calculate", calculate, "Perform calculations")
ai.register_tool("get_ticket_price", get_ticket_price, "Get ticket price information")

# The AI will automatically select the appropriate tool
response = ai.request("How much does a ticket to New York cost?")
print(response)

# Try another query
response = ai.request("What's 125 * 37?")
print(response)
```

## Using Prompt Templates

This example shows how to use the prompt management system.

```python
from src.config.models import Model
from src.prompts import PromptManager
from src.core.tool_enabled_ai import ToolEnabledAI

# Initialize prompt manager
prompt_manager = PromptManager(storage_dir="data/prompts")

# Create a template for customer support
template_id = prompt_manager.create_template(
    name="Customer Support",
    description="Template for answering customer support questions",
    template="You are a customer support agent for {{company}}. Answer this customer question: {{question}}",
    default_values={"company": "Acme Corp"}
)

# Create AI with prompt manager
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    prompt_manager=prompt_manager
)

# Use the template
response = ai.request_with_template(
    template_id=template_id,
    variables={
        "question": "How do I reset my password?",
        "company": "TechGiant Inc."
    }
)
print(response)

# Create an alternative version for A/B testing
prompt_manager.create_version(
    template_id=template_id,
    template_string="As a {{company}} support representative, please help with: {{question}}",
    name="Alternative Wording",
    description="Different wording to test effectiveness"
)

# The A/B testing is handled automatically when user_id is provided
response = ai.request_with_template(
    template_id=template_id,
    variables={
        "question": "How do I cancel my subscription?",
        "company": "TechGiant Inc."
    },
    user_id="user-123"  # This determines which version they get
)
print(response)
```

## Using the Tool Statistics Manager

This example demonstrates how to track and analyze tool usage with the `ToolStatsManager`.

```python
from src.tools.tool_stats_manager import ToolStatsManager
from src.tools.tool_manager import ToolManager

# Option 1: Using ToolStatsManager directly
stats_manager = ToolStatsManager()

# Record tool usage
stats_manager.update_stats(
    tool_name="search_tool",
    success=True,
    duration_ms=250
)

# Retrieve statistics
search_stats = stats_manager.get_stats("search_tool")
print(f"Search tool success rate: {search_stats['successes']/search_stats['uses']*100:.1f}%")

# Save statistics for persistence
stats_manager.save_stats()

# Option 2: Using through ToolManager (recommended approach)
tool_manager = ToolManager()

# Execute a tool - stats tracking happens automatically
result = tool_manager.execute_tool("weather_tool", location="New York")

# Save statistics
tool_manager.save_usage_stats()

# Get tool statistics
tool_info = tool_manager.get_tool_info("weather_tool")
if tool_info and 'usage_stats' in tool_info:
    usage_stats = tool_info['usage_stats']
    print(f"Weather tool used {usage_stats['uses']} times with {usage_stats['successes']} successes")
```

For more detailed examples, see [Tool Statistics Example](examples/tool_stats_example.md).

# Examples

This section provides various examples demonstrating how to use the Agentic AI framework.

## Core Functionality

- **`framework_example.py`**: Shows overall framework setup, configuration, basic agent creation, and interaction.
- **`configuration_example.py`**: Focuses on different ways to configure the framework using `configure()` and `UseCasePreset`.
- **`agent_usage_example.py`**: Demonstrates direct instantiation and interaction with a specific agent (`RequestAnalyzer`).
- **`simple_agent_interaction.py`**: (New) Provides a high-level example of interacting with the main `Coordinator` agent, hiding internal setup details. Shows how text requests, image requests (using direct dispatch), and meta requests are handled.

## Tool Management & Usage

- **`tool_execution_example.py`**: Illustrates how to list available tools (internal and MCP) and execute them directly using `ToolManager`.
- **`mcp_tool_example.py`**: Specific example for configuring and using tools hosted on an external MCP (Model-Centric Protocol) server.
- **`tool_stats_example.py`**: Shows how to enable, track, and view tool usage statistics using `ToolStatsManager`.
- **`tool_test_example.py`**: Example of setting up tests for custom tools.

## Advanced Features

- **`metrics_example.py`**: Demonstrates the usage of the metrics system for tracking performance and other data points.
- **`rag_mcp_example.py`**: Example showcasing Retrieval-Augmented Generation (RAG) potentially combined with MCP tools.
- **`ui_example.py`**: Shows how to integrate the framework with a Gradio-based user interface (`SimpleChatUI`).

## Running Examples

Most examples can be run from the project's root directory using:

```bash
# Ensure necessary environment variables are set (e.g., API keys)
export OPENAI_API_KEY=...
export REPLICATE_API_TOKEN=...

# Run using the module flag
python -m examples.example_script_name
```

Replace `example_script_name` with the name of the Python file (without `.py`). Ensure the `examples/` directory contains an `__init__.py` file.
