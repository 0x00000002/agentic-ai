# ToolFinderAgent and ToolRegistry

## Overview

The ToolFinderAgent is a specialized agent in the Agentic-AI framework that analyzes user requests to identify relevant tools. It works with the ToolRegistry to maintain tool definitions and usage statistics, providing an intelligent way to discover and recommend tools based on user needs.

## Architecture

The ToolFinderAgent architecture consists of three main components:

1. **ToolRegistry**: Maintains tool definitions and usage statistics
2. **ToolFinderAgent**: Analyzes user requests to find relevant tools
3. **ToolManager**: Coordinates tool registration, discovery, and execution

### Process Flow

1. **User Request**: User sends a request to the system
2. **Orchestrator Analysis**: The Orchestrator receives the request
3. **Tool Finding**: The Orchestrator calls the ToolFinderAgent to identify relevant tools
4. **Model Response**: The Orchestrator sends the request to the AI model along with information about the relevant tools
5. **Action Planning**: The model generates a response that may include using the identified tools
6. **Execution**: If the model's response indicates tool usage, the tools are executed
7. **Response**: Results are returned to the user

## Components

### ToolRegistry

The ToolRegistry maintains a registry of available tools and their usage statistics. It provides methods for:

- Registering tools
- Retrieving tool definitions
- Tracking tool usage and effectiveness
- Recommending tools based on usage statistics

```python
# Example: Creating a ToolRegistry
tool_registry = ToolRegistry(logger=logger)

# Example: Registering a tool
tool_registry.register_tool("get_weather", weather_tool)

# Example: Getting tool usage statistics
stats = tool_registry.get_usage_stats("get_weather")
```

### ToolFinderAgent

The ToolFinderAgent is a specialized agent that analyzes user requests to identify relevant tools. It:

- Uses AI to understand the user's intent
- Recommends tools based on the request
- Updates usage statistics for recommended tools

```python
# Example: Creating a ToolFinderAgent
tool_finder_agent = ToolFinderAgent(
    ai_instance=ai,
    tool_registry=tool_registry,
    config_manager=config_manager,
    logger=logger
)

# Example: Processing a request
response = tool_finder_agent.process_request(request)
selected_tools = response.selected_tools
```

### ToolManager

The ToolManager coordinates tool operations, including:

- Tool registration
- Tool discovery using the ToolFinderAgent
- Tool execution
- Tool information retrieval

```python
# Example: Creating a ToolManager
tool_manager = ToolManager(
    logger=logger,
    config_manager=config_manager,
    tool_registry=tool_registry,
    tool_executor=tool_executor,
    agent_factory=agent_factory
)

# Example: Enabling agent-based tool finding
tool_manager.enable_agent_based_tool_finding(ai_instance=ai)

# Example: Finding tools for a prompt
tools = tool_manager.find_tools("What's the weather like in New York?")
```

## Integration with the Orchestrator

The Orchestrator uses the ToolFinderAgent to find relevant tools for user requests. It:

1. Calls the ToolFinderAgent to identify relevant tools
2. Uses the identified tools to create an action plan
3. Routes the request to appropriate specialized agents

```python
# Example: Setting up the Orchestrator with the ToolFinderAgent
orchestrator = Orchestrator(
    ai_instance=ai,
    tool_registry=tool_registry,
    config_manager=config_manager,
    logger=logger
)
orchestrator.set_tool_finder_agent(tool_finder_agent)
```

## Usage Example

See the `examples/tool_finder_agent_example.py` script for a complete example of using the ToolFinderAgent with the ToolRegistry.

## Benefits

1. **Intelligent Tool Selection**: The ToolFinderAgent uses AI to understand the user's intent and select appropriate tools
2. **Usage Statistics**: The ToolRegistry tracks tool usage and effectiveness, enabling better recommendations over time
3. **Consistent Interface**: The ToolFinderAgent follows the same agent interface as other agents
4. **Integration with Multi-Agent System**: The ToolFinderAgent can participate in the agent ecosystem

## Future Enhancements

1. **Learning from Tool Usage**: Enhance the ToolFinderAgent to learn from successful tool usage
2. **Tool Discovery**: Implement automatic discovery of new tools
3. **Tool Composition**: Enable the creation of composite tools from existing tools
4. **Tool Versioning**: Add support for tool versioning and compatibility checking

### Tool Execution Example

```python
from src.core.tool_enabled_ai import ToolEnabledAI
from src.tools.tool_manager import ToolManager
from src.tools.basic_tools import read_file # Example tool

# Initialize ToolManager and AI
tool_manager = ToolManager()
tool_manager.register_tool("read_file", read_file)
ai = ToolEnabledAI(tool_manager=tool_manager)

# Simulate AI deciding to call a tool
response_from_ai = {
    "content": "Okay, I will read the file.",
    "tool_calls": [{
        "id": "call_1",
        "type": "tool",
        "name": "read_file",
        "args": {"filepath": "README.md"}
    }]
}

# Process tool calls (this logic might be inside ToolEnabledAI.request)
if response_from_ai.get("tool_calls"):
    results = []
    for call in response_from_ai["tool_calls"]:
        tool_name = call.get("name")
        tool_args = call.get("args")
        try:
            # Use ToolManager to execute
            result = tool_manager.execute_tool(tool_name, tool_args)
            results.append({"tool_call_id": call["id"], "result": result})
        except Exception as e:
            results.append({"tool_call_id": call["id"], "error": str(e)})

    # Send results back to AI to get final response
    # final_response = ai.request_with_tool_results(original_prompt, response_from_ai, results)
    # print(final_response)
    print(f"Tool results: {results}")
```
