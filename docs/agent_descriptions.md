# Agent Descriptions

This document explains how agent descriptions are managed in the Agentic-AI framework.

## Overview

Agent descriptions are metadata that describe the capabilities and purpose of each agent type in the system. These descriptions are used by the Orchestrator to make intelligent decisions about which agents should handle user requests.

## Configuration

Agent descriptions are stored in the `agents.yml` file under the `agent_descriptions` section:

```yaml
# Agent descriptions for AI prompting and routing
agent_descriptions:
  listener: "Handles audio processing and speech recognition"
  translator: "Translates between languages"
  website_parser: "Searches websites for information"
  content_generator: "Creates multimedia content (images, videos, audio)"
  action_planner: "Breaks complex tasks into subtasks"
  mcp_searcher: "Finds relevant Model-Centric Processes"
  paralleliser: "Executes tasks in parallel"
  tool_finder: "Identifies relevant tools for user requests"
  orchestrator: "Routes requests to appropriate specialized agents"
```

## Accessing Agent Descriptions

The `ConfigManager` provides methods to access agent descriptions:

```python
# Get description for a specific agent
description = config_manager.get_agent_description("tool_finder")
print(description)  # "Identifies relevant tools for user requests"

# Get all agent descriptions
all_descriptions = config_manager.get_all_agent_descriptions()
for agent_type, description in all_descriptions.items():
    print(f"{agent_type}: {description}")
```

## Usage in the Orchestrator

The Orchestrator uses agent descriptions to create prompts for the AI model when analyzing user requests:

```python
# Get agent descriptions from the config manager
agent_descriptions = config_manager.get_all_agent_descriptions()

# Format agent list for prompt
agent_list = ""
for agent in available_agents:
    description = agent_descriptions.get(agent, f"Agent type: {agent}")
    agent_list += f"- {agent}: {description}\n"

# Create prompt for AI
prompt = f"""Analyze this user request and determine which specialized agents should handle it:
Request: {user_request}

Available agents:
{agent_list}

Return a JSON list of [agent_name, confidence] pairs, where confidence is 0.0-1.0.
Only include agents with confidence > {confidence_threshold}. If no agents are appropriate, return [].
"""
```

## Benefits of Configuration-Based Descriptions

1. **Separation of Concerns**: Agent descriptions are configuration data, not implementation details.
2. **Easier Maintenance**: Descriptions can be updated without changing code.
3. **Consistency**: All components access the same descriptions from a single source.
4. **Flexibility**: Descriptions can be customized for different deployments.

## Example

See the `examples/agent_descriptions_example.py` script for a complete demonstration of how to use agent descriptions from the configuration system.
