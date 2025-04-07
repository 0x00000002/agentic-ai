# Agentic-AI Architecture

This document describes the architecture of the Agentic-AI framework, focusing on the core design principles and component relationships.

## Overview

Agentic-AI is a modular framework for building AI applications with tool integration capabilities. The architecture follows solid software engineering principles including:

- **Separation of Concerns**: Each component has a clear, focused responsibility
- **Dependency Injection**: Components receive their dependencies rather than creating them
- **Interface-Based Design**: Components interact through well-defined interfaces
- **Error Handling**: Standardized error handling across the framework
- **Configuration Management**: Modular, externalized configuration

## Core Components

### AI Core

The AI core provides the foundation for interacting with language models:

- **AIBase**: Base implementation of the AI interface.
- **ToolEnabledAI**: Extended implementation of `AIBase` with tool integration capabilities.
- **Providers**: Abstraction layer for different AI providers (`BaseProvider`, `OpenAIProvider`, etc.).
- **ProviderFactory**: Creates provider instances based on configuration.
- **Interfaces**: Clear contracts for component interactions (e.g., `AIInterface`, `ProviderInterface`).

### Multi-Agent System

The multi-agent system enables specialized processing of user requests:

- **Coordinator**: Coordinates the workflow between specialized agents (current implementation).
- **RequestAnalyzer**: Analyzes requests to determine appropriate agents and tools.
- **ResponseAggregator**: Combines responses from multiple agents.
- **BaseAgent**: Common functionality for all agents.

### Tool Subsystem

Handles tool definition, management, and execution:

- **ToolManager**: Coordinates tool registration, discovery (via registry), and execution.
- **ToolRegistry**: Manages tool definitions and provider-specific formatting.
- **ToolExecutor**: Executes tool functions with timeout and retries.
- **Models**: Pydantic models for `ToolDefinition`, `ToolCall`, `ToolResult`.

### Configuration Management

Configuration is modularized for better maintainability:

- **UnifiedConfig**: Central singleton class for accessing merged configuration data.
- **Modular Config Files**: Separate files for models, providers, agents, tools, etc.

### Error Handling

The error handling system provides consistent error management:

- **ErrorHandler**: Centralized error handling with standardized responses
- **Exception Hierarchy**: Well-defined exception types for different error scenarios

## Dependency Management

Dependencies between components are primarily managed through:

- **Dependency Injection**: Components like `Coordinator` and `ToolEnabledAI` receive dependencies (e.g., `ToolManager`, `ProviderInterface`) via their constructors.
- **Factories**: Specific factories like `ProviderFactory` are used to create instances of components (e.g., providers) based on configuration.
- **Singleton Access**: Core configuration (`UnifiedConfig`) and logging (`LoggerFactory`) often use singleton patterns for easy access.

This approach promotes loose coupling and testability.

## Component Relationships

(Diagram needs update to reflect current structure, including ToolManager, ToolEnabledAI, Coordinator, ProviderFactory, UnifiedConfig)

```
┌──────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                  │     │                   │     │                   │
│  ConfigFactory   │◄────┤   AppContainer    │────►│   ErrorHandler    │
│                  │     │                   │     │                   │
└───────┬──────────┘     └───────┬───────────┘     └───────────────────┘
        │                        │
        │                        │
┌───────▼──────────┐     ┌───────▼───────────┐
│                  │     │                   │
│  ProviderFactory │     │   AIBase / AI     │
│                  │     │                   │
└───────┬──────────┘     └───────┬───────────┘
        │                        │
        │                        │
┌───────▼──────────┐     ┌───────▼───────────┐     ┌───────────────────┐
│                  │     │                   │     │                   │
│    Providers     │     │   Orchestrator    │────►│  RequestAnalyzer  │
│                  │     │                   │     │                   │
└──────────────────┘     └───────┬───────────┘     └───────────────────┘
                                 │
                                 │
                         ┌───────▼───────────┐     ┌───────────────────┐
                         │                   │     │                   │
                         │  Specialized      │────►│ ResponseAggregator│
                         │  Agents           │     │                   │
                         └───────────────────┘     └───────────────────┘
```

## Key Architectural Improvements

### 1. Refactored Coordinator / Orchestration

The orchestration logic, now primarily in the `Coordinator` agent, is supported by focused components:

- **Coordinator**: Coordinates the overall workflow (delegating or handling directly).
- **RequestAnalyzer**: Handles request intent classification.
- **ResponseAggregator**: (If used) Handles response aggregation.

This separation improves maintainability.

### 2. Modularized Configuration

Configuration remains separated into domain-specific files (models, providers, agents, tools, etc.) managed by the central `UnifiedConfig`.

### 3. Standardized Error Handling

A consistent error handling approach using a hierarchy based on `AIFrameworkError` and specific exceptions like `AIProviderError`, `AIAuthenticationError`, `AIToolError`, etc., improves error visibility and debugging.

### 4. Standardized Provider Interface

The `BaseProvider` class enforces a standard structure (`_prepare_request_payload`, `_make_api_request`, `_convert_response`) for all provider implementations (OpenAI, Anthropic, Gemini, Ollama), simplifying integration and interaction logic in core AI classes. Providers now return a standardized `ProviderResponse` object.

### 5. Simplified Tool Subsystem

The tool system has been streamlined:

- `ToolManager` coordinates execution via `ToolExecutor` and registration/formatting via `ToolRegistry`.
- Redundant components (`AIToolFinder`, `tool_prompt_builder.py`, `tool_call.py`) were removed.
- `ToolEnabledAI` handles the tool-calling loop using the standardized provider responses and `ToolManager`.

### 6. Dependency Injection and Factories

Explicit dependency injection and factories (like `ProviderFactory`) are used instead of a single container, promoting clearer dependency flows.

## Usage Example

```python
from src.core import ToolEnabledAI
from src.agents import Coordinator
from src.config import configure
# from your_tools_module import get_weather # Example tool function

# Configure if needed (optional, uses defaults otherwise)
# configure(model="gpt-4o")

# Create a ToolEnabledAI instance
ai = ToolEnabledAI(model="gpt-4o") # Or any other configured model

# Tools are typically registered via ToolManager or configuration,
# but if manual registration on ToolEnabledAI's manager is needed:
# tool_def = ToolDefinition(...) # Define your tool
# ai._tool_manager.register_tool("get_weather", tool_def)

# Make a request that might use tools
# Use process_prompt for automatic tool handling
response_content = ai.process_prompt("What's the weather like in Paris today?")
print(response_content)

# Create a Coordinator instance (uses default dependencies)
coordinator = Coordinator()

# Process a request with the coordinator
response = coordinator.process_request({
    "prompt": "Translate this text to French and analyze the sentiment",
    "context": {"source_language": "en"}
})

# Print the final content from the coordinator's response
print(response.get("content"))
```
