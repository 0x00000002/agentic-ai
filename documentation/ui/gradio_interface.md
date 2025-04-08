# Gradio Chat Interface

The Agentic-AI framework includes a user-friendly chat interface built with Gradio (`src/ui/simple_chat.py`). This interface integrates with the agent system, primarily the `Coordinator`, to provide a seamless user experience.

## Features

- Text-based chat interface
- Audio input support (microphone) with language selection
- Integration with the `Coordinator` agent for request processing
- Conversation history display
- Audio transcription status updates

## Architecture

The UI is built around the `SimpleChatUI` class, which:

1. Takes an initialized `Coordinator` agent instance.
2. Sets up the Gradio interface components (chatbot, text input, audio input, buttons).
3. Handles message routing between the UI and the `Coordinator` agent.
4. Processes both text input (`process_message`) and audio input (`process_audio`).

## Agent Integration

The chat interface primarily interacts with the agent system via the `Coordinator`:

- **`Coordinator` Agent**: Receives text prompts or audio transcription requests from the UI and routes them appropriately (e.g., to a default chat agent, the `ListenerAgent` for audio, etc.).

## Example Usage (Conceptual)

While the UI can be run directly, here's a conceptual breakdown of its initialization:

```python
from src.ui.simple_chat import SimpleChatUI
from src.agents.coordinator import Coordinator
from src.config import configure, UseCasePreset
from src.agents.agent_factory import AgentFactory # Needed by Coordinator
from src.agents.agent_registry import AgentRegistry # Needed by Factory

# 1. Configure the framework (optional, defaults exist)
configure(
    model="claude-3-haiku", # Example model
    use_case=UseCasePreset.CHAT
)

# 2. Initialize dependencies for Coordinator
# (These are often created internally by Coordinator if not provided)
registry = AgentRegistry()
# Register necessary agents (like ListenerAgent, ChatAgent) in the registry...
# Example: registry.register("listener_agent", ListenerAgent)
# Example: registry.register("chat_agent", ChatAgent)
agent_factory = AgentFactory(registry=registry)

# 3. Create the Coordinator instance
# It will use the globally configured settings and its dependencies
coordinator = Coordinator(agent_factory=agent_factory)

# 4. Create the UI with the coordinator
chat_ui = SimpleChatUI(coordinator=coordinator)

# 5. Launch the interface
chat_ui.launch(share=True) # share=True creates a public link
```

## Running the UI

The simplest way to run the UI is often via the main execution block within `simple_chat.py` itself, or a dedicated run script if provided.

If run directly via `python src/ui/simple_chat.py`, the `run_simple_chat()` function within the file sets up a default configuration (e.g., using `claude-3-haiku` model and `CHAT` use case) and launches the interface.

Check the `if __name__ == "__main__":` block in `src/ui/simple_chat.py` for potential command-line argument handling (though none seem implemented currently).

## Customization

The UI can be customized in several ways:

1. **CSS Styling**: Gradio interfaces support custom CSS. You might add CSS styling within the `build_interface` method.
2. **Component Layout**: Customize the Gradio layout in the `build_interface` method within `SimpleChatUI`.
3. **Model Selection/Configuration**: Modify the model, use case, etc., by calling `src.config.configure()` before initializing the UI or Coordinator, or by using environment variables/config files if supported by the configuration system.
4. **Agent Configuration**: Update default agents used by the `Coordinator` in the `agents.yml` configuration file.
5. **API Keys**: Ensure necessary API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are set as environment variables.

## Adding New Agent Types

Integrating new agents accessible via the UI typically involves:

1.  Implementing the new agent class.
2.  Registering the new agent with the `AgentRegistry`.
3.  Potentially modifying the `Coordinator` or `RequestAnalyzer` logic if the new agent needs specific routing beyond the default handling.
4.  If the agent requires unique UI elements, modifying the `build_interface` method in `SimpleChatUI`.

## Future Enhancements

Planned enhancements for the UI include:

1. File upload/download support
2. Image/video display capabilities
3. Custom visualization components for specialized agents
4. Persistent conversation history
5. User authentication and profiles
