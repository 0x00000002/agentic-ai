# Gradio Chat Interface

The Agentic-AI framework includes a user-friendly chat interface built with Gradio. This interface integrates with the multi-agent architecture to provide a seamless user experience.

## Features

- Text-based chat interface
- Audio input support (microphone)
- Integration with multiple agents
- Conversation history management
- System status messages

## Architecture

The UI is built around the `AgenticChatUI` class, which:

1. Initializes the necessary agents (Request Rooter, Listener)
2. Sets up the Gradio interface components
3. Handles message routing between the UI and the agent system
4. Processes both text and audio inputs

## Agent Integration

The chat interface works with the multi-agent architecture:

- **Request Rooter Agent**: Analyzes user text requests and routes them to appropriate agents
- **Listener Agent**: Processes audio input and converts it to text

## Example Usage

```python
from src.ui.gradio_chat import AgenticChatUI
from src.config.config_manager import ConfigManager
from src.agents import AgentFactory, AgentRegistry

# Initialize components
config_manager = ConfigManager()
registry = AgentRegistry()
agent_factory = AgentFactory(registry=registry)

# Register agent types
from src.agents.request_rooter import RequestRooterAgent
from src.agents.listener_agent import ListenerAgent
agent_factory.register_agent("request_rooter", RequestRooterAgent)
agent_factory.register_agent("listener", ListenerAgent)

# Create the UI
chat_ui = AgenticChatUI(
    config_manager=config_manager,
    agent_factory=agent_factory,
    enable_audio=True
)

# Launch the interface
chat_ui.launch(share=True)
```

## Running the UI

You can run the UI directly from the examples directory:

```bash
python examples/run_chat_ui.py
```

Command line options:

```
--share          Create a public share link
--no-audio       Disable audio input
--debug          Enable debug mode (shows thinking)
--port PORT      Port to run the server on (default: 7860)
--config CONFIG  Path to custom config file
```

## Customization

The UI can be customized in several ways:

1. **CSS Styling**: Modify the `_get_custom_css` method in the `AgenticChatUI` class
2. **Component Layout**: Customize the Gradio layout in the `build_ui` method
3. **Model Selection**: Choose the AI model for generation.
4. **Agent Configuration**: Update agent parameters in the `agents.yml` file
5. **API Keys**: Ensure necessary API keys are set as environment variables (e.g., `OPENAI_API_KEY`).

## Adding New Agent Types

To add support for new agent types in the UI:

1. Register the agent class with the registry:

   ```python
   agent_factory.register_agent("new_agent_type", NewAgentClass)
   ```

2. Update the request rooter agent to recognize and route to the new agent

3. If needed, add specific UI components to handle the agent's input/output requirements

## Future Enhancements

Planned enhancements for the UI include:

1. File upload/download support
2. Image/video display capabilities
3. Custom visualization components for specialized agents
4. Persistent conversation history
5. User authentication and profiles
