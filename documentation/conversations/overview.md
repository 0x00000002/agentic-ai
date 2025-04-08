# Conversation Management

## Overview

The conversation management system in Agentic-AI handles:

- Maintaining conversation history
- Formatting messages for different providers
- Extracting "thoughts" and other metadata from responses
- Managing context and tool calls

## ConversationManager

The `ConversationManager` class tracks the conversation state:

```python
from src.conversation.conversation_manager import ConversationManager

# Create a conversation manager
conversation = ConversationManager()

# Add messages
conversation.add_message(role="user", content="Hello, how are you?")
conversation.add_message(role="assistant", content="I'm doing well! How can I help you today?")

# Get all messages
messages = conversation.get_messages()

# Get the latest message
last_message = conversation.get_last_message()

# Clear the conversation
conversation.clear()
```

## Working with Thoughts

The system can extract AI "thoughts" from responses to debug reasoning:

```python
# Enable thought extraction in the conversation manager
conversation.add_message(
    role="assistant",
    content="<thinking>Let me consider the best approach here...</thinking>The answer is 42.",
    extract_thoughts=True
)

# Get the processed message (without thoughts)
processed_message = conversation.get_last_message()
# Result: {"role": "assistant", "content": "The answer is 42."}

# Access the thoughts if needed
thoughts = conversation.get_thoughts()
# Result: ["Let me consider the best approach here..."]
```

## Handling Tool Calls

Conversation manager also tracks tool calls and their results:

```python
# Add a message with tool calls
conversation.add_message(
    role="assistant",
    content="I'll check the weather for you.",
    tool_calls=[
        {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
    ]
)

# Add the tool response
conversation.add_message(
    role="tool",
    name="get_weather",
    content="It's sunny and 75°F in New York."
)
```

## Response Parser

The `ResponseParser` processes raw AI responses:

```python
from src.conversation.response_parser import ResponseParser

parser = ResponseParser()

# Parse a response with thoughts
result = parser.parse_response(
    "<thinking>I should search for this information.</thinking>The capital of France is Paris.",
    extract_thoughts=True,
    show_thinking=False  # Hide thoughts in final output
)

# Result: {
#   "content": "The capital of France is Paris.",
#   "thoughts": "I should search for this information."
# }
```

## Sequence Diagram

```
User     AI      ConversationManager    ResponseParser    Provider
 |       |              |                     |               |
 | Request               |                     |               |
 |------>|              |                     |               |
 |       | Add user message                   |               |
 |       |------------->|                     |               |
 |       |              |                     |               |
 |       | Get messages  |                     |               |
 |       |<-------------|                     |               |
 |       |              |                     |               |
 |       | Send to provider                                   |
 |       |-------------------------------------------------->|
 |       |              |                     |               |
 |       | Raw response                                       |
 |       |<--------------------------------------------------|
 |       |              |                     |               |
 |       | Parse response                     |               |
 |       |----------------------------->|     |               |
 |       |              |                     |               |
 |       | Parsed response                    |               |
 |       |<-----------------------------|     |               |
 |       |              |                     |               |
 |       | Add assistant message              |               |
 |       |------------->|                     |               |
 |       |              |                     |               |
 | Response             |                     |               |
 |<------|              |                     |               |
```
