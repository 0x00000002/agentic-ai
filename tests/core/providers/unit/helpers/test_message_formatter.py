"""
Unit tests for MessageFormatter class.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.core.providers.message_formatter import MessageFormatter
from src.utils.logger import LoggerInterface


class TestMessageFormatter:
    """Test suite for the MessageFormatter class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture for a mock logger."""
        return MagicMock(spec=LoggerInterface)
    
    @pytest.fixture
    def default_formatter(self, mock_logger):
        """Fixture for a default MessageFormatter."""
        return MessageFormatter(logger=mock_logger)
    
    @pytest.fixture
    def custom_role_formatter(self, mock_logger):
        """Fixture for a MessageFormatter with custom role mapping."""
        custom_mapping = {
            "system": "system_instruction",
            "user": "human",
            "assistant": "bot",
            "tool": "function"
        }
        return MessageFormatter(role_mapping=custom_mapping, logger=mock_logger)
    
    def test_init_default(self, default_formatter, mock_logger):
        """Test initialization with default parameters."""
        assert default_formatter.logger is mock_logger
        assert default_formatter._role_map == {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool"
        }
    
    def test_init_custom_roles(self, custom_role_formatter, mock_logger):
        """Test initialization with custom role mapping."""
        assert custom_role_formatter.logger is mock_logger
        assert custom_role_formatter._role_map == {
            "system": "system_instruction",
            "user": "human",
            "assistant": "bot",
            "tool": "function"
        }
    
    def test_map_role_default(self, default_formatter):
        """Test role mapping with default formatter."""
        assert default_formatter.map_role("system") == "system"
        assert default_formatter.map_role("user") == "user"
        assert default_formatter.map_role("assistant") == "assistant"
        assert default_formatter.map_role("tool") == "tool"
        assert default_formatter.map_role("unknown") == "user"  # Default to user
    
    def test_map_role_custom(self, custom_role_formatter):
        """Test role mapping with custom formatter."""
        assert custom_role_formatter.map_role("system") == "system_instruction"
        assert custom_role_formatter.map_role("user") == "human"
        assert custom_role_formatter.map_role("assistant") == "bot"
        assert custom_role_formatter.map_role("tool") == "function"
        assert custom_role_formatter.map_role("unknown") == "user"  # Default to user
    
    def test_format_message_basic(self, default_formatter):
        """Test formatting a basic message."""
        message = {"role": "user", "content": "Hello!"}
        formatted = default_formatter.format_message(message)
        assert formatted == {"role": "user", "content": "Hello!"}
    
    def test_format_message_with_tool_fields(self, default_formatter):
        """Test formatting a message with tool-specific fields."""
        message = {
            "role": "tool",
            "name": "calculator",
            "content": "42"
        }
        formatted = default_formatter.format_message(message)
        assert formatted == {
            "role": "tool",
            "name": "calculator",
            "content": "42"
        }
    
    def test_format_message_with_tool_calls(self, default_formatter):
        """Test formatting a message with tool calls."""
        tool_calls = [{"id": "1", "function": {"name": "calculator", "arguments": '{"a": 1}'}}]
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        }
        formatted = default_formatter.format_message(message)
        assert formatted == {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        }
    
    def test_format_messages(self, default_formatter):
        """Test formatting a list of messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        formatted = default_formatter.format_messages(messages)
        assert len(formatted) == 4
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[2]["role"] == "assistant"
        assert formatted[3]["role"] == "user"
    
    def test_format_messages_with_system_prompt(self, default_formatter):
        """Test formatting messages with a system prompt."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        system_prompt = "You are a helpful assistant."
        formatted = default_formatter.format_messages(messages, system_prompt=system_prompt)
        assert len(formatted) == 4
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
    
    def test_format_messages_with_system_prompt_already_present(self, default_formatter):
        """Test formatting messages with a system prompt when one is already present."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        system_prompt = "You are an AI assistant."
        formatted = default_formatter.format_messages(messages, system_prompt=system_prompt)
        assert len(formatted) == 4  # No extra system message
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "You are a helpful assistant."  # Original is preserved
    
    def test_add_tool_message(self, default_formatter):
        """Test adding a tool message to a messages list."""
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "I'll use the calculator."}
        ]
        updated = default_formatter.add_tool_message(messages, "calculator", "4")
        assert len(updated) == 3
        assert updated[2]["role"] == "tool"
        assert updated[2]["name"] == "calculator"
        assert updated[2]["content"] == "4"
    
    def test_post_process_message(self, default_formatter):
        """Test post-processing a message."""
        message = {"role": "user", "content": "Hello!"}
        formatted = {"role": "user", "content": "Hello!"}
        result = default_formatter.post_process_message(formatted, message)
        assert result == formatted  # No changes by default 