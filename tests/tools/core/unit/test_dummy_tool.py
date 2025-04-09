"""Unit tests for the dummy tool."""
import pytest
from src.tools.core.dummy_tool import dummy_tool_function

class TestDummyTool:
    """Tests for the dummy_tool_function."""

    def test_dummy_tool_basic(self):
        """Test the basic functionality of the dummy tool."""
        query = "test input"
        expected_output = f"Dummy tool processed query: {query}"
        result = dummy_tool_function(query)
        assert isinstance(result, str)
        assert result == expected_output

    def test_dummy_tool_empty_input(self):
        """Test the dummy tool with empty input."""
        query = ""
        expected_output = f"Dummy tool processed query: {query}"
        result = dummy_tool_function(query)
        assert result == expected_output 