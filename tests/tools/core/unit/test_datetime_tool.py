"""Unit tests for the datetime tool."""
import pytest
import re
from src.tools.core.datetime_tool import get_datetime

class TestDatetimeTool:
    """Tests for the get_datetime function."""

    def test_get_datetime_returns_string(self):
        """Verify that get_datetime returns a string."""
        result = get_datetime()
        assert isinstance(result, str)

    def test_get_datetime_format_basic(self):
        """Verify the basic ISO 8601 format using regex."""
        result = get_datetime()
        # Basic check for YYYY-MM-DDTHH:MM:SS.ffffff structure
        # This isn't a perfect ISO 8601 validator but checks the essential parts
        iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+$"
        assert re.match(iso_pattern, result) is not None 