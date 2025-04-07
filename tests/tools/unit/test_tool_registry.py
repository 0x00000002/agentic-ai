# tests/tools/unit/test_tool_registry.py
"""
Unit tests for the ToolRegistry class.
"""

import pytest
from unittest.mock import MagicMock, patch

# Import necessary components from src/tools
from src.tools.models import ToolDefinition
from src.tools.tool_registry import ToolRegistry
from src.exceptions import AIToolError

# Example Tool Definitions (can be moved to fixtures)
def sample_tool_func(location: str, unit: str = "celsius") -> str:
    """Gets the weather."""
    return f"Weather in {location} is nice, {unit}"

def sample_stock_func(symbol: str) -> str:
    """Gets stock price."""
    return f"Price for {symbol} is $100"

# Define parameter schemas directly as dicts
WEATHER_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Unit for temperature"}
    },
    "required": ["location"]
}

STOCK_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "The stock ticker symbol"},
    },
    "required": ["symbol"]
}

# Update ToolDefinition instantiation
TOOL_DEF_1 = ToolDefinition(
    name="get_weather",
    description="Get the current weather in a given location",
    function=sample_tool_func,
    parameters_schema=WEATHER_PARAMS_SCHEMA
)

TOOL_DEF_2 = ToolDefinition(
    name="get_stock_price",
    description="Get the current stock price for a symbol",
    function=sample_stock_func, 
    parameters_schema=STOCK_PARAMS_SCHEMA
)

class TestToolRegistry:
    """Test suite for ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Provides a clean ToolRegistry instance for each test."""
        # Prevent built-in tool registration for cleaner tests
        with patch.object(ToolRegistry, '_register_builtin_tools', return_value=None):
             reg = ToolRegistry()
             yield reg

    def test_register_tool_success(self, registry: ToolRegistry):
        """Test successful registration of a tool."""
        # Use keyword arguments for clarity and correctness
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        assert TOOL_DEF_1.name in registry._tools_metadata # Check internal dict
        # Use get_tool, not get_tool_definition
        assert registry.get_tool(TOOL_DEF_1.name) == TOOL_DEF_1

    def test_register_duplicate_tool_raises_error(self, registry: ToolRegistry):
        """Test registering a tool with a duplicate name raises AIToolError."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        with pytest.raises(AIToolError, match=f"TOOL: Tool '{TOOL_DEF_1.name}' already registered"):
            registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)

    def test_register_multiple_tools(self, registry: ToolRegistry):
        """Test registering multiple different tools."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        assert TOOL_DEF_1.name in registry._tools_metadata
        assert TOOL_DEF_2.name in registry._tools_metadata
        assert len(registry._tools_metadata) == 2

    def test_get_tool_definition_not_found(self, registry: ToolRegistry):
        """Test get_tool returns None for non-existent tool."""
        # Use get_tool
        assert registry.get_tool("non_existent_tool") is None

    def test_get_all_definitions(self, registry: ToolRegistry):
        """Test getting all registered tool definitions."""
        # Use get_all_tool_definitions
        assert registry.get_all_tool_definitions() == {}
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        assert registry.get_all_tool_definitions() == {
            TOOL_DEF_1.name: TOOL_DEF_1,
            TOOL_DEF_2.name: TOOL_DEF_2
        }

    # --- Formatting Tests ---
    def test_format_for_openai(self, registry: ToolRegistry):
        """Test formatting tools for the OpenAI provider."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        
        formatted = registry.format_tools_for_provider("OPENAI")
        assert len(formatted) == 2
        
        # Check structure of the first tool
        tool1_formatted = next(t for t in formatted if t['function']['name'] == 'get_weather')
        assert tool1_formatted['type'] == 'function'
        assert tool1_formatted['function']['description'] == TOOL_DEF_1.description
        assert tool1_formatted['function']['parameters'] == TOOL_DEF_1.parameters_schema # Direct access

    def test_format_for_anthropic(self, registry: ToolRegistry):
        """Test formatting tools for the Anthropic provider."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        
        formatted = registry.format_tools_for_provider("ANTHROPIC")
        assert len(formatted) == 2
        
        tool1_formatted = next(t for t in formatted if t['name'] == 'get_weather')
        assert tool1_formatted['description'] == TOOL_DEF_1.description
        assert tool1_formatted['input_schema'] == TOOL_DEF_1.parameters_schema # Direct access

    def test_format_for_gemini(self, registry: ToolRegistry):
        """Test formatting tools for the Gemini provider."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        
        formatted = registry.format_tools_for_provider("GEMINI")
        declarations = formatted # It returns a list of dicts
        assert len(declarations) == 2
        
        # Access name via function_declaration key
        tool1_decl_outer = next(d for d in declarations if d['function_declaration']['name'] == 'get_weather')
        tool1_decl = tool1_decl_outer['function_declaration'] # Get inner dict
        assert tool1_decl['description'] == TOOL_DEF_1.description
        assert tool1_decl['parameters'] == TOOL_DEF_1.parameters_schema # Direct access
        
    def test_format_unknown_provider(self, registry: ToolRegistry):
        """Test formatting for an unknown provider returns default format and logs warning."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        mock_logger = MagicMock()
        registry._logger = mock_logger # Inject mock logger
        
        formatted = registry.format_tools_for_provider("UNKNOWN_PROVIDER")
        # Assert default format is returned, not empty list
        assert len(formatted) == 1 
        assert formatted[0]['name'] == TOOL_DEF_1.name # Check default format keys
        assert formatted[0]['description'] == TOOL_DEF_1.description
        assert formatted[0]['parameters'] == TOOL_DEF_1.parameters_schema
        # Check warning log
        mock_logger.warning.assert_called_once()
        assert "Using default tool format for provider: UNKNOWN_PROVIDER" in mock_logger.warning.call_args[0][0]

    def test_format_specific_tool_subset(self, registry: ToolRegistry):
        """Test formatting only a specific subset of registered tools."""
        registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
        registry.register_tool(tool_name=TOOL_DEF_2.name, tool_definition=TOOL_DEF_2)
        
        # Format only 'get_weather' for OpenAI
        formatted = registry.format_tools_for_provider("OPENAI", tool_names={TOOL_DEF_1.name})
        
        assert len(formatted) == 1
        assert formatted[0]['function']['name'] == TOOL_DEF_1.name
        
    def test_format_empty_registry(self, registry: ToolRegistry):
         """Test formatting an empty registry returns empty list."""
         formatted = registry.format_tools_for_provider("OPENAI")
         assert formatted == []
         
    def test_format_empty_subset(self, registry: ToolRegistry):
         """Test formatting with an empty subset returns empty list."""
         registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
         formatted = registry.format_tools_for_provider("OPENAI", tool_names=set())
         assert formatted == []
         
    def test_format_non_existent_subset(self, registry: ToolRegistry):
         """Test formatting with a subset containing non-existent tools."""
         registry.register_tool(tool_name=TOOL_DEF_1.name, tool_definition=TOOL_DEF_1)
         formatted = registry.format_tools_for_provider("OPENAI", tool_names={"non_existent", TOOL_DEF_1.name})
         assert len(formatted) == 1 # Should only format the existing tool
         assert formatted[0]['function']['name'] == TOOL_DEF_1.name 