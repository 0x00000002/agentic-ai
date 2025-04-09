# tests/tools/unit/test_tool_registry.py
"""
Unit tests for the ToolRegistry class.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Import necessary components from src/tools
from src.tools.models import ToolDefinition
from src.tools.tool_registry import ToolRegistry
from src.exceptions import AIToolError

# Example Tool Functions (can be simple mocks or lambdas for these tests)
def mock_tool_func_1(location: str) -> str:
    return f"Weather in {location}: Sunny"

def mock_tool_func_2(query: str, count: int = 5) -> Dict[str, Any]:
    return {"results": [f"Result {i} for {query}" for i in range(count)]}

# Tool Definitions (Updated for lazy loading)
TOOL_DEF_1 = ToolDefinition(
    name="get_weather",
    description="Gets the weather for a location.",
    parameters_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    },
    module_path="tests.tools.unit.test_tool_registry", # Points to this test file
    function_name="mock_tool_func_1",
    function=None # Keep function None initially for testing loading
)

TOOL_DEF_2 = ToolDefinition(
    name="search_docs",
    description="Searches documents.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    module_path="tests.tools.unit.test_tool_registry", # Points to this test file
    function_name="mock_tool_func_2",
    function=None
)

class TestToolRegistry:
    """Test suite for ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Provides a clean ToolRegistry instance for each test."""
        # Patch config loading during ToolRegistry init to ensure it starts empty
        with patch.object(ToolRegistry, '_load_tools_from_config', return_value=None) as mock_load:
            reg = ToolRegistry()
            # Ensure the patch was effective (optional sanity check)
            # mock_load.assert_called_once()
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