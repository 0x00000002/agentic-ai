# tests/tools/unit/test_tool_registry.py
"""
Unit tests for the ToolRegistry class (focused on internal tools).
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Import necessary components from src/tools
from src.tools.models import ToolDefinition
from src.tools.tool_registry import ToolRegistry
from src.exceptions import AIToolError
from src.config import UnifiedConfig # Needed for mocking config

# Example Tool Functions (can be simple mocks or lambdas for these tests)
def mock_tool_func_1(location: str) -> str:
    return f"Weather in {location}: Sunny"

def mock_tool_func_2(query: str, count: int = 5) -> Dict[str, Any]:
    return {"results": [f"Result {i} for {query}" for i in range(count)]}

# Internal Tool Definitions (Updated for new model)
TOOL_DEF_INTERNAL_1 = ToolDefinition(
    name="get_weather",
    description="Gets the weather for a location.",
    parameters_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    },
    source="internal", # Added source
    speed="fast",      # Added metadata
    safety="native",   # Added metadata
    module="tests.tools.unit.test_tool_registry", # Updated field name
    function="mock_tool_func_1", # Updated field name
    category="weather"
)

TOOL_DEF_INTERNAL_2 = ToolDefinition(
    name="search_docs",
    description="Search documentation files.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    source="internal",
    module="src.tools.core.search", # Dummy
    function="search_docs_func", # Dummy
    category="file_management",
    speed="medium",
    safety="sandboxed"
)

# Example of an invalid definition (e.g., wrong source for this registry)
TOOL_DEF_MCP_INVALID = ToolDefinition(
    name="invalid_mcp_tool",
    description="An MCP tool definition.",
    parameters_schema={},
    source="mcp",
    mcp_server_name="some_server"
)

class TestToolRegistryInternal:
    """Test suite for ToolRegistry focusing on internal tools."""

    @pytest.fixture
    def mock_config(self):
        """Fixture to provide a mock UnifiedConfig."""
        config = MagicMock(spec=UnifiedConfig)
        # Mock the return value for get_tool_config to simulate loading from tools.yml
        config.get_tool_config.return_value = {
            "tools": [
                {
                    "name": TOOL_DEF_INTERNAL_1.name,
                    "description": TOOL_DEF_INTERNAL_1.description,
                    "parameters_schema": TOOL_DEF_INTERNAL_1.parameters_schema,
                    "module": TOOL_DEF_INTERNAL_1.module,
                    "function": TOOL_DEF_INTERNAL_1.function,
                    "category": TOOL_DEF_INTERNAL_1.category,
                    "speed": TOOL_DEF_INTERNAL_1.speed,
                    "safety": TOOL_DEF_INTERNAL_1.safety,
                    # source will be added by _load_internal_tools_from_config
                },
                 {
                    "name": TOOL_DEF_INTERNAL_2.name,
                    "description": TOOL_DEF_INTERNAL_2.description,
                    "parameters_schema": TOOL_DEF_INTERNAL_2.parameters_schema,
                    "module": TOOL_DEF_INTERNAL_2.module,
                    "function": TOOL_DEF_INTERNAL_2.function,
                    "speed": TOOL_DEF_INTERNAL_2.speed,
                    "safety": TOOL_DEF_INTERNAL_2.safety,
                 }
            ],
            "categories": {
                "weather": True, # Example category from config
                "search": {"enabled": True}
            }
        }
        return config

    @pytest.fixture
    def registry(self, mock_config) -> ToolRegistry:
        """Provides a ToolRegistry instance initialized with mocked config."""
        # Patch UnifiedConfig.get_instance to return our mock
        with patch('src.tools.tool_registry.UnifiedConfig.get_instance', return_value=mock_config):
            reg = ToolRegistry()
            return reg

    def test_load_internal_tools_from_config(self, registry: ToolRegistry):
        """Test that tools are loaded correctly from mocked config during init."""
        assert registry.has_tool(TOOL_DEF_INTERNAL_1.name)
        assert registry.has_tool(TOOL_DEF_INTERNAL_2.name)
        assert len(registry.get_tool_names()) == 2
        
        loaded_def1 = registry.get_internal_tool_definition(TOOL_DEF_INTERNAL_1.name)
        assert loaded_def1 is not None
        assert loaded_def1.source == "internal"
        assert loaded_def1.speed == TOOL_DEF_INTERNAL_1.speed
        assert loaded_def1.safety == TOOL_DEF_INTERNAL_1.safety
        assert loaded_def1.category == TOOL_DEF_INTERNAL_1.category
        assert loaded_def1.module == TOOL_DEF_INTERNAL_1.module
        
        # Check categories were loaded and tools assigned
        assert "weather" in registry.get_categories()
        assert TOOL_DEF_INTERNAL_1.name in registry.get_category_tools("weather")
        # Tool 2 didn't have category in mock config, should not be in weather
        assert TOOL_DEF_INTERNAL_2.name not in registry.get_category_tools("weather") 

    def test_register_internal_tool_success(self, mock_config):
        """Test successful registration of a valid internal tool."""
        # Create registry without auto-loading for this test
        with patch('src.tools.tool_registry.UnifiedConfig.get_instance', return_value=mock_config):
             with patch.object(ToolRegistry, '_load_internal_tools_from_config'):
                  registry = ToolRegistry()
        
        registry.register_internal_tool(tool_definition=TOOL_DEF_INTERNAL_1)
        assert registry.has_tool(TOOL_DEF_INTERNAL_1.name)
        assert registry.get_internal_tool_definition(TOOL_DEF_INTERNAL_1.name) == TOOL_DEF_INTERNAL_1
        assert TOOL_DEF_INTERNAL_1.name in registry.get_category_tools("weather")

    def test_register_duplicate_internal_tool_raises_error(self, registry: ToolRegistry):
        """Test registering a duplicate internal tool name raises AIToolError."""
        # Tools are loaded by fixture
        with pytest.raises(AIToolError, match=f"Internal tool '{TOOL_DEF_INTERNAL_1.name}' already registered"):
            registry.register_internal_tool(tool_definition=TOOL_DEF_INTERNAL_1)

    def test_register_mcp_tool_raises_error(self, registry: ToolRegistry):
        """Test registering a tool with source 'mcp' raises AIToolError."""
        with pytest.raises(AIToolError, match=f"Cannot register tool.*source 'mcp'"):
            registry.register_internal_tool(tool_definition=TOOL_DEF_MCP_INVALID)

    def test_get_internal_tool_definition_not_found(self, registry: ToolRegistry):
        """Test get_internal_tool_definition returns None for non-existent tool."""
        assert registry.get_internal_tool_definition("non_existent_tool") is None

    def test_list_internal_definitions(self, registry: ToolRegistry):
        """Test getting all registered internal tool definitions."""
        defs = registry.list_internal_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 2
        
        # Get the names of the tools in the definitions list
        tool_names = [tool_def.name for tool_def in defs]
        assert TOOL_DEF_INTERNAL_1.name in tool_names
        assert TOOL_DEF_INTERNAL_2.name in tool_names
        
        # Check that key attributes match
        for tool_def in defs:
            if tool_def.name == TOOL_DEF_INTERNAL_1.name:
                assert tool_def.description == TOOL_DEF_INTERNAL_1.description
                assert tool_def.module == TOOL_DEF_INTERNAL_1.module
                assert tool_def.function == TOOL_DEF_INTERNAL_1.function
            elif tool_def.name == TOOL_DEF_INTERNAL_2.name:
                assert tool_def.description == TOOL_DEF_INTERNAL_2.description
                assert tool_def.module == TOOL_DEF_INTERNAL_2.module
                assert tool_def.function == TOOL_DEF_INTERNAL_2.function

    def test_get_all_internal_definitions_dict(self, registry: ToolRegistry):
        """Test get_all_internal_tool_definitions returns a dict."""
        defs_dict = registry.get_all_internal_tool_definitions()
        assert isinstance(defs_dict, dict)
        assert len(defs_dict) == 2
        assert defs_dict[TOOL_DEF_INTERNAL_1.name] == TOOL_DEF_INTERNAL_1

    # --- Formatting Tests (only operate on internal tools) ---
    
    # Fixture to provide a registry with tools for formatting tests
    @pytest.fixture
    def populated_registry(self, mock_config) -> ToolRegistry:
         with patch('src.tools.tool_registry.UnifiedConfig.get_instance', return_value=mock_config):
             with patch.object(ToolRegistry, '_load_internal_tools_from_config'): # Prevent auto-load
                registry = ToolRegistry()
                registry.register_internal_tool(TOOL_DEF_INTERNAL_1)
                registry.register_internal_tool(TOOL_DEF_INTERNAL_2)
                return registry

    def test_format_for_openai(self, populated_registry: ToolRegistry):
        """Test formatting internal tools for the OpenAI provider."""
        formatted = populated_registry.format_tools_for_provider("OPENAI")
        assert len(formatted) == 2
        # Check the simpler format
        tool1_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_1.name)
        tool2_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_2.name)

        assert tool1_formatted == {
            "name": TOOL_DEF_INTERNAL_1.name,
            "description": TOOL_DEF_INTERNAL_1.description,
            "input_schema": TOOL_DEF_INTERNAL_1.parameters_schema
        }
        assert tool2_formatted == {
            "name": TOOL_DEF_INTERNAL_2.name,
            "description": TOOL_DEF_INTERNAL_2.description,
            "input_schema": TOOL_DEF_INTERNAL_2.parameters_schema
        }

    def test_format_for_anthropic(self, populated_registry: ToolRegistry):
        """Test formatting internal tools for the Anthropic provider."""
        formatted = populated_registry.format_tools_for_provider("Anthropic") # Case-insensitive check
        assert len(formatted) == 2
        tool1_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_1.name)
        tool2_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_2.name)
        # Expect the same basic format as OpenAI
        assert tool1_formatted == {
            "name": TOOL_DEF_INTERNAL_1.name,
            "description": TOOL_DEF_INTERNAL_1.description,
            "input_schema": TOOL_DEF_INTERNAL_1.parameters_schema
        }
        assert tool2_formatted == {
            "name": TOOL_DEF_INTERNAL_2.name,
            "description": TOOL_DEF_INTERNAL_2.description,
            "input_schema": TOOL_DEF_INTERNAL_2.parameters_schema
        }
        
    def test_format_for_gemini(self, populated_registry: ToolRegistry):
        """Test formatting internal tools for the Gemini provider."""
        formatted = populated_registry.format_tools_for_provider("Gemini")
        assert len(formatted) == 2
        tool1_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_1.name)
        tool2_formatted = next(t for t in formatted if t['name'] == TOOL_DEF_INTERNAL_2.name)
        
        # Gemini expects properties and required fields within parameters
        assert tool1_formatted == {
            'name': TOOL_DEF_INTERNAL_1.name,
            'description': TOOL_DEF_INTERNAL_1.description,
            'parameters': {
                'type': 'object',
                'properties': TOOL_DEF_INTERNAL_1.parameters_schema.get('properties', {}),
                'required': TOOL_DEF_INTERNAL_1.parameters_schema.get('required', [])
            }
        }
        
        # TOOL_DEF_INTERNAL_2 has parameters for query and count
        assert tool2_formatted == {
            'name': TOOL_DEF_INTERNAL_2.name,
            'description': TOOL_DEF_INTERNAL_2.description,
            'parameters': {
                'type': 'object',
                'properties': TOOL_DEF_INTERNAL_2.parameters_schema.get('properties', {}),
                'required': TOOL_DEF_INTERNAL_2.parameters_schema.get('required', [])
            }
        }

    def test_format_unknown_provider(self, populated_registry: ToolRegistry):
        """Test formatting for an unknown provider returns default format."""
        mock_logger = MagicMock()
        populated_registry._logger = mock_logger
        formatted = populated_registry.format_tools_for_provider("UNKNOWN_PROVIDER")
        assert len(formatted) == 2 # Should format both tools with default
        # Check default format (matches Anthropic/base)
        tool1_formatted = next(t for t in formatted if t['name'] == 'get_weather')
        assert tool1_formatted['description'] == TOOL_DEF_INTERNAL_1.description
        assert tool1_formatted['input_schema'] == TOOL_DEF_INTERNAL_1.parameters_schema
        # Check warning log (Warning is no longer logged for unknown providers)
        mock_logger.warning.assert_not_called()

    def test_format_specific_tool_subset(self, populated_registry: ToolRegistry):
        """Test formatting only a specific subset of internal tools."""
        formatted = populated_registry.format_tools_for_provider("OPENAI", tool_names={TOOL_DEF_INTERNAL_1.name})
        assert len(formatted) == 1
        # Check the simpler format
        assert formatted[0]['name'] == TOOL_DEF_INTERNAL_1.name
        assert formatted[0]['description'] == TOOL_DEF_INTERNAL_1.description
        assert formatted[0]['input_schema'] == TOOL_DEF_INTERNAL_1.parameters_schema
        
    def test_format_empty_registry(self):
         """Test formatting an empty registry returns empty list."""
         with patch('src.tools.tool_registry.UnifiedConfig.get_instance', return_value=MagicMock()):
             with patch.object(ToolRegistry, '_load_internal_tools_from_config'):
                  registry = ToolRegistry()
         formatted = registry.format_tools_for_provider("OPENAI")
         assert formatted == [] 

    def test_format_non_existent_tool(self, populated_registry: ToolRegistry, mock_logger):
        """Test formatting when a requested tool name doesn't exist."""
        # ... existing code ...