# tests/tools/unit/test_tool_manager.py
"""
Unit tests for the ToolManager class.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Import necessary components
from src.tools.models import ToolDefinition, ToolCall, ToolResult
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry # Dependency
from src.tools.tool_executor import ToolExecutor # Dependency
from src.exceptions import AIToolError

# Update ToolDefinition instantiation
WEATHER_PARAMS_SCHEMA = {"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]}
STOCK_PARAMS_SCHEMA = {"type": "object", "properties": {"sym": {"type": "string"}}, "required": ["sym"]}

TOOL_DEF_WEATHER = ToolDefinition(
    name="get_weather",
    description="Get weather.",
    function=MagicMock(), # Use a mock for the function itself in manager tests
    parameters_schema=WEATHER_PARAMS_SCHEMA
)
TOOL_DEF_STOCK = ToolDefinition(
    name="get_stock",
    description="Get stock price.",
    function=MagicMock(),
    parameters_schema=STOCK_PARAMS_SCHEMA
)

class TestToolManager:
    """Test suite for ToolManager."""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Provides a mock ToolRegistry."""
        registry = MagicMock(spec=ToolRegistry)
        registry.get_all_tool_definitions.return_value = {}
        return registry

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Provides a mock ToolExecutor."""
        executor = MagicMock(spec=ToolExecutor)
        # Configure execute_tool if needed for specific tests
        # executor.execute_tool = AsyncMock() 
        # ToolManager calls the sync execute method now
        executor.execute = MagicMock()
        return executor

    @pytest.fixture
    def manager_unpatched(self) -> ToolManager:
        """Provides a basic ToolManager instance without patching internal deps."""
        # Patch dependencies during ToolManager init to prevent side effects
        with patch('src.tools.tool_manager.ToolRegistry') as MockReg, \
             patch('src.tools.tool_manager.ToolExecutor') as MockExec, \
             patch('src.tools.tool_manager.UnifiedConfig.get_instance') as MockConfig:
            # Ensure internal instances are mocks
            MockConfig.return_value.get_tool_config.return_value = {} # Default empty tool config
            return ToolManager()

    # Use patching within tests that need to control/assert mock interactions

    # --- Registration Tests ---
    def test_register_tool_delegates_to_registry(self, manager_unpatched: ToolManager):
        """Test that register_tool calls registry.register_tool."""
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.register_tool(tool_name=TOOL_DEF_WEATHER.name, tool_definition=TOOL_DEF_WEATHER)
        manager_unpatched.tool_registry.register_tool.assert_called_once_with(TOOL_DEF_WEATHER.name, TOOL_DEF_WEATHER)
        
    def test_register_tool_delegates_to_registry_correct_args(self, manager_unpatched: ToolManager):
        """Test that register_tool passes correct args to registry."""
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.register_tool(tool_name="get_weather", tool_definition=TOOL_DEF_WEATHER)
        manager_unpatched.tool_registry.register_tool.assert_called_once_with(
            "get_weather", TOOL_DEF_WEATHER
        )

    # --- Execution Tests (REMOVE - Functionality tested doesn't exist) ---
    # @pytest.mark.asyncio
    # async def test_execute_tool_call_delegates_to_executor(...):
    #     pass

    # @pytest.mark.asyncio
    # async def test_execute_multiple_tool_calls(...):
    #     pass

    # --- Formatting/Retrieval Tests (Functionality belongs to Registry) ---
    # REMOVED
        
    # def test_get_tool_definition_delegates_to_registry(...):
    #     pass
        
    # def test_get_all_tool_definitions_delegates_to_registry(...):
    #     pass 