# tests/tools/unit/test_tool_manager.py
"""
Unit tests for the ToolManager class.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Import necessary components
from src.tools.models import ToolDefinition, ToolCall, ToolResult, ToolExecutionStatus
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry # Dependency
from src.tools.tool_executor import ToolExecutor # Dependency
from src.exceptions import AIToolError, ErrorHandler
from src.utils.logger import LoggerInterface

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

    # --- New Tests for Find Tools Method ---
    def test_find_tools_delegates_to_registry(self, manager_unpatched: ToolManager):
        """Test that find_tools delegates to the registry's get_recommended_tools method."""
        # Setup
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_recommended_tools.return_value = ["get_weather", "get_stock"]
        manager_unpatched.tool_config = {"max_recommendations": 5}
        
        # Execute
        result = manager_unpatched.find_tools("What's the weather like?")
        
        # Assert
        manager_unpatched.tool_registry.get_recommended_tools.assert_called_once_with(
            "What's the weather like?", max_tools=5
        )
        assert result == ["get_weather", "get_stock"]
        
    def test_find_tools_handles_exceptions(self, manager_unpatched: ToolManager):
        """Test that find_tools handles exceptions from the registry."""
        # Setup
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_recommended_tools.side_effect = Exception("Registry error")
        manager_unpatched.logger = MagicMock(spec=LoggerInterface)
        
        # Execute
        result = manager_unpatched.find_tools("What's the weather like?")
        
        # Assert
        assert result == []
        manager_unpatched.logger.error.assert_called_once()

    # --- New Tests for Execute Tool Method ---
    def test_execute_tool_calls_executor(self, manager_unpatched: ToolManager):
        """Test that execute_tool calls the executor with correct parameters."""
        # Setup
        tool_def = TOOL_DEF_WEATHER
        mock_result = ToolResult(
            success=True, 
            result="Sunny", 
            tool_name="get_weather",
            error=None
        )
        
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_tool.return_value = tool_def
        
        manager_unpatched.tool_executor = MagicMock(spec=ToolExecutor)
        manager_unpatched.tool_executor.execute.return_value = mock_result
        
        manager_unpatched.config.get_tool_config = MagicMock(return_value={})
        
        # Execute
        result = manager_unpatched.execute_tool("get_weather", loc="San Francisco")
        
        # Assert
        manager_unpatched.tool_executor.execute.assert_called_once_with(
            tool_def, loc="San Francisco"
        )
        assert result == mock_result
        
    def test_execute_tool_adds_config_params(self, manager_unpatched: ToolManager):
        """Test that execute_tool adds tool-specific config parameters."""
        # Setup
        tool_def = TOOL_DEF_WEATHER
        mock_result = ToolResult(
            success=True, 
            result="Sunny", 
            tool_name="get_weather",
            error=None
        )
        
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_tool.return_value = tool_def
        
        manager_unpatched.tool_executor = MagicMock(spec=ToolExecutor)
        manager_unpatched.tool_executor.execute.return_value = mock_result
        
        # Configure a tool-specific config with additional parameters
        manager_unpatched.config.get_tool_config = MagicMock(return_value={"units": "metric"})
        
        # Execute - only pass 'loc' parameter
        result = manager_unpatched.execute_tool("get_weather", loc="San Francisco")
        
        # Assert - should include both user-provided and config parameters
        manager_unpatched.tool_executor.execute.assert_called_once_with(
            tool_def, loc="San Francisco", units="metric"
        )
        
    def test_execute_tool_handles_tool_not_found(self, manager_unpatched: ToolManager):
        """Test that execute_tool handles the case when a tool is not found."""
        # Setup
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_tool.return_value = None
        manager_unpatched.logger = MagicMock(spec=LoggerInterface)
        
        # We need to patch ToolResult to use our fixed version with status correctly
        with patch('src.tools.tool_manager.ToolResult') as MockToolResult:
            # Execute
            manager_unpatched.execute_tool("nonexistent_tool")
            
            # Assert
            MockToolResult.assert_called_once()
            call_args = MockToolResult.call_args[1]
            assert "Tool not found" in call_args["error"]
            assert call_args["tool_name"] == "nonexistent_tool"
            manager_unpatched.logger.error.assert_called_once()
        
    def test_execute_tool_handles_exceptions(self, manager_unpatched: ToolManager):
        """Test that execute_tool handles exceptions from the executor."""
        # Setup
        tool_def = TOOL_DEF_WEATHER
        
        manager_unpatched.tool_registry = MagicMock(spec=ToolRegistry)
        manager_unpatched.tool_registry.get_tool.return_value = tool_def
        
        manager_unpatched.tool_executor = MagicMock(spec=ToolExecutor)
        manager_unpatched.tool_executor.execute.side_effect = Exception("Execution error")
        
        manager_unpatched.config.get_tool_config = MagicMock(return_value={})
        manager_unpatched.logger = MagicMock(spec=LoggerInterface)
        
        # We need to patch AIToolError and ToolResult to use our fixed version
        with patch('src.tools.tool_manager.AIToolError') as MockAIToolError, \
             patch('src.tools.tool_manager.ToolResult') as MockToolResult, \
             patch('src.tools.tool_manager.ErrorHandler.handle_error') as mock_error_handler:
            
            mock_error_handler.return_value = {"message": "Handled error: Execution error"}
            
            # Execute
            manager_unpatched.execute_tool("get_weather", loc="San Francisco")
            
            # Assert AIToolError was created with the correct parameters
            MockAIToolError.assert_called_once()
            assert "Error executing tool get_weather" in MockAIToolError.call_args[0][0]
            assert MockAIToolError.call_args[1]["tool_name"] == "get_weather"
            
            # Assert ErrorHandler.handle_error was called
            mock_error_handler.assert_called_once()
            
            # Assert ToolResult was created with the correct parameters
            MockToolResult.assert_called_once()
            assert "Handled error: Execution error" == MockToolResult.call_args[1]["error"]

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