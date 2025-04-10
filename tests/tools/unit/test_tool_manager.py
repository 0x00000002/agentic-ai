# tests/tools/unit/test_tool_manager.py
"""
Unit tests for the ToolManager class.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import time # Added for stats duration

# Import necessary components
from src.tools.models import ToolDefinition, ToolCall, ToolResult, ToolExecutionStatus
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry # Dependency
from src.tools.tool_executor import ToolExecutor # Dependency
from src.tools.tool_stats_manager import ToolStatsManager # Dependency
from src.exceptions import AIToolError, ErrorHandler
from src.utils.logger import LoggerInterface
from src.config.unified_config import UnifiedConfig

# Update ToolDefinition instantiation
WEATHER_PARAMS_SCHEMA = {"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]}
STOCK_PARAMS_SCHEMA = {"type": "object", "properties": {"sym": {"type": "string"}}, "required": ["sym"]}

# Updated to include module_path and function_name
TOOL_DEF_WEATHER = ToolDefinition(
    name="get_weather", 
    description="Get current weather", 
    parameters_schema={"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]},
    module_path="src.tools.core.weather", # Dummy path for testing
    function_name="get_weather_func",   # Dummy name for testing
    # function=MagicMock() # Function can be None now
)

TOOL_DEF_STOCK = ToolDefinition(
    name="get_stock", 
    description="Get stock price", 
    parameters_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    module_path="src.tools.core.stock", # Dummy path
    function_name="get_stock_func",   # Dummy name
    # function=MagicMock()
)

class TestToolManager:
    """Test suite for ToolManager."""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Provides a mock ToolRegistry."""
        registry = MagicMock(spec=ToolRegistry)
        registry.get_all_tool_definitions.return_value = {}
        registry.get_tool.return_value = None # Default to not found
        return registry

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Provides a mock ToolExecutor."""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = MagicMock() # Default mock for execute
        return executor

    @pytest.fixture
    def mock_stats_manager(self) -> MagicMock:
        """Provides a mock ToolStatsManager."""
        stats_manager = MagicMock(spec=ToolStatsManager)
        stats_manager.get_stats.return_value = None # Default to no stats
        return stats_manager
        
    @pytest.fixture
    def manager(self, mock_registry, mock_executor, mock_stats_manager) -> ToolManager:
        """Provides a ToolManager instance with mocked dependencies."""
        # Patch config loading during init
        with patch('src.tools.tool_manager.UnifiedConfig.get_instance') as MockConfig:
            mock_config_instance = MockConfig.return_value
            
            # Set a default return value for the general config call in __init__
            # Tests needing specific tool config will override this return_value directly.
            mock_config_instance.get_tool_config.return_value = {"stats": {}, "execution": {}}
            # Removed side_effect
            
            # Pass mocks directly to constructor
            manager_instance = ToolManager(
                logger=MagicMock(spec=LoggerInterface),
                unified_config=mock_config_instance,
                tool_registry=mock_registry,
                tool_executor=mock_executor,
                tool_stats_manager=mock_stats_manager
            )
            # Reset the mock call count AFTER __init__ has run, 
            # so tests only see calls made *within* the test itself.
            mock_config_instance.get_tool_config.reset_mock()
            return manager_instance

    # --- Registration Tests ---
    def test_register_tool_delegates_to_registry(self, manager: ToolManager, mock_registry):
        """Test that register_tool calls registry.register_tool."""
        manager.register_tool(tool_name=TOOL_DEF_WEATHER.name, tool_definition=TOOL_DEF_WEATHER)
        mock_registry.register_tool.assert_called_once_with(TOOL_DEF_WEATHER.name, TOOL_DEF_WEATHER)
        
    # --- Execute Tool Tests ---
    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_success(self, manager: ToolManager, mock_registry, mock_executor, mock_stats_manager):
        """Test successful execution flow including stats update."""
        # Setup
        tool_name = "get_weather"
        tool_def = TOOL_DEF_WEATHER
        args = {"loc": "London"}
        exec_args_no_request_id = args.copy() # Args passed to executor shouldn't include request_id
        call_args = {**args, "request_id": "req-123"} # Args passed to manager include request_id
        mock_result = ToolResult(success=True, result="Rainy", tool_name=tool_name)
        
        mock_registry.get_tool.return_value = tool_def
        # Ensure the executor's execute is an AsyncMock if not already set by fixture
        mock_executor.execute = AsyncMock(return_value=mock_result)
        manager.config.get_tool_config.return_value = {} # No specific config

        # Mock time for duration calculation
        with patch('time.monotonic', side_effect=[100.0, 100.5]): # Use monotonic clock
             # Execute with await
             result = await manager.execute_tool(tool_name=tool_name, **call_args)

        # Assert Executor Call (without request_id)
        mock_executor.execute.assert_called_once_with(tool_def, **exec_args_no_request_id)
        assert result == mock_result

        # Assert Stats Update Call
        mock_stats_manager.update_stats.assert_called_once_with(
            tool_name=tool_name,
            success=True,
            duration_ms=500, # 100.5 - 100.0 = 0.5s = 500ms
            request_id="req-123"
        )
        # Check get_tool_config called to find tool-specific config
        manager.config.get_tool_config.assert_called_with(tool_name)
        
    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_failure(self, manager: ToolManager, mock_registry, mock_executor, mock_stats_manager):
        """Test failure execution flow including stats update."""
        # Setup
        tool_name = "get_weather"
        tool_def = TOOL_DEF_WEATHER
        args = {"loc": "London"}
        exec_args_no_request_id = args.copy()
        call_args = {**args, "request_id": "req-abc"}
        mock_result = ToolResult(success=False, error="Executor Failed", tool_name=tool_name)
        
        mock_registry.get_tool.return_value = tool_def
        mock_executor.execute = AsyncMock(return_value=mock_result) # Executor returns failure result
        manager.config.get_tool_config.return_value = {} # No specific config

        with patch('time.monotonic', side_effect=[200.0, 200.8]): # Use monotonic clock
             result = await manager.execute_tool(tool_name=tool_name, **call_args) # Await

        # Assert Executor Call
        mock_executor.execute.assert_called_once_with(tool_def, **exec_args_no_request_id)
        assert result == mock_result

        # Assert Stats Update Call (success=False)
        mock_stats_manager.update_stats.assert_called_once_with(
            tool_name=tool_name,
            success=False,
            duration_ms=800, 
            request_id="req-abc"
        )
        manager.config.get_tool_config.assert_called_with(tool_name)

    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_adds_config_params(self, manager: ToolManager, mock_registry, mock_executor, mock_stats_manager):
        """Test that execute_tool adds tool-specific config parameters."""
        # Setup
        tool_name = "get_weather"
        tool_def = TOOL_DEF_WEATHER
        args = {"loc": "Paris"} # No request_id here
        mock_result = ToolResult(success=True, result="Sunny", tool_name=tool_name)
        
        mock_registry.get_tool.return_value = tool_def
        mock_executor.execute = AsyncMock(return_value=mock_result)
        
        # Mock config to return specific params for this tool
        manager.config.get_tool_config.return_value = {"units": "metric", "source": "api"}

        with patch('time.monotonic', side_effect=[300.0, 300.1]): # Use monotonic clock
            result = await manager.execute_tool(tool_name=tool_name, **args) # Await

        # Assert Executor Call includes merged params
        expected_executor_args = {"loc": "Paris", "units": "metric", "source": "api"}
        mock_executor.execute.assert_called_once_with(tool_def, **expected_executor_args)
        assert result == mock_result

        # Assert Stats Update Call (no request_id)
        mock_stats_manager.update_stats.assert_called_once_with(
            tool_name=tool_name,
            success=True,
            duration_ms=100, 
            request_id=None
        )
        manager.config.get_tool_config.assert_called_with(tool_name)

    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_handles_tool_not_found(self, manager: ToolManager, mock_registry, mock_stats_manager):
        """Test execute_tool handles tool not found before executor or stats calls."""
        # Setup
        tool_name = "nonexistent_tool"
        mock_registry.get_tool.return_value = None # Tool not found
        
        # Execute
        result = await manager.execute_tool(tool_name=tool_name, arg1="val1") # Await
        
        # Assert
        assert result.success is False
        assert "Tool not found" in result.error
        assert result.tool_name == tool_name
        # Executor should NOT have been called
        manager.tool_executor.execute.assert_not_called()
        # Stats manager should NOT have been called
        mock_stats_manager.update_stats.assert_not_called()

    @pytest.mark.asyncio # Mark as async
    async def test_execute_tool_handles_manager_exception(self, manager: ToolManager, mock_registry, mock_stats_manager):
        """Test execute_tool handles exceptions during its own processing (e.g., config access)."""
        # Setup: Make config access raise an error AFTER getting tool def
        tool_name = "get_weather"
        tool_def = TOOL_DEF_WEATHER
        mock_registry.get_tool.return_value = tool_def
        manager.config.get_tool_config.side_effect = Exception("Config Read Error")
        
        # Patch ErrorHandler to check its input
        with patch('src.tools.tool_manager.ErrorHandler.handle_error') as mock_error_handler:
            mock_error_handler.return_value = {"message": "Handled Config Error"}
            
            # Execute
            result = await manager.execute_tool(tool_name=tool_name, loc="Berlin") # Await

        # Assert
        assert result.success is False
        assert result.error == "Handled Config Error"
        assert result.tool_name == tool_name
        # Executor should NOT have been called
        manager.tool_executor.execute.assert_not_called()
        # Stats manager should NOT have been called
        mock_stats_manager.update_stats.assert_not_called()
        # Error handler should have been called
        mock_error_handler.assert_called_once()
        assert isinstance(mock_error_handler.call_args[0][0], AIToolError)
        assert "Config Read Error" in str(mock_error_handler.call_args[0][0])

    # --- Get Tool Info Tests ---
    def test_get_tool_info_success(self, manager: ToolManager, mock_registry, mock_stats_manager):
        """Test getting tool info successfully."""
        # Setup
        tool_name = "get_stock"
        tool_def = TOOL_DEF_STOCK
        mock_registry.get_tool.return_value = tool_def
        mock_stats = {"uses": 10, "successes": 8}
        mock_stats_manager.get_stats.return_value = mock_stats
        
        # Configure the return value specifically for the call *within* get_tool_info
        mock_tool_config_specific = {"api_key": "dummy"}
        manager.config.get_tool_config.return_value = mock_tool_config_specific
        # Side effect is still active, but return_value takes precedence for the next call

        # Execute
        info = manager.get_tool_info(tool_name)

        # Assert
        assert info is not None
        assert info["name"] == tool_name
        assert info["description"] == tool_def.description
        assert info["parameters"] == tool_def.parameters_schema # Check schema attribute
        assert info["usage_stats"] == mock_stats
        assert info["config"] == mock_tool_config_specific # Check the specific config was returned
        mock_registry.get_tool.assert_called_once_with(tool_name)
        mock_stats_manager.get_stats.assert_called_once_with(tool_name)
        # Assert the mock was called once *during this test* with the specific tool name
        manager.config.get_tool_config.assert_called_once_with(tool_name)
        
    def test_get_tool_info_not_found(self, manager: ToolManager, mock_registry):
        """Test get_tool_info when the tool definition is not found."""
        tool_name = "unknown_tool"
        mock_registry.get_tool.return_value = None # Tool not found
        
        info = manager.get_tool_info(tool_name)
        
        assert info is None
        mock_registry.get_tool.assert_called_once_with(tool_name)

    def test_get_tool_info_no_stats(self, manager: ToolManager, mock_registry, mock_stats_manager):
        """Test getting tool info when stats are not available."""
        tool_name = "get_weather"
        tool_def = TOOL_DEF_WEATHER
        mock_registry.get_tool.return_value = tool_def
        mock_stats_manager.get_stats.return_value = None # No stats found
        # Ensure the config call within get_tool_info returns default empty dict
        manager.config.get_tool_config.return_value = {}
        
        info = manager.get_tool_info(tool_name)
        
        assert info is not None
        assert info["usage_stats"] is None
        mock_stats_manager.get_stats.assert_called_once_with(tool_name)
        # Also assert config was checked for the specific tool
        manager.config.get_tool_config.assert_called_once_with(tool_name)

    # --- Get All Tools --- 
    def test_get_all_tools_delegates_to_registry(self, manager: ToolManager, mock_registry):
        """Test get_all_tools delegates to registry.get_all_tool_definitions."""
        mock_defs = {TOOL_DEF_WEATHER.name: TOOL_DEF_WEATHER}
        mock_registry.get_all_tool_definitions.return_value = mock_defs
        
        result = manager.get_all_tools()
        
        assert result == mock_defs
        mock_registry.get_all_tool_definitions.assert_called_once()
        
    # --- Format Tools --- 
    def test_format_tools_for_model_delegates_to_registry(self, manager: ToolManager, mock_registry):
        """Test format_tools_for_model gets provider and delegates to registry."""
        model_id = "gpt-4"
        provider = "openai" # Assume config maps gpt-4 to openai
        tool_names = ["get_weather"]
        tool_names_set = set(tool_names)
        formatted_list = [{"formatted": "openai_weather"}]
        
        manager.config.get_model_config.return_value = {"provider": provider} # Mock config lookup
        mock_registry.format_tools_for_provider.return_value = formatted_list
        
        result = manager.format_tools_for_model(model_id, tool_names)
        
        assert result == formatted_list
        manager.config.get_model_config.assert_called_once_with(model_id)
        mock_registry.format_tools_for_provider.assert_called_once_with(provider.upper(), tool_names_set)
        
    def test_format_tools_for_model_no_provider(self, manager: ToolManager, mock_registry):
        """Test format_tools_for_model handles missing provider info."""
        model_id = "unknown-model"
        manager.config.get_model_config.return_value = {} # No provider info
        manager.logger = MagicMock(spec=LoggerInterface)

        result = manager.format_tools_for_model(model_id)
        
        assert result == []
        mock_registry.format_tools_for_provider.assert_not_called()
        manager.logger.warning.assert_called_once()

    # --- Stats Save/Load Tests ---
    def test_save_usage_stats_delegates(self, manager: ToolManager, mock_stats_manager):
        """Test save_usage_stats delegates to ToolStatsManager."""
        path = "/tmp/stats.json"
        manager.save_usage_stats(path)
        mock_stats_manager.save_stats.assert_called_once_with(path)
        
    def test_save_usage_stats_delegates_no_path(self, manager: ToolManager, mock_stats_manager):
        """Test save_usage_stats delegates with None path."""
        manager.save_usage_stats(None)
        mock_stats_manager.save_stats.assert_called_once_with(None)

    def test_load_usage_stats_delegates(self, manager: ToolManager, mock_stats_manager):
        """Test load_usage_stats delegates to ToolStatsManager."""
        path = "/tmp/stats.json"
        manager.load_usage_stats(path)
        mock_stats_manager.load_stats.assert_called_once_with(path)
        
    def test_load_usage_stats_delegates_no_path(self, manager: ToolManager, mock_stats_manager):
        """Test load_usage_stats delegates with None path."""
        manager.load_usage_stats(None)
        mock_stats_manager.load_stats.assert_called_once_with(None)

    # --- Removed Tests ---
    # Tests for find_tools removed
    # Tests directly testing registry formatting removed (test registry directly)
        
    # def test_get_tool_definition_delegates_to_registry(...):
    #     pass
        
    # def test_get_all_tool_definitions_delegates_to_registry(...):
    #     pass 