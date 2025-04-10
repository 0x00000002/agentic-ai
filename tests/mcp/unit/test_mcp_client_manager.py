"""
Unit tests for the MCPClientManager.
"""

import pytest
import logging
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from contextlib import AsyncExitStack
from typing import List, Dict, Any, Optional

# Import the class to test
from src.mcp.mcp_client_manager import MCPClientManager

# Import related components and exceptions
from src.config import UnifiedConfig
from src.exceptions import AIConfigError, AIToolError

# Import the class we are mocking for spec
from mcp import ClientSession
# Import the model we use
from src.tools.models import ToolDefinition

# Mock the external SDK classes and functions
@pytest.fixture(autouse=True)
def mock_mcp_deps(mocker):
    # Mock stdio_client context manager
    mock_stdio_ctx_manager = AsyncMock() # The object returned by stdio_client()
    mock_stdio_ctx_manager.__aenter__.return_value = (AsyncMock(), AsyncMock()) # read, write handles
    mock_stdio_client_func = mocker.patch('src.mcp.mcp_client_manager.stdio_client', return_value=mock_stdio_ctx_manager)

    # Mock ClientSession context manager and its initialize method
    mock_session_instance = AsyncMock(spec=ClientSession)
    mock_session_instance.initialize = AsyncMock()
    mock_session_ctx_manager = AsyncMock() # The object returned by ClientSession()
    mock_session_ctx_manager.__aenter__.return_value = mock_session_instance
    mock_ClientSession_constructor = mocker.patch('src.mcp.mcp_client_manager.ClientSession', return_value=mock_session_ctx_manager)
    
    # Mock StdioServerParameters (just needs to be constructible)
    mock_StdioServerParameters = mocker.patch('src.mcp.mcp_client_manager.StdioServerParameters')
    
    # We also need to mock AsyncExitStack methods used
    mock_exit_stack = AsyncMock(spec=AsyncExitStack)
    
    # Define a side_effect function for enter_async_context
    async def enter_ctx_side_effect(context_manager):
        if context_manager is mock_stdio_ctx_manager:
            return await mock_stdio_ctx_manager.__aenter__()
        elif context_manager is mock_session_ctx_manager:
            return await mock_session_ctx_manager.__aenter__()
        else:
            # Default fallback if needed, or raise error
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = AsyncMock()
            return await mock_ctx.__aenter__()
            
    mock_exit_stack.enter_async_context.side_effect = enter_ctx_side_effect
    mocker.patch('src.mcp.mcp_client_manager.AsyncExitStack', return_value=mock_exit_stack)
    
    return {
        "stdio_client_func": mock_stdio_client_func,
        "stdio_ctx_manager": mock_stdio_ctx_manager,
        "ClientSession_constructor": mock_ClientSession_constructor,
        "session_ctx_manager": mock_session_ctx_manager,
        "StdioServerParameters": mock_StdioServerParameters,
        "exit_stack": mock_exit_stack,
        "session_instance": mock_session_instance
    }

# Fixture for a mock logger
@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

# Fixture for a valid mock configuration with declared tools
@pytest.fixture
def mock_config_valid():
    config = MagicMock(spec=UnifiedConfig)
    config.get_mcp_config.return_value = {
        "mcp_servers": {
            "server1": {
                "description": "Server One",
                "command": "python",
                "script_path": "path/to/server1.py",
                "env": {"KEY": "VALUE"},
                "provides_tools": [
                    {
                        "name": "mcp_tool_a",
                        "description": "MCP Tool A",
                        "inputSchema": {"type": "object", "properties": {"p1": {"type": "string"}}}, 
                        "speed": "medium",
                        "safety": "external"
                    },
                    {
                        "name": "mcp_tool_b",
                        "description": "MCP Tool B",
                        "inputSchema": {},
                        "speed": "slow"
                        # safety will use default
                    }
                ]
            },
            "server2": {
                "description": "Server Two",
                "command": "node",
                "script_path": "path/to/server2.js",
                "provides_tools": [
                    {
                        "name": "mcp_tool_c",
                        "description": "MCP Tool C",
                        "inputSchema": {}
                        # speed/safety use defaults
                    }
                ]
            }
        }
    }
    return config

# Fixture for empty/missing MCP config
@pytest.fixture
def mock_config_empty():
    config = MagicMock(spec=UnifiedConfig)
    config.get_mcp_config.return_value = {}
    return config

# Fixture for config where mcp_servers is not a dict
@pytest.fixture
def mock_config_invalid_servers_type():
    config = MagicMock(spec=UnifiedConfig)
    config.get_mcp_config.return_value = {"mcp_servers": "not_a_dict"}
    return config
    
# Fixture for config with invalid server definitions
@pytest.fixture
def mock_config_invalid_server_defs():
    config = MagicMock(spec=UnifiedConfig)
    config.get_mcp_config.return_value = {
        "mcp_servers": {
            "server_ok": {"command": "python", "script_path": "path/to/ok.py", "provides_tools": []}, # Valid
            "server_no_script": {"command": "python", "provides_tools": []},          # Missing script_path
            "server_no_command": {"script_path": "path/to/nocmd.py", "provides_tools": []}, # Missing command
            "server_invalid_tools": {"command": "python", "script_path": "path/to/invalid.py", "provides_tools": "not_a_list"}, # Invalid provides_tools
            "server_bad_tool_decl": {
                 "command": "python", "script_path": "path/to/bad_decl.py", 
                 "provides_tools": [
                      {"description": "missing name"}, # Missing name
                      "not_a_dict", # Invalid type
                      {"name": "bad_schema", "inputSchema": "not_a_dict"}, # Invalid schema (Pydantic validation will catch)
                      {"name": "good_tool", "description": "A valid tool", "inputSchema": {}} # Valid tool
                  ]
            }
        }
    }
    return config

# Fixture for config where a server is missing command/script_path (redundant, covered by invalid_server_defs)
# @pytest.fixture
# def mock_config_missing_script(): ...

# --- Test Cases ---

@pytest.mark.asyncio
async def test_init_success_loading(mock_config_valid, mock_logger):
    """Test successful initialization loads server configs and declared tools."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    
    # Check server configs
    assert len(manager._server_configs) == 2
    assert "server1" in manager._server_configs
    assert "server2" in manager._server_configs
    assert manager._server_configs["server1"]["script_path"] == "path/to/server1.py"
    assert manager._server_configs["server1"]["command"] == "python"
    assert manager._server_configs["server1"]["env"] == {"KEY": "VALUE"}
    
    # Check declared tools
    assert len(manager._declared_mcp_tools) == 3
    assert "mcp_tool_a" in manager._declared_mcp_tools
    assert "mcp_tool_b" in manager._declared_mcp_tools
    assert "mcp_tool_c" in manager._declared_mcp_tools
    
    tool_a_def = manager._declared_mcp_tools["mcp_tool_a"]
    assert isinstance(tool_a_def, ToolDefinition)
    assert tool_a_def.source == "mcp"
    assert tool_a_def.mcp_server_name == "server1"
    assert tool_a_def.speed == "medium"
    assert tool_a_def.safety == "external"
    assert tool_a_def.description == "MCP Tool A"
    assert tool_a_def.parameters_schema == {"type": "object", "properties": {"p1": {"type": "string"}}}

    tool_b_def = manager._declared_mcp_tools["mcp_tool_b"]
    assert tool_b_def.source == "mcp"
    assert tool_b_def.mcp_server_name == "server1"
    assert tool_b_def.speed == "slow"
    assert tool_b_def.safety == "native" # Default safety
    
    tool_c_def = manager._declared_mcp_tools["mcp_tool_c"]
    assert tool_c_def.source == "mcp"
    assert tool_c_def.mcp_server_name == "server2"
    assert tool_c_def.speed == "medium" # Default speed
    assert tool_c_def.safety == "native" # Default safety

    # Check logger wasn't used for errors/warnings during valid load
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()

@pytest.mark.asyncio
async def test_init_no_mcp_servers_section(mock_config_empty, mock_logger):
    """Test initialization with no mcp_servers section."""
    manager = MCPClientManager(config=mock_config_empty, logger=mock_logger)
    assert len(manager._server_configs) == 0
    assert len(manager._declared_mcp_tools) == 0
    mock_logger.warning.assert_called_once_with("No server definitions found under 'mcp_servers' key.")

@pytest.mark.asyncio
async def test_init_invalid_mcp_servers_type(mock_config_invalid_servers_type, mock_logger):
    """Test initialization when mcp_servers is not a dict."""
    manager = MCPClientManager(config=mock_config_invalid_servers_type, logger=mock_logger)
    assert len(manager._server_configs) == 0
    assert len(manager._declared_mcp_tools) == 0
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert "Expected a dictionary under 'mcp_servers' key" in args[0]

@pytest.mark.asyncio
async def test_init_invalid_server_definitions(mock_config_invalid_server_defs, mock_logger):
    """Test initialization skips invalid server definitions and tool declarations."""
    manager = MCPClientManager(config=mock_config_invalid_server_defs, logger=mock_logger)
    
    # Only the valid server should be loaded
    assert len(manager._server_configs) == 3
    assert "server_ok" in manager._server_configs
    assert "server_no_script" not in manager._server_configs # Skipped due to missing script_path
    assert "server_no_command" not in manager._server_configs # Skipped due to missing command
    # Correcting this assertion - server_invalid_tools is valid but has invalid tools
    assert "server_invalid_tools" in manager._server_configs # Server config is valid
    assert "server_bad_tool_decl" in manager._server_configs # Server is valid, tools are bad

    # Check which warnings were logged for servers
    server_warnings = [call.args[0] for call in mock_logger.warning.call_args_list if "Skipping MCP server" in call.args[0]]
    assert len(server_warnings) == 2
    assert any("server_no_script" in msg for msg in server_warnings)
    assert any("server_no_command" in msg for msg in server_warnings)
    
    # Check declared tools (only from valid servers with valid declarations)
    assert len(manager._declared_mcp_tools) == 1
    assert "good_tool" in manager._declared_mcp_tools
    assert isinstance(manager._declared_mcp_tools["good_tool"], ToolDefinition)
    assert manager._declared_mcp_tools["good_tool"].mcp_server_name == "server_bad_tool_decl"
    
    # Check warnings for tool declarations
    tool_warnings = [call.args[0] for call in mock_logger.warning.call_args_list if "Skipping invalid tool declaration" in call.args[0] or "'provides_tools' for server" in call.args[0] or "Skipping tool declaration" in call.args[0]]
    assert len(tool_warnings) >= 3 # server_invalid_tools list + 2 bad decls in server_bad_tool_decl
    assert any("'provides_tools' for server 'server_invalid_tools' must be a list" in msg for msg in tool_warnings)
    assert any("Skipping tool declaration under server 'server_bad_tool_decl' (missing 'name')" in msg for msg in tool_warnings)
    assert any("Skipping invalid tool declaration under server 'server_bad_tool_decl' (not a dictionary)" in msg for msg in tool_warnings)
    
    # Check errors for tool declarations
    tool_errors = [call.args[0] for call in mock_logger.error.call_args_list if "Validation failed for declared MCP tool" in call.args[0]]
    assert len(tool_errors) >= 1 # bad_schema fails validation
    assert any("Validation failed for declared MCP tool 'bad_schema'" in msg for msg in tool_errors)


@pytest.mark.asyncio
async def test_list_mcp_tool_definitions(mock_config_valid, mock_logger):
    """Test listing declared MCP tools."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    defs = manager.list_mcp_tool_definitions()
    assert isinstance(defs, list)
    assert len(defs) == 3
    def_names = {d.name for d in defs}
    assert def_names == {"mcp_tool_a", "mcp_tool_b", "mcp_tool_c"}

@pytest.mark.asyncio
async def test_get_tool_client_success_and_cache(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test getting a client session successfully and that it's cached."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    
    # First call for server1
    client1 = await manager.get_tool_client("server1")
    assert client1 is not None
    assert client1 is mock_mcp_deps["session_instance"] # Should be the mocked instance
    
    # Check that StdioServerParameters was called correctly for server1
    mock_mcp_deps["StdioServerParameters"].assert_called_once_with(
        command="python", 
        args=["path/to/server1.py"], 
        env={"KEY": "VALUE"}
    )
    # Check that stdio_client was called with the params
    mock_mcp_deps["stdio_client_func"].assert_called_once_with(mock_mcp_deps["StdioServerParameters"].return_value)
    # Check that ClientSession was called with stdio handles
    stdio_read, stdio_write = mock_mcp_deps["stdio_ctx_manager"].__aenter__.return_value
    mock_mcp_deps["ClientSession_constructor"].assert_called_once_with(stdio_read, stdio_write)
    # Check initialize was called
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()
    assert "server1" in manager._active_sessions

    # Second call for server1
    client2 = await manager.get_tool_client("server1")
    assert client2 is client1 # Should return the exact same instance
    # Ensure the setup functions were not called again
    mock_mcp_deps["StdioServerParameters"].assert_called_once()
    mock_mcp_deps["stdio_client_func"].assert_called_once()
    mock_mcp_deps["ClientSession_constructor"].assert_called_once()
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_tool_client_not_found(mock_config_valid, mock_logger):
    """Test getting a client for a non-existent server name."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    with pytest.raises(AIToolError, match="MCP server 'non_existent_server' not configured."):
        await manager.get_tool_client("non_existent_server")

# Removed test_get_tool_client_missing_script as it's covered by invalid defs test

@pytest.mark.asyncio
async def test_get_tool_client_instantiation_fails(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test getting a client when session instantiation raises an error."""
    # Configure one of the mocked async steps to raise an error (e.g., entering stdio_client context)
    mock_mcp_deps["stdio_ctx_manager"].__aenter__.side_effect = ConnectionError("Failed to launch process")
    
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    
    with pytest.raises(AIToolError, match="Could not create or initialize client for MCP server 'server1'.*Reason: Failed to launch process"):
        await manager.get_tool_client("server1")
    
    # Check error was logged with exc_info=True
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert "Failed to instantiate or initialize MCP ClientSession for server 'server1'" in args[0]
    assert kwargs.get('exc_info') is True
    
    # Ensure it didn't get added to active sessions
    assert "server1" not in manager._active_sessions

# Helper functions for test_close_all_clients
def create_mock_config():
    """Create a basic mock config for testing."""
    config = MagicMock(spec=UnifiedConfig)
    config.get_mcp_config.return_value = {
        "mcp_servers": {
            "test_server": {
                "command": "python",
                "script_path": "path/to/test_server.py",
                "provides_tools": []
            }
        }
    }
    return config

def create_mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock(spec=logging.Logger)
    return logger

@pytest.mark.asyncio
async def test_close_all_clients(mocker):
    """Test that close_all_clients method properly cleans up resources."""
    # Setup
    manager = MCPClientManager(config=create_mock_config(), logger=create_mock_logger())
    
    # Mock the AsyncExitStack.aclose method
    mock_aclose = AsyncMock()
    manager._exit_stack.aclose = mock_aclose
    
    # Manually populate the active sessions dictionary
    mock_session = AsyncMock()
    manager._active_sessions['test_server'] = mock_session
    
    # Verify session is populated
    assert len(manager._active_sessions) > 0
    
    # Call method under test
    await manager.close_all_clients()
    
    # Verify exit stack's aclose was called
    mock_aclose.assert_called_once()
    
    # Verify client cache is cleared
    assert len(manager._active_sessions) == 0
    
    # Verify logging occurred
    manager._logger.info.assert_has_calls([
        mocker.call("Closing all active MCP clients and transports via AsyncExitStack."),
        mocker.call("AsyncExitStack closed.")
    ]) 