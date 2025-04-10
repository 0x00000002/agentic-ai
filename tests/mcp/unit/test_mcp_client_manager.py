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

# ---- Added imports for network mocks ----
# Correct imports for mcp v1.6.0 mocks
# from mcp.client.http import HttpClient
# from mcp.client.websocket import WebSocketClient
import os # For mocking getenv

# --- Custom Mock Classes for Network Clients --- 
# These are no longer strictly needed as we patch functions, but keep for clarity if desired
# class MockHttpClient: ... 
# class MockWebSocketClient: ...
# --- End Custom Mock Classes --- 

# Mock the external SDK classes and functions
@pytest.fixture(autouse=True)
def mock_mcp_deps(mocker):
    # ---- Remove stdio mocks ----
    # mock_stdio_ctx_manager = AsyncMock() 
    # mock_stdio_ctx_manager.__aenter__.return_value = (AsyncMock(), AsyncMock())
    # mock_stdio_client_func = mocker.patch('src.mcp.mcp_client_manager.stdio_client', return_value=mock_stdio_ctx_manager)
    # mock_StdioServerParameters = mocker.patch('src.mcp.mcp_client_manager.StdioServerParameters')

    # ---- Mock sse_client (v1.6.0 function) ----
    mock_sse_ctx_manager = AsyncMock(name="sse_client_context")
    mock_sse_ctx_manager.__aenter__.return_value = (AsyncMock(name="sse_reader"), AsyncMock(name="sse_writer"))
    mock_sse_client_func = mocker.patch('src.mcp.mcp_client_manager.sse_client',
                                        return_value=mock_sse_ctx_manager)
    
    # ---- Mock websocket_client (v1.6.0 function) ----
    mock_ws_ctx_manager = AsyncMock(name="websocket_client_context")
    mock_ws_ctx_manager.__aenter__.return_value = (AsyncMock(name="ws_reader"), AsyncMock(name="ws_writer"))
    mock_websocket_client_func = mocker.patch('src.mcp.mcp_client_manager.websocket_client',
                                             return_value=mock_ws_ctx_manager)

    # ---- Mock ClientSession (unchanged) ----
    mock_session_instance = AsyncMock(spec=ClientSession)
    mock_session_instance.initialize = AsyncMock()
    mock_session_ctx_manager = AsyncMock(name="session_context")
    mock_session_ctx_manager.__aenter__.return_value = mock_session_instance
    mock_ClientSession_constructor = mocker.patch('src.mcp.mcp_client_manager.ClientSession', 
                                               return_value=mock_session_ctx_manager)
    
    # ---- Mock AsyncExitStack (update side effect for client context managers) ----
    mock_exit_stack = AsyncMock(spec=AsyncExitStack)
    async def enter_ctx_side_effect(context_manager):
        if context_manager is mock_sse_ctx_manager:
            return await mock_sse_ctx_manager.__aenter__()
        elif context_manager is mock_ws_ctx_manager:
            return await mock_ws_ctx_manager.__aenter__()
        elif context_manager is mock_session_ctx_manager:
            return await mock_session_ctx_manager.__aenter__()
        else:
            mocker.fail(f"AsyncExitStack entered unexpected context manager: {context_manager}")
            
    mock_exit_stack.enter_async_context.side_effect = enter_ctx_side_effect
    mocker.patch('src.mcp.mcp_client_manager.AsyncExitStack', return_value=mock_exit_stack)
    
    # ---- Mock os.getenv (unchanged) ----
    mock_os_getenv = mocker.patch('src.mcp.mcp_client_manager.os.getenv')

    # ---- Updated return dictionary (v1.6.0 functions) ----
    return {
        # Removed constructor/instance mocks
        # "HttpClient_constructor": mock_HttpClient_constructor,
        # "http_client_instance": mock_http_client_instance,
        # etc.
        "sse_client_func": mock_sse_client_func,
        "sse_ctx_manager": mock_sse_ctx_manager,
        "websocket_client_func": mock_websocket_client_func,
        "ws_ctx_manager": mock_ws_ctx_manager,
        "ClientSession_constructor": mock_ClientSession_constructor,
        "session_ctx_manager": mock_session_ctx_manager,
        "exit_stack": mock_exit_stack,
        "session_instance": mock_session_instance,
        "os_getenv": mock_os_getenv
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
            "http_server_no_auth": { # HTTP, no auth
                "description": "HTTP Server One",
                "url": "http://localhost:8081/mcp",
                # No auth section
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
            "ws_server_with_auth": { # WebSocket, with auth
                "description": "WS Server Two",
                "url": "wss://secure.server.com/mcp/ws",
                "auth": {
                    "type": "bearer",
                    "token_env_var": "WS_SERVER_TOKEN"
                },
                "provides_tools": [
                    {
                        "name": "mcp_tool_c",
                        "description": "MCP Tool C",
                        "inputSchema": {}
                        # speed/safety use defaults
                    }
                ]
            },
            "https_server_auth_no_env": { # HTTPS, auth configured but env var might be missing
                 "description": "HTTPS Server Three",
                 "url": "https://another.service.io/api/mcp",
                 "auth": {
                     "type": "bearer",
                     "token_env_var": "HTTPS_SERVER_TOKEN_MISSING" # Simulate missing env var
                 },
                 "provides_tools": [
                     {
                         "name": "mcp_tool_d",
                         "description": "MCP Tool D",
                         "inputSchema": {}
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
            "server_ok": { # Valid basic server
                "url": "http://valid.com/mcp", 
                "provides_tools": []
            },
            # --- Invalid server configs --- 
            "server_no_url": { # Missing url
                "description": "Missing URL",
                "provides_tools": []
            },         
            "server_invalid_url_scheme": { # Invalid URL scheme
                "url": "ftp://invalid.com",
                "provides_tools": []
            }, 
            "server_bad_url_format": { # Malformed URL
                "url": "http//missing-colon.com",
                "provides_tools": []
            },
            "server_invalid_auth_type": { # Auth is not a dict
                "url": "http://auth.invalid/mcp",
                "auth": "not_a_dict",
                "provides_tools": []
            },
            # --- Valid servers with invalid tool definitions --- 
            "server_invalid_tools_list": { # provides_tools is not a list
                "url": "http://tools.invalid/mcp", 
                "provides_tools": "not_a_list"
            }, 
            "server_bad_tool_decl": { # Valid server, some bad tool declarations
                 "url": "http://badd ecl.valid/mcp", 
                 "provides_tools": [
                      {"description": "missing name"}, # Missing name
                      "not_a_dict", # Invalid type in list
                      {"name": "bad_schema", "inputSchema": "not_a_dict"}, # Invalid schema (Pydantic)
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
    """Test successful initialization loads server configs (url, auth) and declared tools."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    
    # Check server configs (updated for url/auth)
    assert len(manager._server_configs) == 3
    assert "http_server_no_auth" in manager._server_configs
    assert "ws_server_with_auth" in manager._server_configs
    assert "https_server_auth_no_env" in manager._server_configs
    
    http_conf = manager._server_configs["http_server_no_auth"]
    assert http_conf["url"] == "http://localhost:8081/mcp"
    assert http_conf["auth"] is None # No auth configured
    
    ws_conf = manager._server_configs["ws_server_with_auth"]
    assert ws_conf["url"] == "wss://secure.server.com/mcp/ws"
    assert ws_conf["auth"] == {"type": "bearer", "token_env_var": "WS_SERVER_TOKEN"}
    
    https_conf = manager._server_configs["https_server_auth_no_env"]
    assert https_conf["url"] == "https://another.service.io/api/mcp"
    assert https_conf["auth"] == {"type": "bearer", "token_env_var": "HTTPS_SERVER_TOKEN_MISSING"}
    
    # Check declared tools (should reference new server names)
    assert len(manager._declared_mcp_tools) == 4
    assert "mcp_tool_a" in manager._declared_mcp_tools
    assert "mcp_tool_b" in manager._declared_mcp_tools
    assert "mcp_tool_c" in manager._declared_mcp_tools
    assert "mcp_tool_d" in manager._declared_mcp_tools
    
    tool_a_def = manager._declared_mcp_tools["mcp_tool_a"]
    assert isinstance(tool_a_def, ToolDefinition)
    assert tool_a_def.source == "mcp"
    assert tool_a_def.mcp_server_name == "http_server_no_auth" # Updated server name
    assert tool_a_def.speed == "medium"
    assert tool_a_def.safety == "external"
    assert tool_a_def.description == "MCP Tool A"
    assert tool_a_def.parameters_schema == {"type": "object", "properties": {"p1": {"type": "string"}}}

    tool_b_def = manager._declared_mcp_tools["mcp_tool_b"]
    assert tool_b_def.source == "mcp"
    assert tool_b_def.mcp_server_name == "http_server_no_auth" # Updated server name
    assert tool_b_def.speed == "slow"
    assert tool_b_def.safety == "native" # Default safety
    
    tool_c_def = manager._declared_mcp_tools["mcp_tool_c"]
    assert tool_c_def.source == "mcp"
    assert tool_c_def.mcp_server_name == "ws_server_with_auth" # Updated server name
    assert tool_c_def.speed == "medium" # Default speed
    assert tool_c_def.safety == "native" # Default safety

    tool_d_def = manager._declared_mcp_tools["mcp_tool_d"]
    assert tool_d_def.source == "mcp"
    assert tool_d_def.mcp_server_name == "https_server_auth_no_env" # Updated server name

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
    """Test initialization skips invalid server definitions (url, auth) and tool declarations."""
    manager = MCPClientManager(config=mock_config_invalid_server_defs, logger=mock_logger)
    
    # Only the servers with valid definitions should be loaded
    assert len(manager._server_configs) == 3
    assert "server_ok" in manager._server_configs
    assert "server_invalid_tools_list" in manager._server_configs # Server config is valid
    assert "server_bad_tool_decl" in manager._server_configs # Server is valid, tools are bad
    
    # Ensure invalid servers were skipped
    assert "server_no_url" not in manager._server_configs 
    assert "server_invalid_url_scheme" not in manager._server_configs
    assert "server_bad_url_format" not in manager._server_configs
    assert "server_invalid_auth_type" not in manager._server_configs

    # Check which warnings were logged for skipped servers
    server_warnings = [call.args[0] for call in mock_logger.warning.call_args_list if "Skipping MCP server" in call.args[0]]
    assert len(server_warnings) >= 4 # Expect at least 4 warnings for the skipped servers
    assert any("server_no_url': Missing or invalid required field 'url'" in msg for msg in server_warnings)
    assert any("server_invalid_url_scheme': Unsupported URL scheme 'ftp'" in msg for msg in server_warnings)
    assert any("server_bad_url_format'" in msg for msg in server_warnings)
    assert any("server_invalid_auth_type': 'auth' field must be a dictionary" in msg for msg in server_warnings)
    
    # Check declared tools (only from valid servers with valid declarations)
    assert len(manager._declared_mcp_tools) == 1
    assert "good_tool" in manager._declared_mcp_tools
    assert isinstance(manager._declared_mcp_tools["good_tool"], ToolDefinition)
    assert manager._declared_mcp_tools["good_tool"].mcp_server_name == "server_bad_tool_decl"
    
    # Check warnings for tool declarations
    tool_warnings = [call.args[0] for call in mock_logger.warning.call_args_list if "Skipping invalid tool declaration" in call.args[0] or "'provides_tools' for server" in call.args[0] or "Skipping tool declaration" in call.args[0]]
    # Check for specific tool warnings
    assert any("'provides_tools' for server 'server_invalid_tools_list' must be a list" in msg for msg in tool_warnings)
    assert any("Skipping tool declaration under server 'server_bad_tool_decl' (missing 'name')" in msg for msg in tool_warnings)
    assert any("Skipping invalid tool declaration under server 'server_bad_tool_decl' (not a dictionary)" in msg for msg in tool_warnings)
    # Assert the count after checking individual messages
    assert len(tool_warnings) >= 3 

    # Check errors for tool declarations
    tool_errors = [call.args[0] for call in mock_logger.error.call_args_list if "Validation failed for declared MCP tool" in call.args[0]]
    assert len(tool_errors) >= 1 # bad_schema fails validation
    assert any("Validation failed for declared MCP tool 'bad_schema'" in msg for msg in tool_errors)

@pytest.mark.asyncio
async def test_list_mcp_tool_definitions(mock_config_valid, mock_logger):
    """Test listing declared MCP tools after valid loading."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    defs = manager.list_mcp_tool_definitions()
    assert isinstance(defs, list)
    # Use the count from the applied fixture
    expected_tool_count = len(mock_config_valid.get_mcp_config()['mcp_servers']['http_server_no_auth']['provides_tools']) + \
                          len(mock_config_valid.get_mcp_config()['mcp_servers']['ws_server_with_auth']['provides_tools']) + \
                          len(mock_config_valid.get_mcp_config()['mcp_servers']['https_server_auth_no_env']['provides_tools'])
    assert len(defs) == expected_tool_count 
    def_names = {d.name for d in defs}
    assert def_names == {"mcp_tool_a", "mcp_tool_b", "mcp_tool_c", "mcp_tool_d"}

@pytest.mark.asyncio
async def test_get_tool_client_http_success_no_auth(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test successful connection to HTTP server without authentication."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "http_server_no_auth"
    expected_url = "http://localhost:8081/mcp"

    # First call for http_server_no_auth
    client1 = await manager.get_tool_client(server_name)
    assert client1 is mock_mcp_deps["session_instance"]
    
    # Check that sse_client was called correctly (no headers)
    mock_mcp_deps["sse_client_func"].assert_called_once_with(url=expected_url, headers={})
    # Check that the client instance was used as context manager
    mock_mcp_deps["exit_stack"].enter_async_context.assert_any_call(mock_mcp_deps["sse_ctx_manager"])
    # Check that ClientSession was called with http handles
    sse_read, sse_write = mock_mcp_deps["sse_ctx_manager"].__aenter__.return_value
    mock_mcp_deps["ClientSession_constructor"].assert_called_once_with(sse_read, sse_write)
    # Check initialize was called
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()
    assert server_name in manager._active_sessions
    
    # Ensure websocket client constructor was not called
    mock_mcp_deps["websocket_client_func"].assert_not_called()

@pytest.mark.asyncio
async def test_get_tool_client_ws_success_with_auth(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test successful connection to WebSocket server with bearer authentication."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "ws_server_with_auth"
    expected_url = "wss://secure.server.com/mcp/ws"
    token_env_var = "WS_SERVER_TOKEN"
    mock_token = "test-ws-token-123"
    
    # Mock os.getenv to return the token
    mock_mcp_deps["os_getenv"].configure_mock(**{"return_value": mock_token})

    # First call for ws_server_with_auth
    client1 = await manager.get_tool_client(server_name)
    assert client1 is mock_mcp_deps["session_instance"]
    
    # Check os.getenv was called
    mock_mcp_deps["os_getenv"].assert_called_once_with(token_env_var)
    
    # Check that websocket_client was constructed correctly (NO headers in v1.6.0)
    mock_mcp_deps["websocket_client_func"].assert_called_once_with(url=expected_url)
    
    # Check that the WARNING about ignored headers was logged
    mock_logger.warning.assert_any_call(
        f"MCP v1.6.0 websocket_client does not support headers. Authentication configured for server '{server_name}' will be ignored."
    )
    
    # Check that the client instance was used as context manager
    mock_mcp_deps["exit_stack"].enter_async_context.assert_any_call(mock_mcp_deps["ws_ctx_manager"])
    # Check that ClientSession was called with ws handles
    ws_read, ws_write = mock_mcp_deps["ws_ctx_manager"].__aenter__.return_value
    mock_mcp_deps["ClientSession_constructor"].assert_called_once_with(ws_read, ws_write)
    # Check initialize was called
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()
    
    # Ensure http client constructor was not called
    mock_mcp_deps["sse_client_func"].assert_not_called()

@pytest.mark.asyncio
async def test_get_tool_client_https_auth_env_missing(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test connection to HTTPS server attempts without auth when env var is missing."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "https_server_auth_no_env"
    expected_url = "https://another.service.io/api/mcp"
    token_env_var = "HTTPS_SERVER_TOKEN_MISSING"
    
    # Mock os.getenv to return None (simulate missing var)
    mock_mcp_deps["os_getenv"].configure_mock(**{"return_value": None})

    # Call for https_server_auth_no_env
    client1 = await manager.get_tool_client(server_name)
    assert client1 is mock_mcp_deps["session_instance"]
    
    # Check os.getenv was called
    mock_mcp_deps["os_getenv"].assert_called_once_with(token_env_var)
    mock_logger.warning.assert_any_call(
        f"Bearer token environment variable '{token_env_var}' not set for server '{server_name}'. Attempting connection without auth."
    )
    
    # Check that sse_client was called correctly (no headers)
    mock_mcp_deps["sse_client_func"].assert_called_once_with(url=expected_url, headers={})
    # Check that the client instance was used as context manager
    mock_mcp_deps["exit_stack"].enter_async_context.assert_any_call(mock_mcp_deps["sse_ctx_manager"])
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()
    assert server_name in manager._active_sessions

@pytest.mark.asyncio
async def test_get_tool_client_caching(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test that getting a client session is cached."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "http_server_no_auth"
    
    # First call
    client1 = await manager.get_tool_client(server_name)
    assert client1 is mock_mcp_deps["session_instance"]
    
    # Check functions were called once
    mock_mcp_deps["sse_client_func"].assert_called_once()
    mock_mcp_deps["exit_stack"].enter_async_context.assert_any_call(mock_mcp_deps["sse_ctx_manager"])
    mock_mcp_deps["ClientSession_constructor"].assert_called_once()
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()

    # Reset initialize mock for the second check
    mock_mcp_deps["session_instance"].initialize.reset_mock() 

    # Second call for the same server
    client2 = await manager.get_tool_client(server_name)
    assert client2 is client1 # Should return the exact same instance
    
    # Ensure the setup functions were NOT called again
    mock_mcp_deps["sse_client_func"].assert_called_once()
    mock_mcp_deps["exit_stack"].enter_async_context.assert_any_call(mock_mcp_deps["sse_ctx_manager"])
    mock_mcp_deps["ClientSession_constructor"].assert_called_once()
    mock_mcp_deps["session_instance"].initialize.assert_not_awaited() # Initialize should not be called again

    # Check exit_stack was called expected number of times (depends on number of context managers entered on first call)
    # Expected calls: sse_client context, session context
    assert mock_mcp_deps["exit_stack"].enter_async_context.call_count == 2 
    mock_mcp_deps["ClientSession_constructor"].assert_called_once() # Constructor called only once
    mock_mcp_deps["session_instance"].initialize.assert_not_awaited() # Initialize should not be called again

@pytest.mark.asyncio
async def test_get_tool_client_not_found(mock_config_valid, mock_logger):
    """Test getting a client for a non-existent server name."""
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    with pytest.raises(AIToolError, match="MCP server 'non_existent_server' not configured."):
        await manager.get_tool_client("non_existent_server")

# Removed test_get_tool_client_missing_script as config format changed

@pytest.mark.asyncio
async def test_get_tool_client_connection_fails(mock_config_valid, mock_mcp_deps, mock_logger, mocker):
    """Test getting a client when the network client (sse/ws) raises an error on context entry."""
    # Configure the sse_client mock context manager (__aenter__) to raise an error
    mock_mcp_deps["sse_ctx_manager"] = AsyncMock(name="sse_client_context_error")
    mock_mcp_deps["sse_ctx_manager"].__aenter__.side_effect = ConnectionRefusedError("Test connection refused")
    # Re-patch the function to return this specific context manager mock
    mocker.patch('src.mcp.mcp_client_manager.sse_client', return_value=mock_mcp_deps["sse_ctx_manager"])
    # Update exit stack side effect to recognize the new mock
    original_exit_stack_mock = mock_mcp_deps["exit_stack"]
    original_side_effect_func = original_exit_stack_mock.enter_async_context.side_effect
    
    async def new_side_effect(context_manager):
        if context_manager is mock_mcp_deps["sse_ctx_manager"]:
            # Call the __aenter__ which has the side_effect (ConnectionRefusedError)
            return await context_manager.__aenter__() 
        # Call original side effect for other cases (ws_ctx_manager, session_ctx_manager)
        return await original_side_effect_func(context_manager)
        
    mock_mcp_deps["exit_stack"].enter_async_context.side_effect = new_side_effect

    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "http_server_no_auth"
    expected_url = "http://localhost:8081/mcp" # Match the server being tested
    
    with pytest.raises(AIToolError, match=f"Connection refused for MCP server '{server_name}'.*Reason: Test connection refused"):
        await manager.get_tool_client(server_name)
    
    # Check error was logged with exc_info=True
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert f"Connection refused for MCP server '{server_name}' at {expected_url}" in args[0]
    assert kwargs.get('exc_info') is True
    
    # Ensure it didn't get added to active sessions
    assert server_name not in manager._active_sessions
    # Ensure session constructor wasn't called because connection failed before session creation
    mock_mcp_deps["ClientSession_constructor"].assert_not_called()

@pytest.mark.asyncio
async def test_get_tool_client_initialization_fails(mock_config_valid, mock_mcp_deps, mock_logger):
    """Test getting a client when session.initialize() raises an error."""
    # Configure the mocked session instance's initialize to raise an error
    init_error = asyncio.TimeoutError("Initialization timed out")
    mock_mcp_deps["session_instance"].initialize.side_effect = init_error
    
    manager = MCPClientManager(config=mock_config_valid, logger=mock_logger)
    server_name = "http_server_no_auth"
    expected_url = "http://localhost:8081/mcp" # Match the server being tested
    
    with pytest.raises(AIToolError, match=f"Timeout connecting to MCP server '{server_name}'.*Reason: Initialization timed out"):
        await manager.get_tool_client(server_name)
        
    # Check error was logged with exc_info=True
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert f"Timeout connecting to MCP server '{server_name}' at {expected_url}" in args[0]
    assert kwargs.get('exc_info') is True
    
    # Ensure it didn't get added to active sessions
    assert server_name not in manager._active_sessions
    # Check that the client function and session constructor WERE called
    mock_mcp_deps["sse_client_func"].assert_called_once()
    mock_mcp_deps["ClientSession_constructor"].assert_called_once()
    # Check initialize was awaited
    mock_mcp_deps["session_instance"].initialize.assert_awaited_once()

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

@pytest.fixture
def setup_for_close_test(mock_config_valid, mock_logger):
    return MCPClientManager(config=mock_config_valid, logger=mock_logger)

@pytest.mark.asyncio
async def test_close_all_clients(mocker, setup_for_close_test):
    """Test that close_all_clients method properly cleans up resources."""
    # Setup
    manager = setup_for_close_test
    
    # Mock the AsyncExitStack.aclose method
    mock_aclose = AsyncMock()
    manager._exit_stack.aclose = mock_aclose
    
    # Manually populate the active sessions dictionary (even though connection wasn't fully mocked here)
    mock_session = AsyncMock()
    server_name_to_close = "http_server_no_auth" # Use a name from the valid config
    manager._active_sessions[server_name_to_close] = mock_session
    
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