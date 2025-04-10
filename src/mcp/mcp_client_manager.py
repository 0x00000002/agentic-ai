"""
Manages the lifecycle and access to MCP (Model Context Protocol) clients.
"""

import logging
import asyncio
import os # Added for environment variable access
from urllib.parse import urlparse # Added for URL parsing
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional, List

from pydantic import ValidationError

# Import necessary MCP SDK components based on confirmed usage
from mcp import ClientSession
# from mcp.client import http, websocket # Incorrect module structure
# from mcp.client.http import HttpClient # Incorrect class/location
# from mcp.client.websocket import WebSocketClient # Incorrect class/location
# --- Adding correct imports for mcp v1.6.0 --- 
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client

# Import config system and exceptions
from ..config import get_config, UnifiedConfig
from ..exceptions import AIConfigError, AIToolError
from ..utils.logger import LoggerFactory
# Import ToolDefinition model
from ..tools.models import ToolDefinition

# --- Reverting imports as they are incorrect for mcp v1.6.0 --- 
# from mcp.client.http import http_client, HttpServerParameters # Invalid in v1.6.0
# from mcp.client.websocket import websocket_client, WebSocketServerParameters # Invalid in v1.6.0
# --- Adding correct imports for mcp v1.6.0 --- 
# from mcp.client.http import HttpClient
# from mcp.client.websocket import WebSocketClient


class MCPClientManager:
    """
    Handles loading MCP server configurations and declared tool definitions,
    and lazily instantiating/managing ClientSessions for servers.
    """

    def __init__(self, config: Optional[UnifiedConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the MCPClientManager.
        """
        self._config = config or get_config()
        self._logger = logger or LoggerFactory.create(name="mcp_client_manager")
        # Store server configs {server_name: {url: ..., auth: ..., description: ...}} 
        self._server_configs: Dict[str, Dict[str, Any]] = {}
        # Store declared tool definitions {tool_name: ToolDefinition}
        self._declared_mcp_tools: Dict[str, ToolDefinition] = {}
        # Store active sessions {server_name: ClientSession}
        self._active_sessions: Dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()

        self._load_mcp_configs()

    def _load_mcp_configs(self):
        """Loads MCP server configurations and their declared tools from the main config."""
        self._logger.info("Loading MCP server configurations and declared tools...")
        try:
            mcp_config = self._config.get_mcp_config() or {}
            # Expecting top-level key 'mcp_servers'
            server_definitions = mcp_config.get("mcp_servers", {})
            if not isinstance(server_definitions, dict):
                self._logger.error(f"Expected a dictionary under 'mcp_servers' key, got {type(server_definitions)}. Skipping MCP config loading.")
                return

            if not server_definitions:
                self._logger.warning("No server definitions found under 'mcp_servers' key.")
                return

            for server_name, server_conf in server_definitions.items():
                if not isinstance(server_conf, dict):
                    self._logger.warning(f"Skipping invalid MCP server entry for '{server_name}' (not a dictionary): {server_conf}")
                    continue
                
                # --- Refactored Config Loading --- 
                url = server_conf.get("url")
                auth_config = server_conf.get("auth") # Optional auth details
                provided_tools_list = server_conf.get("provides_tools", [])
                
                if not url or not isinstance(url, str):
                    self._logger.warning(f"Skipping MCP server '{server_name}': Missing or invalid required field 'url'.")
                    continue
                
                # Validate URL scheme (basic check)
                try:
                    parsed_url = urlparse(url)
                    if parsed_url.scheme not in ["http", "https", "ws", "wss"]:
                         self._logger.warning(f"Skipping MCP server '{server_name}': Unsupported URL scheme '{parsed_url.scheme}'. Supported: http, https, ws, wss.")
                         continue
                except Exception as e:
                    self._logger.warning(f"Skipping MCP server '{server_name}': Could not parse URL '{url}'. Error: {e}")
                    continue
                    
                # Validate auth config if present
                if auth_config is not None and not isinstance(auth_config, dict):
                    self._logger.warning(f"Skipping MCP server '{server_name}': 'auth' field must be a dictionary if present, got {type(auth_config)}.")
                    continue # Or maybe just ignore auth? For now, skip server.

                # Store server connection config
                self._server_configs[server_name] = {
                    "url": url,
                    "auth": auth_config, # Store the whole auth dict or None
                    "description": server_conf.get("description", ""),
                }
                self._logger.debug(f"Loaded config for MCP server: {server_name} (URL: {url})")
                # --- End Refactored Config Loading --- 

                # Process declared tools for this server
                if not isinstance(provided_tools_list, list):
                    self._logger.warning(f"'provides_tools' for server '{server_name}' must be a list, got {type(provided_tools_list)}. Skipping tool declarations for this server.")
                    continue
                    
                for tool_conf in provided_tools_list:
                    if not isinstance(tool_conf, dict):
                        self._logger.warning(f"Skipping invalid tool declaration under server '{server_name}' (not a dictionary): {tool_conf}")
                        continue
                    
                    tool_name = tool_conf.get("name")
                    if not tool_name:
                         self._logger.warning(f"Skipping tool declaration under server '{server_name}' (missing 'name'): {tool_conf}")
                         continue
                         
                    if tool_name in self._declared_mcp_tools:
                         self._logger.warning(f"Duplicate MCP tool name declaration '{tool_name}' found (from server '{server_name}'). Overwriting previous declaration.")
                    
                    # Prepare data for ToolDefinition validation
                    tool_def_data = tool_conf.copy()
                    tool_def_data['source'] = 'mcp'
                    tool_def_data['mcp_server_name'] = server_name
                    # Handle schema alias
                    if 'inputSchema' in tool_def_data:
                        tool_def_data['parameters_schema'] = tool_def_data.pop('inputSchema')
                        
                    try:
                        tool_def = ToolDefinition(**tool_def_data)
                        self._declared_mcp_tools[tool_name] = tool_def
                        self._logger.debug(f"Declared MCP tool '{tool_name}' from server '{server_name}'.")
                    except ValidationError as e:
                         self._logger.error(f"Validation failed for declared MCP tool '{tool_name}' from server '{server_name}': {e}")
                    except Exception as e:
                         self._logger.error(f"Unexpected error processing declared MCP tool '{tool_name}' from server '{server_name}': {e}", exc_info=True)

        except Exception as e:
            self._logger.error(f"Failed to load MCP configurations: {e}", exc_info=True)

    def list_mcp_tool_definitions(self) -> List[ToolDefinition]:
        """Returns a list of all declared MCP tool definitions loaded from config."""
        return list(self._declared_mcp_tools.values())

    async def get_tool_client(self, mcp_server_name: str) -> ClientSession:
        """
        Gets or creates an MCP ClientSession for the specified server name
        by connecting to its configured network URL (HTTP/WS).
        Handles basic bearer token authentication via environment variables.
        """
        # Check cache first
        if mcp_server_name in self._active_sessions:
            self._logger.debug(f"Returning cached MCP ClientSession for server: {mcp_server_name}")
            return self._active_sessions[mcp_server_name]

        # Check if configuration exists
        if mcp_server_name not in self._server_configs:
            self._logger.error(f"MCP server '{mcp_server_name}' not found in configuration.")
            raise AIToolError(f"MCP server '{mcp_server_name}' not configured.")

        server_config = self._server_configs[mcp_server_name]
        url = server_config.get("url")
        auth_config = server_config.get("auth")

        # URL should have been validated during loading, but double-check
        if not url:
             self._logger.error(f"Internal Error: URL missing for configured MCP server '{mcp_server_name}'.")
             raise AIToolError(f"Internal configuration error for MCP server '{mcp_server_name}'.")

        try:
            self._logger.info(f"Establishing new MCP ClientSession for server: {mcp_server_name} at {url}")
            
            # --- Authentication Handling --- 
            auth_headers = {}
            if auth_config:
                auth_type = auth_config.get("type")
                if auth_type == "bearer":
                    token_env_var = auth_config.get("token_env_var")
                    if token_env_var:
                        token = os.getenv(token_env_var)
                        if token:
                            auth_headers["Authorization"] = f"Bearer {token}"
                            self._logger.debug(f"Using Bearer token authentication for {mcp_server_name} from env var '{token_env_var}'.")
                        else:
                             self._logger.warning(f"Bearer token environment variable '{token_env_var}' not set for server '{mcp_server_name}'. Attempting connection without auth.")
                    else:
                        self._logger.warning(f"Auth type is 'bearer' but 'token_env_var' is not specified for server '{mcp_server_name}'. Attempting connection without auth.")
                else:
                     self._logger.warning(f"Unsupported authentication type '{auth_type}' for server '{mcp_server_name}'. Attempting connection without auth.")
            # --- End Authentication --- 

           # --- Network Client Selection & Connection ---
            parsed_url = urlparse(url)
            transport = None
            client = None # Define client variable
            
            # Prepare headers for the client constructor
            client_headers = auth_headers if auth_headers else {}

            # Instantiate the correct client based on URL scheme
            if parsed_url.scheme in ["http", "https"]:
                self._logger.debug(f"Using sse_client for {url}")
                # sse_client supports headers
                transport = await self._exit_stack.enter_async_context(
                    sse_client(url=url, headers=client_headers)
                )
            elif parsed_url.scheme in ["ws", "wss"]:
                self._logger.debug(f"Using websocket_client for {url}")
                # websocket_client in v1.6.0 does NOT support headers
                if client_headers: 
                    self._logger.warning(
                        f"MCP v1.6.0 websocket_client does not support headers. Authentication configured for server '{mcp_server_name}' will be ignored."
                    )
                transport = await self._exit_stack.enter_async_context(
                    websocket_client(url=url)
                )
            else:
                # This should have been caught during loading, but belts and braces
                raise AIToolError(f"Unsupported URL scheme '{parsed_url.scheme}' for server '{mcp_server_name}'.")
            
            # Use the client instance as an async context manager via the exit stack
            # to get the transport (reader, writer)
            # transport = await self._exit_stack.enter_async_context(client) # Logic moved into if/elif
            
            read_stream, write_stream = transport 
            
            # Enter the ClientSession context using the stack
            session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

            await session.initialize()
            # --- End Network Client Selection & Connection ---
            
            self._active_sessions[mcp_server_name] = session
            self._logger.info(f"MCP ClientSession initialized successfully for server: {mcp_server_name}")
            return session
        except ConnectionRefusedError as e: # More specific error handling
             self._logger.error(f"Connection refused for MCP server '{mcp_server_name}' at {url}. Is the server running and accessible? Error: {e}", exc_info=True)
             raise AIToolError(f"Connection refused for MCP server '{mcp_server_name}'. Reason: {e}")
        except asyncio.TimeoutError as e: # Handle timeouts
             self._logger.error(f"Timeout connecting to MCP server '{mcp_server_name}' at {url}. Error: {e}", exc_info=True)
             raise AIToolError(f"Timeout connecting to MCP server '{mcp_server_name}'. Reason: {e}")
        except Exception as e:
            # Catch potential errors from SDK clients or session initialization
            self._logger.error(f"Failed to establish or initialize MCP ClientSession for server '{mcp_server_name}' at {url}: {e}", exc_info=True)
            # Clean up potentially partially entered contexts if initialization fails
            # await self._exit_stack.aclose() # ExitStack handles this automatically on exception propagation
            raise AIToolError(f"Could not create or initialize client for MCP server '{mcp_server_name}'. Reason: {type(e).__name__}: {e}")

    async def close_all_clients(self):
        """Clean up resources, closing all managed MCP sessions and network transports."""
        self._logger.info("Closing all active MCP clients and transports via AsyncExitStack.")
        await self._exit_stack.aclose()
        self._active_sessions = {} # Clear cache after closing
        self._logger.info("AsyncExitStack closed.")

    # Optional: Add a method to close connections if needed
    # def close_all_clients(self):
    #     self._logger.info("Closing all active MCP clients.")
    #     for name, client in self._active_clients.items():
    #         try:
    #             # Assuming the client has a close() method
    #             if hasattr(client, 'close') and callable(client.close):
    #                 client.close()
    #                 self._logger.debug(f"Closed client for {name}")
    #         except Exception as e:
    #             self._logger.error(f"Error closing client for {name}: {e}", exc_info=True)
    #     self._active_clients = {} 