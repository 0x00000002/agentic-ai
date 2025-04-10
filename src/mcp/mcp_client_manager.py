"""
Manages the lifecycle and access to MCP (Model Context Protocol) clients.
"""

import logging
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional, List

from pydantic import ValidationError

# Assuming MCPClient is the main class from the SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import config system and exceptions
from ..config import get_config, UnifiedConfig
from ..exceptions import AIConfigError, AIToolError
from ..utils.logger import LoggerFactory
# Import ToolDefinition model
from ..tools.models import ToolDefinition


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
        # Store server configs {server_name: {command: ..., script_path: ...}} 
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
                
                command = server_conf.get("command")
                script_path = server_conf.get("script_path")
                provided_tools_list = server_conf.get("provides_tools", [])
                
                if not command or not script_path:
                    self._logger.warning(f"Skipping MCP server '{server_name}': Missing required fields 'command' or 'script_path'.")
                    continue
                
                # Store server launch config
                self._server_configs[server_name] = {
                    "command": command,
                    "script_path": script_path,
                    "env": server_conf.get("env"), # Optional env vars
                    "description": server_conf.get("description", ""),
                }
                self._logger.debug(f"Loaded config for MCP server: {server_name}")

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
        Gets or creates an MCP ClientSession for the specified server name.
        (Renamed parameter for clarity)
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
        command = server_config.get("command")
        script_path = server_config.get("script_path")
        env = server_config.get("env") # Optional environment variables

        # Command/script_path should have been validated during loading, but double-check
        if not command or not script_path:
             # This case should ideally not be reached if loading logic is correct
             self._logger.error(f"Internal Error: Command or script_path missing for configured MCP server '{mcp_server_name}'.")
             raise AIToolError(f"Internal configuration error for MCP server '{mcp_server_name}'.")

        try:
            self._logger.info(f"Instantiating new MCP ClientSession for server: {mcp_server_name} ({command} {script_path})")
            
            server_params = StdioServerParameters(
                command=command,
                args=[script_path],
                env=env # Pass environment variables if provided
            )

            # Enter the context for stdio_client and ClientSession using the stack
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
            stdio_read, stdio_write = stdio_transport
            session = await self._exit_stack.enter_async_context(ClientSession(stdio_read, stdio_write))

            await session.initialize()
            
            self._active_sessions[mcp_server_name] = session
            self._logger.info(f"MCP ClientSession initialized successfully for server: {mcp_server_name}")
            return session
        except Exception as e:
            self._logger.error(f"Failed to instantiate or initialize MCP ClientSession for server '{mcp_server_name}': {e}", exc_info=True)
            raise AIToolError(f"Could not create or initialize client for MCP server '{mcp_server_name}'. Reason: {e}")

    async def close_all_clients(self):
        """Clean up resources, closing all managed MCP sessions and transports."""
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