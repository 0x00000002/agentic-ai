"""
Tool Manager Module

This module provides the central manager for discovering and executing all tools,
whether internal or external (MCP).
"""
from typing import Dict, Any, List, Optional, Set, Union, TYPE_CHECKING
import json
import time
import asyncio

from src.utils.logger import LoggerFactory, LoggerInterface
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_executor import ToolExecutor
from src.tools.tool_stats_manager import ToolStatsManager
from src.tools.models import ToolDefinition, ToolResult, ToolExecutionStatus, ToolCall
from src.exceptions import AIToolError, ErrorHandler
from src.config.unified_config import UnifiedConfig
from src.mcp.mcp_client_manager import MCPClientManager

if TYPE_CHECKING:
    from src.agents.tool_finder_agent import ToolFinderAgent
    from src.agents import AgentFactory


class ToolManager:
    """
    Manager for coordinating tool discovery and execution across all sources.
    
    This class aggregates tool definitions from ToolRegistry (internal) 
    and MCPClientManager (external MCP) and handles dispatching execution.
    """
    
    def __init__(self, unified_config: Optional[UnifiedConfig] = None, 
                 logger: Optional[LoggerInterface] = None, 
                 tool_registry: Optional[ToolRegistry] = None, 
                 tool_executor: Optional[ToolExecutor] = None,
                 tool_stats_manager: Optional[ToolStatsManager] = None,
                 mcp_client_manager: Optional[MCPClientManager] = None):
        """
        Initialize the tool manager.
        """
        self.logger = logger or LoggerFactory.create("tool_manager")
        self.config = unified_config or UnifiedConfig.get_instance()
        self.tool_config = self.config.get_tool_config() # General tool config (execution, stats)
        
        # Initialize dependencies (dependency injection)
        self.tool_registry = tool_registry or ToolRegistry(logger=self.logger)
        self.mcp_client_manager = mcp_client_manager or MCPClientManager(config=self.config, logger=self.logger)
        
        executor_config = self.tool_config.get("execution", {})
        self.tool_executor = tool_executor or ToolExecutor(
            logger=self.logger,
            timeout=executor_config.get("timeout", 30),
            max_retries=executor_config.get("max_retries", 3)
        )
        self.tool_stats_manager = tool_stats_manager or ToolStatsManager(logger=self.logger, unified_config=self.config)
        
        # Aggregate tool definitions from all sources
        self._all_tools: Dict[str, ToolDefinition] = {}
        self._load_all_tool_definitions()
        
        self.logger.info(f"Tool manager initialized with {len(self._all_tools)} tools.")

    def _load_all_tool_definitions(self):
        """Load internal tools from ToolRegistry and declared MCP tools from MCPClientManager."""
        self._all_tools = {}
        
        # Load internal tools
        internal_tools = self.tool_registry.list_internal_tool_definitions()
        for tool_def in internal_tools:
            if tool_def.name in self._all_tools:
                 self.logger.warning(f"Duplicate tool name '{tool_def.name}' found (internal). Overwriting previous definition.")
            self._all_tools[tool_def.name] = tool_def
        self.logger.info(f"Loaded {len(internal_tools)} internal tool definitions.")
            
        # Load declared MCP tools
        mcp_tools = self.mcp_client_manager.list_mcp_tool_definitions()
        for tool_def in mcp_tools:
            if tool_def.name in self._all_tools:
                 self.logger.warning(f"Duplicate tool name '{tool_def.name}' found (MCP tool from server '{tool_def.mcp_server_name}'). Overwriting previous definition (maybe internal?).")
            self._all_tools[tool_def.name] = tool_def
        self.logger.info(f"Loaded {len(mcp_tools)} declared MCP tool definitions.")
        
    def reload_tools(self):
        """Reload tool definitions from configuration sources."""
        self.logger.info("Reloading tool definitions...")
        # Re-initialize registries/managers if they depend on config that might change?
        # For now, just reload the definitions into the manager
        # TODO: Consider if ToolRegistry/MCPClientManager need re-init if config changes
        self._load_all_tool_definitions()
        self.logger.info(f"Tool definitions reloaded. Total tools: {len(self._all_tools)}.")

    # --- Tool Discovery Methods --- 

    def list_available_tools(self) -> List[ToolDefinition]:
        """Returns a list of all available tool definitions (internal and MCP)."""
        return list(self._all_tools.values())

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get the definition for a specific tool by name."""
        return self._all_tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """Get definitions for all registered tools (internal and MCP)."""
        return self._all_tools.copy()

    # --- Tool Execution Method --- 
        
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool based on a ToolCall object asynchronously.
        Dispatches execution to the appropriate handler (internal executor or MCP client).
        
        Args:
            tool_call: The ToolCall object containing name and arguments.
            
        Returns:
            ToolResult object with execution results.
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        request_id = getattr(tool_call, 'id', None) # Use tool_call.id if available
        start_time = time.monotonic()
        
        tool_definition = self.get_tool_definition(tool_name)
        result: Optional[ToolResult] = None # Initialize result
        metadata = {"original_call_id": request_id} if request_id else {} # Prepare metadata

        if not tool_definition:
            error_message = f"Tool not found: {tool_name}"
            self.logger.error(error_message)
            # Update stats even if tool not found
            result = ToolResult(success=False, error=error_message, result=None, tool_name=tool_name, metadata=metadata)
        else:
            # Tool found, proceed with execution
            try:
                if tool_definition.source == "internal":
                    self.logger.debug(f"Executing internal tool '{tool_name}'...")
                    # Pass definition for context if executor needs it, otherwise just module/func
                    # Pass arguments as a dictionary to the 'parameters' argument
                    result = await self.tool_executor.execute(
                        tool_definition=tool_definition,
                        parameters=tool_args # Pass as dict, not **kwargs
                        # **tool_args
                    )
                elif tool_definition.source == "mcp":
                    self.logger.debug(f"Executing MCP tool '{tool_name}' via server '{tool_definition.mcp_server_name}'...")
                    if not tool_definition.mcp_server_name:
                         # This shouldn't happen due to validation, but defensive check
                         raise AIToolError(f"MCP tool '{tool_name}' definition is missing mcp_server_name.")
                    
                    # Get the MCP client session (connects/launches if needed)
                    session = await self.mcp_client_manager.get_tool_client(tool_definition.mcp_server_name)
                    
                    # Call the tool via the session or wrapper
                    # This now works whether session is mcp.ClientSession or _HttpClientWrapper
                    mcp_response = await session.call_tool(tool_name, tool_args)
                    
                    # Convert MCP response / HTTP response dict to our ToolResult format
                    # Check for an error field/attribute in the response FIRST.
                    # Handle both attribute access (ClientSession) and dict access (Wrapper)
                    mcp_error_obj = getattr(mcp_response, 'error', None) # For ClientSession
                    mcp_error_dict_val = mcp_response.get("error") if isinstance(mcp_response, dict) else None # For Wrapper
                    
                    if mcp_error_obj or mcp_error_dict_val:
                         # Extract error message
                         error_message = mcp_error_dict_val or getattr(mcp_error_obj, 'message', str(mcp_error_obj))
                         # Get details if available
                         error_details = mcp_response.get("error_details") if isinstance(mcp_response, dict) else None
                         full_error_message = f"{error_message}" + (f": {error_details}" if error_details else "")
                         
                         self.logger.warning(f"MCP tool '{tool_name}' returned an error: {full_error_message}")
                         result = ToolResult(success=False, error=full_error_message, result=None, tool_name=tool_name, metadata=metadata)
                    else:
                         # Assume successful response content is in a 'result' or 'content' attribute/key
                         result_content = None
                         if isinstance(mcp_response, dict):
                             result_content = mcp_response.get('result', mcp_response.get('content'))
                         else:
                              result_content = getattr(mcp_response, 'result', getattr(mcp_response, 'content', None))
                              
                         self.logger.debug(f"MCP tool '{tool_name}' executed successfully.")
                         result = ToolResult(success=True, result=result_content, error=None, tool_name=tool_name, metadata=metadata)
                         
                else:
                    raise AIToolError(f"Unknown tool source '{tool_definition.source}' for tool '{tool_name}'")

            except asyncio.TimeoutError: # Specific handling for timeouts
                self.logger.error(f"Timeout executing tool '{tool_name}'.")
                result = ToolResult(success=False, error=f"Timeout executing tool {tool_name}", result=None, tool_name=tool_name, metadata=metadata)
            # Add specific exception handling for MCP connection errors if the library provides them
            # except MCPConnectionError as conn_err: 
            #    self.logger.error(f"Connection error during MCP tool '{tool_name}' execution: {conn_err}", exc_info=True)
            #    result = ToolResult(success=False, error=f"Connection error for tool {tool_name}: {conn_err}", result=None, tool_name=tool_name)
            except Exception as e:
                self.logger.error(f"Error during ToolManager execution dispatch for '{tool_name}': {e}", exc_info=True)
                # Preserve the original exception type/message if possible
                tool_error = AIToolError(f"Error executing tool {tool_name}: {type(e).__name__}: {str(e)}", tool_name=tool_name) 
                error_response = ErrorHandler.handle_error(tool_error, self.logger)
                result = ToolResult(success=False, error=error_response["message"], result=None, tool_name=tool_name, metadata=metadata)

        # Ensure we always have a result object IF tool was found
        if result is None:
             self.logger.error(f"Tool execution for '{tool_name}' resulted in None result. Returning error.")
             result = ToolResult(success=False, error="Tool execution failed to produce a result.", result=None, tool_name=tool_name, metadata=metadata)

        # Record stats (this will now run even if tool_definition was None)
        end_time = time.monotonic()
        execution_time_ms = int((end_time - start_time) * 1000)
        self.tool_stats_manager.update_stats(
            tool_name=tool_name,
            success=result.success,
            duration_ms=execution_time_ms,
            request_id=request_id 
        )
            
        return result
        
    # --- Other Methods (Potentially delegate more to specific managers) --- 

    def format_tools_for_model(self, model_id: str, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tools for use with a specific model.
        Considers tools from all sources.
        """
        model_config = self.config.get_model_config(model_id)
        if not model_config or "provider" not in model_config:
            self.logger.warning(f"Unable to determine provider for model {model_id}, cannot format tools.")
            return []
            
        provider_name = model_config["provider"].upper()
            
        target_tool_names = set(tool_names) if tool_names else set(self._all_tools.keys())
        
        formatted_tools = []
        for name in target_tool_names:
            tool_def = self.get_tool_definition(name)
            if not tool_def:
                self.logger.warning(f"Tool '{name}' requested for formatting not found.")
                continue

            try:
                # Basic format suitable for most providers (OpenAI, Anthropic)
                # Uses alias 'input_schema' for parameters_schema field in ToolDefinition
                provider_format = tool_def.model_dump(by_alias=True, include={'name', 'description', 'parameters_schema'})
                
                # Provider-specific adjustments
                if "GEMINI" in provider_name:
                     # Reformat parameters for Gemini's FunctionDeclaration structure
                     provider_format = {
                         "name": tool_def.name,
                         "description": tool_def.description,
                         "parameters": {
                             "type": "object",
                             "properties": tool_def.parameters_schema.get('properties', {}),
                             "required": tool_def.parameters_schema.get('required', [])
                         }
                     }
                elif "OPENAI" in provider_name:
                     # OpenAI expects {"type": "function", "function": {...}} structure
                     provider_format = {"type": "function", "function": provider_format}
                
                # elif "ANTHROPIC" in provider_name:
                # Anthropic seems compatible with the base format (name, desc, input_schema)

                formatted_tools.append(provider_format)
            except Exception as e:
                 self.logger.error(f"Error formatting tool '{name}' for provider '{provider_name}': {e}", exc_info=True)

        return formatted_tools
    
    # Delegate stats methods directly to stats manager
    def save_usage_stats(self, file_path: Optional[str] = None) -> None:
        self.tool_stats_manager.save_stats(file_path)
    
    def load_usage_stats(self, file_path: Optional[str] = None) -> None:
        self.tool_stats_manager.load_stats(file_path)
        
    # get_tool_info might need updating if stats/config keys change
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        tool_definition = self.get_tool_definition(tool_name)
        if not tool_definition:
            return None
        usage_stats = self.tool_stats_manager.get_stats(tool_name)
        # Tool-specific config might need rethinking - how is it stored/retrieved now?
        # Let's just return definition + stats for now
        return {
            "definition": tool_definition.model_dump(), # Return the full definition
            "usage_stats": usage_stats,
        }
        
    # Remove old register_tool method - registration happens in ToolRegistry/MCPClientManager
    # def register_tool(self, ...) -> None: ... 