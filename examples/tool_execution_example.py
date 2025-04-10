# examples/tool_execution_example.py
"""
Example script demonstrating how to list and execute internal and MCP tools
using the ToolManager.
"""
import asyncio
import os
from pathlib import Path
import sys
import logging

# --- Setup Path ---
# Add the project root to the Python path to allow importing from 'src'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# Set environment variable for config loading (optional, but can help)
os.environ['AGENTIC_AI_APP_ROOT'] = str(project_root)

# --- Imports ---
from src.config import configure, get_config, UnifiedConfig
from src.tools.tool_manager import ToolManager
from src.tools.models import ToolCall, ToolResult
from src.utils.logger import LoggerFactory

# Configure logging for the example
logger = LoggerFactory.create("tool_example")

# --- Main Execution Logic ---
async def main():
    logger.info("--- Initializing ToolManager ---")
    # Ensure configuration is loaded (uses defaults from src/config if not called)
    # configure() # Uncomment to load potentially custom user configs if needed

    # ToolManager uses the UnifiedConfig singleton and creates its own dependencies
    # (ToolRegistry, MCPClientManager, etc.) if they are not provided.
    try:
        tool_manager = ToolManager()
        logger.info("ToolManager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ToolManager: {e}", exc_info=True)
        logger.error("Ensure your configuration files (tools.yml, mcp.yml) are correctly placed and formatted.")
        return

    logger.info("\n--- Listing Available Tools ---")
    try:
        available_tools = tool_manager.list_available_tools()
        logger.info(f"Found {len(available_tools)} tools:")
        for tool_def in available_tools:
            logger.info(f"  - Name: {tool_def.name:<25} Source: {tool_def.source:<10} Server: {tool_def.mcp_server_name or 'N/A'}")
    except Exception as e:
        logger.error(f"Failed to list available tools: {e}", exc_info=True)
        return

    # --- Execute Internal Tool ---
    logger.info("\n--- Attempting to Execute Internal Tool (calculator) ---")
    internal_tool_name = "calculator"
    if tool_manager.get_tool_definition(internal_tool_name):
        internal_tool_call = ToolCall(
            id="call-internal-example-1",
            name=internal_tool_name,
            arguments={"expression": "10 * (5 + 2)"}
        )
        logger.info(f"Executing internal tool '{internal_tool_name}' with args: {internal_tool_call.arguments}")
        try:
            start_time = asyncio.get_event_loop().time()
            internal_result: ToolResult = await tool_manager.execute_tool(internal_tool_call)
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(f"Internal Tool Result (took {duration:.2f} ms):")
            logger.info(f"  Success: {internal_result.success}")
            if internal_result.success:
                logger.info(f"  Result: {internal_result.result}")
            else:
                logger.error(f"  Error: {internal_result.error}")
        except Exception as e:
            logger.error(f"Unexpected error executing internal tool '{internal_tool_name}': {e}", exc_info=True)
    else:
         logger.warning(f"Internal tool '{internal_tool_name}' not found in configuration. Skipping execution.")

    # --- Execute MCP Tool ---
    # IMPORTANT: This part requires the MCP server defined in 'mcp.yml'
    # (e.g., 'execute_python_code' pointing to a python script) to be runnable.
    # If the server cannot be started or communicated with, this will fail.
    logger.info("\n--- Attempting to Execute MCP Tool (execute_python_code) ---")
    mcp_tool_name = "execute_python_code" # Change if using a different MCP tool
    if tool_manager.get_tool_definition(mcp_tool_name):
        mcp_tool_call = ToolCall(
            id="call-mcp-example-1",
            name=mcp_tool_name,
            arguments={"code": "import platform; print(f'Hello from MCP on {platform.system()}!')"}
        )
        logger.info(f"Executing MCP tool '{mcp_tool_name}' with args: {mcp_tool_call.arguments}")
        try:
            start_time = asyncio.get_event_loop().time()
            mcp_result: ToolResult = await tool_manager.execute_tool(mcp_tool_call)
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(f"MCP Tool Result (took {duration:.2f} ms):")
            logger.info(f"  Success: {mcp_result.success}")
            if mcp_result.success:
                logger.info(f"  Result: {mcp_result.result}")
            else:
                logger.error(f"  Error: {mcp_result.error}")

        except Exception as e:
            logger.error(f"Unexpected error executing MCP tool '{mcp_tool_name}': {e}", exc_info=True)
            logger.error("Ensure the MCP server definition in src/config/mcp.yml is correct and the server script exists/runs.")
            logger.error("Check MCP server logs for more details if the connection was established but failed.")
    else:
         logger.warning(f"MCP tool '{mcp_tool_name}' not found in configuration. Skipping MCP execution example.")

    # --- Cleanup ---
    # Important: Clean up MCP client sessions if any were started
    logger.info("\n--- Closing MCP Client Sessions ---")
    try:
        # Access the manager created by ToolManager if needed, or ToolManager handles it internally now.
        # Check ToolManager.close_mcp_sessions - it does exist and calls mcp_client_manager.close_all_clients()
        # Correct way: Access the MCPClientManager instance via the ToolManager
        if hasattr(tool_manager, 'mcp_client_manager') and tool_manager.mcp_client_manager:
            await tool_manager.mcp_client_manager.close_all_clients()
            logger.info("MCP sessions closed.")
        else:
            logger.info("No MCPClientManager found or needed to close.")
    except Exception as e:
        logger.error(f"Error closing MCP sessions: {e}", exc_info=True)
        logger.error(f"Error closing MCP sessions: {e}")

if __name__ == "__main__":
    logger.info("Running Tool Execution Example...")
    asyncio.run(main())
    logger.info("Example finished.")