#!/usr/bin/env python
"""
Example demonstrating the use of RAG (Retrieval-Augmented Generation) 
P tools provided by an MCP server through the ToolEnabledAI interface.

This script assumes:
1. An MCP server providing 'rag_add_document' and 'rag_query' tools is configured 
   in src/config/mcp.yml (like the 'rag_server' example).
2. The MCP server specified in the config ('http://localhost:8080/mcp' by default 
   for 'rag_server') is running and accessible.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# --- Setup Path --- 
# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['AGENTIC_AI_APP_ROOT'] = str(project_root)

# --- Imports --- 
from src.config import configure, get_config, UseCasePreset
from src.core.tool_enabled_ai import ToolEnabledAI
from src.tools.tool_manager import ToolManager
from src.utils.logger import LoggerFactory

# --- Logging --- 
logger = LoggerFactory.create("rag_mcp_example", use_real_logger=True)


async def main():
    logger.info("--- Initializing RAG MCP Example ---")

    try:
        # Configure the framework (adjust model if needed)
        # Using a tool-capable model is important here.
        logger.info("Configuring framework...")
        configure(
            model="claude-3-5-sonnet", 
            use_case=UseCasePreset.CODING, # Use case might influence default prompts/tool availability
            show_thinking=True
        )
        config = get_config()

        # Initialize ToolManager - this loads tools from tools.yml and mcp.yml
        logger.info("Initializing ToolManager...")
        tool_manager = ToolManager(unified_config=config, logger=logger)
        
        # Verify the RAG tools are loaded (optional check)
        rag_query_tool = tool_manager.get_tool_definition("rag_query")
        rag_add_tool = tool_manager.get_tool_definition("rag_add_document")
        if not rag_query_tool or not rag_add_tool:
            logger.error("RAG tools ('rag_query', 'rag_add_document') not found.")
            logger.error("Ensure 'rag_server' is correctly defined in src/config/mcp.yml")
            return
        logger.info(f"Found RAG tools provided by MCP server: {rag_query_tool.mcp_server_name}")

        # Create ToolEnabledAI instance, passing the ToolManager
        logger.info("Creating ToolEnabledAI instance...")
        ai = ToolEnabledAI(
            model=config.get_default_model(),
            # Provide a system prompt that encourages using the knowledge base
            system_prompt="You are an assistant capable of managing and querying a knowledge base. Use your tools to add information and answer questions based on it.",
            logger=logger,
            tool_manager=tool_manager # Crucial step to enable tools
        )

        # 1. Add a document to the knowledge base via AI request
        logger.info("--- Adding Document via AI Request ---")
        add_request = "Please add this information to the knowledge base: 'The user's name is Alex and he is a blockchain engineer'"
        logger.info(f"Sending request: {add_request}")
        
        start_time = asyncio.get_event_loop().time()
        # The AI should interpret this and call the 'rag_add_document' tool
        add_response = await ai.request(add_request)
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(f"Received response in {duration:.2f} ms.")
        print(f"\n[Add Document Response]\n{add_response}\n")

        # Allow some time for the document to be potentially processed by the RAG server
        await asyncio.sleep(1) 

        # 2. Query the knowledge base via AI request
        logger.info("--- Querying Knowledge Base via AI Request ---")
        query_request = "Tell me something about the user"
        logger.info(f"Sending request: {query_request}")

        start_time = asyncio.get_event_loop().time()
        # The AI should interpret this and call the 'rag_query' tool
        query_response = await ai.request(query_request)
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(f"Received response in {duration:.2f} ms.")
        print(f"\n[Query Response]\n{query_response}\n")

    except Exception as e:
        logger.error(f"An error occurred during the RAG example: {e}", exc_info=True)
        logger.error("Ensure the MCP server for RAG tools is running and accessible at the URL defined in mcp.yml.")

    finally:
        # Important: Clean up MCP client sessions if ToolManager was used
        if 'tool_manager' in locals() and tool_manager and hasattr(tool_manager, 'close_mcp_sessions'):
            logger.info("--- Closing MCP Client Sessions ---")
            await tool_manager.close_mcp_sessions()
            logger.info("MCP sessions closed.")

if __name__ == "__main__":
    logger.info("Running RAG MCP Usage Example...")
    asyncio.run(main())
    logger.info("Example finished.") 