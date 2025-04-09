#!/usr/bin/env python
"""
Example script demonstrating that tools from tools.yml can be properly loaded and used.
This example shows how to:
  1. Initialize the ToolRegistry and ToolManager
  2. Load tools from the configuration
  3. Check available tools
  4. Execute tools directly
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tool_test_example")

from src.config import configure
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_manager import ToolManager
from src.tools.tool_stats_manager import ToolStatsManager

def main():
    """Run the example to test tools from configuration."""
    print("\n=== Tool Test Example ===")
    
    # Configure the framework with default settings
    configure()
    
    # Create the tool registry and manager
    print("\nInitializing ToolRegistry and ToolManager...")
    tool_registry = ToolRegistry(logger=logger)
    # Pass the tool registry to the ToolManager so they share the same registry
    tool_manager = ToolManager(logger=logger, tool_registry=tool_registry)
    
    # Get and display available tools
    tool_definitions = tool_registry.get_all_tool_definitions()
    print(f"\nFound {len(tool_definitions)} tools in the configuration:")
    
    for name, definition in tool_definitions.items():
        print(f"  - {name}: {definition.description}")
    
    # Try executing some tools if they exist
    if "calculator" in tool_definitions:
        print("\nExecuting calculator tool...")
        result = tool_manager.execute_tool("calculator", expression="2 + 2 * 3")
        print(f"Calculator result: {result.result}")
    
    if "get_current_datetime" in tool_definitions:
        print("\nExecuting datetime tool...")
        result = tool_manager.execute_tool("get_current_datetime")
        print(f"Current datetime: {result.result}")
    
    if "dummy_tool" in tool_definitions:
        print("\nExecuting dummy tool...")
        result = tool_manager.execute_tool("dummy_tool", query="Hello, dummy tool!")
        print(f"Dummy tool result: {result.result}")
    
    # Get tool usage statistics
    print("\nChecking tool statistics...")
    stats_manager = ToolStatsManager(logger=logger)
    all_stats = stats_manager.get_all_stats()
    
    if all_stats:
        print("Tool usage statistics:")
        for tool_name, stats in all_stats.items():
            print(f"  - {tool_name}: {stats['uses']} uses, {stats['successes']} successes")
    else:
        print("No tool usage statistics found.")
    
    # Save the statistics
    stats_manager.save_stats()
    print(f"\nStatistics saved to: {stats_manager.stats_storage_path}")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main() 