#!/usr/bin/env python
"""
Example script demonstrating the usage of ToolStatsManager for tracking tool usage statistics.
This example shows how to:
  1. Initialize the ToolStatsManager
  2. Update statistics for tool usage (successes and failures)
  3. Save statistics to a file
  4. Load statistics from a file
  5. Retrieve and analyze tool usage data

Note: When running framework_example.py, you may see a warning:
  "No tool definitions found under 'tools' key in configuration."
This is normal for the example since it's not using tools directly, 
and ToolStatsManager will still work correctly even without tool definitions.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
import json
from pprint import pprint
from datetime import datetime
import time
import random
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ToolStatsManager and related dependencies
from src.tools.tool_stats_manager import ToolStatsManager
from src.utils.logger import LoggerFactory
from src.config.unified_config import UnifiedConfig
from src.tools.models import ToolExecutionStatus

# Set up a simple temporary path for this example
TEMP_STATS_PATH = "examples/temp/example_tool_stats.json"

def ensure_temp_dir():
    """Create the temporary directory if it doesn't exist."""
    os.makedirs(os.path.dirname(TEMP_STATS_PATH), exist_ok=True)
    
def clear_temp_file():
    """Remove the temporary stats file if it exists."""
    if os.path.exists(TEMP_STATS_PATH):
        os.remove(TEMP_STATS_PATH)
        print(f"Removed existing stats file: {TEMP_STATS_PATH}")
        
def simulate_tool_usage(stats_manager, tool_name, num_calls=10, success_rate=0.8):
    """
    Simulate multiple tool usages to demonstrate stats collection.
    
    Args:
        stats_manager: The ToolStatsManager instance
        tool_name: Name of the tool to simulate
        num_calls: Number of calls to simulate
        success_rate: Probability of success for each call
    """
    print(f"\nSimulating {num_calls} calls to {tool_name} (success rate: {success_rate*100}%)...")
    
    for i in range(num_calls):
        # Simulate random success/failure based on success_rate
        success = random.random() < success_rate
        
        # Simulate random execution time between 50-500ms
        duration_ms = int(random.uniform(50, 500))
        
        # Add a small delay to ensure timestamps are different
        time.sleep(0.01)
        
        # Update statistics
        stats_manager.update_stats(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms
        )
        
        status = "SUCCESS" if success else "FAILURE"
        print(f"  Call {i+1}: {status} - Duration: {duration_ms}ms")
    
    print(f"Completed simulation for {tool_name}")

def print_tool_stats(stats_manager, tool_name):
    """Print statistics for a specific tool."""
    stats = stats_manager.get_stats(tool_name)
    
    if stats is None:
        print(f"No statistics available for tool: {tool_name}")
        return
    
    print(f"\nStatistics for tool: {tool_name}")
    print("-" * 40)
    print(f"Total uses:     {stats['uses']}")
    print(f"Successes:      {stats['successes']} ({stats['successes']/stats['uses']*100:.1f}%)")
    print(f"Failures:       {stats['failures']} ({stats['failures']/stats['uses']*100:.1f}%)")
    print(f"First used:     {stats['first_used']}")
    print(f"Last used:      {stats['last_used']}")
    print(f"Avg duration:   {stats['avg_duration_ms']:.2f}ms")
    print(f"Total duration: {stats['total_duration_ms']}ms")
    print("-" * 40)

def create_mock_config():
    """Create a mock configuration object for ToolStatsManager."""
    mock_config = MagicMock()
    # Set up the mock to return our custom stats configuration
    mock_config.get_tool_config.return_value = {
        "stats": {
            "track_usage": True,
            "storage_path": TEMP_STATS_PATH
        }
    }
    return mock_config

def main():
    """Main function demonstrating ToolStatsManager usage."""
    # Ensure our temp directory exists and stats file is cleared
    ensure_temp_dir()
    clear_temp_file()
    
    print("\n=== ToolStatsManager Example ===")
    
    # 1. Initialize ToolStatsManager with custom configuration
    print("\nInitializing ToolStatsManager with custom configuration...")
    
    # Create a logger
    logger = LoggerFactory.create("tool_stats_example")
    
    # Create a mock config
    mock_config = create_mock_config()
    
    # Create a ToolStatsManager instance with our mock config
    stats_manager = ToolStatsManager(
        logger=logger,
        unified_config=mock_config
    )
    
    print(f"ToolStatsManager initialized with stats path: {stats_manager.stats_storage_path}")
    
    # 2. Simulate usage for different tools
    simulate_tool_usage(stats_manager, "weather_tool", num_calls=5, success_rate=0.8)
    simulate_tool_usage(stats_manager, "calculator_tool", num_calls=10, success_rate=0.9)
    simulate_tool_usage(stats_manager, "search_tool", num_calls=8, success_rate=0.75)
    
    # 3. Save statistics to file
    print("\nSaving tool statistics to file...")
    stats_manager.save_stats()
    print(f"Statistics saved to: {stats_manager.stats_storage_path}")
    
    # Display file content
    with open(stats_manager.stats_storage_path, 'r') as f:
        saved_data = json.load(f)
        print("\nSaved JSON content:")
        pprint(saved_data, indent=2, width=100)
    
    # 4. Create a new instance and load statistics
    print("\nCreating a new ToolStatsManager instance and loading saved statistics...")
    new_stats_manager = ToolStatsManager(
        logger=logger,
        unified_config=mock_config
    )
    
    # 5. Retrieve and print statistics for each tool
    print("\n=== Tool Usage Statistics ===")
    
    all_stats = new_stats_manager.get_all_stats()
    for tool_name in all_stats.keys():
        print_tool_stats(new_stats_manager, tool_name)
    
    # 6. Add more usage data to demonstrate updates
    print("\nAdding more usage data for weather_tool...")
    simulate_tool_usage(new_stats_manager, "weather_tool", num_calls=3, success_rate=0.67)
    
    # Print updated stats
    print_tool_stats(new_stats_manager, "weather_tool")
    
    # Save the updated statistics
    print("\nSaving updated statistics...")
    new_stats_manager.save_stats()
    
    print("\n=== Example Complete ===")
    print(f"Tool statistics are saved at: {TEMP_STATS_PATH}")

if __name__ == "__main__":
    main() 