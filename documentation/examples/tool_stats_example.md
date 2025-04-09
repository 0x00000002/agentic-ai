# Tool Usage Statistics Example

This example demonstrates how to use the `ToolStatsManager` to track, analyze, and persist tool usage statistics.

## Overview

The `ToolStatsManager` allows you to:

- Track successful and failed tool executions
- Record and calculate performance metrics (execution duration)
- Save statistics to a JSON file for later analysis
- Load previously saved statistics
- Query statistics for specific tools or all tools

## Basic Usage

### Initialization

```python
from src.tools.tool_stats_manager import ToolStatsManager
from src.utils.logger import LoggerFactory
from src.config import UnifiedConfig

# Initialize with default configuration
stats_manager = ToolStatsManager()

# Or initialize with custom components
stats_manager = ToolStatsManager(
    logger=LoggerFactory.create("my_stats_logger"),
    unified_config=UnifiedConfig.get_instance()
)
```

### Recording Tool Usage

```python
# Record a successful tool execution with duration
stats_manager.update_stats(
    tool_name="weather_tool",
    success=True,
    duration_ms=250  # milliseconds
)

# Record a failed tool execution
stats_manager.update_stats(
    tool_name="search_tool",
    success=False
)

# You can optionally include a request ID for cross-referencing
stats_manager.update_stats(
    tool_name="calculator_tool",
    success=True,
    duration_ms=50,
    request_id="req-12345"
)
```

### Retrieving Statistics

```python
# Get statistics for a specific tool
weather_stats = stats_manager.get_stats("weather_tool")
if weather_stats:
    print(f"Weather tool:")
    print(f"- Used {weather_stats['uses']} times")
    print(f"- Success rate: {weather_stats['successes']/weather_stats['uses']*100:.1f}%")
    print(f"- Average duration: {weather_stats['avg_duration_ms']:.2f}ms")

# Get statistics for all tools
all_stats = stats_manager.get_all_stats()
for tool_name, stats in all_stats.items():
    print(f"{tool_name}: {stats['uses']} uses, {stats['successes']} successes")
```

### Saving and Loading Statistics

```python
# Save statistics to the default path (from configuration)
stats_manager.save_stats()

# Save to a custom path
stats_manager.save_stats("/path/to/custom_stats.json")

# Load statistics from the default path
stats_manager.load_stats()

# Load from a custom path
stats_manager.load_stats("/path/to/custom_stats.json")
```

## Complete Example

Here's a comprehensive example showing the full workflow:

```python
import time
import random
from src.tools.tool_stats_manager import ToolStatsManager
from src.utils.logger import LoggerFactory

# Create a ToolStatsManager with default configuration
stats_manager = ToolStatsManager(
    logger=LoggerFactory.create("example_logger")
)

# Function to simulate tool executions
def simulate_tool_usage(tool_name, num_calls, success_rate):
    print(f"Simulating {num_calls} calls to {tool_name}...")

    for i in range(num_calls):
        # Simulate success/failure based on success_rate
        success = random.random() < success_rate

        # Simulate execution time (50-500ms)
        duration_ms = int(random.uniform(50, 500))

        # Update statistics
        stats_manager.update_stats(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms
        )

        # Small delay for better timestamps
        time.sleep(0.01)

    print(f"Completed simulation for {tool_name}")

# Simulate usage of different tools
simulate_tool_usage("weather_tool", 5, 0.8)    # 80% success rate
simulate_tool_usage("calculator_tool", 10, 0.9) # 90% success rate
simulate_tool_usage("search_tool", 8, 0.75)     # 75% success rate

# Save statistics
stats_manager.save_stats()
print(f"Statistics saved to: {stats_manager.stats_storage_path}")

# Print statistics for analysis
all_stats = stats_manager.get_all_stats()
print("\nTool Usage Summary:")
print("-" * 60)
print(f"{'Tool Name':<20} {'Uses':<6} {'Success Rate':<15} {'Avg Duration':<15}")
print("-" * 60)
for tool_name, stats in all_stats.items():
    success_rate = stats['successes'] / stats['uses'] * 100 if stats['uses'] > 0 else 0
    print(f"{tool_name:<20} {stats['uses']:<6} {success_rate:>6.1f}%        {stats['avg_duration_ms']:>8.2f}ms")
```

Example output:

```
Simulating 5 calls to weather_tool...
Completed simulation for weather_tool
Simulating 10 calls to calculator_tool...
Completed simulation for calculator_tool
Simulating 8 calls to search_tool...
Completed simulation for search_tool
Statistics saved to: data/tool_stats.json

Tool Usage Summary:
------------------------------------------------------------
Tool Name            Uses   Success Rate     Avg Duration
------------------------------------------------------------
weather_tool         5      80.0%            210.50ms
calculator_tool      10     90.0%            275.78ms
search_tool          8      75.0%            226.33ms
```

## Integration with ToolManager

In practice, you would normally use the `ToolStatsManager` indirectly through the `ToolManager`, which acts as a facade for the entire tools system:

```python
from src.tools.tool_manager import ToolManager

tool_manager = ToolManager()

# Execute a tool - stats tracking happens automatically
result = tool_manager.execute_tool("weather_tool", location="New York")

# Save statistics when needed
tool_manager.save_usage_stats()

# Get tool information including usage statistics
tool_info = tool_manager.get_tool_info("weather_tool")
print(f"Weather tool used {tool_info['usage_stats']['uses']} times")
```

## Configuration

Tool statistics tracking can be configured in the `tools.yml` configuration file:

```yaml
stats:
  track_usage: true # Set to false to disable tracking
  storage_path: "data/tool_stats.json" # Custom storage path
```

For more information, see the [Tool Usage Statistics](/documentation/tools/overview.md#tool-usage-statistics) section in the Tools documentation.
