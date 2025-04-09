"""
Manages usage statistics for tools.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from ..utils.logger import LoggerInterface, LoggerFactory
from ..config import UnifiedConfig

class ToolStatsManager:
    """Handles loading, updating, and saving tool usage statistics."""

    DEFAULT_STATS_PATH = "data/tool_stats.json"

    def __init__(self, logger: Optional[LoggerInterface] = None, 
                 unified_config: Optional[UnifiedConfig] = None):
        """
        Initialize the ToolStatsManager.

        Args:
            logger: Optional logger instance.
            unified_config: Optional UnifiedConfig instance.
        """
        self._logger = logger or LoggerFactory.create("tool_stats_manager")
        self._config = unified_config or UnifiedConfig.get_instance()
        self.usage_stats: Dict[str, Dict[str, Any]] = {}

        # Determine storage path from config
        tool_config = self._config.get_tool_config() or {}
        stats_config = tool_config.get("stats", {})
        self.stats_storage_path = stats_config.get("storage_path", self.DEFAULT_STATS_PATH)
        self._track_usage = stats_config.get("track_usage", True)

        self._logger.info(f"ToolStatsManager initialized. Tracking enabled: {self._track_usage}. Path: '{self.stats_storage_path}'")

        # Attempt to load existing stats if tracking enabled
        if self._track_usage:
            self.load_stats()

    def update_stats(self, 
                     tool_name: str, 
                     success: bool, 
                     duration_ms: Optional[int] = None, 
                     request_id: Optional[str] = None) -> None:
        """
        Update the usage statistics for a specific tool.

        Args:
            tool_name: The name of the tool that was used.
            success: Boolean indicating if the execution was successful.
            duration_ms: Optional execution duration in milliseconds.
            request_id: Optional request ID associated with the execution.
        """
        if not self._track_usage:
            return # Don't update if tracking is disabled

        if tool_name not in self.usage_stats:
            # Initialize stats for a tool if it's the first time seeing it
            # This might happen if a tool was added after initial load
            self.usage_stats[tool_name] = {
                "uses": 0,
                "successes": 0,
                "failures": 0, # Adding failures explicitly
                "last_used": None,
                "first_used": datetime.now().isoformat(),
                "total_duration_ms": 0,
                "avg_duration_ms": 0.0
            }
            self._logger.info(f"Initialized usage stats for new tool: {tool_name}")

        stats = self.usage_stats[tool_name]
        stats["uses"] += 1
        stats["last_used"] = datetime.now().isoformat()

        if success:
            stats["successes"] += 1
            if duration_ms is not None:
                 stats["total_duration_ms"] = stats.get("total_duration_ms", 0) + duration_ms
                 # Recalculate average duration on success
                 stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["successes"] if stats["successes"] > 0 else 0.0
        else:
            stats["failures"] = stats.get("failures", 0) + 1

        # Optionally log request_id if needed for detailed tracing
        # self._logger.debug(f"Updated stats for {tool_name} (Request: {request_id})")

    def save_stats(self, file_path: Optional[str] = None) -> None:
        """
        Save current usage statistics to a JSON file.

        Args:
            file_path: Optional path to save the stats. Uses configured path if None.
        """
        if not self._track_usage:
            self._logger.debug("Skipping saving stats: tracking is disabled.")
            return
            
        path = file_path or self.stats_storage_path
        if not path:
             self._logger.error("Cannot save stats: No storage path specified.")
             return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.usage_stats, f, indent=4)
            self._logger.info(f"Tool usage statistics saved to {path}")

        except IOError as e:
            self._logger.error(f"Failed to save tool usage statistics to {path}: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving tool stats: {e}", exc_info=True)

    def load_stats(self, file_path: Optional[str] = None) -> None:
        """
        Load usage statistics from a JSON file.

        Args:
            file_path: Optional path to load stats from. Uses configured path if None.
        """
        path = file_path or self.stats_storage_path
        if not path:
             self._logger.warning("Cannot load stats: No storage path specified.")
             return
             
        if not os.path.exists(path):
            self._logger.info(f"Tool usage statistics file not found at {path}. Starting with empty stats.")
            self.usage_stats = {}
            return

        try:
            with open(path, 'r') as f:
                loaded_stats = json.load(f)
                # Basic validation: check if it's a dictionary
                if isinstance(loaded_stats, dict):
                    self.usage_stats = loaded_stats
                    self._logger.info(f"Tool usage statistics loaded from {path}")
                else:
                    self._logger.error(f"Failed to load stats: Expected a dictionary in {path}, got {type(loaded_stats)}. Keeping existing stats.")

        except (IOError, json.JSONDecodeError) as e:
            self._logger.error(f"Failed to load or parse tool usage statistics from {path}: {e}. Starting with empty stats.")
            self.usage_stats = {} # Reset on load error
        except Exception as e:
             self._logger.error(f"Unexpected error loading tool stats: {e}", exc_info=True)
             self.usage_stats = {} # Reset on load error

    def get_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            A dictionary containing the stats, or None if the tool has no stats.
        """
        return self.usage_stats.get(tool_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for all tools.

        Returns:
            A dictionary mapping tool names to their statistics.
        """
        return self.usage_stats.copy() # Return a copy to prevent external modification 