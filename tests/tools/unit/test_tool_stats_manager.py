import json
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

from src.tools.tool_stats_manager import ToolStatsManager
from src.config.unified_config import UnifiedConfig
from src.utils.logger import LoggerInterface

class TestToolStatsManager:
    """Test suite for ToolStatsManager."""

    @pytest.fixture
    def mock_logger(self):
        """Mock LoggerInterface."""
        return MagicMock(spec=LoggerInterface)

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Mock UnifiedConfig instance with stats tracking enabled."""
        mock_config = MagicMock(spec=UnifiedConfig)
        stats_file = tmp_path / "test_stats.json"
        mock_config.get_tool_config.return_value = {
            "stats": {
                "track_usage": True,
                "storage_path": str(stats_file)
            }
        }
        return mock_config

    @pytest.fixture
    def stats_manager(self, mock_logger, mock_config):
        """Instance of ToolStatsManager with tracking enabled."""
        with patch.object(ToolStatsManager, 'load_stats') as mock_load:
            manager = ToolStatsManager(
                logger=mock_logger,
                unified_config=mock_config
            )
            # Verify load_stats was called during initialization
            mock_load.assert_called_once()
            return manager

    @pytest.fixture
    def stats_manager_no_tracking(self, mock_logger):
        """Instance of ToolStatsManager with tracking disabled."""
        mock_config = MagicMock(spec=UnifiedConfig)
        mock_config.get_tool_config.return_value = {
            "stats": {
                "track_usage": False
            }
        }
        return ToolStatsManager(
            logger=mock_logger,
            unified_config=mock_config
        )

    @pytest.fixture
    def stats_manager_no_path(self, mock_logger):
        """Instance of ToolStatsManager with tracking enabled but no path."""
        mock_config = MagicMock(spec=UnifiedConfig)
        mock_config.get_tool_config.return_value = {
            "stats": {
                "track_usage": True,
                "storage_path": None
            }
        }
        return ToolStatsManager(
            logger=mock_logger,
            unified_config=mock_config
        )

    # Tests for __init__
    def test_init_with_tracking_enabled(self, stats_manager, mock_config):
        """Test initialization with tracking enabled."""
        assert stats_manager._track_usage is True
        assert stats_manager.stats_storage_path == mock_config.get_tool_config().get("stats").get("storage_path")
        
    def test_init_with_tracking_disabled(self, stats_manager_no_tracking):
        """Test initialization with tracking disabled."""
        assert stats_manager_no_tracking._track_usage is False
        
    def test_init_with_no_path(self, stats_manager_no_path):
        """Test initialization with no path specified."""
        assert stats_manager_no_path._track_usage is True
        assert stats_manager_no_path.stats_storage_path is None

    # Tests for update_stats
    def test_update_stats_new_tool(self, stats_manager):
        """Test updating stats for a new tool (success)."""
        tool_name = "new_tool"
        
        stats_manager.update_stats(tool_name, True, 500)
        
        assert tool_name in stats_manager.usage_stats
        assert stats_manager.usage_stats[tool_name]["uses"] == 1
        assert stats_manager.usage_stats[tool_name]["successes"] == 1
        assert stats_manager.usage_stats[tool_name]["failures"] == 0
        assert "first_used" in stats_manager.usage_stats[tool_name]
        assert "last_used" in stats_manager.usage_stats[tool_name]
        assert stats_manager.usage_stats[tool_name]["total_duration_ms"] == 500
        assert stats_manager.usage_stats[tool_name]["avg_duration_ms"] == 500.0

    def test_update_stats_existing_tool(self, stats_manager):
        """Test updating stats for an existing tool (success)."""
        tool_name = "existing_tool"
        # First update
        stats_manager.update_stats(tool_name, True, 500)
        # Second update
        stats_manager.update_stats(tool_name, True, 700)
        
        assert stats_manager.usage_stats[tool_name]["uses"] == 2
        assert stats_manager.usage_stats[tool_name]["successes"] == 2
        assert stats_manager.usage_stats[tool_name]["failures"] == 0
        assert stats_manager.usage_stats[tool_name]["total_duration_ms"] == 1200
        assert stats_manager.usage_stats[tool_name]["avg_duration_ms"] == 600.0

    def test_update_stats_tool_failure(self, stats_manager):
        """Test updating stats for a failure."""
        tool_name = "failing_tool"
        # Update with success=False
        stats_manager.update_stats(tool_name, False, 300)
        
        assert stats_manager.usage_stats[tool_name]["uses"] == 1
        assert stats_manager.usage_stats[tool_name]["successes"] == 0
        assert stats_manager.usage_stats[tool_name]["failures"] == 1
        assert stats_manager.usage_stats[tool_name]["total_duration_ms"] == 0
        assert stats_manager.usage_stats[tool_name]["avg_duration_ms"] == 0.0

    def test_update_stats_avg_duration_calculation(self, stats_manager):
        """Test correct calculation of avg_duration_ms."""
        tool_name = "duration_test"
        # First update
        stats_manager.update_stats(tool_name, True, 100)
        # Second update
        stats_manager.update_stats(tool_name, True, 200)
        # Failure should not affect duration
        stats_manager.update_stats(tool_name, False, 300)
        # Another success
        stats_manager.update_stats(tool_name, True, 300)
        
        assert stats_manager.usage_stats[tool_name]["uses"] == 4
        assert stats_manager.usage_stats[tool_name]["successes"] == 3
        assert stats_manager.usage_stats[tool_name]["failures"] == 1
        assert stats_manager.usage_stats[tool_name]["total_duration_ms"] == 600
        assert stats_manager.usage_stats[tool_name]["avg_duration_ms"] == 200.0

    def test_update_stats_tracking_disabled(self, stats_manager_no_tracking):
        """Test that update_stats does nothing if tracking is disabled."""
        tool_name = "disabled_tracking_tool"
        
        stats_manager_no_tracking.update_stats(tool_name, True, 500)
        
        assert tool_name not in stats_manager_no_tracking.usage_stats

    # Tests for save_stats
    def test_save_stats_success(self, stats_manager, tmp_path):
        """Test successful save to the configured path."""
        tool_name = "save_test"
        stats_manager.update_stats(tool_name, True, 500)
        
        stats_manager.save_stats()
        
        path = stats_manager.stats_storage_path
        assert os.path.exists(path)
        
        with open(path, 'r') as f:
            saved_stats = json.load(f)
            assert tool_name in saved_stats
            assert saved_stats[tool_name]["uses"] == 1
            assert saved_stats[tool_name]["avg_duration_ms"] == 500.0

    def test_save_stats_custom_path(self, stats_manager, tmp_path):
        """Test save to a specific provided path."""
        tool_name = "custom_path_test"
        stats_manager.update_stats(tool_name, True, 500)
        
        custom_path = str(tmp_path / "custom_stats.json")
        stats_manager.save_stats(custom_path)
        
        assert os.path.exists(custom_path)
        
        with open(custom_path, 'r') as f:
            saved_stats = json.load(f)
            assert tool_name in saved_stats

    def test_save_stats_tracking_disabled(self, stats_manager_no_tracking, mock_logger):
        """Test save does nothing if tracking is disabled."""
        stats_manager_no_tracking.save_stats()
        mock_logger.debug.assert_called_with("Skipping saving stats: tracking is disabled.")

    def test_save_stats_no_path(self, stats_manager_no_path, mock_logger):
        """Test save logs error if no path is available."""
        stats_manager_no_path.stats_storage_path = None
        stats_manager_no_path.save_stats()
        mock_logger.error.assert_called_with("Cannot save stats: No storage path specified.")

    def test_save_stats_io_error(self, stats_manager, mock_logger):
        """Test handling IOError during save."""
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = IOError("File error")
            stats_manager.save_stats()
            mock_logger.error.assert_called_once()

    # Tests for load_stats
    def test_load_stats_nonexistent_file(self, stats_manager, mock_logger):
        """Test loading from a non-existent file (should result in empty stats)."""
        non_existent_path = "/non/existent/path.json"
        
        with patch('os.path.exists', return_value=False):
            stats_manager.load_stats(non_existent_path)
            
        assert stats_manager.usage_stats == {}
        mock_logger.info.assert_called()

    def test_load_stats_valid_file(self, stats_manager, tmp_path):
        """Test loading from a valid existing file."""
        # Create test stats data
        test_stats = {
            "test_tool": {
                "uses": 5,
                "successes": 4,
                "failures": 1,
                "first_used": "2023-01-01T12:00:00",
                "last_used": "2023-01-02T12:00:00",
                "total_duration_ms": 2000,
                "avg_duration_ms": 500.0
            }
        }
        
        # Save test data to file
        test_path = tmp_path / "test_load.json"
        with open(test_path, 'w') as f:
            json.dump(test_stats, f)
            
        # Load the file
        stats_manager.load_stats(str(test_path))
        
        assert "test_tool" in stats_manager.usage_stats
        assert stats_manager.usage_stats["test_tool"]["uses"] == 5
        assert stats_manager.usage_stats["test_tool"]["avg_duration_ms"] == 500.0

    def test_load_stats_invalid_json(self, stats_manager, mock_logger, tmp_path):
        """Test loading from a corrupted/invalid JSON file."""
        invalid_path = tmp_path / "invalid.json"
        
        # Create invalid JSON file
        with open(invalid_path, 'w') as f:
            f.write("{ not valid json }")
            
        stats_manager.load_stats(str(invalid_path))
        
        assert stats_manager.usage_stats == {}
        mock_logger.error.assert_called()

    def test_load_stats_io_error(self, stats_manager, mock_logger):
        """Test handling IOError during load."""
        # Looking at the implementation, we need to make sure we're testing the correct error path
        # The exception is caught, but error is logged under a specific message pattern
        with patch('os.path.exists', return_value=True):  # Make it think the file exists
            with patch("builtins.open") as mock_file:
                # Make open throw IOError when called
                mock_file.side_effect = IOError("File error")
                
                # Call load_stats
                stats_manager.load_stats("test_path.json")
                
                # Check that error was logged with the expected message pattern
                # The actual implementation doesn't use exc_info parameter
                mock_logger.error.assert_called_with(
                    "Failed to load or parse tool usage statistics from test_path.json: File error. Starting with empty stats."
                )

    def test_load_stats_invalid_data_type(self, stats_manager, mock_logger, tmp_path):
        """Test loading non-dictionary data."""
        invalid_path = tmp_path / "invalid_type.json"
        
        # Create JSON with wrong type
        with open(invalid_path, 'w') as f:
            json.dump(["list", "not", "dict"], f)
            
        stats_manager.load_stats(str(invalid_path))
        
        # Should not update stats with invalid data
        mock_logger.error.assert_called()

    def test_load_stats_no_path(self, stats_manager_no_path, mock_logger):
        """Test loading does nothing if no path is available."""
        stats_manager_no_path.stats_storage_path = None
        stats_manager_no_path.load_stats()
        mock_logger.warning.assert_called_with("Cannot load stats: No storage path specified.")

    # Tests for get_stats and get_all_stats
    def test_get_stats_existing_tool(self, stats_manager):
        """Test getting stats for an existing tool."""
        tool_name = "get_test"
        stats_manager.update_stats(tool_name, True, 500)
        
        stats = stats_manager.get_stats(tool_name)
        
        assert stats is not None
        assert stats["uses"] == 1
        assert stats["avg_duration_ms"] == 500.0

    def test_get_stats_nonexistent_tool(self, stats_manager):
        """Test getting stats for a non-existent tool (should return None)."""
        stats = stats_manager.get_stats("nonexistent_tool")
        assert stats is None

    def test_get_all_stats_returns_copy(self, stats_manager):
        """Test getting all stats returns a copy of the internal dictionary."""
        tool_name = "copy_test"
        stats_manager.update_stats(tool_name, True, 500)
        
        all_stats = stats_manager.get_all_stats()
        
        # Verify it's a copy
        assert all_stats is not stats_manager.usage_stats
        
        # The implementation returns a shallow copy (dict.copy()), not a deep copy
        # So we can only verify that the top-level dictionaries are different objects
        assert isinstance(all_stats, dict)
        assert tool_name in all_stats
        assert all_stats[tool_name]["uses"] == 1