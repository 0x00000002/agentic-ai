import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import os
import json

# Modules to test
from src.prompts.prompt_manager import PromptManager

# Dependencies to mock
from src.prompts.prompt_template import PromptTemplate
from src.prompts.prompt_version import PromptVersion
from src.prompts.metrics import PromptMetrics
from src.utils.logger import LoggerInterface, LoggerFactory


class TestPromptManager:
    """Unit tests for the PromptManager class."""

    @patch('src.prompts.prompt_manager.LoggerFactory')
    @patch('src.prompts.prompt_manager.PromptMetrics')
    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('src.prompts.prompt_manager.os.makedirs')
    @patch.object(PromptManager, '_load_data') # Patch internal method
    def test_init_defaults(self, mock_load_data, mock_makedirs, mock_exists, mock_PromptMetrics, mock_LoggerFactory):
        """Test PromptManager initialization with default arguments."""
        # Arrange
        mock_logger_instance = MagicMock(spec=LoggerInterface)
        mock_LoggerFactory.create.return_value = mock_logger_instance
        mock_metrics_instance = MagicMock(spec=PromptMetrics)
        mock_PromptMetrics.return_value = mock_metrics_instance
        
        # Act
        manager = PromptManager()

        # Assert
        assert manager._storage_dir is None
        assert manager._metrics == mock_metrics_instance
        mock_PromptMetrics.assert_called_once_with() # Called with no args
        assert manager._logger == mock_logger_instance
        mock_LoggerFactory.create.assert_called_once_with() # Called with no args
        assert manager._auto_save is True
        assert manager._templates == {}
        assert manager._versions == {}
        assert manager._active_versions == {}
        assert manager._test_allocations == {}
        
        mock_exists.assert_not_called()
        mock_makedirs.assert_not_called()
        mock_load_data.assert_not_called() # No storage_dir, so shouldn't load

    @patch('src.prompts.prompt_manager.LoggerFactory')
    @patch('src.prompts.prompt_manager.PromptMetrics')
    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('src.prompts.prompt_manager.os.makedirs')
    @patch.object(PromptManager, '_load_data')
    def test_init_with_new_storage_dir(self, mock_load_data, mock_makedirs, mock_exists, mock_PromptMetrics, mock_LoggerFactory, tmp_path: Path):
        """Test __init__ when storage_dir is provided and doesn't exist."""
        # Arrange
        storage_dir = str(tmp_path / "prompts")
        mock_exists.return_value = False # Directory does not exist
        mock_logger_instance = MagicMock(spec=LoggerInterface)
        mock_LoggerFactory.create.return_value = mock_logger_instance
        mock_metrics_instance = MagicMock(spec=PromptMetrics)
        mock_PromptMetrics.return_value = mock_metrics_instance
        
        # Act
        manager = PromptManager(storage_dir=storage_dir, auto_save=False)

        # Assert
        assert manager._storage_dir == storage_dir
        assert manager._metrics == mock_metrics_instance
        assert manager._logger == mock_logger_instance
        assert manager._auto_save is False
        
        mock_exists.assert_called_once_with(storage_dir)
        mock_makedirs.assert_called_once_with(storage_dir)
        mock_load_data.assert_called_once_with() # Called because storage_dir exists

    @patch('src.prompts.prompt_manager.LoggerFactory')
    @patch('src.prompts.prompt_manager.PromptMetrics')
    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('src.prompts.prompt_manager.os.makedirs')
    @patch.object(PromptManager, '_load_data')
    def test_init_with_existing_storage_dir(self, mock_load_data, mock_makedirs, mock_exists, mock_PromptMetrics, mock_LoggerFactory, tmp_path: Path):
        """Test __init__ when storage_dir is provided and already exists."""
         # Arrange
        storage_dir = str(tmp_path / "prompts")
        mock_exists.return_value = True # Directory *does* exist
        mock_logger_instance = MagicMock(spec=LoggerInterface)
        mock_LoggerFactory.create.return_value = mock_logger_instance
        mock_metrics_instance = MagicMock(spec=PromptMetrics)
        mock_PromptMetrics.return_value = mock_metrics_instance
        
        # Act
        manager = PromptManager(storage_dir=storage_dir, auto_save=True)

        # Assert
        assert manager._storage_dir == storage_dir
        assert manager._metrics == mock_metrics_instance
        assert manager._logger == mock_logger_instance
        assert manager._auto_save is True
        
        mock_exists.assert_called_once_with(storage_dir)
        mock_makedirs.assert_not_called() # Should not be called if dir exists
        mock_load_data.assert_called_once_with() # Called because storage_dir exists

    @patch.object(PromptManager, '_load_data') # Still need to patch load
    def test_init_with_provided_deps(self, mock_load_data):
        """Test __init__ when metrics and logger are provided."""
        # Arrange
        mock_logger = MagicMock(spec=LoggerInterface)
        mock_metrics = MagicMock(spec=PromptMetrics)
        
        # Act
        manager = PromptManager(metrics=mock_metrics, logger=mock_logger)

        # Assert
        assert manager._storage_dir is None
        assert manager._metrics == mock_metrics # Should use provided instance
        assert manager._logger == mock_logger # Should use provided instance
        assert manager._auto_save is True
        mock_load_data.assert_not_called()

        assert manager._auto_save is True
        mock_load_data.assert_not_called()

    # --- Tests for _save_data --- 

    @patch('src.prompts.prompt_manager.json.dump')
    @patch('builtins.open')
    @patch.object(PromptManager, '_load_data') # Still need to patch load during init
    @patch('src.prompts.prompt_manager.os.path.exists', return_value=True) # Assume dir exists for this test
    def test_save_data_success(self, mock_exists, mock_load_data, mock_open, mock_json_dump, tmp_path: Path):
        """Test _save_data successfully saves templates, versions, and metrics."""
        # Arrange
        storage_dir = str(tmp_path / "save_test")
        mock_logger = MagicMock(spec=LoggerInterface)
        mock_metrics = MagicMock(spec=PromptMetrics)
        
        manager = PromptManager(storage_dir=storage_dir, logger=mock_logger, metrics=mock_metrics)
        
        # Populate with some data
        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"name": "Test Template"}
        manager._templates = {"t1": mock_template}
        
        mock_version = MagicMock()
        mock_version.to_dict.return_value = {"version": "v1"}
        manager._versions = {"t1": [mock_version]}
        
        # Mock file handling
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        # Act
        manager._save_data()

        # Assert
        # Remove checks for templates.json
        # expected_templates_file = os.path.join(storage_dir, "templates.json")
        expected_versions_file = os.path.join(storage_dir, "versions.json")
        expected_metrics_file = os.path.join(storage_dir, "metrics.json")
        
        # Check open calls (only versions.json should be opened)
        # open_calls = [
        #     call(expected_templates_file, 'w'),
        #     call(expected_versions_file, 'w')
        # ]
        # mock_open.assert_has_calls(open_calls, any_order=True)
        mock_open.assert_called_once_with(expected_versions_file, 'w')
        
        # Check json.dump calls (only versions.json data)
        # expected_templates_data = {"t1": {"name": "Test Template"}}
        expected_versions_data = {"t1": [{"version": "v1"}]}
        # dump_calls = [
        #     call(expected_templates_data, mock_file_handle, indent=2),
        #     call(expected_versions_data, mock_file_handle, indent=2)
        # ]
        # assert call(expected_templates_data, mock_open.return_value.__enter__.return_value, indent=2) in mock_json_dump.call_args_list
        # assert call(expected_versions_data, mock_open.return_value.__enter__.return_value, indent=2) in mock_json_dump.call_args_list
        # assert mock_json_dump.call_count == 2
        mock_json_dump.assert_called_once_with(expected_versions_data, mock_file_handle, indent=2)

        # Check metrics save call
        mock_metrics.save_to_file.assert_called_once_with(expected_metrics_file)
        
        # Check logging (update message)
        # mock_logger.info.assert_called_with(f"Saved prompt data to {storage_dir}")
        mock_logger.info.assert_called_with(f"Saved prompt versions and metrics to {storage_dir}")

    @patch('src.prompts.prompt_manager.json.dump')
    @patch('builtins.open')
    @patch('src.prompts.prompt_manager.PromptMetrics') # Mock metrics creation
    def test_save_data_no_storage_dir(self, mock_PromptMetrics, mock_open, mock_json_dump):
        """Test _save_data does nothing if storage_dir is None."""
        # Arrange
        manager = PromptManager() # No storage_dir
        manager._metrics = MagicMock() # Assign a mock metrics to check save call
        
        # Act
        manager._save_data()
        
        # Assert
        mock_open.assert_not_called()
        mock_json_dump.assert_not_called()
        manager._metrics.save_to_file.assert_not_called()

    # --- Tests for _load_data --- 

    @patch('src.prompts.prompt_manager.PromptMetrics.load_from_file')
    @patch('src.prompts.prompt_manager.PromptVersion.from_dict')
    @patch('src.prompts.prompt_manager.json.load')
    @patch('builtins.open')
    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('src.prompts.prompt_manager.os.makedirs') # Patch makedirs for init
    def test_load_data_success(self,
                               mock_makedirs,
                               mock_os_path_exists,
                               mock_open,
                               mock_json_load,
                               mock_version_from_dict,
                               mock_metrics_load,
                               tmp_path: Path):
        """Test _load_data successfully loads VERSIONS and METRICS from existing files during init."""
        # Arrange
        storage_dir = str(tmp_path / "load_test")
        # Don't create dir here, let PromptManager potentially do it (mocked)
        # os.makedirs(storage_dir) 

        versions_file = os.path.join(storage_dir, "versions.json")
        metrics_file = os.path.join(storage_dir, "metrics.json")

        # Simulate files existing AND the storage directory itself
        mock_os_path_exists.side_effect = lambda path: path in [storage_dir, versions_file, metrics_file]
        
        # Mock data to be loaded
        version_data_active = {"version": "v1", "is_active": True}
        version_data_inactive = {"version": "v0", "is_active": False}
        versions_data = {"t1": [version_data_active, version_data_inactive]}
        metrics_data = {"some_metric": 1}
        
        # Configure json.load mock 
        def json_load_side_effect(fp):
            file_path = fp.name 
            if file_path == versions_file:
                return versions_data
            # Include handling for metrics file if PromptMetrics.load_from_file uses json.load
            elif file_path == metrics_file: 
                 return metrics_data
            raise FileNotFoundError(f"Unexpected file for json.load: {file_path}")
        mock_json_load.side_effect = json_load_side_effect

        # Mock file handle for open context manager
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value = mock_file_handle
        def open_side_effect(path, mode):
            # Need to handle potential open call for metrics file if PromptMetrics does it
            mock_file_handle.name = path 
            return mock_file_handle
        mock_open.side_effect = open_side_effect
        
        # Mock return objects from factory methods
        mock_version_obj_active = MagicMock(is_active=True, version="v1")
        mock_version_obj_inactive = MagicMock(is_active=False, version="v0")
        mock_version_from_dict.side_effect = [mock_version_obj_active, mock_version_obj_inactive]
        mock_metrics_obj = MagicMock()
        mock_metrics_load.return_value = mock_metrics_obj
        mock_logger = MagicMock(spec=LoggerInterface)

        # Act
        # Initialize manager - REAL _load_data is called implicitly by __init__
        manager = PromptManager(storage_dir=storage_dir, logger=mock_logger)
            
        # Assert
        # Check os.path.exists calls
        templates_file_path_for_check = os.path.join(storage_dir, "templates.json")
        # Should be called for dir itself (in __init__), versions, and metrics (in _load_data)
        exists_calls = [call(storage_dir), call(versions_file), call(metrics_file)] 
        mock_os_path_exists.assert_has_calls(exists_calls, any_order=True)
        # Verify templates.json was NOT checked
        assert call(templates_file_path_for_check) not in mock_os_path_exists.call_args_list
        
        # Check open calls (only versions file should be opened by _load_data)
        # NOTE: If PromptMetrics.load_from_file opens metrics.json, this needs adjustment
        open_calls = [call(versions_file, 'r')] 
        mock_open.assert_has_calls(open_calls, any_order=True) # Check versions file was opened
        assert call(metrics_file, 'r') not in mock_open.call_args_list # Assert _load_data didn't open metrics
        
        # Check json.load calls (expect 1 for versions.json by _load_data,
        # potentially 1 inside PromptMetrics.load_from_file)
        assert mock_json_load.call_count >= 1 # Allow 1 or 2 calls
        # Check the call specifically for the versions file handle
        mock_json_load.assert_any_call(mock_file_handle)

        # Check factory method calls
        version_calls = [call(version_data_active), call(version_data_inactive)]
        mock_version_from_dict.assert_has_calls(version_calls)
        # This was called once implicitly during __init__ -> _load_data
        mock_metrics_load.assert_called_once_with(metrics_file) 

        # Check internal state 
        assert manager._templates == {} 
        assert manager._versions == {"t1": [mock_version_obj_active, mock_version_obj_inactive]}
        assert manager._active_versions == {"t1": mock_version_obj_active}
        assert manager._metrics == mock_metrics_obj
        # _load_data logs info, __init__ might log info too
        mock_logger.info.assert_any_call(f"Loaded prompt versions and metrics from {storage_dir}")

    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('builtins.open')
    @patch('src.prompts.prompt_manager.PromptMetrics.load_from_file')
    def test_load_data_files_not_found(self, mock_metrics_load, mock_open, mock_os_path_exists, tmp_path: Path):
        """Test _load_data handles non-existent files gracefully."""
        # Arrange
        storage_dir = str(tmp_path / "load_missing")
        # Ensure os.path.exists returns False for files, maybe True for dir during init
        mock_os_path_exists.side_effect = lambda p: p == storage_dir # Only dir exists
        mock_logger = MagicMock(spec=LoggerInterface)

        # Initialise Manager (patching os.makedirs during init)
        with patch('src.prompts.prompt_manager.os.makedirs') as init_makedirs:
             manager = PromptManager(storage_dir=storage_dir, logger=mock_logger)
             init_makedirs.assert_not_called() # Dir exists

        # Reset mock side effect for the actual load call
        mock_os_path_exists.side_effect = lambda p: False # Now files don't exist

        # Act
        manager._load_data()

        # Assert
        templates_file = os.path.join(storage_dir, "templates.json")
        versions_file = os.path.join(storage_dir, "versions.json")
        metrics_file = os.path.join(storage_dir, "metrics.json")
        # Remove templates.json call from expected list
        exists_calls = [call(versions_file), call(metrics_file)]
        mock_os_path_exists.assert_has_calls(exists_calls, any_order=True)
        # Verify templates.json was NOT checked (during the _load_data call phase)
        # Note: it might have been checked during __init__ depending on side_effect, 
        # but the important part is it's not checked within the _load_data logic path here.
        # Check calls *after* the side_effect was reset
        calls_during_load = [c for c in mock_os_path_exists.call_args_list if c != call(storage_dir)] # Filter out potential init call
        assert call(templates_file) not in calls_during_load

        mock_open.assert_not_called()
        mock_metrics_load.assert_not_called()
        assert manager._templates == {}
        assert manager._versions == {}
        assert manager._active_versions == {}
        assert isinstance(manager._metrics, PromptMetrics)
        mock_logger.info.assert_called_with(f"Loaded prompt versions and metrics from {storage_dir}")

    @patch('src.prompts.prompt_manager.os.path.exists')
    @patch('builtins.open')
    @patch('src.prompts.prompt_manager.PromptMetrics.load_from_file')
    def test_load_data_no_storage_dir(self, mock_metrics_load, mock_open, mock_os_path_exists):
        """Test _load_data does nothing if storage_dir is None."""
        # Arrange
        with patch('src.prompts.prompt_manager.LoggerFactory'), \
             patch('src.prompts.prompt_manager.PromptMetrics'):
            manager = PromptManager(storage_dir=None)
        
        # Act
        manager._load_data()
        
        # Assert
        mock_os_path_exists.assert_not_called()
        mock_open.assert_not_called()
        mock_metrics_load.assert_not_called()

    # --- Tests for create_template ---

    @patch('src.prompts.prompt_manager.PromptVersion.create_new_version')
    @patch('src.prompts.prompt_manager.PromptTemplate')
    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data') # Patch load during init
    def test_create_template_success_autosave_on(self, mock_load_data, mock_save_data, mock_PromptTemplate, mock_create_new_version):
        """Test successful template creation with auto_save enabled (default)."""
        # Arrange
        mock_logger = MagicMock(spec=LoggerInterface)
        manager = PromptManager(logger=mock_logger, auto_save=True) # Explicitly True

        template_name = "Test Template"
        template_desc = "A test description"
        template_str = "Hello {{name}}"
        default_vals = {"name": "World"}

        mock_pt_instance = MagicMock()
        mock_pt_instance.template_id = "t-123"
        mock_pt_instance.name = template_name # For logging
        mock_PromptTemplate.return_value = mock_pt_instance

        mock_pv_instance = MagicMock()
        mock_pv_instance.version = "v-abc"
        mock_create_new_version.return_value = mock_pv_instance

        # Act
        template_id = manager.create_template(
            name=template_name,
            description=template_desc,
            template=template_str,
            default_values=default_vals
        )

        # Assert
        assert template_id == "t-123"
        mock_PromptTemplate.assert_called_once_with(
            name=template_name,
            description=template_desc,
            template=template_str,
            default_values=default_vals
        )
        mock_create_new_version.assert_called_once_with(
            template_id="t-123",
            previous_version=None,
            content={
                "template": template_str,
                "default_values": default_vals
            },
            name="Initial Version",
            description="Initial version of the template"
        )
        
        # Check internal state
        assert manager._templates == {"t-123": mock_pt_instance}
        assert manager._versions == {"t-123": [mock_pv_instance]}
        assert manager._active_versions == {"t-123": mock_pv_instance}
        assert mock_pv_instance.is_active is True # Should be set to active

        # Check logging
        mock_logger.info.assert_called_with(f"Created template: {template_name} ({template_id})")

        # Check auto-save
        mock_save_data.assert_called_once_with() 

    @patch('src.prompts.prompt_manager.PromptVersion.create_new_version')
    @patch('src.prompts.prompt_manager.PromptTemplate')
    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data') # Patch load during init
    def test_create_template_success_autosave_off(self, mock_load_data, mock_save_data, mock_PromptTemplate, mock_create_new_version):
        """Test successful template creation with auto_save disabled."""
        # Arrange
        mock_logger = MagicMock(spec=LoggerInterface)
        manager = PromptManager(logger=mock_logger, auto_save=False) # Explicitly False

        template_name = "Test Template 2"
        template_desc = "Another test description"
        template_str = "Hi {{user}}"
        
        mock_pt_instance = MagicMock()
        mock_pt_instance.template_id = "t-456"
        mock_pt_instance.name = template_name
        mock_PromptTemplate.return_value = mock_pt_instance
        mock_pv_instance = MagicMock()
        mock_create_new_version.return_value = mock_pv_instance

        # Act
        template_id = manager.create_template(
            name=template_name,
            description=template_desc,
            template=template_str
            # default_values=None # Test optional arg
        )

        # Assert
        assert template_id == "t-456"
        mock_PromptTemplate.assert_called_once_with(
            name=template_name,
            description=template_desc,
            template=template_str,
            default_values=None # Check default arg
        )
        mock_create_new_version.assert_called_once_with(
            template_id="t-456",
            previous_version=None,
            content={
                "template": template_str,
                "default_values": {} # Check default content
            },
            name="Initial Version",
            description="Initial version of the template"
        )
        
        # Check internal state (abbreviated checks)
        assert "t-456" in manager._templates
        assert manager._versions["t-456"] == [mock_pv_instance]
        assert manager._active_versions["t-456"] == mock_pv_instance
        assert mock_pv_instance.is_active is True

        # Check logging
        mock_logger.info.assert_called_once()

        # Check NO auto-save
        mock_save_data.assert_not_called()

    # --- Tests for get_template ---
    
    @patch.object(PromptManager, '_load_data')
    def test_get_template_found(self, mock_load_data):
        """Test get_template returns the correct template object."""
        # Arrange
        manager = PromptManager()
        mock_template = MagicMock()
        manager._templates = {"t1": mock_template}
        
        # Act
        result = manager.get_template("t1")
        
        # Assert
        assert result == mock_template

    @patch.object(PromptManager, '_load_data')
    def test_get_template_not_found(self, mock_load_data):
        """Test get_template returns None for a non-existent ID."""
        # Arrange
        manager = PromptManager()
        manager._templates = {"t1": MagicMock()} # Add some other template
        
        # Act
        result = manager.get_template("t-not-exist")
        
        # Assert
        assert result is None

    # --- Tests for list_templates ---

    @patch.object(PromptManager, '_load_data')
    def test_list_templates_empty(self, mock_load_data):
        """Test list_templates returns empty list when no templates exist."""
        # Arrange
        manager = PromptManager()
        
        # Act
        result = manager.list_templates()
        
        # Assert
        assert result == []

    @patch.object(PromptManager, '_load_data')
    def test_list_templates_multiple(self, mock_load_data):
        """Test list_templates returns correct data for multiple templates."""
        # Arrange
        manager = PromptManager()
        
        # Remove spec to allow adding attributes directly
        mock_template1 = MagicMock() # Removed spec
        mock_template1.template_id = "t1" 
        mock_template1.name = "Template 1" 
        mock_template1.description = "Desc 1"
        mock_template1.created_at = MagicMock(isoformat=lambda: "2023-01-01T10:00:00")
        
        # Remove spec to allow adding attributes directly
        mock_template2 = MagicMock() # Removed spec 
        mock_template2.template_id = "t2" 
        mock_template2.name = "Template 2" 
        mock_template2.description = "Desc 2"
        mock_template2.created_at = MagicMock(isoformat=lambda: "2023-01-02T12:00:00")
        
        manager._templates = {"t1": mock_template1, "t2": mock_template2}
        
        # Remove spec for versions as well for consistency, although not strictly needed for this failure
        mock_version1a = MagicMock() # Removed spec 
        mock_version1a.version = "v1a"
        mock_version1b = MagicMock() # Removed spec
        mock_version1b.version = "v1b" # Active
        mock_version2a = MagicMock() # Removed spec
        mock_version2a.version = "v2a" # Active
        
        manager._versions = {
            "t1": [mock_version1a, mock_version1b],
            "t2": [mock_version2a]
        }
        manager._active_versions = {
            "t1": mock_version1b,
            "t2": mock_version2a
        }

        # Act
        result = manager.list_templates()
        
        # Assert
        assert len(result) == 2
        
        expected_result = [
            {
                "template_id": "t1",
                "name": "Template 1",
                "description": "Desc 1",
                "created_at": "2023-01-01T10:00:00",
                "version_count": 2,
                "active_version": "v1b"
            },
            {
                "template_id": "t2",
                "name": "Template 2",
                "description": "Desc 2",
                "created_at": "2023-01-02T12:00:00",
                "version_count": 1,
                "active_version": "v2a"
            }
        ]
        
        # Compare ignoring order
        assert sorted(result, key=lambda x: x['template_id']) == sorted(expected_result, key=lambda x: x['template_id'])

    # --- Tests for update_template ---

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_update_template_success_autosave_on(self, mock_load_data, mock_save_data):
        """Test updating name and description successfully with auto-save."""
        # Arrange
        manager = PromptManager(auto_save=True)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        mock_template = MagicMock() 
        template_id = "t-update"
        manager._templates = {template_id: mock_template}
        
        new_name = "Updated Name"
        new_desc = "Updated Desc"

        # Act
        result = manager.update_template(template_id, name=new_name, description=new_desc)

        # Assert
        assert result is True
        mock_logger.info.assert_called_once()
        assert f"Metadata update requested for template: {template_id}" in mock_logger.info.call_args[0][0]
        mock_save_data.assert_called_once_with()

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_update_template_partial_update(self, mock_load_data, mock_save_data):
        """Test updating only the name (logs intent, doesn't modify source)."""
        # Arrange
        manager = PromptManager(auto_save=True)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        mock_template = MagicMock()
        template_id = "t-partial"
        manager._templates = {template_id: mock_template}
        new_name = "Partial Update"

        # Act
        result = manager.update_template(template_id, name=new_name)

        # Assert
        assert result is True
        mock_logger.info.assert_called_once()
        assert f"Metadata update requested for template: {template_id}" in mock_logger.info.call_args[0][0]
        mock_save_data.assert_called_once_with()

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_update_template_not_found(self, mock_load_data, mock_save_data):
        """Test updating a non-existent template."""
        # Arrange
        manager = PromptManager(auto_save=True)
        manager._templates = {"t-exists": MagicMock()}
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger

        # Act
        result = manager.update_template("t-not-exists", name="New Name")

        # Assert
        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Attempted to update non-existent template ID: t-not-exists" in mock_logger.warning.call_args[0][0]
        mock_logger.info.assert_not_called()
        mock_save_data.assert_not_called()

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_update_template_success_autosave_off(self, mock_load_data, mock_save_data):
        """Test updating successfully without auto-save (logs intent, doesn't modify source)."""
        # Arrange
        manager = PromptManager(auto_save=False)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        mock_template = MagicMock()
        template_id = "t-update-nosave"
        manager._templates = {template_id: mock_template}
        new_name = "Updated No Save"

        # Act
        result = manager.update_template(template_id, name=new_name)

        # Assert
        assert result is True
        mock_logger.info.assert_called_once()
        assert f"Metadata update requested for template: {template_id}" in mock_logger.info.call_args[0][0]
        mock_save_data.assert_not_called()

    # --- Tests for delete_template ---

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_delete_template_success_autosave_on(self, mock_load_data, mock_save_data):
        """Test deleting an existing template with auto_save."""
        # Arrange
        manager = PromptManager(auto_save=True)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        template_id = "t-delete"
        manager._templates = {template_id: MagicMock()}
        manager._versions = {template_id: [MagicMock()]}
        manager._active_versions = {template_id: MagicMock()}
        # Note: Test allocations might also need clearing if relevant

        # Act
        result = manager.delete_template(template_id)

        # Assert
        assert result is True
        assert template_id not in manager._templates
        assert template_id not in manager._versions
        assert template_id not in manager._active_versions
        mock_logger.info.assert_called_with(f"Deleted template: {template_id}")
        mock_save_data.assert_called_once_with()

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_delete_template_not_found(self, mock_load_data, mock_save_data):
        """Test deleting a non-existent template."""
        # Arrange
        manager = PromptManager(auto_save=True)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        template_id_exists = "t-exists"
        template_id_delete = "t-not-exists"
        manager._templates = {template_id_exists: MagicMock()}
        manager._versions = {template_id_exists: [MagicMock()]}
        manager._active_versions = {template_id_exists: MagicMock()}
        initial_template_count = len(manager._templates)

        # Act
        result = manager.delete_template(template_id_delete)

        # Assert
        assert result is False
        assert len(manager._templates) == initial_template_count # No change
        assert template_id_exists in manager._templates
        mock_logger.info.assert_not_called()
        mock_save_data.assert_not_called()

    @patch.object(PromptManager, '_save_data')
    @patch.object(PromptManager, '_load_data')
    def test_delete_template_success_autosave_off(self, mock_load_data, mock_save_data):
        """Test deleting an existing template without auto_save."""
        # Arrange
        manager = PromptManager(auto_save=False)
        mock_logger = MagicMock(spec=LoggerInterface)
        manager._logger = mock_logger
        template_id = "t-delete-nosave"
        manager._templates = {template_id: MagicMock()}
        manager._versions = {template_id: [MagicMock()]}
        manager._active_versions = {template_id: MagicMock()}

        # Act
        result = manager.delete_template(template_id)

        # Assert
        assert result is True
        assert template_id not in manager._templates
        assert template_id not in manager._versions
        assert template_id not in manager._active_versions
        mock_logger.info.assert_called_with(f"Deleted template: {template_id}")
        mock_save_data.assert_not_called() # Should not save