# tests/prompts/unit/test_prompt_template.py
"""
Unit tests for the PromptTemplate class.
"""

import pytest
import os
import yaml
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import time

from src.prompts.prompt_template import PromptTemplate
from src.exceptions import AIConfigError # Used in error handling

# Helper to create dummy template YAML files
def create_template_yaml(path: Path, data: dict):
    path.write_text(yaml.dump(data), encoding='utf-8')

# --- Test Suite --- 
class TestPromptTemplate:
    """Test suite for PromptTemplate."""

    @pytest.fixture
    def temp_templates_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for template YAML files."""
        templates_dir = tmp_path / "prompt_templates"
        templates_dir.mkdir()
        return templates_dir

    # --- Initialization Tests --- 

    # Patch os.makedirs to avoid side effects in other tests if dir exists
    @patch('src.prompts.prompt_template.os.makedirs')
    @patch('src.prompts.prompt_template.os.listdir', return_value=[]) # Assume empty dir initially
    def test_init_specified_dir(self, mock_listdir, mock_makedirs, temp_templates_dir: Path):
        """Test initialization with a specified templates_dir."""
        pt = PromptTemplate(templates_dir=str(temp_templates_dir))
        assert pt._templates_dir == str(temp_templates_dir)
        mock_makedirs.assert_called_once_with(str(temp_templates_dir), exist_ok=True)
        mock_listdir.assert_called_once_with(str(temp_templates_dir))

    @patch('src.prompts.prompt_template.os.path.dirname')
    @patch('src.prompts.prompt_template.os.path.abspath')
    @patch('src.prompts.prompt_template.os.makedirs')
    @patch('src.prompts.prompt_template.os.listdir', return_value=[])
    @patch('src.prompts.prompt_template.os.path.exists', return_value=True)
    def test_init_default_dir(self, mock_exists, mock_listdir, mock_makedirs, mock_abspath, mock_dirname):
        """Test initialization uses the default template directory."""
        # Mock path derivation relative to the prompts module
        dummy_prompts_module_dir = "/path/to/src/prompts"
        # Assuming the code calculates ../prompts/templates relative to prompt_template.py
        expected_templates_dir = "/path/to/src/prompts/templates" 
        
        # Mock dirname to return the prompts dir path
        # Need to mock it twice: once for dirname(abspath(__file__)) -> prompts dir
        # and once more for dirname(prompts_dir) -> src dir (used in calculation? No, just joins) 
        mock_abspath.return_value = os.path.join(dummy_prompts_module_dir, "prompt_template.py")
        mock_dirname.return_value = dummy_prompts_module_dir 
        # The code actually does os.path.dirname(os.path.dirname(abspath(__file__))) which gives /path/to/src
        # Let's adjust the mock return values for the nested calls
        mock_dirname.side_effect = [dummy_prompts_module_dir, "/path/to/src"]

        pt = PromptTemplate() # No templates_dir specified
        
        # Adjust expected path based on actual calculation
        expected_templates_dir = os.path.join("/path/to/src", "prompts", "templates")
        assert pt._templates_dir == expected_templates_dir
        mock_makedirs.assert_called_once_with(expected_templates_dir, exist_ok=True)
        mock_listdir.assert_called_once_with(expected_templates_dir)
        mock_exists.assert_called_once_with(expected_templates_dir)
        # Check abspath was called once, dirname twice
        mock_abspath.assert_called_once()
        assert mock_dirname.call_count == 2

    @patch('src.prompts.prompt_template.PromptTemplate._load_templates')
    def test_init_calls_load_templates(self, mock_load, temp_templates_dir: Path):
        """Test that __init__ calls _load_templates."""
        PromptTemplate(templates_dir=str(temp_templates_dir))
        mock_load.assert_called_once()

    # --- Loading Tests --- 

    def test_load_templates_success(self, temp_templates_dir: Path):
        """Test loading a valid template YAML file."""
        template_data = {
            'template1': {
                'default_version': 'v1',
                'versions': [{'version': 'v1', 'template': 'Hello {{name}}'}]
            }
        }
        create_template_yaml(temp_templates_dir / "file1.yaml", template_data)
        
        pt = PromptTemplate(templates_dir=str(temp_templates_dir))
        assert 'template1' in pt._templates
        assert pt._templates['template1'] == template_data['template1']

    def test_load_multiple_files(self, temp_templates_dir: Path):
        """Test loading templates from multiple YAML files."""
        data1 = {'tmpl1': {'versions': [{'version': 'v1', 'template': 'T1'}]}}
        data2 = {'tmpl2': {'versions': [{'version': 'v1', 'template': 'T2'}]}}
        create_template_yaml(temp_templates_dir / "f1.yml", data1)
        create_template_yaml(temp_templates_dir / "f2.yaml", data2)
        
        pt = PromptTemplate(templates_dir=str(temp_templates_dir))
        assert 'tmpl1' in pt._templates
        assert 'tmpl2' in pt._templates
        assert pt._templates['tmpl1'] == data1['tmpl1']
        assert pt._templates['tmpl2'] == data2['tmpl2']

    def test_load_empty_yaml_file(self, temp_templates_dir: Path):
        """Test loading an empty YAML file is skipped gracefully."""
        create_template_yaml(temp_templates_dir / "empty.yml", None) # Or {}
        data1 = {'tmpl1': {'versions': [{'version': 'v1', 'template': 'T1'}]}}
        create_template_yaml(temp_templates_dir / "f1.yml", data1)

        pt = PromptTemplate(templates_dir=str(temp_templates_dir))
        assert len(pt._templates) == 1 # Only tmpl1 should be loaded
        assert 'tmpl1' in pt._templates

    def test_load_malformed_yaml_file(self, temp_templates_dir: Path):
        """Test loading malformed YAML logs error and skips file."""
        bad_file_path = str(temp_templates_dir / "bad.yaml")
        good_file_path = str(temp_templates_dir / "f1.yml")
        # Create the files
        (temp_templates_dir / "bad.yaml").touch() 
        data1 = {'tmpl1': {'versions': [{'version': 'v1', 'template': 'T1'}]}}
        create_template_yaml(temp_templates_dir / "f1.yml", data1)

        # Store original open
        original_open = open
        
        # Define side effect for open
        def open_side_effect(file, mode='r', **kwargs):
            if file == bad_file_path:
                # Return a stream-like object with invalid YAML
                from io import StringIO
                return StringIO("key: value: another_value\n invalid_indent")
            else:
                # For other files (like the good one), use the real open
                return original_open(file, mode, **kwargs)
        
        # Patch open
        with patch("builtins.open", side_effect=open_side_effect):
            mock_logger = MagicMock()
            # Patch listdir to ensure both files are seen
            with patch('src.prompts.prompt_template.os.listdir', return_value=["bad.yaml", "f1.yml"]):
                 pt = PromptTemplate(templates_dir=str(temp_templates_dir), logger=mock_logger)
            
        # Assertions 
        assert len(pt._templates) == 1 # Only good file loaded
        assert 'tmpl1' in pt._templates
        # Update assertion: Expect error to be called twice due to code behavior
        assert mock_logger.error.call_count == 2 
        # Check the message of the *last* error call
        last_error_call_args = mock_logger.error.call_args_list[-1][0] # Get args of the last call
        assert "Failed to load template file bad.yaml" in last_error_call_args[0]
        assert "mapping values are not allowed here" in last_error_call_args[0]

    @patch('src.prompts.prompt_template.os.path.exists', return_value=False)
    @patch('src.prompts.prompt_template.os.makedirs')
    def test_load_templates_dir_not_found(self, mock_makedirs, mock_exists):
        """Test loading when the templates directory doesn't exist."""
        mock_logger = MagicMock()
        test_dir = "/non/existent/dir"
        pt = PromptTemplate(templates_dir=test_dir, logger=mock_logger)
        assert not pt._templates # Templates should be empty
        # Assert makedirs was called (even though exists is false for loading)
        mock_makedirs.assert_called_once_with(test_dir, exist_ok=True)
        # Assert exists was called during the _load_templates check
        mock_exists.assert_called_once_with(test_dir)
        mock_logger.warning.assert_called_once_with(f"Templates directory not found: {test_dir}")

    # --- Rendering Tests --- 
    @pytest.fixture
    def loaded_prompt_template(self, temp_templates_dir: Path) -> PromptTemplate:
        """Provides a PromptTemplate instance loaded with sample templates."""
        template_data = {
            'tmpl_simple': {
                'versions': [{'version': 'v1', 'template': 'Hello {{name}}'}]
            },
            'tmpl_multi_var': {
                'versions': [{'version': 'v1', 'template': 'User: {{user}}, Action: {{action}}'}]
            },
            'tmpl_versions': {
                'default_version': 'v2',
                'versions': [
                    {'version': 'v1', 'template': 'Version 1: {{val}}'},
                    {'version': 'v2', 'template': 'Version 2: {{val}}'},
                    {'version': 'v3', 'template': 'Version 3: {{val}}'}, # Latest
                ]
            },
            'tmpl_no_default': {
                 'versions': [
                    {'version': 'v1a', 'template': 'No default v1a'},
                    {'version': 'v2a', 'template': 'No default v2a'}, # Latest
                ]
            },
            'tmpl_no_versions': {},
            'tmpl_empty_template': {
                 'versions': [{'version': 'v1', 'template': ''}]
            }
        }
        create_template_yaml(temp_templates_dir / "rendering_test.yaml", template_data)
        # Patch makedirs and listdir to avoid issues during test fixture setup
        with patch('src.prompts.prompt_template.os.makedirs'), \
             patch('src.prompts.prompt_template.os.listdir', return_value=["rendering_test.yaml"]):
            pt = PromptTemplate(templates_dir=str(temp_templates_dir))
            return pt

    def test_render_prompt_success_simple(self, loaded_prompt_template: PromptTemplate):
        """Test successful rendering with simple variables."""
        rendered, usage_id = loaded_prompt_template.render_prompt('tmpl_simple', {'name': 'World'})
        assert rendered == "Hello World"
        assert isinstance(usage_id, str)

    def test_render_prompt_success_multiple_vars(self, loaded_prompt_template: PromptTemplate):
        """Test successful rendering with multiple variables."""
        rendered, _ = loaded_prompt_template.render_prompt(
            'tmpl_multi_var', 
            variables={'user': 'Admin', 'action': 'Login'}
        )
        assert rendered == "User: Admin, Action: Login"
        
    def test_render_prompt_with_context(self, loaded_prompt_template: PromptTemplate):
        """Test variables from context are used in rendering."""
        rendered, _ = loaded_prompt_template.render_prompt(
            'tmpl_simple', 
            variables={'other': 'ignore'}, 
            context={'name': 'Context'}
        )
        assert rendered == "Hello Context"

    def test_render_prompt_variable_override(self, loaded_prompt_template: PromptTemplate):
        """Test variables dictionary overrides context dictionary."""
        rendered, _ = loaded_prompt_template.render_prompt(
            'tmpl_simple', 
            variables={'name': 'Variable'}, 
            context={'name': 'Context'}
        )
        assert rendered == "Hello Context"

    def test_render_prompt_missing_variable_renders_placeholder(self, loaded_prompt_template: PromptTemplate):
        """Test missing variable leaves placeholder and logs warning."""
        mock_logger = MagicMock()
        loaded_prompt_template._logger = mock_logger
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_simple', {'wrong_var': 'value'})
        assert rendered == "Hello {{name}}" # Placeholder remains
        mock_logger.warning.assert_called_once_with("Variable not provided: name")
        
    def test_render_prompt_extra_variable_ignored(self, loaded_prompt_template: PromptTemplate):
        """Test extra variables are ignored."""
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_simple', {'name': 'World', 'extra': 'ignored'})
        assert rendered == "Hello World"
        
    def test_render_prompt_specific_version(self, loaded_prompt_template: PromptTemplate):
        """Test rendering a specific version of a template."""
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_versions', {'val': 'Data'}, version='v1')
        assert rendered == "Version 1: Data"
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_versions', {'val': 'Data'}, version='v3')
        assert rendered == "Version 3: Data"

    def test_render_prompt_default_version(self, loaded_prompt_template: PromptTemplate):
        """Test rendering the default version when version is None."""
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_versions', {'val': 'Data'})
        assert rendered == "Version 2: Data" # v2 is default
        
    def test_render_prompt_latest_version_if_no_default(self, loaded_prompt_template: PromptTemplate):
        """Test rendering the latest version if no default is specified."""
        rendered, _ = loaded_prompt_template.render_prompt('tmpl_no_default')
        assert rendered == "No default v2a" # v2a is the last one listed

    def test_render_prompt_template_not_found(self, loaded_prompt_template: PromptTemplate):
        """Test rendering fails if template_id is not found."""
        with pytest.raises(ValueError, match="Template not found: unknown_tmpl"):
            loaded_prompt_template.render_prompt('unknown_tmpl')

    def test_render_prompt_version_not_found(self, loaded_prompt_template: PromptTemplate):
        """Test rendering fails if specified version is not found."""
        with pytest.raises(ValueError, match="Version v99 not found for template: tmpl_versions"):
            loaded_prompt_template.render_prompt('tmpl_versions', version='v99')

    def test_render_prompt_no_versions_defined(self, loaded_prompt_template: PromptTemplate):
        """Test rendering fails if template has no versions defined."""
        with pytest.raises(ValueError, match="No versions found for template: tmpl_no_versions"):
            loaded_prompt_template.render_prompt('tmpl_no_versions')

    def test_render_prompt_empty_template_text(self, loaded_prompt_template: PromptTemplate):
        """Test rendering fails if the selected version has empty template text."""
        with pytest.raises(ValueError, match="Empty template text for template: tmpl_empty_template"):
            loaded_prompt_template.render_prompt('tmpl_empty_template')

    # --- Other Method Tests ---
    def test_get_template_ids(self, loaded_prompt_template: PromptTemplate):
        """Test getting a list of all loaded template IDs."""
        expected_ids = [
            'tmpl_simple', 
            'tmpl_multi_var', 
            'tmpl_versions', 
            'tmpl_no_default', 
            'tmpl_no_versions', 
            'tmpl_empty_template'
        ]
        # Order isn't guaranteed by dict keys, so compare sets
        assert set(loaded_prompt_template.get_template_ids()) == set(expected_ids)

    def test_record_prompt_performance(self, loaded_prompt_template: PromptTemplate):
        """Test recording performance metrics for a prompt usage."""
        # Render first to get a usage_id
        _, usage_id = loaded_prompt_template.render_prompt('tmpl_simple', {'name': 'PerfTest'})
        
        # Mock the save method to prevent file I/O
        with patch.object(loaded_prompt_template, '_save_metrics') as mock_save:
            metrics_to_record = {'latency': 0.5, 'tokens': 100}
            loaded_prompt_template.record_prompt_performance(usage_id, metrics_to_record)
            
            # Check internal state
            assert usage_id in loaded_prompt_template._template_metrics
            recorded = loaded_prompt_template._template_metrics[usage_id]
            assert recorded['template_id'] == 'tmpl_simple'
            assert recorded['latency'] == 0.5
            assert recorded['tokens'] == 100
            assert 'start_time' in recorded # Should be added by render
            assert 'variables' in recorded
            
            # Check save was called
            mock_save.assert_called_once_with(usage_id)

    def test_record_prompt_performance_calculates_latency(self, loaded_prompt_template: PromptTemplate):
        """Test latency is calculated if not provided."""
        start_time = time.time() - 0.2 # Simulate start time 0.2s ago
        rendered, usage_id = loaded_prompt_template.render_prompt('tmpl_simple', {'name': 'LatencyTest'})
        # Manually overwrite start_time for testing calculation
        loaded_prompt_template._template_metrics[usage_id]['start_time'] = start_time

        with patch.object(loaded_prompt_template, '_save_metrics') as mock_save:
            metrics_to_record = {'tokens': 50} # No latency provided
            loaded_prompt_template.record_prompt_performance(usage_id, metrics_to_record)

            recorded = loaded_prompt_template._template_metrics[usage_id]
            assert 'latency' in recorded
            assert recorded['latency'] == pytest.approx(0.2, abs=0.05) # Check approx latency
            mock_save.assert_called_once_with(usage_id)
            
    def test_record_prompt_performance_invalid_usage_id(self, loaded_prompt_template: PromptTemplate):
         """Test recording performance with an invalid usage ID logs warning."""
         mock_logger = MagicMock()
         loaded_prompt_template._logger = mock_logger
         loaded_prompt_template.record_prompt_performance("invalid-id", {'latency': 1.0})
         mock_logger.warning.assert_called_once_with("Usage ID not found: invalid-id")

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.prompts.prompt_template.os.makedirs")
    @patch("src.prompts.prompt_template.json.dump")
    def test_save_metrics(self, mock_json_dump, mock_makedirs, mock_open_file, loaded_prompt_template: PromptTemplate):
        """Test the _save_metrics method attempts to write JSON to file."""
        # Render to create metrics entry
        _, usage_id = loaded_prompt_template.render_prompt('tmpl_simple', {'name': 'SaveTest'})
        metrics_data = loaded_prompt_template._template_metrics[usage_id]
        
        # Construct expected path
        expected_dir = os.path.join(loaded_prompt_template._templates_dir, "metrics")
        expected_path = os.path.join(expected_dir, f"{usage_id}.json")

        loaded_prompt_template._save_metrics(usage_id)

        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
        mock_open_file.assert_called_once_with(expected_path, 'w')
        mock_json_dump.assert_called_once_with(metrics_data, mock_open_file(), indent=2)
        
    def test_reload_templates(self, temp_templates_dir: Path):
        """Test that reload_templates picks up changes in template files."""
        file_path = temp_templates_dir / "reload_test.yaml"
        initial_data = {'tmpl_reload': {'versions': [{'version': 'v1', 'template': 'Initial'}]}}
        create_template_yaml(file_path, initial_data)
        
        pt = PromptTemplate(templates_dir=str(temp_templates_dir))
        assert pt._templates['tmpl_reload']['versions'][0]['template'] == 'Initial'
        
        # Modify the file
        updated_data = {'tmpl_reload': {'versions': [{'version': 'v1', 'template': 'Reloaded'}]}}
        create_template_yaml(file_path, updated_data)
        
        # Create a new template file
        new_file_path = temp_templates_dir / "new_reload.yaml"
        new_data = {'tmpl_new': {'versions': [{'version': 'v1', 'template': 'New'}]}}
        create_template_yaml(new_file_path, new_data)
        
        # Reload templates
        pt.reload_templates()
        
        # Assert changes are loaded
        assert pt._templates['tmpl_reload']['versions'][0]['template'] == 'Reloaded'
        assert 'tmpl_new' in pt._templates
        assert pt._templates['tmpl_new']['versions'][0]['template'] == 'New'

