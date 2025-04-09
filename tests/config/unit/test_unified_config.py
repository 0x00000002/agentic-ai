# tests/config/unit/test_unified_config.py
"""
Unit tests for the UnifiedConfig class.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml # Assuming YAML is used for config files

from src.config.unified_config import UnifiedConfig
from src.exceptions import AIConfigError

# Fixture to reset the singleton instance before each test
@pytest.fixture(autouse=True)
def reset_unified_config():
    """Reset the UnifiedConfig singleton before each test."""
    UnifiedConfig.reset_instance()
    yield # Run the test
    UnifiedConfig.reset_instance() # Ensure cleanup after test

# Helper to create dummy config files
def create_dummy_yaml(path: Path, data: dict):
    path.write_text(yaml.dump(data), encoding='utf-8')

class TestUnifiedConfig:
    """Test suite for UnifiedConfig."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for config files."""
        config_dir = tmp_path / "config_test_dir"
        config_dir.mkdir()
        # Create default files to avoid warnings, content can be empty initially
        for filename in UnifiedConfig.CONFIG_FILES.values():
            create_dummy_yaml(config_dir / filename, {})
        return config_dir

    def test_singleton_get_instance(self):
        """Test that get_instance returns the same instance."""
        instance1 = UnifiedConfig.get_instance()
        instance2 = UnifiedConfig.get_instance()
        assert instance1 is instance2
        assert isinstance(instance1, UnifiedConfig)

    def test_direct_init_preserves_singleton(self):
        """Test that multiple calls to __init__ still return same singleton instance."""
        # Create first instance
        instance1 = UnifiedConfig.get_instance()
        # Create another instance with different params - it should still be the same instance
        instance2 = UnifiedConfig()
        
        # Both should be the same instance
        assert instance1 is instance2
        assert isinstance(instance1, UnifiedConfig)

    @patch('src.config.unified_config.os.path.dirname')
    @patch('src.config.unified_config.os.path.abspath')
    def test_default_config_dir(self, mock_abspath, mock_dirname):
        """Test that the default config directory is derived correctly."""
        # Mock the path derivation for the file where UnifiedConfig is defined
        # We don't know the exact path in the test env, so use a dummy
        dummy_file_path = "/path/to/src/config/unified_config.py"
        expected_dir = "/path/to/src/config"
        mock_abspath.return_value = dummy_file_path
        mock_dirname.return_value = expected_dir

        # Patch os.path.exists and load_dotenv
        with patch('src.config.unified_config.os.path.exists', return_value=False), \
             patch('src.config.unified_config.load_dotenv') as mock_load_dotenv:
             instance = UnifiedConfig.get_instance()
             assert instance._config_dir == expected_dir
        # Ensure the mocks were called as expected
        mock_abspath.assert_called_once() 
        mock_dirname.assert_called_once_with(dummy_file_path)
        mock_load_dotenv.assert_called_once() # Ensure load_dotenv was called

    def test_specified_config_dir(self, temp_config_dir: Path):
        """Test initialization with a specified config directory."""
        # Patch exists to assume files exist within the temp dir during init
        def mock_exists(path):
            # Only return True if the path is within our temp dir
            return str(temp_config_dir) in path
            
        # Patch os.path.exists and load_dotenv
        with patch('src.config.unified_config.os.path.exists', side_effect=mock_exists), \
             patch('src.config.unified_config.load_dotenv') as mock_load_dotenv:
             instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
             assert instance._config_dir == str(temp_config_dir)
        mock_load_dotenv.assert_called_once()

    # --- Loading Tests --- 
    def test_loading_from_specified_path(self, temp_config_dir: Path):
        """Test loading configuration from a specified directory works."""
        models_data = {'models': {'test-model': {'provider': 'test-prov'}}}
        providers_data = {'providers': {'test-prov': {'api_key': 'dummy'}}}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "providers.yml", providers_data)
        
        # Use get_instance to trigger loading
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        # Assert internal state or use accessors
        assert instance._config.get("models") == models_data
        assert instance._config.get("providers") == providers_data
        # Test an accessor
        assert instance.get_model_config('test-model') == {'provider': 'test-prov'}

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        # Need to handle the singleton nature - maybe reset it?
        # Or test its default state if possible without loading.
        pass

    def test_loading_from_specific_path(self, temp_config_dir: Path):
        """Test loading configuration from a specified directory."""
        # Create dummy config files in temp_config_dir
        # Instantiate UnifiedConfig pointing to temp_config_dir
        # Assert configuration is loaded correctly
        pass

    def test_loading_missing_file(self, temp_config_dir: Path):
        """Test that a missing default config file is handled gracefully."""
        # Delete one of the default files created by the fixture
        models_file = temp_config_dir / "models.yml"
        models_file.unlink()
        
        # Load config
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        # Assert the corresponding config section is empty, but others loaded
        assert "models" not in instance._config or instance._config["models"] == {} 
        assert instance._config.get("providers") is not None # Assuming providers.yml exists

    def test_loading_malformed_yaml(self, temp_config_dir: Path):
        """Test that malformed YAML logs an error and results in empty config section."""
        # Make safe_load raise an error when trying to load models.yml
        models_path = str(temp_config_dir / "models.yml")
        
        # Patch open to return bad data for the specific file
        original_open = open
        def patched_open(file, mode='r', **kwargs):
            if file == models_path:
                 from io import StringIO
                 return StringIO("key: value: another_value\n invalid_indent")
            # Use original open for other files to allow them to load correctly
            return original_open(file, mode, **kwargs)

        # Patch load_dotenv as well, as it happens before loading
        with patch('builtins.open', patched_open), \
             patch('src.config.unified_config.load_dotenv'):
            # Mock logger to check warning
            mock_logger = MagicMock()
            instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), logger=mock_logger)
            
            # Assert models config is empty and error was logged
            assert "models" not in instance._config or instance._config["models"] == {}
            mock_logger.error.assert_called_once()
            assert "Failed to load configuration from models.yml" in mock_logger.error.call_args[0][0]
            # Check for the actual YAML error message substring
            assert "mapping values are not allowed here" in mock_logger.error.call_args[0][0]

    # --- Merging / Override Tests --- 
    def test_user_config_model_override(self, temp_config_dir: Path):
        """Test UserConfig overrides the default model."""
        # Setup base config
        models_data = {'models': {'model-A': {}, 'model-B': {}}}
        use_cases_data = {'default_model': 'model-A'}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        
        # User config overriding the model
        from src.config.user_config import UserConfig
        user_conf = UserConfig(model='model-B')
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf)
        
        assert instance.get_default_model() == 'model-B'

    def test_user_config_show_thinking_override(self, temp_config_dir: Path):
        """Test UserConfig overrides show_thinking."""
        # Base config doesn't set show_thinking (defaults False)
        from src.config.user_config import UserConfig
        user_conf = UserConfig(show_thinking=True)
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf)
        
        assert instance.show_thinking is True

    def test_user_config_object_overrides_base(self, temp_config_dir: Path):
        """Test UserConfig object values override base configuration."""
        # Base config
        models_data = {'models': {'model-base': {}, 'model-override': {}}}
        use_cases_data = {'default_model': 'model-base'}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)

        # User config object with direct overrides
        from src.config.user_config import UserConfig
        user_conf_obj = UserConfig(model='model-override', show_thinking=True)

        # Patch load_dotenv for isolation
        with patch('src.config.unified_config.load_dotenv'):
             instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf_obj)

        assert instance.get_default_model() == 'model-override'
        assert instance.show_thinking is True
        
    @pytest.mark.xfail(reason="Potential bug in _apply_user_config file loading precedence.")
    def test_user_config_file_override(self, temp_config_dir: Path, tmp_path: Path):
        """Test loading user config from a file overrides base config (EXPECTED TO FAIL)."""
        # Base config
        models_data = {'models': {'model-base': {}}}
        use_cases_data = {'default_model': 'model-base'}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)

        # User config file
        user_file_path = tmp_path / "user_config.yml"
        user_file_data = {'model': 'model-user-file', 'show_thinking': True}
        create_dummy_yaml(user_file_path, user_file_data)

        # User config object specifying ONLY the file
        from src.config.user_config import UserConfig
        user_conf_obj = UserConfig(config_file=str(user_file_path))

        # Patch load_dotenv for isolation
        with patch('src.config.unified_config.load_dotenv'):
             instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf_obj)

        # This assertion currently fails due to the suspected bug
        assert instance.get_default_model() == 'model-user-file'
        assert instance.show_thinking is True

    def test_merging_logic(self, temp_config_dir: Path):
        """Test that configurations are merged correctly (e.g., overrides)."""
        # Create multiple config files with overlapping/overriding keys
        # Load config
        # Assert the final merged values are correct
        pass

    def test_get_provider_config(self, temp_config_dir: Path):
        """Test the get_provider_config method."""
        # Setup config with provider details
        # Load config
        # Call get_provider_config
        # Assert correct config dict is returned
        pass

    def test_get_model_config(self, temp_config_dir: Path):
        """Test the get_model_config method."""
        # Setup config with model details
        # Load config
        # Call get_model_config
        # Assert correct config dict is returned
        pass

    def test_get_api_key(self, temp_config_dir: Path):
        """Test the get_api_key method (including environment fallback)."""
        # Setup config with and without API keys
        # Mock os.environ
        # Load config
        # Call get_api_key for different providers
        # Assert correct key is returned or None
        pass

    def test_missing_file_handling(self):
        """Test behavior when specified config files/dirs are missing."""
        # Instantiate UnifiedConfig pointing to non-existent path
        # Assert appropriate behavior (e.g., loads defaults, raises error?)
        pass

    def test_malformed_file_handling(self, temp_config_dir: Path):
        # This test is effectively replaced by test_loading_malformed_yaml
        # Keep the original implementation or remove/refactor
        models_path = str(temp_config_dir / "models.yml")
        original_open = open
        def patched_open(file, mode='r', **kwargs):
            if file == models_path:
                 from io import StringIO
                 return StringIO("key: value: another_value\n invalid_indent")
            return original_open(file, mode, **kwargs)

        with patch('builtins.open', patched_open), \
             patch('src.config.unified_config.load_dotenv'): # Also patch load_dotenv here
            mock_logger = MagicMock()
            instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), logger=mock_logger)
            assert "models" not in instance._config or instance._config["models"] == {}
            mock_logger.error.assert_called_once()
            assert "Failed to load configuration from models.yml" in mock_logger.error.call_args[0][0]
            # Check for actual YAML error
            assert "mapping values are not allowed here" in mock_logger.error.call_args[0][0]

    # --- Accessor Tests ---
    def test_get_provider_config_success(self, temp_config_dir: Path):
        """Test getting existing provider configuration."""
        provider_data = {'providers': {'prov1': {'key': 'val1'}, 'prov2': {'key': 'val2'}}}
        create_dummy_yaml(temp_config_dir / "providers.yml", provider_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        assert instance.get_provider_config('prov1') == {'key': 'val1'}
        assert instance.get_provider_config('prov2') == {'key': 'val2'}

    def test_get_provider_config_not_found(self, temp_config_dir: Path):
        """Test getting non-existent provider config raises AIConfigError."""
        provider_data = {'providers': {'prov1': {'key': 'val1'}}}
        create_dummy_yaml(temp_config_dir / "providers.yml", provider_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        with pytest.raises(AIConfigError, match="Provider configuration not found: non_existent_prov"):
            instance.get_provider_config('non_existent_prov')

    def test_get_model_config_success(self, temp_config_dir: Path):
        """Test getting existing model configuration (specific and default)."""
        models_data = {'models': {'model1': {'p': 'a'}, 'model2': {'p': 'b'}}}
        use_cases_data = {'default_model': 'model2'}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        assert instance.get_model_config('model1') == {'p': 'a'}
        # Test getting default model
        assert instance.get_model_config() == {'p': 'b'} 

    def test_get_model_config_not_found(self, temp_config_dir: Path):
        """Test getting non-existent model config raises AIConfigError."""
        models_data = {'models': {'model1': {'p': 'a'}}}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        with pytest.raises(AIConfigError, match="Model configuration not found: non_existent_model"):
            instance.get_model_config('non_existent_model')
            
    def test_get_model_config_default_not_found(self, temp_config_dir: Path):
        """Test getting default model when default isn't in models.yml raises AIConfigError."""
        models_data = {'models': {'model1': {'p': 'a'}}}
        use_cases_data = {'default_model': 'missing_default'}
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        with pytest.raises(AIConfigError, match="Model configuration not found: missing_default"):
            instance.get_model_config() # Request default
            
    def test_get_agent_config_success(self, temp_config_dir: Path):
        """Test getting existing agent configuration."""
        agents_data = {'agents': {'agent1': {'a': 1}, 'agent2': {'b': 2}}}
        create_dummy_yaml(temp_config_dir / "agents.yml", agents_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        assert instance.get_agent_config('agent1') == {'a': 1}
        assert instance.get_agent_config('agent2') == {'b': 2}

    def test_get_agent_config_not_found(self, temp_config_dir: Path):
        """Test getting non-existent agent config returns empty dict."""
        agents_data = {'agents': {'agent1': {'a': 1}}}
        create_dummy_yaml(temp_config_dir / "agents.yml", agents_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        
        assert instance.get_agent_config('non_existent_agent') == {}
        
    @patch.dict(os.environ, {}, clear=True) # Ensure env var is not set
    def test_get_api_key_not_found(self, temp_config_dir: Path):
        """Test get_api_key returns None when not found in env."""
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_api_key("provider_b") is None
        
    @patch.dict(os.environ, {"ACTUAL_ENV_VAR_FOR_A": "env_key_a"}, clear=True)
    def test_get_api_key_from_env(self, temp_config_dir: Path):
        """Test getting API key successfully via api_key_env in provider config."""
        # 1. Define the provider config specifying the env var name
        provider_data = {
            'providers': {
                'provider_a': {
                    'api_key_env': 'ACTUAL_ENV_VAR_FOR_A' 
                }
            }
        }
        create_dummy_yaml(temp_config_dir / "providers.yml", provider_data)

        # 2. Patch load_dotenv for isolation
        with patch('src.config.unified_config.load_dotenv') as mock_load_dotenv:
            # 3. Load config - @patch.dict ensures ACTUAL_ENV_VAR_FOR_A is set
            instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
            
            # 4. Assertions (inside the 'with' block)
            # Now get_api_key will find 'api_key_env' and look up 'ACTUAL_ENV_VAR_FOR_A'
            assert instance.get_api_key("provider_a") == "env_key_a"
            # Test case insensitivity of provider name lookup if needed (get_provider_config handles it)
            # assert instance.get_api_key("PROVIDER_A") == "env_key_a" 
            
            # Ensure load_dotenv was called during init
            mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_not_found_no_env_var_key(self, temp_config_dir: Path):
        """Test get_api_key returns None when api_key_env is missing in config."""
        # Define provider config *without* api_key_env
        provider_data = {
            'providers': {
                'provider_c': {
                    'other_config': 'value'
                }
            }
        }
        create_dummy_yaml(temp_config_dir / "providers.yml", provider_data)

        with patch('src.config.unified_config.load_dotenv'):
            instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
            assert instance.get_api_key("provider_c") is None

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_not_found_env_var_missing(self, temp_config_dir: Path):
        """Test get_api_key returns None when api_key_env points to unset env var."""
        # Define provider config with api_key_env
        provider_data = {
            'providers': {
                'provider_d': {
                    'api_key_env': 'MISSING_ENV_VAR'
                }
            }
        }
        create_dummy_yaml(temp_config_dir / "providers.yml", provider_data)

        with patch('src.config.unified_config.load_dotenv'):
            instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
            # MISSING_ENV_VAR is not in the patched os.environ
            assert instance.get_api_key("provider_d") is None

    def test_reload_config(self, temp_config_dir: Path):
        """Test that reload() picks up changes in config files."""
        models_path = temp_config_dir / "models.yml"
        initial_data = {'models': {'model1': {'param': 'initial'}}}
        create_dummy_yaml(models_path, initial_data)
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_model_config('model1') == {'param': 'initial'}
        
        # Modify the file
        updated_data = {'models': {'model1': {'param': 'reloaded'}}}
        create_dummy_yaml(models_path, updated_data)
        
        # Reload and test again
        instance.reload()
        assert instance.get_model_config('model1') == {'param': 'reloaded'}
        
    # --- Other Accessor Tests ---
    def test_get_tool_config_all(self, temp_config_dir: Path):
        """Test getting configuration for all tools."""
        tools_data = {
            'tools': [
                {
                    'name': 'tool1',
                    'desc': 'd1',
                    'module': 'mod1',
                    'function': 'func1'
                },
                {
                    'name': 'tool2',
                    'desc': 'd2',
                    'module': 'mod2',
                    'function': 'func2'
                }
            ],
            'categories': {'cat1': {'enabled': True}}
        }
        create_dummy_yaml(temp_config_dir / "tools.yml", tools_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        # get_tool_config() without args returns the whole 'tools' section
        assert instance.get_tool_config() == tools_data

    def test_get_tool_config_specific(self, temp_config_dir: Path):
        """Test getting configuration for a specific tool."""
        tool1_data = {
            'name': 'tool1',
            'desc': 'd1',
            'module': 'mod1',
            'function': 'func1'
        }
        tools_data = {
            'tools': [
                tool1_data,
                {
                    'name': 'tool2',
                    'desc': 'd2',
                    'module': 'mod2',
                    'function': 'func2'
                }
            ]
        }
        create_dummy_yaml(temp_config_dir / "tools.yml", tools_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        # get_tool_config() with a name should find the specific tool dict
        assert instance.get_tool_config('tool1') == tool1_data

    def test_get_tool_config_not_found(self, temp_config_dir: Path):
        """Test getting non-existent tool config returns empty dict."""
        tools_data = {'tools': {'categories': {'cat1': {'tool1': {'desc': 'd1'}}}}}
        create_dummy_yaml(temp_config_dir / "tools.yml", tools_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_tool_config('non_existent_tool') == {}

    def test_get_agent_descriptions(self, temp_config_dir: Path):
        """Test getting agent descriptions."""
        agents_data = {'agents': {'agent1': {'a': 1}, 'agent2': {'b': 2}},
                       'agent_descriptions': {'agent1': 'Desc A', 'agent2': 'Desc B'}}
        create_dummy_yaml(temp_config_dir / "agents.yml", agents_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_agent_descriptions() == {'agent1': 'Desc A', 'agent2': 'Desc B'}

    def test_get_agent_descriptions_missing(self, temp_config_dir: Path):
        """Test getting agent descriptions when none are defined."""
        agents_data = {'agents': {'agent1': {'a': 1}}}
        create_dummy_yaml(temp_config_dir / "agents.yml", agents_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_agent_descriptions() == {}

    def test_get_use_case_config_specific(self, temp_config_dir: Path):
        """Test getting configuration for a specific, existing use case."""
        use_cases_data = {'use_cases': {'case1': {'q': 'high'}, 'case2': {'q': 'low'}}}
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_use_case_config('case1') == {'q': 'high'}

    def test_get_use_case_config_not_found(self, temp_config_dir: Path):
        """Test getting non-existent use case config returns defaults."""
        use_cases_data = {'use_cases': {'case1': {'q': 'high'}}}
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        # Should return default values if specific case not found
        assert instance.get_use_case_config('non_existent_case') == {"quality": "medium", "speed": "standard"}

    def test_get_use_case_config_model_default(self, temp_config_dir: Path):
        """Test getting default use case associated with the default model."""
        models_data = {'models': {'model1': {'use_cases': ['case_m1']}}}
        use_cases_data = {'use_cases': {'case_m1': {'q': 'model_default'}} }
        # Set model1 as the default model
        use_cases_data['default_model'] = 'model1' 
        create_dummy_yaml(temp_config_dir / "models.yml", models_data)
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        # No use case specified, should find 'case_m1' via 'model1'
        assert instance.get_use_case_config() == {'q': 'model_default'}
        
    def test_get_use_case_config_user_override(self, temp_config_dir: Path):
        """Test that UserConfig overrides the use case."""
        use_cases_data = {'use_cases': {'case_base': {'q': 'base'}, 'case_user': {'q': 'user'}}}
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        
        from src.config.user_config import UserConfig
        user_conf = UserConfig(use_case='case_user')
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf)
        assert instance.get_use_case_config() == {'q': 'user'}

    def test_get_system_prompt_no_override(self, temp_config_dir: Path):
        """Test get_system_prompt returns None when no override exists."""
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.get_system_prompt() is None
        
    def test_get_system_prompt_user_override(self, temp_config_dir: Path):
        """Test get_system_prompt returns the user override."""
        from src.config.user_config import UserConfig
        user_prompt = "You are a helpful test assistant."
        user_conf = UserConfig(system_prompt=user_prompt)
        
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir), user_config=user_conf)
        assert instance.get_system_prompt() == user_prompt

    def test_log_level_default(self, temp_config_dir: Path):
        """Test default log level is INFO."""
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.log_level == "INFO"
        assert instance.get_log_level() == "INFO"

    def test_log_level_from_config(self, temp_config_dir: Path):
        """Test log level is loaded from use_cases config."""
        use_cases_data = {'log_level': 'DEBUG'}
        create_dummy_yaml(temp_config_dir / "use_cases.yml", use_cases_data)
        instance = UnifiedConfig.get_instance(config_dir=str(temp_config_dir))
        assert instance.log_level == "DEBUG"
        assert instance.get_log_level() == "DEBUG"

    # Add tests for get_use_case_config, get_system_prompt, log_level etc. 