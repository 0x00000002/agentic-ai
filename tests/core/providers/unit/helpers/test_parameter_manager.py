"""
Unit tests for ParameterManager class.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.core.providers.parameter_manager import ParameterManager
from src.utils.logger import LoggerInterface


class TestParameterManager:
    """Test suite for the ParameterManager class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture for a mock logger."""
        return MagicMock(spec=LoggerInterface)
    
    @pytest.fixture
    def default_manager(self, mock_logger):
        """Fixture for a default ParameterManager."""
        return ParameterManager(logger=mock_logger)
    
    @pytest.fixture
    def populated_manager(self, mock_logger):
        """Fixture for a ParameterManager with various parameters set."""
        default_params = {"temperature": 0.7, "max_tokens": 2048}
        model_params = {"temperature": 0.8, "top_p": 0.95}
        allowed_params = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"}
        param_mapping = {"max_tokens": "max_tokens_to_sample"}
        return ParameterManager(
            default_parameters=default_params,
            model_parameters=model_params,
            allowed_parameters=allowed_params,
            parameter_mapping=param_mapping,
            logger=mock_logger
        )
    
    def test_init_default(self, default_manager, mock_logger):
        """Test initialization with default parameters."""
        assert default_manager.logger is mock_logger
        assert default_manager.default_parameters == {}
        assert default_manager.model_parameters == {}
        assert default_manager.parameters == {}
        assert default_manager.allowed_parameters == set()
        assert default_manager.parameter_mapping == {}
    
    def test_init_with_params(self, populated_manager, mock_logger):
        """Test initialization with specific parameters."""
        assert populated_manager.logger is mock_logger
        assert populated_manager.default_parameters == {"temperature": 0.7, "max_tokens": 2048}
        assert populated_manager.model_parameters == {"temperature": 0.8, "top_p": 0.95}
        
        # Base parameters should be defaults updated with model parameters
        assert populated_manager.parameters == {"temperature": 0.8, "max_tokens": 2048, "top_p": 0.95}
        
        assert populated_manager.allowed_parameters == {
            "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"
        }
        assert populated_manager.parameter_mapping == {"max_tokens": "max_tokens_to_sample"}
    
    def test_get_default_parameters(self, populated_manager):
        """Test getting default parameters."""
        params = populated_manager.get_default_parameters()
        assert params == {"temperature": 0.7, "max_tokens": 2048}
        # Check it's a copy
        params["new_param"] = "value"
        assert "new_param" not in populated_manager.default_parameters
    
    def test_get_model_parameters(self, populated_manager):
        """Test getting model parameters."""
        params = populated_manager.get_model_parameters()
        assert params == {"temperature": 0.8, "top_p": 0.95}
        # Check it's a copy
        params["new_param"] = "value"
        assert "new_param" not in populated_manager.model_parameters
    
    def test_get_base_parameters(self, populated_manager):
        """Test getting base parameters."""
        params = populated_manager.get_base_parameters()
        assert params == {"temperature": 0.8, "max_tokens": 2048, "top_p": 0.95}
        # Check it's a copy
        params["new_param"] = "value"
        assert "new_param" not in populated_manager.parameters
    
    def test_merge_parameters(self, populated_manager):
        """Test merging parameters."""
        runtime_options = {"temperature": 0.5, "presence_penalty": 0.2}
        merged = populated_manager.merge_parameters(runtime_options)
        
        # Should include base parameters overridden by runtime options
        assert merged == {
            "temperature": 0.5,  # From runtime
            "max_tokens": 2048,  # From defaults
            "top_p": 0.95,  # From model
            "presence_penalty": 0.2  # From runtime
        }
    
    def test_merge_parameters_enforce_allowed(self, populated_manager):
        """Test enforcing allowed parameters during merge."""
        runtime_options = {
            "temperature": 0.5,  # Allowed
            "presence_penalty": 0.2,  # Allowed
            "disallowed_param": "value"  # Not allowed
        }
        merged = populated_manager.merge_parameters(runtime_options, enforce_allowed=True)
        
        # Should filter out disallowed parameters
        assert merged == {
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 0.95,
            "presence_penalty": 0.2
        }
        assert "disallowed_param" not in merged
    
    def test_merge_parameters_no_enforce(self, populated_manager):
        """Test not enforcing allowed parameters during merge."""
        runtime_options = {
            "temperature": 0.5,
            "disallowed_param": "value"
        }
        merged = populated_manager.merge_parameters(runtime_options, enforce_allowed=False)
        
        # Should include disallowed parameters
        assert merged == {
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 0.95,
            "disallowed_param": "value"
        }
    
    def test_map_parameters(self, populated_manager):
        """Test mapping parameter names."""
        params = {
            "temperature": 0.7,
            "max_tokens": 100,  # This should be mapped
            "top_p": 0.9
        }
        mapped = populated_manager.map_parameters(params)
        
        # max_tokens should be mapped to max_tokens_to_sample
        assert mapped == {
            "temperature": 0.7,
            "max_tokens_to_sample": 100,  # Mapped
            "top_p": 0.9
        }
    
    def test_map_parameters_no_mapping(self, default_manager):
        """Test mapping parameters with empty mapping."""
        params = {"temperature": 0.7, "max_tokens": 100}
        mapped = default_manager.map_parameters(params)
        
        # Should be unchanged
        assert mapped == params
    
    def test_extract_special_parameters(self, populated_manager):
        """Test extracting special parameters."""
        params = {
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
            "tools": ["calculator", "weather"],
            "max_tokens": 100
        }
        special_keys = ["system_prompt", "tools"]
        special_params = populated_manager.extract_special_parameters(params, special_keys)
        
        # Should extract only the specified keys
        assert special_params == {
            "system_prompt": "You are a helpful assistant.",
            "tools": ["calculator", "weather"]
        }
    
    def test_prepare_request_payload(self, populated_manager):
        """Test preparing a request payload."""
        runtime_options = {
            "temperature": 0.5,
            "system_prompt": "You are a helpful assistant.",
            "tools": ["calculator"],
            "max_tokens": 100,
            "presence_penalty": 0.2,  # Add this to runtime options
            "disallowed_param": "value"
        }
        special_keys = ["system_prompt", "tools"]
        
        params, special_params = populated_manager.prepare_request_payload(
            runtime_options=runtime_options,
            special_keys=special_keys
        )
        
        # Check mapped parameters - note that presence_penalty is now expected
        assert params == {
            "temperature": 0.5,
            "max_tokens_to_sample": 100,  # Mapped
            "top_p": 0.95,  # From base
            "presence_penalty": 0.2  # From runtime
        }
        
        # Check special parameters
        assert special_params == {
            "system_prompt": "You are a helpful assistant.",
            "tools": ["calculator"]
        }
        
        # Disallowed param should be filtered out
        assert "disallowed_param" not in params
    
    def test_update_defaults(self, default_manager):
        """Test updating default parameters."""
        default_manager.update_defaults({"temperature": 0.8, "max_tokens": 1000})
        
        # Should update both default_parameters and parameters
        assert default_manager.default_parameters == {"temperature": 0.8, "max_tokens": 1000}
        assert default_manager.parameters == {"temperature": 0.8, "max_tokens": 1000}
    
    def test_set_allowed_parameters(self, default_manager):
        """Test setting allowed parameters."""
        allowed = {"temperature", "max_tokens", "top_p"}
        default_manager.set_allowed_parameters(allowed)
        assert default_manager.allowed_parameters == allowed
    
    def test_set_parameter_mapping(self, default_manager):
        """Test setting parameter mapping."""
        mapping = {"max_tokens": "max_tokens_to_sample", "temperature": "temp"}
        default_manager.set_parameter_mapping(mapping)
        assert default_manager.parameter_mapping == mapping 