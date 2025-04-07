"""
Step definitions for provider configuration tests.
"""
from behave import given, when, then
from unittest.mock import MagicMock, patch
import os
import json
from src.config.unified_config import UnifiedConfig
from src.core.providers.configuration import ConfigurationError

@given('the AI framework is initialized')
def step_impl(context):
    """Initialize the AI framework."""
    context.unified_config = UnifiedConfig()
    context.unified_config.providers = {
        "openai": {
            "api_key": "test-openai-key",
            "timeout": 30,
            "models": {
                "gpt-4": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "anthropic": {
            "api_key": "test-anthropic-key",
            "timeout": 30,
            "models": {
                "claude-3-opus": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "gemini": {
            "api_key": "test-gemini-key",
            "timeout": 30,
            "models": {
                "gemini-pro": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": 30,
            "models": {
                "llama3": {
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            }
        }
    }

@given('provider configurations are available in the unified config')
def step_impl(context):
    """Verify provider configurations are available."""
    assert context.unified_config is not None
    assert "openai" in context.unified_config.providers
    assert "anthropic" in context.unified_config.providers
    assert "gemini" in context.unified_config.providers
    assert "ollama" in context.unified_config.providers

@given('a provider type "{provider_type}"')
def step_impl(context, provider_type):
    """Set the provider type."""
    context.provider_type = provider_type

@given('a malformed provider configuration')
def step_impl(context):
    """Create a malformed provider configuration."""
    context.malformed_config = {
        "api_key": "",  # Empty API key
        "timeout": "invalid",  # Invalid timeout value
        "models": {}  # Empty models
    }

@given('environment variables for provider configuration')
def step_impl(context):
    """Set environment variables for provider configuration."""
    context.env_vars = {
        "OPENAI_API_KEY": "env-openai-key",
        "OPENAI_TIMEOUT": "60",
        "OPENAI_MODEL": "gpt-4-turbo"
    }
    for key, value in context.env_vars.items():
        os.environ[key] = value

@given('an existing provider configuration')
def step_impl(context):
    """Create an existing provider configuration."""
    context.original_config = {
        "api_key": "original-key",
        "timeout": 30,
        "models": {
            "gemini-pro": {
                "max_tokens": 2000,
                "temperature": 0.7
            }
        }
    }
    context.config_file = "test_config.json"
    with open(context.config_file, "w") as f:
        json.dump(context.original_config, f)

@given('provider-specific configuration requirements')
def step_impl(context):
    """Set provider-specific configuration requirements."""
    context.required_settings = {
        "base_url": "http://localhost:11434",
        "model_path": "/path/to/model",
        "context_length": 4096,
        "gpu_layers": 32
    }

@given('a non-existent configuration file')
def step_impl(context):
    """Set a non-existent configuration file path."""
    context.config_file = "nonexistent_config.json"
    assert not os.path.exists(context.config_file)

@given('a malformed configuration file')
def step_impl(context):
    """Create a malformed configuration file."""
    context.config_file = "malformed_config.json"
    with open(context.config_file, "w") as f:
        f.write("invalid json content")

@given('multiple configuration sources')
def step_impl(context):
    """Create multiple configuration sources."""
    context.config_sources = {
        "default": {
            "api_key": "default-key",
            "timeout": 30
        },
        "environment": {
            "api_key": "env-key",
            "timeout": 60
        },
        "file": {
            "api_key": "file-key",
            "timeout": 45
        }
    }

@when('I load the provider configuration')
def step_impl(context):
    """Load the provider configuration."""
    context.config = context.unified_config.providers[context.provider_type]

@when('I attempt to validate the configuration')
def step_impl(context):
    """Attempt to validate the configuration."""
    try:
        context.unified_config.validate_provider_config(
            context.provider_type,
            context.malformed_config
        )
        context.error = None
    except ConfigurationError as e:
        context.error = e

@when('I update the configuration file')
def step_impl(context):
    """Update the configuration file."""
    context.new_config = {
        "api_key": "new-key",
        "timeout": 60,
        "models": {
            "gemini-pro": {
                "max_tokens": 4000,
                "temperature": 0.8
            }
        }
    }
    with open(context.config_file, "w") as f:
        json.dump(context.new_config, f)

@when('I reload the configuration')
def step_impl(context):
    """Reload the configuration."""
    context.unified_config.reload_provider_config(context.provider_type)

@when('I validate the configuration')
def step_impl(context):
    """Validate the configuration."""
    context.validation_result = context.unified_config.validate_provider_config(
        context.provider_type,
        context.required_settings
    )

@when('I attempt to load the configuration')
def step_impl(context):
    """Attempt to load the configuration."""
    try:
        with open(context.config_file, "r") as f:
            context.config = json.load(f)
        context.error = None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        context.error = e

@when('I load the provider configuration from multiple sources')
def step_impl(context):
    """Load configuration from multiple sources."""
    context.final_config = context.unified_config.merge_configurations(
        context.config_sources
    )

@then('the configuration should contain required fields')
def step_impl(context):
    """Verify required configuration fields."""
    assert "api_key" in context.config
    assert "timeout" in context.config
    assert "models" in context.config
    assert isinstance(context.config["timeout"], int)
    assert isinstance(context.config["models"], dict)

@then('the configuration should be properly validated')
def step_impl(context):
    """Verify configuration validation."""
    assert context.unified_config.validate_provider_config(
        context.provider_type,
        context.config
    )

@then('the configuration should be accessible through UnifiedConfig')
def step_impl(context):
    """Verify configuration accessibility."""
    assert context.unified_config.get_provider_config(context.provider_type) == context.config

@then('a ConfigurationError should be raised')
def step_impl(context):
    """Verify ConfigurationError was raised."""
    assert context.error is not None
    assert isinstance(context.error, ConfigurationError)

@then('the error message should indicate the specific validation failure')
def step_impl(context):
    """Verify error message content."""
    assert "validation" in str(context.error).lower()
    assert "required" in str(context.error).lower()

@then('the environment variables should override default values')
def step_impl(context):
    """Verify environment variable overrides."""
    assert context.config["api_key"] == context.env_vars["OPENAI_API_KEY"]
    assert str(context.config["timeout"]) == context.env_vars["OPENAI_TIMEOUT"]

@then('the overridden values should be properly validated')
def step_impl(context):
    """Verify overridden value validation."""
    assert context.unified_config.validate_provider_config(
        context.provider_type,
        context.config
    )

@then('the new configuration should be loaded')
def step_impl(context):
    """Verify new configuration loading."""
    assert context.unified_config.providers[context.provider_type] == context.new_config

@then('existing provider instances should use the new configuration')
def step_impl(context):
    """Verify provider instance configuration update."""
    provider = context.unified_config.get_provider(context.provider_type)
    assert provider.config == context.new_config

@then('all provider-specific settings should be present')
def step_impl(context):
    """Verify provider-specific settings presence."""
    for key in context.required_settings:
        assert key in context.validation_result

@then('the settings should meet provider requirements')
def step_impl(context):
    """Verify settings meet requirements."""
    assert context.validation_result is True

@then('the error should indicate the missing file')
def step_impl(context):
    """Verify missing file error."""
    assert isinstance(context.error, FileNotFoundError)
    assert context.config_file in str(context.error)

@then('the error should indicate the format issue')
def step_impl(context):
    """Verify format error."""
    assert isinstance(context.error, json.JSONDecodeError)

@then('the configurations should be properly merged')
def step_impl(context):
    """Verify configuration merging."""
    assert context.final_config is not None
    assert "api_key" in context.final_config
    assert "timeout" in context.final_config

@then('conflicts should be resolved according to priority')
def step_impl(context):
    """Verify conflict resolution."""
    assert context.final_config["api_key"] == context.config_sources["file"]["api_key"]
    assert context.final_config["timeout"] == context.config_sources["environment"]["timeout"]

@then('the final configuration should be consistent')
def step_impl(context):
    """Verify final configuration consistency."""
    assert isinstance(context.final_config["timeout"], int)
    assert context.final_config["api_key"] in [
        source["api_key"] for source in context.config_sources.values()
    ] 