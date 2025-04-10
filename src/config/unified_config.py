"""
Unified Configuration Manager for Agentic AI

This module provides a single, comprehensive configuration system that replaces
the previous ConfigFactory and ConfigManager implementations.
"""
from typing import Dict, Any, Optional, Union, List, Set, Type, TypeVar
import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

from .user_config import UserConfig
from ..exceptions import AIConfigError
from ..utils.logger import LoggerFactory, LoggerInterface


# Define a metaclass for the Singleton pattern
class SingletonMeta(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[Type, Any] = {}
    
    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def clear_instance(cls, target_class):
        """Clear the singleton instance for testing purposes."""
        if target_class in cls._instances:
            del cls._instances[target_class]


class UnifiedConfig(metaclass=SingletonMeta):
    """
    Unified configuration manager for the AI framework.
    
    This class is a singleton that manages all configuration for the AI framework.
    It loads configuration from modular YAML files, environment variables, and user overrides.
    """
    
    # Configuration file paths relative to the src/config directory
    CONFIG_FILES = {
        "models": "models.yml",
        "providers": "providers.yml",
        "agents": "agents.yml",
        "use_cases": "use_cases.yml",
        "tools": "tools.yml",
        "mcp": "mcp.yml"  # Add MCP configuration file
    }
    
    @classmethod
    def get_instance(cls, user_config: Optional[UserConfig] = None, 
                     config_dir: Optional[str] = None,
                     logger: Optional[LoggerInterface] = None) -> 'UnifiedConfig':
        """
        Get the singleton instance of the configuration manager.
        
        Args:
            user_config: Optional user configuration overrides
            config_dir: Optional directory containing configuration files
            logger: Optional logger instance
        
        Returns:
            UnifiedConfig instance
        """
        # Get the instance (created if it doesn't exist)
        instance = cls.__call__(config_dir=config_dir, logger=logger)
        
        # Apply user config if provided
        if user_config is not None:
            instance.apply_user_config(user_config)
            
        return instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing purposes)."""
        SingletonMeta.clear_instance(cls)
    
    def __init__(self, config_dir: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Optional directory containing configuration files
            logger: Optional logger instance
        """
        # Load environment variables first
        load_dotenv()
        
        self._logger = logger or LoggerFactory.create(name="unified_config")
        
        # Determine configuration directory
        if config_dir is None:
            # Default to the directory containing this file
            self._config_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._config_dir = config_dir
            
        self._logger.debug(f"Using configuration directory: {self._config_dir}")
        
        # Default settings
        self._show_thinking = False
        self._default_model = "phi4"  # Fallback default
        self._user_overrides = {}
        
        # Load base configuration
        self._config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load all configuration files."""
        for config_key, filename in self.CONFIG_FILES.items():
            try:
                filepath = os.path.join(self._config_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        self._config[config_key] = yaml.safe_load(f)
                    self._logger.debug(f"Loaded configuration from {filename}")
                else:
                    self._logger.warning(f"Configuration file not found: {filepath}")
                    self._config[config_key] = {}
            except Exception as e:
                # Log error but continue with empty config
                self._logger.error(f"Failed to load configuration from {filename}: {str(e)}")
                self._config[config_key] = {}
        
        # Set default model from use_cases config or use the fallback
        use_cases_config = self._config.get("use_cases", {})
        self._default_model = use_cases_config.get("default_model", self._default_model)
    
    def apply_user_config(self, user_config: UserConfig) -> None:
        """
        Apply user configuration overrides.
        
        Args:
            user_config: User configuration instance
        """
        # This is a public method now to make it explicitly callable
        self._apply_user_config(user_config)
    
    def _apply_user_config(self, user_config: UserConfig) -> None:
        """
        Internal implementation of apply_user_config.
        Merges user overrides into the main configuration structure.
        
        Args:
            user_config: User configuration instance
        """
        # Load from external file first (recursive call)
        if user_config.config_file and user_config.config_file.exists():
            self._logger.debug(f"Loading user configuration from file: {user_config.config_file}")
            try:
                file_config = UserConfig.from_file(user_config.config_file)
                self._apply_user_config(file_config) # Apply file config first
            except Exception as e:
                self._logger.error(f"Failed to load user configuration from file: {str(e)}")
        
        # Get overrides from the current user_config object
        overrides = user_config.to_dict()
        
        # Store raw overrides for simple property access (like show_thinking)
        self._user_overrides.update(overrides)
        
        # --- Intelligent Merging ---
        target_model_id = overrides.get("model") or self._default_model
        
        # Merge top-level settings that affect the framework
        if "show_thinking" in overrides:
            self._show_thinking = overrides["show_thinking"]
            self._logger.debug(f"Applied show_thinking override: {self._show_thinking}")
        
        if "model" in overrides:
            # Check if the model exists before setting it as default
            available_models = self._config.get("models", {}).get("models", {}).keys()
            if overrides["model"] in available_models:
                self._default_model = overrides["model"]
                self._logger.debug(f"Applied default model override: {self._default_model}")
            else:
                self._logger.warning(f"User override model '{overrides['model']}' not found. Ignoring default model override.")
        
        # Merge parameters into the specific model's config
        if target_model_id in self._config.get("models", {}).get("models", {}):
            model_cfg = self._config["models"]["models"][target_model_id]
            if "parameters" not in model_cfg:
                model_cfg["parameters"] = {}
            
            # --- Create/Update runtime_parameters --- 
            # Initialize if it doesn't exist
            if "runtime_parameters" not in model_cfg:
                model_cfg["runtime_parameters"] = {}
            
            # 1. Set default temperature from model's top level (if not already set)
            if 'temperature' not in model_cfg["runtime_parameters"] and 'temperature' in model_cfg:
                model_cfg["runtime_parameters"]["temperature"] = model_cfg['temperature']
            
            # 2. Apply user override for temperature
            if "temperature" in overrides:
                model_cfg["runtime_parameters"]["temperature"] = overrides["temperature"]
                self._logger.debug(f"Applied temperature override ({overrides['temperature']}) to model '{target_model_id}' runtime parameters")
            
            # Add other potential parameter overrides here (e.g., max_tokens, top_p)
            # Example:
            # if "max_tokens" in overrides:
            #     model_cfg["runtime_parameters"]["max_tokens"] = overrides["max_tokens"]
            #     self._logger.debug(f"Merged max_tokens override ({overrides['max_tokens']}) into model '{target_model_id}'")
        
        else:
            if "model" in overrides: # Only warn if a specific model was requested
                self._logger.warning(f"Cannot apply parameter overrides: Target model '{target_model_id}' not found in configuration.")
        
        # Merge system prompt override (could be global or model-specific if needed)
        if "system_prompt" in overrides:
            # For now, assume it's a global override?
            # Or potentially merge into model_cfg["system_prompt"] ?
            # Let's store it in _user_overrides for now, accessed via get_system_prompt()
            self._logger.debug("Stored system_prompt override.")
            pass # Already stored in self._user_overrides
        
        # Handle Use Case Preset (applies quality/speed defaults)
        if "use_case" in overrides and overrides["use_case"]:
            self._apply_use_case(overrides["use_case"], target_model_id)
    
    def _apply_use_case(self, use_case: Union[str, Any], target_model_id: str) -> None:
        """
        Applies configuration settings based on a use case preset.
        Placeholder implementation.

        Args:
            use_case: The use case identifier or preset object.
            target_model_id: The model being configured.
        """
        use_case_name = str(use_case) # Simple conversion for now
        self._logger.debug(f"Applying use case '{use_case_name}' settings for model '{target_model_id}' (Placeholder)")
        # TODO: Implement actual logic to merge use case defaults 
        # (e.g., quality, speed) into the model or global config if needed.
        use_cases_config = self._config.get("use_cases", {}).get("use_cases", {})
        if use_case_name in use_cases_config:
            uc_config = use_cases_config[use_case_name]
            # Example: Merge quality/speed into the model's parameters if not already set
            # model_cfg = self._config["models"]["models"][target_model_id]
            # if "parameters" not in model_cfg: model_cfg["parameters"] = {}
            # if "quality" not in model_cfg["parameters"]:
            #     model_cfg["parameters"]["quality"] = uc_config.get("quality")
            # ... etc ... 
            pass # Keep it simple for now
        else:
            self._logger.warning(f"Use case '{use_case_name}' not found in use_cases.yml")
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration
            
        Raises:
            AIConfigError: If provider configuration is not found
        """
        providers = self._config.get("providers", {}).get("providers", {})
        if provider not in providers:
            raise AIConfigError(f"Provider configuration not found: {provider}", config_name="providers")
        return providers[provider]
    
    def get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the fully merged configuration for a specific model.
        
        Args:
            model_id: Model ID or None for default model
            
        Returns:
            Model configuration (already merged with overrides)
            
        Raises:
            AIConfigError: If model configuration is not found
        """
        # Use default model if none specified
        if model_id is None:
            model_id = self._default_model
        
        models = self._config.get("models", {}).get("models", {})
        if model_id not in models:
            # Log the specific model ID that was not found
            self._logger.error(f"Configuration for model '{model_id}' not found. Available models: {list(models.keys())}")
            raise AIConfigError(f"Model configuration not found: {model_id}", config_name="models")
        
        # Return the configuration directly - merging happens in _apply_user_config
        # Return a copy to prevent modification of the internal state
        return dict(models[model_id])
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration for all models.
        
        Returns:
            Dictionary of model configurations
        """
        return self._config.get("models", {}).get("models", {})
    
    def get_model_names(self) -> List[str]:
        """
        Get list of all available model names.
        
        Returns:
            List of model names
        """
        return list(self.get_all_models().keys())
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent configuration
        """
        agents = self._config.get("agents", {}).get("agents", {})
        if agent_id not in agents:
            return {}  # Return empty dict instead of raising error for backward compatibility
        return agents[agent_id]
    
    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all agents.
        
        Returns:
            Dictionary of agent descriptions
        """
        return self._config.get("agents", {}).get("agent_descriptions", {})
    
    def get_use_case_config(self, use_case: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific use case or the default config for the current model.
        
        Args:
            use_case: Use case name or None for default
            
        Returns:
            Use case configuration
        """
        # Handle user-specified use case
        user_use_case = self._user_overrides.get('use_case')
        if user_use_case and use_case is None:
            use_case = user_use_case
            self._logger.debug(f"Using user-specified use case: {use_case}")
        
        # If still no use case, look for use cases defined for the current model
        if use_case is None:
            current_model = self._default_model
            model_config = self.get_model_config(current_model)
            model_use_cases = model_config.get('use_cases', [])
            
            if model_use_cases and len(model_use_cases) > 0:
                # Use the first use case associated with the model
                use_case = model_use_cases[0]
                self._logger.debug(f"Using model's default use case: {use_case}")
        
        # Get use case configuration
        use_cases = self._config.get("use_cases", {}).get("use_cases", {})
        
        # If use case is specified and exists, return it
        if use_case and use_case in use_cases:
            return use_cases[use_case]
        
        # Otherwise, return default parameters
        self._logger.warning(f"Use case '{use_case}' not found or not specified, using defaults")
        return {
            "quality": "medium",
            "speed": "standard"
        }
    
    @property
    def default_model(self) -> str:
        """Get the default model ID."""
        return self._default_model
    
    def get_default_model(self) -> str:
        """
        Get the default model ID.
        
        Returns:
            Default model ID
        """
        return self._default_model
    
    @property
    def config_dir(self) -> str:
        """Get the configuration directory path."""
        return self._config_dir
    
    def get_log_level(self) -> str:
        """
        Get the log level.
        
        Returns:
            Log level string
        """
        return self._config.get("use_cases", {}).get("log_level", "INFO")
    
    @property
    def log_level(self) -> str:
        """Get the configured log level."""
        return self.get_log_level()
    
    @property
    def show_thinking(self) -> bool:
        """Get the show_thinking setting."""
        return self._show_thinking
        
    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        """Set the show_thinking setting."""
        self._show_thinking = value
    
    @property
    def user_overrides(self) -> Dict[str, Any]:
        """Get the current user configuration overrides."""
        return self._user_overrides.copy()  # Return a copy to prevent modification
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider_name: The provider name
            
        Returns:
            API key string or None if not found
        """
        try:
            provider_config = self.get_provider_config(provider_name)
            env_var = provider_config.get("api_key_env")
            if env_var:
                return os.environ.get(env_var)
            return None
        except Exception as e:
            self._logger.error(f"Error getting API key for provider {provider_name}: {str(e)}")
            return None
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._logger.debug("Reloading configuration")
        self._load_config()
        
        # Re-apply user overrides if they exist
        user_overrides = self._user_overrides
        if user_overrides:
            user_config = UserConfig(**user_overrides)
            self._apply_user_config(user_config)
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """
        Get the MCP configuration.
        
        Returns:
            Dictionary containing the MCP configuration, or an empty dict if not loaded.
        """
        return self._config.get("mcp", {})
    
    def get_tool_config(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for tools or a specific tool.
        
        Args:
            tool_name: Optional tool name for specific tool configuration
            
        Returns:
            Tool configuration dictionary
        """
        # Get the entire tools section from the config
        tools_config = self._config.get("tools", {})
        
        # Return all tool configurations if no specific tool is requested
        if tool_name is None:
            return tools_config
            
        # If a specific tool is requested, look for it in the tools list
        if "tools" in tools_config and isinstance(tools_config["tools"], list):
            for tool_def in tools_config["tools"]:
                if isinstance(tool_def, dict) and tool_def.get("name") == tool_name:
                    return tool_def
                
        # Return empty dictionary if tool not found
        self._logger.warning(f"Tool configuration not found: {tool_name}")
        return {}
    
    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt, with user override if provided.
        
        Returns:
            System prompt string or None if not specified
        """
        # Check for user override
        if 'system_prompt' in self._user_overrides:
            return self._user_overrides['system_prompt']
        
        # Otherwise return None - the model classes should provide defaults
        return None 