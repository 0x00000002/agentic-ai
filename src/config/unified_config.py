"""
Unified Configuration Manager for Agentic AI

This module provides a single, comprehensive configuration system that replaces
the previous ConfigFactory and ConfigManager implementations.
"""
from typing import Dict, Any, Optional, Union, List, Set
import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

from .user_config import UserConfig
from ..exceptions import AIConfigError
from ..utils.logger import LoggerFactory, LoggerInterface


class UnifiedConfig:
    """
    Unified configuration manager for the AI framework.
    
    This class is a singleton that manages all configuration for the AI framework.
    It loads configuration from modular YAML files, environment variables, and user overrides.
    """
    
    _instance = None
    
    # Configuration file paths relative to the src/config directory
    CONFIG_FILES = {
        "models": "models.yml",
        "providers": "providers.yml",
        "agents": "agents.yml",
        "use_cases": "use_cases.yml",
        "tools": "tools.yml"
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
        if cls._instance is None:
            cls._instance = UnifiedConfig(user_config=user_config, 
                                          config_dir=config_dir,
                                          logger=logger)
        elif user_config is not None:
            # Update existing instance with new user configuration
            cls._instance._apply_user_config(user_config)
        return cls._instance
    
    def __init__(self, user_config: Optional[UserConfig] = None, 
                 config_dir: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the configuration manager.
        
        Args:
            user_config: Optional user configuration overrides
            config_dir: Optional directory containing configuration files
            logger: Optional logger instance
        """
        # Singleton check
        if UnifiedConfig._instance is not None:
            raise RuntimeError("UnifiedConfig is a singleton. Use get_instance() instead.")
        
        # Load environment variables first
        load_dotenv()
        
        self._logger = logger or LoggerFactory.create(name="unified_config")
        
        # Determine configuration directory
        if config_dir is None:
            # Default to the directory containing this file
            self._config_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._config_dir = config_dir
            
        self._logger.info(f"Using configuration directory: {self._config_dir}")
        
        # Default settings
        self._show_thinking = False
        self._default_model = "phi4"  # Fallback default
        
        # Load base configuration
        self._config = {}
        self._load_config()
        
        # Apply user configuration if provided
        if user_config is not None:
            self._apply_user_config(user_config)
    
    def _load_config(self) -> None:
        """Load all configuration files."""
        for config_key, filename in self.CONFIG_FILES.items():
            try:
                filepath = os.path.join(self._config_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        self._config[config_key] = yaml.safe_load(f)
                    self._logger.info(f"Loaded configuration from {filename}")
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
    
    def _apply_user_config(self, user_config: UserConfig) -> None:
        """
        Apply user configuration overrides.
        
        Args:
            user_config: User configuration instance
        """
        # Load from external file if specified
        if user_config.config_file and user_config.config_file.exists():
            self._logger.info(f"Loading user configuration from file: {user_config.config_file}")
            try:
                file_config = UserConfig.from_file(user_config.config_file)
                self._apply_user_config(file_config)
            except Exception as e:
                self._logger.error(f"Failed to load user configuration from file: {str(e)}")
        
        # Apply specific overrides
        config_dict = user_config.to_dict()
        
        # Handle model override
        if "model" in config_dict:
            override_model_key = config_dict["model"]
            available_models = self.get_all_models().keys()
            if override_model_key in available_models:
                self._default_model = override_model_key
                self._logger.info(f"User override for default model: {self._default_model}")
            else:
                self._logger.warning(f"User override model '{override_model_key}' not found in available models. Ignoring override.")
        
        # Handle show_thinking override
        if "show_thinking" in config_dict:
            self._show_thinking = config_dict["show_thinking"]
            self._logger.info(f"User override for show_thinking: {self._show_thinking}")
        
        # Store all overrides for later use
        self._user_overrides = config_dict
    
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
        Get configuration for a specific model.
        
        Args:
            model_id: Model ID or None for default model
            
        Returns:
            Model configuration
            
        Raises:
            AIConfigError: If model configuration is not found
        """
        # Use default model if none specified
        if model_id is None:
            model_id = self._default_model
        
        models = self._config.get("models", {}).get("models", {})
        if model_id not in models:
            raise AIConfigError(f"Model configuration not found: {model_id}", config_name="models")
        
        # Start with base model config
        model_config = dict(models[model_id])
        
        # Apply temperature override if specified and for this specific model
        if hasattr(self, '_user_overrides') and self._user_overrides.get('model') == model_id:
            if 'temperature' in self._user_overrides:
                model_config['temperature'] = self._user_overrides['temperature']
                self._logger.info(f"Applied user temperature override for model {model_id}: {model_config['temperature']}")
        
        return model_config
    
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
        user_use_case = getattr(self, '_user_overrides', {}).get('use_case')
        if user_use_case and use_case is None:
            use_case = user_use_case
            self._logger.info(f"Using user-specified use case: {use_case}")
        
        # If still no use case, look for use cases defined for the current model
        if use_case is None:
            current_model = self._default_model
            model_config = self.get_model_config(current_model)
            model_use_cases = model_config.get('use_cases', [])
            
            if model_use_cases and len(model_use_cases) > 0:
                # Use the first use case associated with the model
                use_case = model_use_cases[0]
                self._logger.info(f"Using model's default use case: {use_case}")
        
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
    
    def get_default_model(self) -> str:
        """
        Get the default model ID.
        
        Returns:
            Default model ID
        """
        return self._default_model
    
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
        self._logger.info("Reloading configuration")
        self._load_config()
        
        # Re-apply user overrides if they exist
        if hasattr(self, '_user_overrides'):
            user_config = UserConfig(**self._user_overrides)
            self._apply_user_config(user_config)
    
    def get_tool_config(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for tools or a specific tool.
        
        Args:
            tool_name: Optional tool name for specific tool configuration
            
        Returns:
            Tool configuration dictionary
        """
        tools = self._config.get("tools", {}).get("tools", {})
        
        # Return all tool configurations if no specific tool is requested
        if tool_name is None:
            return tools
            
        # Check if the specific tool exists in any category
        for category, category_config in tools.get("categories", {}).items():
            if tool_name in category_config:
                return category_config[tool_name]
                
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
        if hasattr(self, '_user_overrides') and 'system_prompt' in self._user_overrides:
            return self._user_overrides['system_prompt']
        
        # Otherwise return None - the model classes should provide defaults
        return None 