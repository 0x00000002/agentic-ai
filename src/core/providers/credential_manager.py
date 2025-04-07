"""
Credential management for AI providers.
Handles API keys, authentication, and other credential-related functionality.
"""
from typing import Dict, Any, Optional, Union, List
import os
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import AICredentialsError, AIAuthenticationError
from ...config.unified_config import UnifiedConfig


class CredentialManager:
    """
    Manages credentials for AI providers.
    
    This class is responsible for:
    - Managing API keys and other authentication credentials
    - Loading credentials from different sources (config, environment)
    - Supporting credential rotation if needed
    - Validating credentials
    """
    
    def __init__(self, 
                 provider_name: str,
                 provider_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None,
                 config: Optional[UnifiedConfig] = None):
        """
        Initialize the credential manager.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            provider_config: Provider-specific configuration dictionary
            logger: Optional logger instance
            config: Optional UnifiedConfig instance
        """
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.logger = logger or LoggerFactory.create(name=f"{provider_name}_credentials")
        self.config = config or UnifiedConfig.get_instance()
        
        # Store credentials as they are loaded
        self.credentials: Dict[str, Any] = {}
        
    def get_api_key(self) -> Optional[str]:
        """
        Get the API key for the provider.
        
        Tries multiple sources in order:
        1. Already loaded credentials
        2. Provider config's direct api_key value
        3. Environment variable specified in provider config
        4. UnifiedConfig's get_api_key method
        
        Returns:
            API key or None if not found
        """
        # Return from cache if already loaded
        if "api_key" in self.credentials:
            return self.credentials["api_key"]
            
        # Try direct api_key in provider config
        api_key = self.provider_config.get("api_key")
        if api_key:
            self.credentials["api_key"] = api_key
            return api_key
            
        # Try environment variable
        env_var_name = self.provider_config.get("api_key_env")
        if env_var_name:
            api_key = os.environ.get(env_var_name)
            if api_key:
                self.credentials["api_key"] = api_key
                return api_key
                
        # Try UnifiedConfig's method
        api_key = self.config.get_api_key(self.provider_name)
        if api_key:
            self.credentials["api_key"] = api_key
            return api_key
            
        # Not found
        return None
        
    def get_credential(self, credential_name: str) -> Optional[Any]:
        """
        Get a specific credential by name.
        
        Args:
            credential_name: Name of the credential
            
        Returns:
            Credential value or None if not found
        """
        # Return from cache if already loaded
        if credential_name in self.credentials:
            return self.credentials[credential_name]
            
        # Try direct value in provider config
        value = self.provider_config.get(credential_name)
        if value is not None:
            self.credentials[credential_name] = value
            return value
            
        # Try environment variable
        env_var_name = self.provider_config.get(f"{credential_name}_env")
        if env_var_name:
            value = os.environ.get(env_var_name)
            if value:
                self.credentials[credential_name] = value
                return value
                
        # Not found
        return None
        
    def load_credentials(self, required_credentials: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load all credentials needed for the provider.
        
        Args:
            required_credentials: List of credential names that are required
            
        Returns:
            Dictionary of loaded credentials
            
        Raises:
            AICredentialsError: If a required credential is not found
        """
        # Default to just api_key if no list provided
        required_credentials = required_credentials or ["api_key"]
        
        # Load all required credentials
        for name in required_credentials:
            value = self.get_credential(name)
            if value is None and name in required_credentials:
                self.logger.error(f"Required credential '{name}' not found for provider '{self.provider_name}'")
                raise AICredentialsError(f"Missing required credential: {name}", provider=self.provider_name)
                
        return self.credentials
        
    def validate_credentials(self) -> bool:
        """
        Validate that the credentials are valid.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        # Basic validation - just check if api_key exists
        return bool(self.get_api_key())
        
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of authentication headers
        """
        api_key = self.get_api_key()
        if not api_key:
            self.logger.error(f"No API key available for provider '{self.provider_name}'")
            raise AIAuthenticationError(f"No API key available", provider=self.provider_name)
            
        # Standard Bearer token format
        return {"Authorization": f"Bearer {api_key}"}
        
    def get_client_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for client initialization.
        
        Returns:
            Dictionary of client initialization arguments
        """
        kwargs = {}
        
        # Add API key
        api_key = self.get_api_key()
        if api_key:
            kwargs["api_key"] = api_key
            
        # Add base URL if present
        base_url = self.get_credential("base_url")
        if base_url:
            kwargs["base_url"] = base_url
            
        # Add organization if present (for OpenAI)
        org_id = self.get_credential("organization")
        if org_id:
            kwargs["organization"] = org_id
            
        return kwargs
        
    def rotate_credentials(self) -> None:
        """
        Rotate credentials if supported by the provider.
        
        Raises:
            NotImplementedError: If credential rotation is not supported
        """
        raise NotImplementedError(f"Credential rotation not supported for provider '{self.provider_name}'") 