"""
Parameter management for AI providers.
Handles standardization, merging, and mapping of request parameters.
"""
from typing import Dict, Any, Optional, List, Set
from ...utils.logger import LoggerInterface, LoggerFactory


class ParameterManager:
    """
    Handles parameter management for AI providers.
    
    This class is responsible for managing parameters for API requests, including:
    - Storing default parameters
    - Merging parameters from different sources (defaults, model config, runtime)
    - Mapping standard parameters to provider-specific names
    - Filtering allowed parameters for each provider
    """
    
    def __init__(self, 
                 default_parameters: Optional[Dict[str, Any]] = None,
                 model_parameters: Optional[Dict[str, Any]] = None,
                 allowed_parameters: Optional[Set[str]] = None,
                 parameter_mapping: Optional[Dict[str, str]] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the parameter manager.
        
        Args:
            default_parameters: Default parameters for the provider
            model_parameters: Model-specific parameters
            allowed_parameters: Set of parameter names allowed by the provider
            parameter_mapping: Mapping from standard parameter names to provider-specific names
            logger: Optional logger instance
        """
        self.logger = logger or LoggerFactory.create(name="parameter_manager")
        
        # Initialize parameters in priority order (defaults < model config)
        self.default_parameters = {} if default_parameters is None else default_parameters.copy()
        
        # Store model parameters separately to allow merging in the right order
        self.model_parameters = {} if model_parameters is None else model_parameters.copy()
        
        # Combine defaults and model parameters for base parameters
        self.parameters = self.default_parameters.copy()
        self.parameters.update(self.model_parameters)
        
        # Store allowed parameters and mapping
        self.allowed_parameters = allowed_parameters or set()
        self.parameter_mapping = parameter_mapping or {}

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters.
        
        Returns:
            Default parameters dictionary
        """
        return self.default_parameters.copy()
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the model-specific parameters.
        
        Returns:
            Model parameters dictionary
        """
        return self.model_parameters.copy()
    
    def get_base_parameters(self) -> Dict[str, Any]:
        """
        Get the base parameters (defaults + model).
        
        Returns:
            Base parameters dictionary
        """
        return self.parameters.copy()
    
    def merge_parameters(self, 
                         runtime_options: Dict[str, Any],
                         enforce_allowed: bool = True) -> Dict[str, Any]:
        """
        Merge parameters from different sources.
        
        Priority: defaults < model config < runtime options
        
        Args:
            runtime_options: Runtime options to merge
            enforce_allowed: Whether to filter out disallowed parameters
            
        Returns:
            Merged parameters dictionary
        """
        # Start with base parameters
        merged = self.parameters.copy()
        
        # Update with runtime options
        merged.update(runtime_options)
        
        # Filter out disallowed parameters if requested
        if enforce_allowed and self.allowed_parameters:
            merged = {k: v for k, v in merged.items() if k in self.allowed_parameters}
            
        return merged
    
    def map_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map standard parameter names to provider-specific names.
        
        Args:
            parameters: Parameters to map
            
        Returns:
            Mapped parameters dictionary
        """
        if not self.parameter_mapping:
            return parameters
            
        mapped = {}
        for key, value in parameters.items():
            # Map the key if it exists in the mapping, otherwise keep it as is
            mapped_key = self.parameter_mapping.get(key, key)
            mapped[mapped_key] = value
            
        return mapped
    
    def extract_special_parameters(self, 
                                  parameters: Dict[str, Any], 
                                  special_keys: List[str]) -> Dict[str, Any]:
        """
        Extract special parameters from the parameters dictionary.
        
        Args:
            parameters: Parameters to extract from
            special_keys: List of special parameter keys to extract
            
        Returns:
            Dictionary of extracted special parameters
        """
        extracted = {}
        for key in special_keys:
            if key in parameters:
                extracted[key] = parameters[key]
                
        return extracted
    
    def prepare_request_payload(self, 
                               runtime_options: Dict[str, Any],
                               special_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare the request payload by merging, mapping, and processing parameters.
        
        Args:
            runtime_options: Runtime options for the request
            special_keys: List of special parameter keys to extract
            
        Returns:
            Processed parameters for the request payload
        """
        # Extract special parameters from runtime options directly
        special_params = {}
        if special_keys:
            # Extract special parameters from runtime options
            special_params = {key: runtime_options[key] for key in special_keys if key in runtime_options}
            # Create a copy of runtime options without special keys for merging
            filtered_runtime_options = {k: v for k, v in runtime_options.items() if k not in special_params}
        else:
            filtered_runtime_options = runtime_options
        
        # Merge parameters (without special keys)
        merged_params = self.merge_parameters(filtered_runtime_options)
        
        # Map parameters to provider-specific names
        mapped_params = self.map_parameters(merged_params)
        
        self.logger.debug(f"Prepared request parameters: {list(mapped_params.keys())}")
        
        return mapped_params, special_params
    
    def update_defaults(self, new_defaults: Dict[str, Any]) -> None:
        """
        Update the default parameters.
        
        Args:
            new_defaults: New default parameters
        """
        self.default_parameters.update(new_defaults)
        # Update base parameters too
        self.parameters.update(new_defaults)
        
    def set_allowed_parameters(self, allowed: Set[str]) -> None:
        """
        Set the allowed parameters.
        
        Args:
            allowed: Set of allowed parameter names
        """
        self.allowed_parameters = allowed
        
    def set_parameter_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set the parameter mapping.
        
        Args:
            mapping: Mapping from standard names to provider-specific names
        """
        self.parameter_mapping = mapping 