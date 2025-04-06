"""
Prompt Template service.
Provides functionality for template-based prompts with variable substitution,
versioning, and performance tracking.
"""
from typing import Dict, Any, Optional, Tuple, List
import os
import uuid
import re
import time
import json
import yaml
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIConfigError, ErrorHandler


class PromptTemplate:
    """
    Service for managing prompt templates with versioning and performance tracking.
    """
    
    def __init__(self, 
                 templates_dir: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the prompt template service.
        
        Args:
            templates_dir: Directory containing template files (or None for default)
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create(name="prompt_template")
        
        # Determine templates directory
        if templates_dir is None:
            # Default to 'templates' subdirectory in the prompts directory
            self._templates_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "prompts",
                "templates"
            )
        else:
            self._templates_dir = templates_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self._templates_dir, exist_ok=True)
            
        self._logger.info(f"Using templates directory: {self._templates_dir}")
        
        # Initialize template cache
        self._templates = {}
        self._template_metrics = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from the templates directory."""
        if not os.path.exists(self._templates_dir):
            self._logger.warning(f"Templates directory not found: {self._templates_dir}")
            return
            
        for filename in os.listdir(self._templates_dir):
            if filename.endswith(('.yml', '.yaml')):
                try:
                    filepath = os.path.join(self._templates_dir, filename)
                    with open(filepath, 'r') as f:
                        templates = yaml.safe_load(f)
                    
                    if not templates:
                        continue
                        
                    for template_id, template_data in templates.items():
                        self._templates[template_id] = template_data
                        self._logger.info(f"Loaded template: {template_id} from {filename}")
                except Exception as e:
                    error_response = ErrorHandler.handle_error(
                        AIConfigError(f"Failed to load template file {filename}: {str(e)}", config_name="templates"),
                        self._logger
                    )
                    self._logger.error(f"Template loading error: {error_response['message']}")
    
    def render_prompt(self, 
                     template_id: str, 
                     variables: Optional[Dict[str, Any]] = None,
                     version: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Render a prompt template with variables.
        
        Args:
            template_id: ID of the template to use
            variables: Variables to substitute in the template
            version: Specific template version to use (or None for default/latest)
            context: Additional context for template rendering
            
        Returns:
            Tuple of (rendered prompt, usage ID)
            
        Raises:
            ValueError: If template not found or invalid
        """
        self._logger.debug(f"Available templates before rendering '{template_id}': {list(self._templates.keys())}")
        if template_id not in self._templates:
            raise ValueError(f"Template not found: {template_id}")
            
        template_data = self._templates[template_id]
        
        # Get versions
        versions = template_data.get("versions", [])
        if not versions:
            raise ValueError(f"No versions found for template: {template_id}")
            
        # Select version
        template_version = None
        if version:
            # Find specific version
            for v in versions:
                if v.get("version") == version:
                    template_version = v
                    break
            if not template_version:
                raise ValueError(f"Version {version} not found for template: {template_id}")
        else:
            # Use default or latest version
            default_version = template_data.get("default_version")
            if default_version:
                for v in versions:
                    if v.get("version") == default_version:
                        template_version = v
                        break
            
            # If no default or not found, use the latest version
            if not template_version:
                template_version = versions[-1]
        
        # Get template text
        template_text = template_version.get("template", "")
        if not template_text:
            raise ValueError(f"Empty template text for template: {template_id}")
        
        # Generate usage ID for tracking
        usage_id = str(uuid.uuid4())
        
        # Prepare variables
        vars_to_use = {}
        if variables:
            vars_to_use.update(variables)
        if context:
            vars_to_use.update(context)
        
        # Render template
        rendered_text = self._render_template_text(template_text, vars_to_use)
        
        # Store start time for metrics
        self._template_metrics[usage_id] = {
            "template_id": template_id,
            "version": template_version.get("version"),
            "start_time": time.time(),
            "variables": vars_to_use
        }
        
        return rendered_text, usage_id
    
    def _render_template_text(self, template_text: str, variables: Dict[str, Any]) -> str:
        """
        Render a template string by substituting variables.
        
        Args:
            template_text: Template text with placeholders
            variables: Variables to substitute
            
        Returns:
            Rendered template string
        """
        # Simple variable substitution with {{variable}} syntax
        rendered = template_text
        
        # Find all variables in the template
        var_pattern = r'\{\{([a-zA-Z0-9_]+)\}\}'
        matches = re.findall(var_pattern, template_text)
        
        # Substitute each variable
        for var_name in matches:
            if var_name in variables:
                placeholder = f"{{{{{var_name}}}}}"
                value = str(variables[var_name])
                rendered = rendered.replace(placeholder, value)
            else:
                self._logger.warning(f"Variable not provided: {var_name}")
        
        return rendered
    
    def record_prompt_performance(self, 
                                 usage_id: str, 
                                 metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics for a prompt usage.
        
        Args:
            usage_id: Usage ID returned from render_prompt
            metrics: Metrics to record (latency, tokens, etc.)
        """
        if usage_id not in self._template_metrics:
            self._logger.warning(f"Usage ID not found: {usage_id}")
            return
            
        # Calculate latency if not provided
        if "latency" not in metrics and "start_time" in self._template_metrics[usage_id]:
            metrics["latency"] = time.time() - self._template_metrics[usage_id]["start_time"]
            
        # Update metrics
        self._template_metrics[usage_id].update(metrics)
        
        # Log metrics
        self._logger.info(f"Prompt metrics for {usage_id}: {metrics}")
        
        # Save metrics to file (in a real system, you'd want to use a database)
        self._save_metrics(usage_id)
    
    def _save_metrics(self, usage_id: str) -> None:
        """
        Save metrics to file.
        
        Args:
            usage_id: Usage ID to save metrics for
        """
        if usage_id not in self._template_metrics:
            return
            
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(self._templates_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save metrics to file
        metrics_file = os.path.join(metrics_dir, f"{usage_id}.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self._template_metrics[usage_id], f, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to save metrics: {str(e)}")
    
    def get_template_ids(self) -> List[str]:
        """
        Get a list of all available template IDs.
        
        Returns:
            List of template IDs
        """
        return list(self._templates.keys())
    
    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template information
            
        Raises:
            ValueError: If template not found
        """
        if template_id not in self._templates:
            raise ValueError(f"Template not found: {template_id}")
            
        return self._templates[template_id]
    
    def reload_templates(self) -> None:
        """Reload all templates from the templates directory."""
        self._templates = {}
        self._load_templates()
        self._logger.info("Templates reloaded")
    
    def add_template(self, template_id: str, template_data: Dict[str, Any]) -> None:
        """
        Add a new template or update an existing one.
        
        Args:
            template_id: ID of the template
            template_data: Template data
        """
        self._templates[template_id] = template_data
        self._logger.info(f"Added/updated template: {template_id}")
        
        # Save template to file
        self._save_template(template_id, template_data)
    
    def _save_template(self, template_id: str, template_data: Dict[str, Any]) -> None:
        """
        Save a template to file.
        
        Args:
            template_id: ID of the template
            template_data: Template data
        """
        # Determine file path
        file_path = os.path.join(self._templates_dir, f"{template_id}.yml")
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump({template_id: template_data}, f, default_flow_style=False)
            self._logger.info(f"Saved template to file: {file_path}")
        except Exception as e:
            self._logger.error(f"Failed to save template: {str(e)}") 