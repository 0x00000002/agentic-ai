"""
Prompt manager implementation.
Main interface for the prompt management system.
"""
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import os
import json
import random
from datetime import datetime
import uuid

from .prompt_template import PromptTemplate
from .prompt_version import PromptVersion
from .metrics import PromptMetrics
from ..utils.logger import LoggerInterface, LoggerFactory


class PromptManager:
    """
    Main class for prompt management.
    Handles template creation, versioning, A/B testing, and optimization.
    """
    
    def __init__(self, 
                storage_dir: Optional[str] = None,
                metrics: Optional[PromptMetrics] = None,
                logger: Optional[LoggerInterface] = None,
                auto_save: bool = True):
        """
        Initialize the prompt manager.
        
        Args:
            storage_dir: Directory to store prompt templates and metrics
            metrics: Metrics object (or None to create new)
            logger: Logger instance
            auto_save: Whether to automatically save changes
        """
        self._templates = {}  # Template ID -> PromptTemplate
        self._versions = {}   # Template ID -> [PromptVersion, ...]
        self._active_versions = {}  # Template ID -> PromptVersion
        self._test_allocations = {}  # Template ID -> {user_id -> version_id}
        
        # Set up metrics
        self._metrics = metrics or PromptMetrics()
        
        # Set up logger
        self._logger = logger or LoggerFactory.create()
        
        # Set up storage
        self._storage_dir = storage_dir
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        self._auto_save = auto_save
        
        # Load existing data if available
        if storage_dir:
            self._load_data()
    
    def _save_data(self) -> None:
        """Save versions and metrics data to disk."""
        if not self._storage_dir:
            return
            
        # Remove templates.json saving
        # templates_file = os.path.join(self._storage_dir, "templates.json")
        versions_file = os.path.join(self._storage_dir, "versions.json")
        metrics_file = os.path.join(self._storage_dir, "metrics.json")
        
        # Save templates - REMOVED
        # templates_data = {t_id: template.to_dict() for t_id, template in self._templates.items()}
        # with open(templates_file, 'w') as f:
        #     json.dump(templates_data, f, indent=2)
        
        # Save versions (Assuming PromptVersion.to_dict() exists and is correct)
        versions_data = {}
        for t_id, versions in self._versions.items():
            versions_data[t_id] = [v.to_dict() for v in versions]
        
        # Handle potential errors during version serialization if needed
        try:
            with open(versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
        except TypeError as e:
            self._logger.error(f"Failed to serialize versions to JSON: {e}")
            # Decide how to handle: skip saving versions, raise error, etc.
            
        # Save metrics
        self._metrics.save_to_file(metrics_file)
        
        self._logger.info(f"Saved prompt versions and metrics to {self._storage_dir}")
    
    def _load_data(self) -> None:
        """Load versions and metrics data from disk."""
        if not self._storage_dir:
            return
            
        # Remove templates.json handling
        # templates_file = os.path.join(self._storage_dir, "templates.json") 
        versions_file = os.path.join(self._storage_dir, "versions.json")
        metrics_file = os.path.join(self._storage_dir, "metrics.json")
        
        # Load templates - REMOVED
        # if os.path.exists(templates_file):
        #     try:
        #         with open(templates_file, 'r') as f:
        #             templates_data = json.load(f)
        #         for t_id, template_data in templates_data.items():
                     # Incorrect: PromptTemplate doesn't have from_dict
        #             # self._templates[t_id] = PromptTemplate.from_dict(template_data) 
        #             # Placeholder: Templates should be loaded by PromptTemplate instance itself
        #             self._logger.warning(f"Skipping loading template '{t_id}' from templates.json (deprecated)")
        #     except Exception as e:
        #         self._logger.error(f"Failed to load templates from {templates_file}: {e}")

        # Reset internal state that should not be loaded here
        self._templates = {} 
        self._active_versions = {} # Reset, will be repopulated below
        self._versions = {} # Reset, will be repopulated below
        
        # Load versions (Assuming PromptVersion.from_dict() exists and is correct)
        if os.path.exists(versions_file):
            try:
                with open(versions_file, 'r') as f:
                    versions_data = json.load(f)
                    
                for t_id, version_list in versions_data.items():
                    self._versions[t_id] = []
                    for version_data in version_list:
                        try:
                            version = PromptVersion.from_dict(version_data)
                            self._versions[t_id].append(version)
                            
                            # Set as active if marked
                            if version.is_active:
                                # Simple assignment (might need logic for multiple actives)
                                self._active_versions[t_id] = version 
                        except Exception as e:
                             self._logger.error(f"Failed to load version data for template {t_id}: {version_data}. Error: {e}")
            except Exception as e:
                self._logger.error(f"Failed to load or parse versions from {versions_file}: {e}")
        
        # Load metrics
        if os.path.exists(metrics_file):
            try:
                # Assuming PromptMetrics.load_from_file is the correct way
                self._metrics = PromptMetrics.load_from_file(metrics_file) 
            except Exception as e:
                 self._logger.error(f"Failed to load metrics from {metrics_file}: {e}")
            
        self._logger.info(f"Loaded prompt versions and metrics from {self._storage_dir}")
    
    def create_template(self, 
                       name: str,
                       description: str,
                       template: str,
                       default_values: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new prompt template.
        
        Args:
            name: Name of the template
            description: Description of the template
            template: Template string with variable placeholders
            default_values: Default values for variables
            
        Returns:
            Template ID
        """
        # Create template
        prompt_template = PromptTemplate(
            name=name,
            description=description,
            template=template,
            default_values=default_values
        )
        
        template_id = prompt_template.template_id
        self._templates[template_id] = prompt_template
        
        # Create initial version
        initial_version = PromptVersion.create_new_version(
            template_id=template_id,
            previous_version=None,
            content={
                "template": template,
                "default_values": default_values or {}
            },
            name="Initial Version",
            description="Initial version of the template"
        )
        
        self._versions[template_id] = [initial_version]
        
        # Set as active version
        initial_version.is_active = True
        self._active_versions[template_id] = initial_version
        
        self._logger.info(f"Created template: {name} ({template_id})")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return template_id
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template or None if not found
        """
        return self._templates.get(template_id)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template info dictionaries
        """
        return [
            {
                "template_id": t_id,
                "name": template.name,
                "description": template.description,
                "created_at": template.created_at.isoformat(),
                "version_count": len(self._versions.get(t_id, [])),
                "active_version": self._active_versions.get(t_id).version if t_id in self._active_versions else None
            }
            for t_id, template in self._templates.items()
        ]
    
    def update_template(self,
                       template_id: str,
                       name: Optional[str] = None,
                       description: Optional[str] = None) -> bool:
        """
        Update a template's metadata reference within the manager.
        NOTE: This method currently DOES NOT update the underlying template source file (YAML).
        Metadata changes must be made directly to the source files.
        This method mainly serves to log the intent and potentially trigger auto-save 
        for related manager state (like versions) if needed in the future.
        
        Args:
            template_id: Template ID to check existence for.
            name: New name (informational only).
            description: New description (informational only).
            
        Returns:
            True if template ID exists in the manager's internal cache, False otherwise.
        """
        template = self.get_template(template_id) # Checks if ID exists in self._templates
        if not template:
            self._logger.warning(f"Attempted to update non-existent template ID: {template_id}")
            return False
            
        # Log the intent to update.
        self._logger.info(f"Metadata update requested for template: {template_id} (Name: {name}, Desc: {description}) - Note: Underlying template source NOT modified by this action.")

        # The actual template object (likely loaded from YAML by PromptTemplate) is not updated here.
        # template.update(name=name, description=description) # This line is incorrect and removed.
        
        # Auto-save if enabled (might save versions/metrics if they changed elsewhere)
        if self._auto_save:
            self._save_data()
            
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template and all its versions.
        
        Args:
            template_id: Template ID
            
        Returns:
            True if successful, False if template not found
        """
        if template_id not in self._templates:
            return False
            
        # Remove template
        del self._templates[template_id]
        
        # Remove versions
        if template_id in self._versions:
            del self._versions[template_id]
            
        # Remove active version
        if template_id in self._active_versions:
            del self._active_versions[template_id]
            
        # Remove test allocations
        if template_id in self._test_allocations:
            del self._test_allocations[template_id]
            
        self._logger.info(f"Deleted template: {template_id}")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return True
    
    def create_version(self,
                      template_id: str,
                      template_string: str,
                      default_values: Optional[Dict[str, Any]] = None,
                      name: Optional[str] = None,
                      description: Optional[str] = None,
                      set_active: bool = False) -> Optional[str]:
        """
        Create a new version of a template.
        
        Args:
            template_id: Template ID
            template_string: New template string
            default_values: New default values (or None to inherit)
            name: Version name
            description: Version description
            set_active: Whether to set as active version
            
        Returns:
            Version ID if successful, None if template not found
        """
        if template_id not in self._templates:
            return None
            
        # Get previous version
        previous_version = None
        if template_id in self._versions and self._versions[template_id]:
            previous_version = self._versions[template_id][-1]
            
        # If no new default values, inherit from previous version
        if default_values is None and previous_version:
            default_values = previous_version.content.get("default_values", {})
            
        # Create new version
        new_version = PromptVersion.create_new_version(
            template_id=template_id,
            previous_version=previous_version,
            content={
                "template": template_string,
                "default_values": default_values or {}
            },
            name=name,
            description=description
        )
        
        # Add to versions list
        if template_id not in self._versions:
            self._versions[template_id] = []
        self._versions[template_id].append(new_version)
        
        # Set as active if requested
        if set_active:
            # Deactivate current active version
            if template_id in self._active_versions:
                self._active_versions[template_id].is_active = False
                
            new_version.is_active = True
            self._active_versions[template_id] = new_version
            
            # Update template
            template = self._templates[template_id]
            template.update(template=template_string, default_values=default_values)
        
        self._logger.info(f"Created version {new_version.version} for template: {template_id}")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return new_version.version_id
    
    def get_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions for a template.
        
        Args:
            template_id: Template ID
            
        Returns:
            List of version info dictionaries
        """
        if template_id not in self._versions:
            return []
            
        return [
            {
                "version_id": version.version_id,
                "version": version.version,
                "name": version.name,
                "description": version.description,
                "created_at": version.created_at.isoformat(),
                "is_active": version.is_active
            }
            for version in self._versions[template_id]
        ]
    
    def set_active_version(self, template_id: str, version_id: str) -> bool:
        """
        Set a version as the active version for a template.
        
        Args:
            template_id: Template ID
            version_id: Version ID
            
        Returns:
            True if successful, False if not found
        """
        if template_id not in self._versions:
            return False
            
        # Find the version
        target_version = None
        for version in self._versions[template_id]:
            if version.version_id == version_id:
                target_version = version
                break
                
        if not target_version:
            return False
            
        # Deactivate current active version
        if template_id in self._active_versions:
            self._active_versions[template_id].is_active = False
            
        # Activate new version
        target_version.is_active = True
        self._active_versions[template_id] = target_version
        
        # Update template
        template = self._templates[template_id]
        template.update(
            template=target_version.content.get("template", ""),
            default_values=target_version.content.get("default_values", {})
        )
        
        self._logger.info(f"Set version {target_version.version} as active for template: {template_id}")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return True
    
    def start_ab_test(self,
                     template_id: str,
                     version_ids: List[str],
                     weights: Optional[List[float]] = None) -> bool:
        """
        Start an A/B test between multiple versions.
        
        Args:
            template_id: Template ID
            version_ids: List of version IDs to test
            weights: Relative weights for each version (or None for equal)
            
        Returns:
            True if successful, False if not found
        """
        if template_id not in self._versions:
            return False
            
        # Validate all versions exist
        versions = []
        for version_id in version_ids:
            found = False
            for version in self._versions[template_id]:
                if version.version_id == version_id:
                    versions.append(version)
                    found = True
                    break
                    
            if not found:
                return False
                
        # Set up weights if not provided
        if weights is None:
            weights = [1.0 / len(versions)] * len(versions)
            
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Store test configuration
        self._test_allocations[template_id] = {
            "versions": version_ids,
            "weights": normalized_weights,
            "allocations": {}  # Will be populated as users interact
        }
        
        self._logger.info(f"Started A/B test for template: {template_id} with {len(version_ids)} versions")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return True
    
    def stop_ab_test(self, template_id: str, winning_version_id: Optional[str] = None) -> bool:
        """
        Stop an A/B test and optionally set a winning version.
        
        Args:
            template_id: Template ID
            winning_version_id: ID of winning version to set as active
            
        Returns:
            True if successful, False if not found
        """
        if template_id not in self._test_allocations:
            return False
            
        # Remove test allocations
        del self._test_allocations[template_id]
        
        # Set winning version as active if specified
        if winning_version_id:
            self.set_active_version(template_id, winning_version_id)
            
        self._logger.info(f"Stopped A/B test for template: {template_id}")
        
        # Auto-save if enabled
        if self._auto_save:
            self._save_data()
            
        return True
    
    def render_prompt(self,
                     template_id: str,
                     variables: Optional[Dict[str, Any]] = None,
                     user_id: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        """
        Render a prompt from a template, with A/B testing support.
        
        Args:
            template_id: Template ID
            variables: Variables for the template
            user_id: User ID for A/B testing
            context: Additional context for metrics
            
        Returns:
            Tuple of (rendered prompt, version ID used)
            
        Raises:
            ValueError: If template not found
        """
        if template_id not in self._templates:
            raise ValueError(f"Template not found: {template_id}")
            
        # Determine which version to use
        version_id = None
        version = None
        
        # If A/B testing is active for this template
        if template_id in self._test_allocations and user_id:
            test_data = self._test_allocations[template_id]
            
            # If user already has an allocation, use it
            if user_id in test_data.get("allocations", {}):
                version_id = test_data["allocations"][user_id]
            else:
                # Randomly assign a version based on weights
                version_id = random.choices(
                    test_data["versions"],
                    weights=test_data["weights"],
                    k=1
                )[0]
                
                # Store the allocation
                test_data["allocations"][user_id] = version_id
            
            # Find the version object
            for v in self._versions[template_id]:
                if v.version_id == version_id:
                    version = v
                    break
                    
        # Otherwise use active version
        if not version and template_id in self._active_versions:
            version = self._active_versions[template_id]
            version_id = version.version_id
            
        # Fallback to latest version if still not found
        if not version and template_id in self._versions and self._versions[template_id]:
            version = self._versions[template_id][-1]
            version_id = version.version_id
            
        if not version:
            raise ValueError(f"No versions found for template: {template_id}")
            
        # Get template string and defaults from version
        template_string = version.content.get("template", "")
        default_values = version.content.get("default_values", {})
        
        # Create a temporary template for rendering
        temp_template = PromptTemplate(
            template=template_string,
            default_values=default_values
        )
        
        # Render the template
        rendered = temp_template.render(variables)
        
        # Record usage for metrics
        usage_id = self._metrics.record_usage(
            template_id=template_id,
            version_id=version_id,
            user_id=user_id,
            context=context
        )
        
        return rendered, usage_id
    
    def record_prompt_performance(self,
                                 usage_id: str,
                                 metrics: Dict[str, Union[float, int, str]],
                                 feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Record performance metrics for a prompt.
        
        Args:
            usage_id: Usage ID from render_prompt
            metrics: Performance metrics (latency, tokens, etc.)
            feedback: User or system feedback
        """
        self._metrics.record_performance(
            usage_id=usage_id,
            metrics=metrics,
            feedback=feedback
        )
        
        # Auto-save if enabled
        if self._auto_save and self._storage_dir:
            metrics_file = os.path.join(self._storage_dir, "metrics.json")
            self._metrics.save_to_file(metrics_file)
    
    def optimize_prompt(self,
                       template_id: str,
                       metric_key: str,
                       higher_is_better: bool = True,
                       min_usage_count: int = 10) -> Optional[str]:
        """
        Optimize a prompt by finding the best-performing version.
        
        Args:
            template_id: Template ID
            metric_key: Metric to optimize for
            higher_is_better: Whether higher values are better
            min_usage_count: Minimum usage count for consideration
            
        Returns:
            ID of best version, or None if not enough data
        """
        best_version_id = self._metrics.recommend_version(
            template_id=template_id,
            optimization_metric=metric_key,
            higher_is_better=higher_is_better,
            min_usage_count=min_usage_count
        )
        
        if best_version_id:
            # Automatically set as active
            self.set_active_version(template_id, best_version_id)
            self._logger.info(f"Optimized template {template_id} based on {metric_key}")
            
        return best_version_id
    
    def get_template_metrics(self,
                           template_id: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get metrics for a template.
        
        Args:
            template_id: Template ID
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Metrics data
        """
        return self._metrics.get_metrics_for_template(
            template_id=template_id,
            start_time=start_time,
            end_time=end_time
        )
    
    def compare_versions(self,
                        template_id: str,
                        version_ids: List[str],
                        metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare metrics between versions.
        
        Args:
            template_id: Template ID
            version_ids: Version IDs to compare
            metric_keys: Metrics to focus on
            
        Returns:
            Comparison data
        """
        return self._metrics.compare_versions(
            template_id=template_id,
            version_ids=version_ids,
            metric_keys=metric_keys
        ) 