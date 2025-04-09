"""
Tool Manager Module

This module provides a manager for coordinating tool operations in the Agentic-AI framework.
"""
from typing import Dict, Any, List, Optional, Set, Union, TYPE_CHECKING
import json
import time

from src.utils.logger import LoggerFactory, LoggerInterface
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_executor import ToolExecutor
from src.tools.tool_stats_manager import ToolStatsManager
from src.tools.models import ToolDefinition, ToolResult, ToolExecutionStatus
from src.exceptions import AIToolError, ErrorHandler
from src.config.unified_config import UnifiedConfig

if TYPE_CHECKING:
    from src.agents.tool_finder_agent import ToolFinderAgent
    from src.agents import AgentFactory


class ToolManager:
    """
    Manager for coordinating tool operations in the Agentic-AI framework.
    
    This class coordinates tool registration, discovery, and execution.
    It works with the ToolRegistry for definitions and ToolStatsManager for usage statistics.
    (Removed ToolFinderAgent dependency)
    """
    
    def __init__(self, unified_config: Optional[UnifiedConfig] = None, 
                 logger: Optional[LoggerInterface] = None, 
                 tool_registry: Optional[ToolRegistry] = None, 
                 tool_executor: Optional[ToolExecutor] = None,
                 tool_stats_manager: Optional[ToolStatsManager] = None):
        """
        Initialize the tool manager.
        
        Args:
            unified_config: Optional UnifiedConfig instance
            logger: Optional logger instance
            tool_registry: Optional tool registry
            tool_executor: Optional tool executor
            tool_stats_manager: Optional stats manager instance
        """
        self.logger = logger or LoggerFactory.create("tool_manager")
        self.config = unified_config or UnifiedConfig.get_instance()
        
        # Load tool configuration
        self.tool_config = self.config.get_tool_config()
        
        # Initialize registry and executor
        self.tool_registry = tool_registry or ToolRegistry(logger=self.logger)
        
        # Configure tool executor with settings from config
        executor_config = self.tool_config.get("execution", {})
        self.tool_executor = tool_executor or ToolExecutor(
            logger=self.logger,
            timeout=executor_config.get("timeout", 30),
            max_retries=executor_config.get("max_retries", 3)
        )
        
        # Initialize stats manager (it handles loading stats internally if configured)
        self.tool_stats_manager = tool_stats_manager or ToolStatsManager(logger=self.logger, unified_config=self.config)
        
        self.logger.info("Tool manager initialized (AIToolFinder removed)")
        
    def register_tool(self, tool_name: str, tool_definition: ToolDefinition) -> None:
        """
        Register a tool with the tool registry.
        
        Args:
            tool_name: Name of the tool
            tool_definition: Tool definition object
        """
        self.tool_registry.register_tool(tool_name, tool_definition)
        
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            ToolResult object with execution results
        """
        try:
            # Get the tool definition
            tool_definition = self.tool_registry.get_tool(tool_name)
            
            if not tool_definition:
                error_message = f"Tool not found: {tool_name}"
                self.logger.error(error_message)
                return ToolResult(
                    success=False,
                    error=error_message,
                    result=None,
                    tool_name=tool_name
                )
                
            # Execute the tool using configs from tool_config if applicable
            tool_specific_config = self.config.get_tool_config(tool_name)
            execution_params = {**kwargs}
            
            # Add any tool-specific configuration parameters
            if tool_specific_config:
                for param, value in tool_specific_config.items():
                    if param not in execution_params:
                        execution_params[param] = value
            
            # Track start time for metrics
            start_time = time.time()
            
            # Extract request_id if present for metrics tracking
            request_id = kwargs.get("request_id")
            
            # Execute the tool
            result = self.tool_executor.execute(tool_definition, **execution_params)
            
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Delegate stats update to ToolStatsManager
            # ToolStatsManager checks internally if tracking is enabled
            self.tool_stats_manager.update_stats(
                tool_name=tool_name,
                success=result.success,
                duration_ms=execution_time_ms,
                request_id=request_id # Pass request_id if needed by stats manager
            )
            
            return result
        except Exception as e:
            # Use error handler for standardized error handling
            tool_error = AIToolError(f"Error executing tool {tool_name}: {str(e)}", tool_name=tool_name)
            error_response = ErrorHandler.handle_error(tool_error, self.logger)
            
            # Return a tool result with the error
            return ToolResult(
                success=False,
                error=error_response["message"],
                result=None,
                tool_name=tool_name
            )
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool information or None if not found
        """
        tool_definition = self.tool_registry.get_tool(tool_name)
        
        if not tool_definition:
            return None
            
        # Get usage stats
        usage_stats = self.tool_stats_manager.get_stats(tool_name)
        
        # Get any additional tool configuration
        tool_config = self.config.get_tool_config(tool_name)
        
        return {
            "name": tool_name,
            "description": tool_definition.description,
            "parameters": tool_definition.parameters_schema,
            "usage_stats": usage_stats,
            "config": tool_config
        }
    
    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """
        Get definitions for all registered tools.
        
        Returns:
            Dictionary mapping tool names to ToolDefinition objects.
        """
        # Directly return the definitions from the registry
        return self.tool_registry.get_all_tool_definitions()
    
    def format_tools_for_model(self, model_id: str, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tools for use with a specific model.
        
        Args:
            model_id: Model identifier
            tool_names: Optional list of tool names to format
            
        Returns:
            List of formatted tools for the model's provider
        """
        # Get the model configuration
        model_config = self.config.get_model_config(model_id)
        if not model_config or "provider" not in model_config:
            self.logger.warning(f"Unable to determine provider for model {model_id}, using default tool format")
            return []
            
        # Get the provider from the model configuration
        provider_name = model_config["provider"].upper()
            
        # Convert list to set if provided
        tool_names_set = set(tool_names) if tool_names else None
        
        # Format tools for the provider
        return self.tool_registry.format_tools_for_provider(provider_name, tool_names_set)
    
    def save_usage_stats(self, file_path: Optional[str] = None) -> None:
        """
        Save usage statistics to a file.
        
        Args:
            file_path: Optional path to save stats. Uses configured path if None.
        """
        # Delegate to ToolStatsManager
        self.tool_stats_manager.save_stats(file_path)
    
    def load_usage_stats(self, file_path: Optional[str] = None) -> None:
        """
        Load usage statistics from a file.
        
        Args:
            file_path: Optional path to load stats from. Uses configured path if None.
        """
        # Delegate to ToolStatsManager
        self.tool_stats_manager.load_stats(file_path) 