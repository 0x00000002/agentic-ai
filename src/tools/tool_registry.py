"""
Tool registry for managing available tools.
"""
from typing import Dict, List, Any, Optional, Set, Callable, Union
from .interfaces import ToolStrategy
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError
from .models import ToolDefinition, ToolResult
from ..config import get_config
from ..config.provider import Provider
import json
import os
from datetime import datetime


# Dummy function for the built-in tool
def dummy_tool_function(query: str) -> str:
    """A simple dummy tool for testing."""
    return f"Dummy tool processed query: {query}"

class ToolRegistry:
    """
    Registry for managing tool definitions and implementations.
    """
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the tool registry.
        
        Args:
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create()
        self._config = get_config()
        self._tools: Dict[str, ToolStrategy] = {}
        self._tools_metadata: Dict[str, ToolDefinition] = {}
        self._tool_categories: Dict[str, Set[str]] = {}  # Category name to set of tool names
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load tool categories from configuration
        self._load_categories()
        
        # Register built-in tools if enabled
        if self._is_builtin_enabled():
            self._register_builtin_tools()
    
    def _load_categories(self) -> None:
        """Load tool categories from configuration."""
        tool_config = self._config.get_tool_config()
        categories = tool_config.get("categories", {})
        
        for category_name, category_config in categories.items():
            # Initialize empty set for each category
            if category_config.get("enabled", True):
                self._tool_categories[category_name] = set()
                self._logger.debug(f"Loaded tool category: {category_name}")
    
    def _is_builtin_enabled(self) -> bool:
        """Check if built-in tools are enabled in configuration."""
        tool_config = self._config.get_tool_config()
        builtin_config = tool_config.get("built_in", {})
        return builtin_config.get("enabled", True)
    
    def _register_builtin_tools(self) -> None:
        """Register any built-in tools."""
        # Define a simple dummy tool
        dummy_tool_def = ToolDefinition(
            name="dummy_tool",
            description="A simple built-in tool for demonstration and testing purposes.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The input query for the dummy tool."}
                },
                "required": ["query"]
            },
            function=dummy_tool_function # Reference the actual function
        )
        try:
            # Register the dummy tool, assign to a default category if desired
            self.register_tool(
                tool_name=dummy_tool_def.name, 
                tool_definition=dummy_tool_def,
                category="built_in" # Assign to a 'built_in' category
            )
            self._logger.info("Registered built-in dummy_tool.")
        except AIToolError as e:
            # Log if registration fails (e.g., if already registered somehow)
             self._logger.warning(f"Could not register built-in dummy_tool: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error registering built-in dummy_tool: {e}", exc_info=True)
    
    def register_tool(self, 
                     tool_name: str, 
                     tool_definition: ToolDefinition,
                     category: Optional[str] = None) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool_name: Unique name for the tool
            tool_definition: Tool definition object
            category: Optional category to assign the tool to
            
        Raises:
            AIToolError: If registration fails (e.g., name already exists)
        """
        if tool_name in self._tools_metadata:
            raise AIToolError(f"Tool '{tool_name}' already registered")
        
        try:
            # Store the definition
            self._tools_metadata[tool_name] = tool_definition
            
            # Add to category if specified
            if category:
                if category not in self._tool_categories:
                    # Create category if it doesn't exist (or wasn't in config)
                    self._tool_categories[category] = set()
                    self._logger.debug(f"Created new tool category during registration: {category}")
                
                self._tool_categories[category].add(tool_name)
                self._logger.debug(f"Tool {tool_name} added to category {category}")
            
            self._logger.info(f"Tool registered: {tool_name}")
            
            # Initialize usage stats if not already present
            if tool_name not in self.usage_stats:
                self.usage_stats[tool_name] = {
                    "uses": 0,
                    "successes": 0,
                    "last_used": None,
                    "first_used": datetime.now().isoformat()
                }
            
        except Exception as e:
            self._logger.error(f"Tool registration failed for {tool_name}: {str(e)}", exc_info=True)
            # Remove potentially partially registered data
            if tool_name in self._tools_metadata:
                del self._tools_metadata[tool_name]
            if category and tool_name in self._tool_categories.get(category, set()):
                 self._tool_categories[category].remove(tool_name)
                 
            raise AIToolError(f"Failed to register tool {tool_name}: {str(e)}")
    
    def get_tool_names(self) -> List[str]:
        """
        Get names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools_metadata.keys())
    
    def get_category_tools(self, category: str) -> List[str]:
        """
        Get all tool names in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of tool names in the category
        """
        return list(self._tool_categories.get(category, set()))
    
    def get_categories(self) -> List[str]:
        """
        Get all available tool categories.
        
        Returns:
            List of category names
        """
        return list(self._tool_categories.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool exists, False otherwise
        """
        return tool_name in self._tools_metadata
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools_metadata.get(tool_name)
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        Get the description of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool description or None if not found
        """
        tool_def = self.get_tool(tool_name)
        return tool_def.description if tool_def else None
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the parameters schema of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool parameters schema or None if not found
        """
        tool_def = self.get_tool(tool_name)
        return tool_def.parameters_schema if tool_def else None
    
    def get_all_tools(self) -> Dict[str, ToolStrategy]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to implementations
        """
        self._logger.warning("get_all_tools() called, returning internal _tools dict. Consider using get_all_tool_definitions().")
        # This internal dict _tools was populated by DefaultToolStrategy, which was removed.
        # This method might be dead code or needs refactoring if still used.
        # Returning empty for safety as _tools isn't populated anymore.
        return {} # Return empty as _tools is no longer populated correctly
    
    def get_all_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """
        Get all tool definitions.
        
        Returns:
            Dictionary mapping tool names to definitions
        """
        return self._tools_metadata.copy()
    
    def format_tools_for_provider(self,
                                 provider_name: str,
                                 tool_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tool definitions for a specific provider's API.

        Args:
            provider_name: Name of the provider (e.g., "ANTHROPIC", "OPENAI", "GOOGLE").
            tool_names: Optional set of specific tools to format. If None, format all.

        Returns:
            List of tool definitions formatted for the provider.
            Returns an empty list if the provider is unknown or formatting fails.
        """
        formatted_tools = []

        # Filter tools if specific ones requested
        tools_to_format = {
            name: defn for name, defn in self._tools_metadata.items()
            if tool_names is None or name in tool_names
        }

        for tool_name, tool_def in tools_to_format.items():
            try:
                # Common structure
                base_tool_format = {
                    "name": tool_name,
                    "description": tool_def.description,
                }

                # Provider-specific parameter schema structure
                if "OPENAI" in provider_name:
                    base_tool_format["parameters"] = tool_def.parameters_schema
                    formatted_tools.append({"type": "function", "function": base_tool_format})
                elif "ANTHROPIC" in provider_name:
                     # Anthropic uses 'input_schema'
                    base_tool_format["input_schema"] = tool_def.parameters_schema
                    formatted_tools.append(base_tool_format)
                elif "GEMINI" in provider_name:
                     # Gemini uses 'parameters'
                    base_tool_format["parameters"] = tool_def.parameters_schema
                    # Wrap in 'function_declaration' for Gemini
                    formatted_tools.append({"function_declaration": base_tool_format})
                else:
                    # Fallback for other providers or default
                    self._logger.warning(f"Using default tool format for provider: {provider_name}")
                    base_tool_format["parameters"] = tool_def.parameters_schema
                    formatted_tools.append(base_tool_format)

            except Exception as e:
                 self._logger.error(f"Failed to format tool '{tool_name}' for provider '{provider_name}': {e}")
                 # Optionally skip this tool or return [] depending on desired robustness

        return formatted_tools

    def update_usage_stats(self, tool_name: str, success: bool, request_id: Optional[str] = None, duration_ms: Optional[int] = None) -> None:
        """
        Update usage statistics for a tool.
        
        Args:
            tool_name: Name of the tool
            success: Whether the tool execution was successful
            request_id: Optional request ID for metrics tracking
            duration_ms: Optional duration in milliseconds
        """
        if tool_name not in self.usage_stats:
            self.usage_stats[tool_name] = {
                "uses": 0,
                "successes": 0,
                "last_used": None,
                "first_used": datetime.now().isoformat()
            }
            
        self.usage_stats[tool_name]["uses"] += 1
        if success:
            self.usage_stats[tool_name]["successes"] += 1
        self.usage_stats[tool_name]["last_used"] = datetime.now().isoformat()
        
        self._logger.debug(f"Updated usage stats for {tool_name}: uses={self.usage_stats[tool_name]['uses']}, successes={self.usage_stats[tool_name]['successes']}")
        
        # Track in metrics service if request_id is provided
        if request_id:
            try:
                from ..metrics.request_metrics import RequestMetricsService
                metrics_service = RequestMetricsService()
                
                # Track tool usage in metrics service
                metrics_service.track_tool_usage(
                    request_id=request_id,
                    tool_id=tool_name,
                    duration_ms=duration_ms,
                    success=success
                )
            except Exception as e:
                self._logger.warning(f"Failed to track tool usage in metrics service: {str(e)}")
    
    def get_usage_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics for tools.
        
        Args:
            tool_name: Optional name of a specific tool
            
        Returns:
            Dictionary of usage statistics
        """
        if tool_name:
            return self.usage_stats.get(tool_name, {})
        return self.usage_stats
    
    def get_recommended_tools(self, prompt: str, max_tools: int = 5) -> List[str]:
        """
        Get recommended tools based on usage statistics.
        
        This is a simple implementation that returns the most frequently used tools.
        In a real implementation, this could use AI or more sophisticated heuristics.
        
        Args:
            prompt: User prompt to analyze
            max_tools: Maximum number of tools to recommend
            
        Returns:
            List of recommended tool names
        """
        # Sort tools by usage count
        sorted_tools = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]["uses"],
            reverse=True
        )
        
        # Return the top N tools
        return [tool_name for tool_name, _ in sorted_tools[:max_tools]]
    
    def save_stats(self, file_path: str) -> None:
        """
        Save usage statistics to a file.
        
        Args:
            file_path: Path to save the statistics to
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
            self._logger.info(f"Saved usage statistics to {file_path}")
        except Exception as e:
            self._logger.error(f"Failed to save usage statistics: {str(e)}")
    
    def load_stats(self, file_path: str) -> None:
        """
        Load usage statistics from a file.
        
        Args:
            file_path: Path to load the statistics from
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.usage_stats = json.load(f)
                self._logger.info(f"Loaded usage statistics from {file_path}")
        except Exception as e:
            self._logger.error(f"Failed to load usage statistics: {str(e)}") 