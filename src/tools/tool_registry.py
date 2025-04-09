"""
Tool registry for managing available tools.
"""
from typing import Dict, List, Any, Optional, Set, Callable, Union
import importlib # Added import
import yaml # Added import

from .interfaces import ToolStrategy
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError
from .models import ToolDefinition, ToolResult
from ..config import get_config, UnifiedConfig # Modified import
from ..config.provider import Provider


# Removed dummy_tool_function
# def dummy_tool_function(query: str) -> str:
#     """A simple dummy tool for testing."""
#     return f"Dummy tool processed query: {query}"


class ToolRegistry:
    """
    Registry for managing tool definitions and implementations.
    Focuses on loading, storing, and formatting tool definitions.
    """
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the tool registry.
        
        Args:
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create()
        # Use UnifiedConfig instance directly for consistency
        self._config = UnifiedConfig.get_instance() 
        self._tools: Dict[str, ToolStrategy] = {} # This seems unused, maybe remove later?
        self._tools_metadata: Dict[str, ToolDefinition] = {}
        self._tool_categories: Dict[str, Set[str]] = {}  # Category name to set of tool names
        
        # Load tool categories from configuration first
        self._load_categories()
        
        # Load tools from the configuration file
        self._load_tools_from_config() # Changed from _register_builtin_tools
    
    def _load_categories(self) -> None:
        """Load tool categories from configuration."""
        # Use get_tool_config() which likely returns the merged dict for 'tools'
        tool_config = self._config.get_tool_config() or {} 
        categories = tool_config.get("categories", {})
        
        for category_name, category_config in categories.items():
            # Initialize empty set for each category
            if isinstance(category_config, dict) and category_config.get("enabled", True):
                self._tool_categories[category_name] = set()
                self._logger.debug(f"Loaded tool category: {category_name}")
            elif isinstance(category_config, bool) and category_config: # Allow simple boolean enablement
                 self._tool_categories[category_name] = set()
                 self._logger.debug(f"Loaded tool category (simple enablement): {category_name}")


    def _load_tools_from_config(self) -> None:
        """Loads tool definitions from the 'tools' section of the merged configuration."""
        self._logger.info("Loading tools from configuration...")
        try:
            # Use get_tool_config() which likely returns the merged dict for 'tools'
            tools_config = self._config.get_tool_config() or {}
            tool_definitions_list = tools_config.get('tools', []) # Access the 'tools' list directly

            if not isinstance(tool_definitions_list, list):
                 self._logger.error(f"Expected a list under 'tools' key in tool config, got {type(tool_definitions_list)}. Skipping tool loading.")
                 return
                 
            if not tool_definitions_list:
                self._logger.warning("No tool definitions found under 'tools' key in configuration.")
                return

            for tool_conf in tool_definitions_list:
                if not isinstance(tool_conf, dict):
                     self._logger.warning(f"Skipping invalid tool config entry (not a dictionary): {tool_conf}")
                     continue
                     
                tool_name = tool_conf.get("name")
                module_path = tool_conf.get("module")
                function_name = tool_conf.get("function")
                description = tool_conf.get("description", "")
                schema = tool_conf.get("parameters_schema", {})
                category = tool_conf.get("category")

                if not all([tool_name, module_path, function_name]):
                    self._logger.error(f"Skipping tool: Missing required fields (name, module, function) in config entry: {tool_conf}")
                    continue

                try:
                    # Don't import/get function here, just store paths
                    # module = importlib.import_module(module_path)
                    # function_callable = getattr(module, function_name)
                    
                    tool_def = ToolDefinition(
                        name=tool_name,
                        description=description,
                        parameters_schema=schema,
                        module_path=module_path,        # Store module path
                        function_name=function_name,    # Store function name
                        function=None                   # Set function to None initially
                    )
                    
                    self.register_tool(tool_name=tool_name, tool_definition=tool_def, category=category)

                # Remove specific import/attribute errors as they won't happen here
                # except ImportError:
                #     self._logger.error(f"Failed to import module '{module_path}' for tool '{tool_name}'. Skipping.")
                # except AttributeError:
                #     self._logger.error(f"Failed to find function '{function_name}' in module '{module_path}' for tool '{tool_name}'. Skipping.")
                except AIToolError as e:
                     self._logger.warning(f"Skipping registration for tool '{tool_name}' from config: {e}") # Changed to warning
                except Exception as e:
                    self._logger.error(f"Unexpected error creating tool definition '{tool_name}' from config: {e}", exc_info=True)

        except Exception as e:
            self._logger.error(f"Failed to load tools from configuration: {e}", exc_info=True)


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