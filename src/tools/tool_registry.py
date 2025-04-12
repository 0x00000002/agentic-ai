"""
Tool registry for managing available **internal** tools.
"""
from typing import Dict, List, Any, Optional, Set, Callable, Union
import importlib # Added import
import yaml # Added import
from pydantic import ValidationError

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
    Registry for managing internal tool definitions.
    Focuses on loading, storing, and formatting **internal** tool definitions from tools.yml.
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
        self._tools_metadata: Dict[str, ToolDefinition] = {}
        self._tool_categories: Dict[str, Set[str]] = {}  # Keep category logic for internal tools
        
        # Load tool categories from configuration first
        self._load_categories()
        
        # Load internal tool definitions from the configuration file
        self._load_internal_tools_from_config()
    
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


    def _load_internal_tools_from_config(self) -> None:
        """Loads internal tool definitions from the 'tools' section of the merged configuration."""
        self._logger.info("Loading internal tools from configuration...")
        try:
            # Use get_tool_config() which likely returns the merged dict for 'tools'
            tools_config = self._config.get_tool_config() or {}
            tool_definitions_list = tools_config.get('tools', []) # Access the 'tools' list directly

            if not isinstance(tool_definitions_list, list):
                 self._logger.error(f"Expected a list under 'tools' key in tool config, got {type(tool_definitions_list)}. Skipping internal tool loading.")
                 return
                 
            if not tool_definitions_list:
                self._logger.warning("No internal tool definitions found under 'tools' key in configuration.")
                return

            for tool_conf in tool_definitions_list:
                if not isinstance(tool_conf, dict):
                     self._logger.warning(f"Skipping invalid internal tool config entry (not a dictionary): {tool_conf}")
                     continue
                
                # Add source explicitly for validation
                tool_conf['source'] = 'internal' 
                tool_name = tool_conf.get("name") # Get name for logging

                try:
                    # Validate and create ToolDefinition using Pydantic
                    tool_def = ToolDefinition(**tool_conf)
                    
                    # Register the validated internal tool definition
                    self.register_internal_tool(tool_definition=tool_def)

                except ValidationError as e:
                    self._logger.error(f"Validation failed for internal tool '{tool_name or 'unnamed'}': {e}")
                except AIToolError as e:
                     self._logger.warning(f"Skipping registration for internal tool '{tool_name}' from config: {e}")
                except Exception as e:
                    self._logger.error(f"Unexpected error creating internal tool definition '{tool_name}' from config: {e}", exc_info=True)

        except Exception as e:
            self._logger.error(f"Failed to load internal tools from configuration: {e}", exc_info=True)


    def register_internal_tool(self, tool_definition: ToolDefinition) -> None:
        """
        Register an internal tool definition in the registry.
        Ensures the source is 'internal'.
        
        Args:
            tool_definition: Tool definition object (source must be 'internal')
            
        Raises:
            AIToolError: If registration fails (e.g., name exists, wrong source)
        """
        if tool_definition.source != 'internal':
             raise AIToolError(f"Cannot register tool '{tool_definition.name}' with source '{tool_definition.source}' in Internal ToolRegistry.")
             
        tool_name = tool_definition.name
        category = tool_definition.category # Get category from definition
        
        if tool_name in self._tools_metadata:
            raise AIToolError(f"Internal tool '{tool_name}' already registered")
        
        try:
            # Store the definition
            self._tools_metadata[tool_name] = tool_definition
            
            # Add to category if specified
            if category:
                if category not in self._tool_categories:
                    # Create category if it doesn't exist (or wasn't in config)
                    self._tool_categories[category] = set()
                    self._logger.debug(f"Created new internal tool category during registration: {category}")
                
                self._tool_categories[category].add(tool_name)
                self._logger.debug(f"Internal tool {tool_name} added to category {category}")
            
            self._logger.info(f"Internal tool registered: {tool_name}")
            
        except Exception as e:
            self._logger.error(f"Internal tool registration failed for {tool_name}: {str(e)}", exc_info=True)
            # Remove potentially partially registered data
            if tool_name in self._tools_metadata:
                del self._tools_metadata[tool_name]
            if category and tool_name in self._tool_categories.get(category, set()):
                 self._tool_categories[category].remove(tool_name)
                 
            raise AIToolError(f"Failed to register internal tool {tool_name}: {str(e)}")

    # --- Methods to retrieve internal tools/definitions ---

    def list_internal_tool_definitions(self) -> List[ToolDefinition]:
        """Returns a list of all registered internal tool definitions."""
        return list(self._tools_metadata.values())

    def get_internal_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get an internal tool definition by name.
        
        Args:
            tool_name: Name of the internal tool to retrieve
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools_metadata.get(tool_name)

    # --- Deprecate or remove methods that don't make sense for internal-only registry? ---
    # get_tool_names, has_tool could be kept, others might be removed/renamed
    # get_tool_description, get_tool_schema can use get_internal_tool_definition
    # get_category_tools, get_categories relate to internal tools

    # Keep for now, but they only operate on internal tools:
    def get_tool_names(self) -> List[str]:
        return list(self._tools_metadata.keys())
    
    def get_category_tools(self, category: str) -> List[str]:
        return list(self._tool_categories.get(category, set()))
    
    def get_categories(self) -> List[str]:
        return list(self._tool_categories.keys())
    
    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools_metadata
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        tool_def = self.get_internal_tool_definition(tool_name)
        return tool_def.description if tool_def else None
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        tool_def = self.get_internal_tool_definition(tool_name)
        # Use .model_dump() for schema if Pydantic v2, or access directly
        return tool_def.parameters_schema if tool_def else None
    
    # Remove get_all_tools as it returned ToolStrategy which seems unused
    # def get_all_tools(self) -> Dict[str, ToolStrategy]: ...
    
    # Rename get_all_tool_definitions
    def get_all_internal_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Gets a dictionary of all registered internal tool definitions."""
        return self._tools_metadata.copy()
    
    def format_tools_for_provider(self,
                                 provider_name: str,
                                 tool_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Formats the specified internal tool definitions for a given provider.
        # ... (rest of the logic - ensure it uses self._tools_metadata)
        """
        formatted_tools = []
        target_tool_names = tool_names if tool_names is not None else self.get_tool_names()

        # Normalize provider name for matching (e.g., handle case, map synonyms)
        provider_name = provider_name.upper()

        for name in target_tool_names:
            tool_def = self.get_internal_tool_definition(name)
            if not tool_def:
                self._logger.warning(f"Tool '{name}' requested for formatting not found in registry.")
                continue

            try:
                # Default format (adjust per provider below)
                provider_format = {}
                
                # Provider-specific adjustments (Example: Gemini needs 'functionDeclarations')
                if "GEMINI" in provider_name:
                    # Gemini requires a specific FunctionDeclaration structure
                    # Note: This might need the 'google-generativeai' library or manual construction
                     provider_format = {
                         "name": tool_def.name,
                         "description": tool_def.description,
                         "parameters": {
                             "type": "object",
                             "properties": tool_def.parameters_schema.get('properties', {}),
                             "required": tool_def.parameters_schema.get('required', [])
                         }
                     }
                elif "OPENAI" in provider_name:
                    # OpenAI requires type: function and function details nested
                    provider_format = {
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": tool_def.parameters_schema # Use the actual schema dict
                        }
                    }
                elif "ANTHROPIC" in provider_name:
                     # Anthropic uses a simpler, flat structure
                     provider_format = {
                         "name": tool_def.name,
                         "description": tool_def.description,
                         "input_schema": tool_def.parameters_schema # Anthropic uses input_schema
                     }
                else:
                    # Fallback/Default if provider unknown (similar to Anthropic?)
                    self._logger.warning(f"Using default tool format for unknown provider: {provider_name}")
                    provider_format = {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "input_schema": tool_def.parameters_schema
                    }

                # Add the formatted tool to the list if not empty
                if provider_format:
                    formatted_tools.append(provider_format)
            except Exception as e:
                 self._logger.error(f"Error formatting tool '{name}' for provider '{provider_name}': {e}", exc_info=True)

        return formatted_tools 