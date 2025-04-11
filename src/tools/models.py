"""
Models for tool-related functionality.
"""
from typing import Dict, Any, List, Optional, Callable, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from enum import Enum


class ToolCall(BaseModel):
    """Model for a tool call requested by the AI."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolExecutionStatus(Enum):
    """Enum for tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    TIMEOUT = "timeout"


class ToolDefinition(BaseModel):
    """Represents the definition of a tool, either internal or external (MCP)."""
    name: str = Field(..., description="Unique name/ID for the tool.")
    description: str = Field(..., description="Description of what the tool does.")
    parameters_schema: Dict[str, Any] = Field(alias="inputSchema", default_factory=dict, description="JSON Schema for the tool's input parameters.")
    source: Literal["internal", "mcp"] = Field(..., description="Source of the tool ('internal' Python function or 'mcp' external server).")
    speed: Optional[Literal["instant", "fast", "medium", "slow", "variable"]] = Field("medium", description="Estimated execution speed.")
    safety: Optional[Literal["native", "sandboxed", "external"]] = Field("native", description="Estimated safety/trust level.")

    # --- Internal Tool Specific --- 
    module: Optional[str] = Field(None, description="Module path for internal tools (e.g., src.tools.core.calculator). Required if source is 'internal'.")
    function: Optional[str] = Field(None, description="Function name for internal tools (e.g., calculate). Required if source is 'internal'.")
    category: Optional[str] = Field(None, description="Optional category for internal tools.")

    # --- MCP Tool Specific ---
    mcp_server_name: Optional[str] = Field(None, description="The name of the MCP server definition in mcp.yml that provides this tool. Required if source is 'mcp'.")
    
    # Private fields (not part of schema)
    _function_cache: Optional[Callable] = None

    @model_validator(mode='before')
    @classmethod
    def check_conditional_fields(cls, values):
        source = values.get('source')
        if source == 'internal':
            if not values.get('module') or not values.get('function'):
                raise ValueError("'module' and 'function' are required for internal tools")
            if values.get('mcp_server_name'):
                 raise ValueError("'mcp_server_name' cannot be set for internal tools")
        elif source == 'mcp':
            if not values.get('mcp_server_name'):
                raise ValueError("'mcp_server_name' is required for mcp tools")
            if values.get('module') or values.get('function') or values.get('category'):
                 raise ValueError("'module', 'function', and 'category' cannot be set for mcp tools")
        return values

    model_config = ConfigDict(
        populate_by_name = True, # Allows using inputSchema alias
        use_enum_values = True, # Ensure Literal values are used correctly
        extra = 'ignore' # Ignore extra fields if they come from config
    )


class ToolCallRequest(BaseModel):
    """Model for a complete tool call request."""
    tool_calls: List[ToolCall]
    content: str
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """Model for a tool execution result."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None 