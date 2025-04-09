"""
Models for tool-related functionality.
"""
from typing import Dict, Any, List, Optional, Callable, Union
from pydantic import BaseModel, Field, ConfigDict
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
    """Model for a tool definition."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    # Add module path and function name for lazy loading
    module_path: str
    function_name: str
    # Make function optional, default to None
    function: Optional[Callable[..., Any]] = Field(default=None, exclude=True) # Exclude from serialization

    # Use ConfigDict instead of nested class Config
    model_config = ConfigDict(arbitrary_types_allowed=True)


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