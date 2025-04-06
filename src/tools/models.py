"""
Models for tool-related functionality.
"""
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum


class ToolCall(BaseModel):
    """Model representing a request from the AI to call a specific tool."""
    name: str = Field(..., description="The name of the tool to call.")
    arguments: Dict[str, Any] = Field(..., description="The arguments to pass to the tool, usually as a dictionary.")
    id: Optional[str] = Field(default=None, description="An optional unique identifier for the tool call, provided by some APIs (e.g., OpenAI).")


class ToolExecutionStatus(str, Enum):
    """Enum for tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolDefinition(BaseModel):
    """Model for tool metadata and configuration."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(..., description="JSON schema for tool parameters")
    function: Callable = Field(..., description="The actual tool function to execute")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional tool metadata")


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