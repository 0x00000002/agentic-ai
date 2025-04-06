# src/core/models.py
"""
Core Pydantic models for AI framework components.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
# Assuming ToolCall is defined in src/tools/models.py
from ..tools.models import ToolCall 

class TokenUsage(BaseModel):
    """Model for token usage statistics."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ProviderResponse(BaseModel):
    """Standardized response object returned by AI providers."""
    content: Optional[str] = Field(default=None, description="Primary text content of the response.")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="List of tool calls requested by the model.")
    stop_reason: Optional[str] = Field(default=None, description="Reason why the model stopped generating (e.g., 'stop', 'tool_use', 'max_tokens').")
    usage: Optional[TokenUsage] = Field(default=None, description="Token usage statistics for the request.")
    model: Optional[str] = Field(default=None, description="Identifier of the model that generated the response.")
    error: Optional[str] = Field(default=None, description="Error message if the request failed.")
    # Optional field to include the raw response for debugging
    raw_response: Optional[Any] = Field(default=None, exclude=True) # Exclude from standard serialization

    # Optional: Add a property to easily check if tool calls exist
    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls) 