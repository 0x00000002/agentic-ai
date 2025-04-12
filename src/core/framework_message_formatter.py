"""
Framework Message Formatter

Formats internal framework events (tool results, errors, status updates) 
into user-readable messages.
"""

from typing import Dict, Any, Optional
from ..utils.logger import LoggerInterface, LoggerFactory

class FrameworkMessageFormatter:
    """Formats structured framework event data into presentable strings."""
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the formatter.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or LoggerFactory.create(name="framework_message_formatter")

    def format_message(self, event_data: Dict[str, Any]) -> str:
        """
        Formats the event data into a user-readable string.

        Args:
            event_data: Dictionary containing the event details.
                        Expected keys: 'type' (e.g., 'tool_result', 'error', 'status'),
                        'data' (content specific to the type).

        Returns:
            A formatted string representation of the event.
        """
        event_type = event_data.get("type", "info")
        data = event_data.get("data", {})
        
        try:
            if event_type == "tool_result":
                tool_name = data.get("tool_name", "Unknown tool")
                tool_output = data.get("output", "No output")
                # Consider adding input args if available and relevant
                # Limit output length? Add formatting?
                return f"Tool `{tool_name}` executed:\n```\n{str(tool_output)}\n```"
            
            elif event_type == "tool_error":
                tool_name = data.get("tool_name", "Unknown tool")
                error_message = data.get("error_message", "An unspecified error occurred during tool execution.")
                return f"Error executing tool `{tool_name}`: {error_message}"
                
            elif event_type == "error":
                error_message = data.get("error_message", "An unspecified error occurred.")
                component = data.get("component", "System")
                return f"An error occurred in {component}: {error_message}"
                
            elif event_type == "status":
                 status_message = data.get("status_message", "Update")
                 details = data.get("details", "")
                 return f"**{status_message}**\n{details}".strip()
                 
            elif event_type == "info":
                message = data.get("message", "Information message.")
                return f"Info: {message}"
                
            else:
                # Fallback for unknown types
                self.logger.warning(f"Formatting unknown event type: {event_type}")
                return f"Received event: {event_type}\nData: {str(data)}"
                
        except Exception as e:
            self.logger.error(f"Error formatting message (type: {event_type}): {e}")
            # Fallback to raw data presentation on formatting error
            return f"[Error formatting message]\nEvent Type: {event_type}\nData: {str(data)}" 