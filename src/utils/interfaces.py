"""
Interfaces for logging implementations.
"""
from typing import Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class LoggerInterface(Protocol):
    """Interface for logging implementations."""
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        ...
    
    def info(self, message: str) -> None:
        """Log an info message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...
    
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message. Set exc_info=True to include exception info."""
        ...
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        ... 