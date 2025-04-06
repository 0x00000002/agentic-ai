"""
Exceptions module for the AI framework.

This module contains custom exception classes for the AI framework.
It organizes exceptions into a hierarchy based on their domain and purpose.
"""
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import traceback


class AIFrameworkError(Exception):
    """Base class for all AI framework exceptions."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
        """
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    def __str__(self):
        """String representation of the error."""
        if self.error_code:
            return f"{self.error_code}: {self.message}"
        return self.message


# Alias for backward compatibility
AIError = AIFrameworkError


class AISetupError(AIFrameworkError):
    """Error during AI setup or initialization."""
    
    def __init__(self, message: str, component: Optional[str] = None):
        """
        Initialize the setup error.
        
        Args:
            message: Error message
            component: Component that failed to initialize
        """
        self.component = component
        error_code = f"SETUP_{component.upper()}" if component else "SETUP"
        super().__init__(message, error_code)


class AIConfigError(AIFrameworkError):
    """Error in configuration."""
    
    def __init__(self, message: str, config_name: Optional[str] = None):
        """
        Initialize the config error.
        
        Args:
            message: Error message
            config_name: Name of the configuration that caused the error
        """
        self.config_name = config_name
        error_code = f"CONFIG_{config_name.upper()}" if config_name else "CONFIG"
        super().__init__(message, error_code)


class AIProcessingError(AIFrameworkError):
    """Error during AI request processing."""
    
    def __init__(self, message: str, component: Optional[str] = None):
        """
        Initialize the processing error.
        
        Args:
            message: Error message
            component: Component where the error occurred
        """
        self.component = component
        error_code = f"PROCESSING_{component.upper()}" if component else "PROCESSING"
        super().__init__(message, error_code)


class AIRequestError(AIFrameworkError):
    """Error during AI request execution."""
    
    def __init__(self, message: str, provider: Optional[str] = None, original_error: Optional[Exception] = None):
        """
        Initialize the request error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            original_error: Original exception that caused this error
        """
        self.provider = provider
        self.original_error = original_error
        error_code = f"REQUEST_{provider.upper()}" if provider else "REQUEST"
        super().__init__(message, error_code)


class AIProviderError(AIFrameworkError):
    """Error from an AI provider."""
    
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None):
        """
        Initialize the provider error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            status_code: HTTP status code if available
        """
        self.provider = provider
        self.status_code = status_code
        error_code = f"PROVIDER_{provider.upper()}" if provider else "PROVIDER"
        if status_code:
            error_code = f"{error_code}_{status_code}"
        super().__init__(message, error_code)


class AITimeoutError(AIFrameworkError):
    """Error when a request times out."""
    
    def __init__(self, message: str, provider: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize the timeout error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            timeout: Timeout value in seconds
        """
        self.provider = provider
        self.timeout = timeout
        error_code = f"TIMEOUT_{provider.upper()}" if provider else "TIMEOUT"
        super().__init__(message, error_code)


class AIToolError(AIFrameworkError):
    """Error during tool execution."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None):
        """
        Initialize the tool error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed
        """
        self.tool_name = tool_name
        error_code = f"TOOL_{tool_name.upper()}" if tool_name else "TOOL"
        super().__init__(message, error_code)


class AIAgentError(AIFrameworkError):
    """Error during agent execution."""
    
    def __init__(self, message: str, agent_id: Optional[str] = None):
        """
        Initialize the agent error.
        
        Args:
            message: Error message
            agent_id: ID of the agent that failed
        """
        self.agent_id = agent_id
        error_code = f"AGENT_{agent_id.upper()}" if agent_id else "AGENT"
        super().__init__(message, error_code)


class AIAuthenticationError(AIFrameworkError):
    """Error during authentication with a provider."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        """
        Initialize the authentication error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
        """
        self.provider = provider
        error_code = f"AUTH_{provider.upper()}" if provider else "AUTH"
        super().__init__(message, error_code)


class AICredentialsError(AIFrameworkError):
    """Error related to missing or invalid credentials."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        """
        Initialize the credentials error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
        """
        self.provider = provider
        error_code = f"CREDENTIALS_{provider.upper()}" if provider else "CREDENTIALS"
        super().__init__(message, error_code)


class AIRateLimitError(AIFrameworkError):
    """Error when a rate limit is exceeded."""
    
    def __init__(self, message: str, provider: Optional[str] = None, retry_after: Optional[int] = None):
        """
        Initialize the rate limit error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            retry_after: Seconds to wait before retrying
        """
        self.provider = provider
        self.retry_after = retry_after
        error_code = f"RATE_LIMIT_{provider.upper()}" if provider else "RATE_LIMIT"
        super().__init__(message, error_code)


class AIInvalidArgumentError(AIFrameworkError):
    """Error for invalid arguments passed to functions."""
    
    def __init__(self, message: str, argument: Optional[str] = None):
        """
        Initialize the invalid argument error.
        
        Args:
            message: Error message
            argument: Name of the invalid argument
        """
        self.argument = argument
        error_code = f"INVALID_ARG_{argument.upper()}" if argument else "INVALID_ARG"
        super().__init__(message, error_code)


class AIResponseError(AIFrameworkError):
    """Error related to AI response processing."""
    
    def __init__(self, message: str, response_id: Optional[str] = None):
        """
        Initialize the response error.
        
        Args:
            message: Error message
            response_id: ID of the response that caused the error
        """
        self.response_id = response_id
        error_code = f"RESPONSE_{response_id.upper()}" if response_id else "RESPONSE"
        super().__init__(message, error_code)


class ConversationError(AIFrameworkError):
    """Error related to conversation management."""
    
    def __init__(self, message: str, conversation_id: Optional[str] = None):
        """
        Initialize the conversation error.
        
        Args:
            message: Error message
            conversation_id: ID of the conversation where the error occurred
        """
        self.conversation_id = conversation_id
        error_code = f"CONVERSATION_{conversation_id.upper()}" if conversation_id else "CONVERSATION"
        super().__init__(message, error_code)


class ErrorHandler:
    """
    Centralized error handling for consistent error management.
    """
    
    @staticmethod
    def handle_error(error: Exception, logger=None) -> Dict[str, Any]:
        """
        Handle an exception and return a standardized error response.
        
        Args:
            error: The exception to handle
            logger: Optional logger to log the error
            
        Returns:
            Standardized error response
        """
        # Get error details
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        # Log the error if logger provided
        if logger:
            logger.error(f"Error: {str(error)}")
            logger.debug(f"Stack trace: {''.join(stack_trace)}")
        
        # Create a standard error response
        if isinstance(error, AIFrameworkError):
            error_response = {
                "status": "error",
                "error_type": error.__class__.__name__,
                "error_code": error.error_code,
                "message": error.message,
                "stack_trace": stack_trace if logger and hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG) else None
            }
            
            # Add specific information based on error type
            if isinstance(error, AIProviderError):
                error_response["provider"] = error.provider
                error_response["status_code"] = error.status_code
            elif isinstance(error, AITimeoutError):
                error_response["provider"] = error.provider
                error_response["timeout"] = error.timeout
            elif isinstance(error, AIToolError):
                error_response["tool_name"] = error.tool_name
            elif isinstance(error, AIAgentError):
                error_response["agent_id"] = error.agent_id
            
        else:
            # Generic error handling for non-framework errors
            error_response = {
                "status": "error",
                "error_type": error.__class__.__name__,
                "message": str(error),
                "stack_trace": stack_trace if logger and hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG) else None
            }
        
        return error_response
    
    @staticmethod
    def format_error(error_response: Dict[str, Any], format_type: str = "json") -> str:
        """
        Format an error response for output.
        
        Args:
            error_response: The error response from handle_error
            format_type: Output format ("json", "text", "html")
            
        Returns:
            Formatted error message
        """
        if format_type == "json":
            return json.dumps(error_response, indent=2)
        elif format_type == "text":
            result = f"Error: {error_response['error_type']}"
            if error_response.get("error_code"):
                result += f" ({error_response['error_code']})"
            result += f"\nMessage: {error_response['message']}"
            
            # Add additional context based on error type
            if error_response.get("provider"):
                result += f"\nProvider: {error_response['provider']}"
            if error_response.get("tool_name"):
                result += f"\nTool: {error_response['tool_name']}"
            if error_response.get("agent_id"):
                result += f"\nAgent: {error_response['agent_id']}"
                
            return result
        elif format_type == "html":
            html = f"<div class='error'><h3>Error: {error_response['error_type']}</h3>"
            html += f"<p>{error_response['message']}</p>"
            
            if error_response.get("stack_trace"):
                trace_html = "".join(error_response["stack_trace"]).replace("\n", "<br>")
                html += f"<details><summary>Stack Trace</summary><pre>{trace_html}</pre></details>"
                
            html += "</div>"
            return html
        else:
            return str(error_response)


# --- Add New Specific Provider Errors ---

class InvalidRequestError(AIProviderError):
    """ Provider indicated the request was invalid (e.g., 400 Bad Request). """
    def __init__(self, message: str, provider: Optional[str] = None, status_code: int = 400):
        # Defaults to status_code 400
        super().__init__(message, provider=provider, status_code=status_code)

class ContentModerationError(AIProviderError):
    """ Request blocked due to content safety filters. """
    def __init__(self, message: str, provider: Optional[str] = None, reason: Optional[str] = None):
        self.reason = reason
        # Use a specific status code if applicable, e.g., 451 or 400
        super().__init__(message, provider=provider, status_code=400)

class ModelNotFoundError(AIProviderError):
    """ Requested model not found or available for the provider. """
    def __init__(self, message: str, provider: Optional[str] = None, model_id: Optional[str] = None):
        self.model_id = model_id
        # Use a specific status code like 404
        super().__init__(message, provider=provider, status_code=404)

# --- End New Specific Provider Errors --- 