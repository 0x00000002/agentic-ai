"""
Anthropic provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
from ..interfaces import ProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.tool_call import ToolCall
from .base_provider import BaseProvider
import json
import os

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider, ToolCapableProviderInterface):
    """Provider implementation for Anthropic Claude models with tools support."""
    
    # Add property for tool support
    supports_tools = True
    
    # Role mapping for Anthropic API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "assistant"  # Anthropic has no tool role
    }
    
    def __init__(self, 
                 model_id: str,
                 provider_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_id: Model identifier
            provider_config: Provider configuration dictionary
            model_config: Model configuration dictionary
            logger: Logger instance
        """
        if not ANTHROPIC_AVAILABLE:
            raise AIProviderError("Anthropic Python SDK not installed. Run 'pip install anthropic'")
            
        super().__init__(model_id, provider_config, model_config, logger)
        
        # Initialize model parameters from self.model_config (set by super)
        self.parameters = {
            "max_tokens": self.model_config.get('output_limit', 4096), # Default from Anthropic docs
            "temperature": self.model_config.get('temperature', 0.7), # Default
        }
        # Store the specific API model ID (already set in self.model_id by super)
        self.full_model_id = self.model_id
        
        self.logger.info(f"Initialized Anthropic provider with model {model_id} (full model ID: {self.full_model_id})")
        self.logger.info(f"Model parameters: {self.parameters}")
    
    def _initialize_credentials(self) -> None:
        """Initialize Anthropic API credentials."""
        try:
            # Get API key from configuration
            api_key = self.provider_config.get("api_key")
            if not api_key:
                api_key = self.config.get_api_key("anthropic")
                
            if not api_key:
                # Try getting from environment directly
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                
            if not api_key:
                self.logger.error("No Anthropic API key found in configuration or environment")
                raise AICredentialsError("No Anthropic API key found")
                
            self.logger.info("Found Anthropic API key")
                
            # Set up Anthropic client
            try:
                self.client = Anthropic(api_key=api_key)
                self.logger.info("Successfully initialized Anthropic client")
            except Exception as e:
                self.logger.error(f"Failed to create Anthropic client: {str(e)}")
                raise AICredentialsError(f"Failed to create Anthropic client: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic credentials: {str(e)}")
            raise AICredentialsError(f"Failed to initialize Anthropic credentials: {str(e)}")
    
    def _map_role(self, role: str) -> str:
        """Map standard role to Anthropic role."""
        return self._ROLE_MAP.get(role, "user")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Anthropic API."""
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map role to Anthropic format
            anthropic_role = self._map_role(role)
            
            # Handle special roles
            if role == "system":
                # For Claude, system message is handled separately
                continue
                
            # Create formatted message
            formatted_message = {
                "role": anthropic_role,
                "content": content
            }
            
            formatted_messages.append(formatted_message)
            
        return formatted_messages
    
    def _extract_system_message(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Extract system message from messages list."""
        for message in messages:
            if message.get("role") == "system":
                return message.get("content", "")
        return None
    
    def request(self, messages: List[Dict[str, str]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Response content as a string, or a dictionary with content and tool calls
        """
        try:
            # Format messages for Anthropic API
            formatted_messages = self._format_messages(messages)
            
            # Extract system message
            system = self._extract_system_message(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-Anthropic parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "top_k", 
                              "stop_sequences", "tools", "tool_choice"]:
                    del params[key]
            
            # Extract tools if provided
            tools = params.pop("tools", None)
            
            # Make the API call
            if tools:
                return self._request_with_tools(messages[-1].get("content", ""), tools, 
                                               conversation=[(m.get("role"), m.get("content")) 
                                                             for m in messages[:-1] if m.get("role") != "system"],
                                               system_prompt=system)
            
            # Standard request
            response = self.client.messages.create(
                model=self.full_model_id,  # Use the full model ID from configuration
                messages=formatted_messages,
                system=system,
                max_tokens=params.get("max_tokens", 4096),
                temperature=params.get("temperature", 0.7)
            )
            
            return response.content[0].text
            
        except anthropic.APIError as e:
            raise AIRequestError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making Anthropic request: {str(e)}")
    
    def request_with_tools(self, 
                          prompt: str, 
                          tools: List[Dict[str, Any]],
                          conversation: Optional[List[Tuple[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          structured_tools: bool = True) -> Dict[str, Any]:
        """
        Make a request that can use tools.
        
        Args:
            prompt: User prompt
            tools: List of tool definitions
            conversation: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt
            structured_tools: Whether to use structured tools format
            
        Returns:
            Dictionary with content and tool calls
        """
        try:
            # Build messages for Anthropic
            anthropic_messages = []
            
            # Add conversation history
            if conversation:
                for role, content in conversation:
                    if role != "system":
                        anthropic_messages.append({
                            "role": self._map_role(role),
                            "content": content
                        })
            
            # Add current prompt
            anthropic_messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Format tools for Anthropic API
            anthropic_tools = []
            for tool in tools:
                # Ensure required fields are present
                if not all(key in tool for key in ["name", "description", "parameters"]):
                    self.logger.warning(f"Skipping tool {tool.get('name', 'unknown')} due to missing required fields")
                    continue
                    
                anthropic_tool = {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"]
                }
                anthropic_tools.append(anthropic_tool)
            
            if not anthropic_tools:
                self.logger.warning("No valid tools provided for the request")
                return {"content": "I cannot use tools at the moment.", "tool_calls": []}
            
            # Make API call with tool configuration
            response = self.client.messages.create(
                model=self.full_model_id,  # Use the full model ID from configuration
                messages=anthropic_messages,
                system=system_prompt,
                tools=anthropic_tools,
                tool_choice="auto",
                max_tokens=self.parameters.get("max_tokens", 4096),
                temperature=self.parameters.get("temperature", 0.7)
            )
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Extract tool calls
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        tool_calls.append(
                            ToolCall(
                                name=block.name,
                                arguments=json.dumps(block.input),
                                id=block.id
                            )
                        )
                    except Exception as e:
                        self.logger.error(f"Error processing tool call: {str(e)}")
                        continue
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
            
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error in tool request: {str(e)}")
            raise AIRequestError(f"Anthropic API error in tool request: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error making Anthropic tools request: {str(e)}")
            raise AIProviderError(f"Error making Anthropic tools request: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]], **options) -> str:
        """
        Stream a response from the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Format messages for Anthropic API
            formatted_messages = self._format_messages(messages)
            
            # Extract system message
            system = self._extract_system_message(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-Anthropic parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "top_k", 
                              "stop_sequences", "tools", "tool_choice"]:
                    del params[key]
            
            # Make the streaming API call
            response = self.client.messages.create(
                model=self.model_id,
                messages=formatted_messages,
                system=system,
                stream=True,
                **params
            )
            
            # Collect the chunks
            chunks = []
            
            for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    chunks.append(chunk.delta.text)
                    
            # Join all chunks
            return "".join(chunks)
            
        except anthropic.APIError as e:
            raise AIRequestError(f"Anthropic API error in streaming: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error streaming from Anthropic: {str(e)}")
    
    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content)
        })
        return messages 