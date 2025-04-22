# Enhanced Development Plan: Personal AI Assistant Framework

This document enhances the original development plan with specific technical details and considerations for the Agentic-AI framework. The plan maintains the original phases while adding guidance and strategic considerations to make it more actionable.

## Phase 1: Core Framework - Persistence Layer

### 1. ConversationStore Interface Design

- **Design Recommendations**:

  - Create a clean async interface with methods: `load_conversation`, `save_conversation`, `delete_conversation`, and `list_conversations`
  - Include proper error handling via custom exceptions (e.g., `ConversationError`)
  - Add pagination support for listing conversations (limit/offset parameters)
  - Include clear documentation for each method with types and error expectations

- **Interface Extension Considerations**:
  - Consider adding optional metadata methods for conversation stats or user preferences
  - Plan for future extensions like conversation search or tagging
  - Add a method for data export/import to facilitate migrations

### 2. FileConversationStore Implementation

- **Key Features**:

  - Use `aiofiles` for non-blocking I/O operations
  - Implement path sanitization to prevent directory traversal
  - Create a directory structure that scales well with many conversations
  - Add file locking mechanism to prevent concurrent write issues
  - Implement automatic cleanup of old/inactive conversations

- **Performance Considerations**:
  - Consider implementing an in-memory cache for frequently accessed conversations
  - Add configurable compression for large conversations
  - Implement chunking for very large conversations to avoid memory issues

### 3. DynamoDBConversationStore Implementation

- **Core Implementation**:

  - Utilize the `boto3` library for AWS DynamoDB integration
  - Handle the 400KB DynamoDB item size limit through automatic compression
  - Create a robust schema with `conversation_id` as partition key
  - Add `last_updated` attribute for better sorting/filtering
  - Consider using TTL for automatic conversation expiration

- **Advanced Features**:

  - Implement efficient conversation listing with a GSI on `last_updated`
  - Create intelligent splitting strategy for conversations exceeding size limits
  - Add automatic retry logic for DynamoDB throttling
  - Implement write batching for improved performance

- **Cost Optimization**:
  - Configure automatic scaling for DynamoDB based on usage patterns
  - Implement read/write unit monitoring and alerting
  - Consider using DynamoDB Accelerator (DAX) for high-traffic scenarios

### 4. ConversationManager Integration

- **Integration Strategy**:

  - Add `conversation_id` and `store` parameters to `ConversationManager.__init__()`
  - Implement lazy loading of conversation data on first access
  - Update all message methods to be async for compatibility with storage layer
  - Add state tracking to minimize unnecessary saves

- **Implementation Considerations**:

  - Add a method to switch between storage implementations
  - Implement graceful fallbacks if persistent storage fails
  - Add message validation before storage
  - Implement conversation forking/merging capabilities

- **Prompt for AI Implementation Assistant**:

```
You are an expert Python developer tasked with updating the ConversationManager class to support persistent storage. The class currently uses an in-memory list to store conversation messages.

Your task:
1. Modify the class to accept a ConversationStore implementation and conversation_id parameter
2. Convert relevant methods to be async to work with the async storage interface
3. Implement lazy loading to defer fetching conversation history until needed
4. Add efficient save logic to minimize storage operations
5. Ensure backward compatibility with existing code

Remember that the ConversationManager is used extensively throughout the codebase, particularly in ToolEnabledAI and BaseAgent, so changes must be non-disruptive to existing functionality.
```

## Phase 2: Tooling & Telegram Integration

### 1. Telegram Bot Integration

- **Bot Architecture**:

  - Create a dedicated `TelegramBot` class that manages the Telegram interaction
  - Use `python-telegram-bot` library's application-based approach for modern async handling
  - Implement command handlers for `/start`, `/help`, and `/reset`
  - Add conversation context tracking via Telegram's `chat_id`
  - Create robust error handling with user-friendly messages

- **Feature Considerations**:

  - Implement typing indicators during processing for better UX
  - Add throttling/rate limiting for high-volume users
  - Implement proper shutdown handling for graceful termination
  - Consider keyboard shortcuts for common actions
  - Support for inline queries and message editing

- **Voice Handling**:

  - Implement audio file download and temporary storage
  - Add voice transcription via existing listener agent
  - Handle different audio formats and quality
  - Implement automatic cleanup of temporary files

- **Prompt for AI Telegram Implementation Assistant**:

```
You are a Telegram bot implementation expert. Design a robust, production-ready Telegram bot for our AI assistant framework with these requirements:

1. Asynchronous handling of all Telegram interactions
2. Integration with our existing Coordinator agent as the central processing unit
3. Persistent conversation storage using our ConversationStore interface
4. Support for text and voice message processing
5. Error handling that's both robust and user-friendly
6. Clean separation between Telegram-specific logic and core AI functionality

Additionally, outline any security considerations and performance optimizations relevant to a Telegram bot deployment.
```

### 2. MCP Tools Integration

- **MCP Configuration**:

  - Create a dedicated `MCPConfig` class to manage MCP configuration
  - Support loading configuration from YAML file or environment variables
  - Include proper validation of MCP endpoint configurations
  - Store connection details, authentication, and endpoint parameters

- **MCP Tool Implementation**:

  - Develop a generic `MCPTool` class that handles HTTP communication
  - Use `aiohttp` for non-blocking HTTP requests
  - Implement timeout handling with configurable defaults
  - Add retry logic for transient network issues
  - Create proper error handling and reporting

- **Integration with Tool System**:

  - Register MCP tools in the existing ToolManager system
  - Generate appropriate ToolDefinitions from MCP configuration
  - Map parameters correctly between AI model and MCP endpoints
  - Implement proper serialization/deserialization of values

- **Configuration Example**:

```yaml
# mcp.yml example
tools:
  web_search:
    url: "https://mcp-search-service.example.com/search"
    description: "Search the web for information"
    timeout: 10
    parameters:
      query:
        type: "string"
        description: "The search query"
      limit:
        type: "integer"
        description: "Maximum number of results"
        default: 5

  email_sender:
    url: "https://mcp-email-service.example.com/send"
    description: "Send an email to a recipient"
    requires_auth: true
    timeout: 15
    parameters:
      to:
        type: "string"
        description: "Recipient's email address"
      subject:
        type: "string"
        description: "Email subject"
      body:
        type: "string"
        description: "Email body content"
```

## Phase 3: Deployment & Operations (VM Focus)

### 1. VM Provisioning & Configuration

- **Provider Selection**:

  - Recommend AWS Lightsail or DigitalOcean for simpler management
  - For EC2, use t3a.small (2vCPU, 2GB RAM) as a minimum configuration
  - Consider spot instances for cost optimization if appropriate
  - Set up automatic backup for the VM

- **Base System Setup**:

  - Use Ubuntu 22.04 LTS for long-term stability
  - Set up UFW firewall allowing only SSH and specific application ports
  - Implement automatic security updates
  - Configure proper swap space (at least 4GB)
  - Set up monitoring with CloudWatch or Prometheus

- **Resource Sizing Guidelines**:
  - RAM: Minimum 2GB, recommended 4GB for production use
  - Storage: 20GB SSD minimum, with separate volume for conversation data
  - CPU: 2 vCPU minimum for responsiveness
  - Network: Standard is sufficient, monitor bandwidth usage

### 2. Dockerization & Deployment

- **Docker Image Design**:

  - Use multi-stage builds to minimize image size
  - Base image: python:3.9-slim for smaller footprint
  - Create non-root user for security
  - Cache pip dependencies for faster builds
  - Include only necessary files and directories

- **Docker Compose Configuration**:

  - Create separate services for the main application, Nginx, and possibly Redis
  - Set up named volumes for persistent data
  - Configure environment-specific compose files
  - Set resource limits appropriately
  - Configure logging driver (json-file with rotation)

- **Deployment Management**:

  - Create simple deployment script for pull, build, up cycle
  - Implement health checks for all containers
  - Add monitoring endpoints (but secure them properly)
  - Set appropriate restart policies
  - Consider Watchtower for automatic image updates

- **Prompt for AI Docker Implementation Assistant**:

```
You are a Docker and containerization expert. Create a production-ready Docker deployment strategy for our Python-based AI assistant with these requirements:

1. Efficient Dockerfile that minimizes image size and build time
2. Docker Compose configuration for coordinating multiple services
3. Proper volume management for persistent data
4. Security hardening for the containerized application
5. Resource allocation recommendations
6. Logging and monitoring strategy

The application is a Python 3.9 async application that uses significant RAM when processing requests but has low baseline resource usage when idle.
```

### 3. Security & Monitoring

- **Security Hardening**:

  - Implement SSH key-only authentication
  - Use AWS IAM roles when possible instead of static credentials
  - Configure AWS Security Groups or VPC for network isolation
  - Regularly rotate all credentials
  - Enable CloudTrail for AWS API activity logging

- **Monitoring Setup**:

  - Set up CloudWatch for basic resource monitoring
  - Create custom CloudWatch metrics for application-specific metrics
  - Set alarms for memory, CPU, and disk usage
  - Implement log forwarding to CloudWatch Logs
  - Create a dashboard for quick system status overview

- **Backup Strategy**:
  - Daily backups of conversation data to S3
  - Implement backup rotation policy (keep 7 daily, 4 weekly)
  - Periodic backup verification
  - Document restore procedures and test them

### 4. Operational Considerations

- **Scaling Strategy**:

  - Plan for vertical scaling path (larger VM) as primary approach
  - Document horizontal scaling options if needed later
  - Identify potential bottlenecks (e.g., token rate limits, database IOPS)

- **Cost Management**:

  - Set up budget alerts in AWS
  - Monitor DynamoDB usage carefully
  - Consider reserved instances for long-term cost savings
  - Implement auto-scaling to reduce costs during low-usage periods

- **Maintenance Procedures**:
  - Document update process for the application
  - Create runbooks for common operations
  - Establish maintenance window for regular updates
  - Implement blue-green deployment for zero-downtime updates

## Phase 4: Future Enhancements (Post-MVP)

### 1. Performance Optimizations

- **Caching Strategy**:

  - Implement Redis for LLM response caching
  - Add cache warming for common queries
  - Create intelligent cache invalidation rules
  - Consider embedding caching for RAG operations

- **Resource Optimization**:
  - Implement auto-scaling based on time-of-day patterns
  - Add dynamic model selection based on query complexity
  - Optimize token usage through prompt engineering
  - Implement batching for tool operations when possible

### 2. Advanced Capabilities

- **Enhanced Agent Selection**:

  - Develop more sophisticated agent routing logic
  - Implement specialized agents for vertical domains
  - Create agent collaboration patterns for complex tasks
  - Add reinforcement learning for improved agent selection

- **Conversation Memory Improvements**:

  - Implement semantic indexing of conversation history
  - Add long-term memory via RAG patterns
  - Create user preference learning capabilities
  - Implement conversation summarization for context management

- **Tool Ecosystem Expansion**:

  - Create tool discovery mechanism
  - Implement tool composition for complex operations
  - Add user-specific tool authorization
  - Create tool usage analytics for optimization

- **Prompt for AI Advanced Features Assistant**:

```
You are a strategic AI product architect. Outline a detailed roadmap for advanced features that could be added to our personal AI assistant framework after the MVP. Focus on:

1. Conversation intelligence and memory management
2. Multi-agent collaboration patterns
3. Advanced tool composition and discovery
4. User personalization and adaptation
5. Performance optimizations at scale

For each feature area, describe the user value, technical approach, implementation complexity, and potential challenges. Prioritize features that would deliver the most immediate value while setting up for longer-term capabilities.
```

### 3. Admin & Monitoring Dashboard

- **Dashboard Design**:

  - Create simple web-based admin interface
  - Add conversation search and browsing capabilities
  - Implement user management features
  - Add usage analytics and reporting
  - Create system health monitoring view

- **Operational Metrics**:
  - Track response times, token usage, and completion rates
  - Implement conversation quality metrics
  - Monitor tool usage patterns and success rates
  - Track error rates and patterns
  - Add cost allocation reporting

### 4. CI/CD Pipeline

- **Build Pipeline**:

  - Set up GitHub Actions for automated testing
  - Implement linting and code quality checks
  - Add automatic Docker image building
  - Create semantic versioning workflow

- **Deployment Pipeline**:
  - Implement staging environment
  - Add automated deployment to staging on merge to develop
  - Create production deployment approval workflow
  - Add post-deployment verification steps
  - Implement automatic rollback on failure

## Examples:

### 1. ConversationStore Interface Design

```python
# src/conversation/store/base_store.py
import abc
from typing import Dict, List, Any, Optional

class ConversationStore(abc.ABC):
    """Abstract base class for conversation storage implementations."""

    @abc.abstractmethod
    async def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation history from storage.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            List of message dictionaries in the format used by ConversationManager

        Raises:
            ConversationError: If loading fails
        """
        pass

    @abc.abstractmethod
    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Persist conversation to storage.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries to store

        Raises:
            ConversationError: If saving fails
        """
        pass

    @abc.abstractmethod
    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Remove conversation from storage.

        Args:
            conversation_id: Unique identifier for the conversation

        Raises:
            ConversationError: If deletion fails
        """
        pass

    @abc.abstractmethod
    async def list_conversations(self, limit: int = 100, offset: int = 0) -> List[str]:
        """
        List available conversation IDs.

        Args:
            limit: Maximum number of conversation IDs to return
            offset: Starting offset for pagination

        Returns:
            List of conversation IDs

        Raises:
            ConversationError: If listing fails
        """
        pass
```

### 2. FileConversationStore Implementation

```python
# src/conversation/store/file_store.py
import os
import json
import aiofiles
from typing import Dict, List, Any, Optional
from .base_store import ConversationStore
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import ConversationError

class FileConversationStore(ConversationStore):
    """File-based implementation of ConversationStore using JSON files."""

    def __init__(self,
                 storage_dir: str = "./conversations",
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the file-based conversation store.

        Args:
            storage_dir: Directory path to store conversation files
            logger: Logger instance
        """
        self._storage_dir = storage_dir
        self._logger = logger or LoggerFactory.create(name="file_conversation_store")

        # Ensure storage directory exists
        os.makedirs(self._storage_dir, exist_ok=True)
        self._logger.info(f"FileConversationStore initialized with storage dir: {self._storage_dir}")

    def _get_file_path(self, conversation_id: str) -> str:
        """Get the file path for a conversation ID."""
        # Sanitize conversation_id to prevent path traversal
        safe_id = conversation_id.replace('/', '_').replace('\\', '_')
        return os.path.join(self._storage_dir, f"{safe_id}.json")

    async def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load conversation from a JSON file."""
        file_path = self._get_file_path(conversation_id)
        self._logger.debug(f"Loading conversation from {file_path}")

        try:
            if not os.path.exists(file_path):
                self._logger.debug(f"Conversation file not found: {file_path}")
                return []

            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            error_msg = f"Error loading conversation {conversation_id}: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save conversation to a JSON file."""
        file_path = self._get_file_path(conversation_id)
        self._logger.debug(f"Saving conversation to {file_path}")

        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(messages, indent=2))
        except Exception as e:
            error_msg = f"Error saving conversation {conversation_id}: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation file."""
        file_path = self._get_file_path(conversation_id)
        self._logger.debug(f"Deleting conversation file: {file_path}")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            error_msg = f"Error deleting conversation {conversation_id}: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def list_conversations(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List all conversation IDs based on filenames."""
        self._logger.debug(f"Listing conversations (limit={limit}, offset={offset})")

        try:
            files = [f for f in os.listdir(self._storage_dir)
                    if f.endswith('.json') and os.path.isfile(os.path.join(self._storage_dir, f))]

            # Sort by modification time (newest first)
            files.sort(key=lambda f: os.path.getmtime(os.path.join(self._storage_dir, f)), reverse=True)

            # Apply pagination
            paginated_files = files[offset:offset+limit]

            # Remove .json extension to get IDs
            return [os.path.splitext(f)[0] for f in paginated_files]
        except Exception as e:
            error_msg = f"Error listing conversations: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e
```

### 3. DynamoDBConversationStore Implementation

```python
# src/conversation/store/dynamodb_store.py
import os
import json
import boto3
import zlib
from typing import Dict, List, Any, Optional
from .base_store import ConversationStore
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import ConversationError
from ...config.unified_config import UnifiedConfig

class DynamoDBConversationStore(ConversationStore):
    """DynamoDB-based implementation of ConversationStore with automatic compression."""

    # DynamoDB item size limit in bytes
    ITEM_SIZE_LIMIT = 400000

    def __init__(self,
                 table_name: Optional[str] = None,
                 region_name: Optional[str] = None,
                 compression_threshold: int = 100000,  # Compress if > 100KB
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the DynamoDB conversation store.

        Args:
            table_name: DynamoDB table name (or None to use config)
            region_name: AWS region (or None to use config)
            compression_threshold: Byte threshold for compressing data
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create(name="dynamodb_conversation_store")

        # Load configuration
        config = UnifiedConfig.get_instance()
        dynamodb_config = config.get_config_section("dynamodb") or {}

        # Set table name and region
        self._table_name = table_name or dynamodb_config.get("conversation_table_name")
        self._region_name = region_name or dynamodb_config.get("region_name") or "us-east-1"

        if not self._table_name:
            raise ValueError("DynamoDB table name not provided and not found in configuration")

        # Compression settings
        self._compression_threshold = compression_threshold

        # Initialize DynamoDB client
        self._dynamodb = boto3.resource('dynamodb', region_name=self._region_name)
        self._table = self._dynamodb.Table(self._table_name)

        self._logger.info(f"DynamoDBConversationStore initialized with table: {self._table_name}")

    async def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load conversation from DynamoDB."""
        self._logger.debug(f"Loading conversation from DynamoDB: {conversation_id}")

        try:
            response = self._table.get_item(Key={'conversation_id': conversation_id})
            item = response.get('Item')

            if not item:
                self._logger.debug(f"Conversation not found: {conversation_id}")
                return []

            # Check if the data is compressed
            is_compressed = item.get('compressed', False)
            messages_data = item['messages']

            if is_compressed:
                # Decompress the data
                compressed_bytes = bytes(messages_data, 'utf-8')
                decompressed_bytes = zlib.decompress(compressed_bytes)
                messages_json = decompressed_bytes.decode('utf-8')
            else:
                messages_json = messages_data

            return json.loads(messages_json)
        except Exception as e:
            error_msg = f"Error loading conversation {conversation_id} from DynamoDB: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save conversation to DynamoDB with compression if needed."""
        self._logger.debug(f"Saving conversation to DynamoDB: {conversation_id}")

        try:
            # Convert messages to JSON
            messages_json = json.dumps(messages)
            is_compressed = False

            # Check if the size exceeds threshold for compression
            if len(messages_json) > self._compression_threshold:
                self._logger.debug(f"Compressing large conversation: {conversation_id} ({len(messages_json)} bytes)")
                compressed_data = zlib.compress(messages_json.encode('utf-8'))
                messages_data = compressed_data.decode('utf-8')
                is_compressed = True

                # Check if still exceeds DynamoDB item size limit
                if len(messages_data) > self.ITEM_SIZE_LIMIT:
                    error_msg = f"Compressed conversation size ({len(messages_data)} bytes) exceeds DynamoDB limit"
                    self._logger.error(error_msg)
                    raise ConversationError(error_msg)
            else:
                messages_data = messages_json

            # Save to DynamoDB
            self._table.put_item(Item={
                'conversation_id': conversation_id,
                'messages': messages_data,
                'compressed': is_compressed,
                'message_count': len(messages),
                'last_updated': int(time.time())
            })
        except Exception as e:
            error_msg = f"Error saving conversation {conversation_id} to DynamoDB: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation from DynamoDB."""
        self._logger.debug(f"Deleting conversation from DynamoDB: {conversation_id}")

        try:
            self._table.delete_item(Key={'conversation_id': conversation_id})
        except Exception as e:
            error_msg = f"Error deleting conversation {conversation_id} from DynamoDB: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e

    async def list_conversations(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List conversations from DynamoDB using scan operation."""
        self._logger.debug(f"Listing conversations from DynamoDB (limit={limit}, offset={offset})")

        try:
            # Note: DynamoDB scan with pagination is not ideal for large datasets
            # Consider implementing a GSI on last_updated if listing is frequently used
            response = self._table.scan(
                ProjectionExpression="conversation_id,last_updated",
                Limit=limit + offset
            )

            items = response.get('Items', [])

            # Sort by last_updated (newest first)
            items.sort(key=lambda x: x.get('last_updated', 0), reverse=True)

            # Apply offset
            if offset > 0:
                items = items[offset:]

            # Apply limit
            items = items[:limit]

            return [item['conversation_id'] for item in items]
        except Exception as e:
            error_msg = f"Error listing conversations from DynamoDB: {str(e)}"
            self._logger.error(error_msg)
            raise ConversationError(error_msg) from e
```

### 4. ConversationManager Integration

```python
# Enhanced src/conversation/conversation_manager.py
from typing import Dict, List, Any, Optional, Union
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import ConversationError
from .response_parser import ResponseParser
from ..prompts.prompt_template import PromptTemplate
from dataclasses import dataclass
from .store.base_store import ConversationStore

@dataclass
class Message:
    """Represents a message in a conversation."""
    role: str
    # Content can be a string or a list (for structured results like Anthropic tool results)
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    thoughts: Optional[str] = None
    # Add tool_call_id for OpenAI compatibility if role is 'tool'
    tool_call_id: Optional[str] = None

class ConversationManager:
    """Manages conversation history and state with persistent storage."""

    def __init__(self,
                 conversation_id: Optional[str] = None,
                 store: Optional[ConversationStore] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the conversation manager.

        Args:
            conversation_id: Unique identifier for this conversation
            store: ConversationStore instance for persistence
            logger: Logger instance for logging operations
        """
        self._logger = logger or LoggerFactory.create()
        self._conversation_id = conversation_id or "default"
        self._store = store
        self._messages: List[Message] = []
        self._metadata: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
        self._response_parser = ResponseParser(logger=self._logger)
        self._prompt_template = PromptTemplate(logger=self._logger)
        self._loaded = False

        self._logger.debug(f"Initialized ConversationManager with ID: {self._conversation_id}")
        if self._store:
            self._logger.debug(f"Using persistent store: {type(self._store).__name__}")

    async def _ensure_loaded(self) -> None:
        """Ensure conversation is loaded from storage if available."""
        if self._store and not self._loaded:
            try:
                loaded_messages = await self._store.load_conversation(self._conversation_id)
                if loaded_messages:
                    # Convert loaded dict messages to Message objects
                    self._messages = [
                        Message(
                            role=msg["role"],
                            content=msg["content"],
                            name=msg.get("name"),
                            thoughts=msg.get("thoughts"),
                            tool_call_id=msg.get("tool_call_id")
                        )
                        for msg in loaded_messages
                    ]
                    self._logger.info(f"Loaded {len(self._messages)} messages from storage for conversation {self._conversation_id}")
                else:
                    self._logger.debug(f"No existing messages found for conversation {self._conversation_id}")

                self._loaded = True
            except Exception as e:
                self._logger.error(f"Error loading conversation: {e}", exc_info=True)
                # Continue with empty messages list
                self._loaded = True

    async def _save_to_store(self) -> None:
        """Save conversation to persistent storage if available."""
        if self._store:
            try:
                # Convert Message objects to dictionaries
                messages_dict = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        **({"name": msg.name} if msg.name is not None else {}),
                        **({"thoughts": msg.thoughts} if msg.thoughts is not None else {}),
                        **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id is not None else {})
                    }
                    for msg in self._messages
                ]

                await self._store.save_conversation(self._conversation_id, messages_dict)
                self._logger.debug(f"Saved {len(self._messages)} messages to storage for conversation {self._conversation_id}")
            except Exception as e:
                self._logger.error(f"Error saving conversation: {e}", exc_info=True)

    async def add_message(self,
                   role: str,
                   content: Union[str, List[Dict[str, Any]]],
                   name: Optional[str] = None,
                   extract_thoughts: bool = True,
                   show_thinking: bool = False,
                   tool_call_id: Optional[str] = None,
                   **kwargs) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant', 'tool')
            content: The message content (string or list of dicts for structured messages)
            name: Optional name for the message (e.g., tool name)
            extract_thoughts: Whether to extract thoughts from the content (if string)
            show_thinking: Whether to include thoughts in the response content (if string)
            tool_call_id: Optional ID for tool messages (OpenAI)
            **kwargs: Additional message metadata (ignored for now)
        """
        # Ensure conversation is loaded
        await self._ensure_loaded()

        thoughts = None
        processed_content = content

        # Only parse thoughts if content is string and role is assistant
        if isinstance(content, str) and extract_thoughts and role == "assistant":
            parsed = self._response_parser.parse_response(
                content,
                extract_thoughts=extract_thoughts,
                show_thinking=show_thinking
            )
            processed_content = parsed["content"]
            thoughts = parsed.get("thoughts")

        # Create the message object
        message = Message(
            role=role,
            content=processed_content, # Can be string or list
            name=name,
            thoughts=thoughts,
            tool_call_id=tool_call_id # Store tool_call_id
        )

        self._messages.append(message)
        self._logger.debug(f"Added {role} message to conversation (Content type: {type(processed_content).__name__})")

        # Save to persistent storage
        await self._save_to_store()

    async def add_interaction(self,
                       user_message: str,
                       assistant_message: str,
                       extract_thoughts: bool = True,
                       show_thinking: bool = False) -> None:
        """
        Add a user-assistant interaction to the conversation.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            extract_thoughts: Whether to extract thoughts from the assistant's response
            show_thinking: Whether to include thoughts in the response content
        """
        await self.add_message(role="user", content=user_message)
        await self.add_message(
            role="assistant",
            content=assistant_message,
            extract_thoughts=extract_thoughts,
            show_thinking=show_thinking
        )
        self._logger.debug("Added user-assistant interaction to conversation")

    async def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the conversation, formatted as dictionaries.
        Handles both string and list content.

        Returns:
            List of message dictionaries ready for most provider APIs.
        """
        # Ensure conversation is loaded
        await self._ensure_loaded()

        output_messages = []
        for msg in self._messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content # Pass content directly (string or list)
            }
            # Add optional fields if they exist
            if msg.name is not None:
                 message_dict["name"] = msg.name
            if msg.tool_call_id is not None:
                 message_dict["tool_call_id"] = msg.tool_call_id
            output_messages.append(message_dict)
        return output_messages

    async def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the conversation.

        Returns:
            The last message dictionary or None if no messages
        """
        # Ensure conversation is loaded
        await self._ensure_loaded()

        if not self._messages:
            return None
        last_msg = self._messages[-1]
        return {
            "role": last_msg.role,
            "content": last_msg.content,
            "name": last_msg.name,
            "thoughts": last_msg.thoughts
        }

    async def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        # Ensure conversation is loaded first
        await self._ensure_loaded()

        self._messages.clear()
        self._logger.debug("Cleared conversation messages")

        # Update persistent storage
        await self._save_to_store()

    async def delete_conversation(self) -> None:
        """
        Delete the conversation from persistent storage.
        """
        if self._store:
            try:
                await self._store.delete_conversation(self._conversation_id)
                self._logger.info(f"Deleted conversation {self._conversation_id} from storage")
                self._messages.clear()
                self._loaded = False
            except Exception as e:
                self._logger.error(f"Error deleting conversation: {e}", exc_info=True)
```

## Phase 2: Tooling & Telegram Integration

### 1. Telegram Bot Implementation

```python
# telegram_bot.py
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List

from telegram import Update, Bot, Message as TelegramMessage
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.conversation.store.base_store import ConversationStore
from src.conversation.store.file_store import FileConversationStore
from src.conversation.store.dynamodb_store import DynamoDBConversationStore
from src.config.unified_config import UnifiedConfig
from src.agents.coordinator import Coordinator
from src.utils.logger import LoggerFactory, LoggerInterface

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot integration for the Agentic-AI framework."""

    def __init__(self, token: str, conversation_store: Optional[ConversationStore] = None):
        """
        Initialize the Telegram bot.

        Args:
            token: Telegram bot token
            conversation_store: ConversationStore implementation for persistence
        """
        self.token = token
        self.store = conversation_store
        self.config = UnifiedConfig.get_instance()
        self.logger = LoggerFactory.create("telegram_bot")

        # Create application
        self.application = Application.builder().token(token).build()

        # Setup handlers
        self._setup_handlers()

        self.logger.info("Telegram bot initialized")

    def _setup_handlers(self):
        """Setup command and message handlers."""
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("reset", self.handle_reset))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_error_handler(self.handle_error)

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        await update.message.reply_text(
            "Hello! I'm your AI assistant. How can I help you today?"
        )

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        help_text = (
            "I'm your personal AI assistant. You can:\n"
            "• Ask me questions\n"
            "• Request tasks\n"
            "• Send voice messages\n"
            "• Use /reset to clear our conversation history"
        )
        await update.message.reply_text(help_text)

    async def handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /reset command to clear conversation history."""
        chat_id = str(update.effective_chat.id)

        # Create a coordinator with the appropriate conversation_id
        coordinator = await self._create_coordinator(chat_id)

        # Clear conversation history
        if self.store:
            try:
                await self.store.delete_conversation(chat_id)
                await update.message.reply_text("Conversation history has been reset.")
            except Exception as e:
                self.logger.error(f"Error resetting conversation: {e}", exc_info=True)
                await update.message.reply_text("Sorry, I couldn't reset our conversation. Please try again.")
        else:
            await update.message.reply_text("No persistent storage configured. Session-only conversation reset.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        chat_id = str(update.effective_chat.id)
        user_message = update.message.text
        user_name = update.message.from_user.first_name

        # Log incoming message
        self.logger.info(f"Received message from {user_name} (chat_id: {chat_id}): {user_message[:50]}...")

        # Notify user that processing is happening
        typing_message = await update.message.reply_text("Thinking...")

        try:
            # Create a coordinator with the appropriate conversation_id
            coordinator = await self._create_coordinator(chat_id)

            # Process the request
            request = {
                "prompt": user_message,
                "metadata": {
                    "user_id": update.message.from_user.id,
                    "user_name": user_name,
                    "platform": "telegram",
                    "chat_id": chat_id
                }
            }

            # Process the request (this will save conversation history via ConversationManager)
            response = await coordinator.process_request(request)

            # Delete the "Thinking..." message
            await typing_message.delete()

            # Send the response
            if response and "content" in response:
                await update.message.reply_text(response["content"])
            else:
                await update.message.reply_text("Sorry, I couldn't process your request.")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            await typing_message.delete()
            await update.message.reply_text("Sorry, I encountered an error processing your request. Please try again.")

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice messages."""
        chat_id = str(update.effective_chat.id)

        # Notify user that processing is happening
        processing_message = await update.message.reply_text("Processing audio...")

        try:
            # Get voice file details
            voice = update.message.voice
            voice_file = await context.bot.get_file(voice.file_id)

            # Create temporary file path
            file_path = f"temp_voice_{voice.file_id}.ogg"

            # Download voice file
            await voice_file.download_to_drive(file_path)

            # Create coordinator
            coordinator = await self._create_coordinator(chat_id)

            # Process as audio transcription request
            request = {
                "type": "audio_transcription",
                "file_path": file_path,
                "metadata": {
                    "user_id": update.message.from_user.id,
                    "user_name": update.message.from_user.first_name,
                    "platform": "telegram",
                    "chat_id": chat_id,
                    "duration": voice.duration
                }
            }

            # Process the request
            response = await coordinator.process_request(request)

            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Delete processing message
            await processing_message.delete()

            # Send the transcription
            if response and "content" in response:
                # First send the transcription
                transcription = response.get("metadata", {}).get("transcription", "")
                if transcription:
                    await update.message.reply_text(f"🎤 *Transcription:*\n{transcription}", parse_mode="Markdown")

                # Then send the AI response to the transcription
                await update.message.reply_text(response["content"])
            else:
                await update.message.reply_text("Sorry, I couldn't transcribe your audio message.")
        except Exception as e:
            self.logger.error(f"Error processing voice message: {e}", exc_info=True)
            await processing_message.delete()
            await update.message.reply_text("Sorry, I encountered an error processing your voice message.")
            # Clean up the temporary file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)

    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the telegram bot."""
        self.logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)

        # Notify user of the error
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, something went wrong. Please try again later."
            )

    async def _create_coordinator(self, conversation_id: str) -> Coordinator:
        """
        Create a coordinator instance with appropriate conversation store.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Initialized Coordinator instance
        """
        # Create a logger for this coordinator
        coordinator_logger = LoggerFactory.create(f"telegram_coordinator_{conversation_id}")

        # Create a coordinator with the conversation store
        coordinator = Coordinator(
            unified_config=self.config,
            logger=coordinator_logger
        )

        # Here we would initialize the ConversationManager with the store and conversation_id
        # This logic would need to be integrated into the BaseAgent and ToolEnabledAI classes
        # to pass the conversation_id and store to the ConversationManager

        return coordinator

    async def run(self):
        """Run the bot until it's stopped."""
        self.logger.info("Starting Telegram bot")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        try:
            # Keep the bot running until Ctrl+C is pressed
            await self.application.updater.stop_on_signal()
        finally:
            await self.application.stop()
            await self.application.shutdown()
            self.logger.info("Telegram bot stopped")

# Main entry point
if __name__ == "__main__":
    # Set up configuration
    config = UnifiedConfig.get_instance()

    # Get bot token from environment variable or config
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or config.get_config_value("telegram", "bot_token")
    if not token:
        raise ValueError("Telegram bot token not found. Set TELEGRAM_BOT_TOKEN environment variable.")

    # Determine store type from config
    store_type = config.get_config_value("conversation", "store_type", default="file")

    # Create the appropriate store based on configuration
    store = None
    if store_type == "file":
        store_dir = config.get_config_value("conversation", "file_store_dir", default="./conversations")
        store = FileConversationStore(storage_dir=store_dir)
    elif store_type == "dynamodb":
        table_name = config.get_config_value("dynamodb", "conversation_table_name")
        region = config.get_config_value("dynamodb", "region_name", default="us-east-1")
        store = DynamoDBConversationStore(table_name=table_name, region_name=region)
    else:
        logger.warning(f"Unknown store type: {store_type}. Proceeding without persistent storage.")

    # Create and run the bot
    bot = TelegramBot(token=token, conversation_store=store)
    asyncio.run(bot.run())
```

### 2. MCP Integration

```python
# src/mcp/mcp_config.py
from typing import Dict, Any, List, Optional
import yaml
import os
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import ConfigError

class MCPConfig:
    """Configuration manager for MCP (Multi-tool Command Protocol) integrations."""

    def __init__(self,
                 config_path: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize MCP configuration.

        Args:
            config_path: Path to MCP configuration YAML file
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create("mcp_config")
        self._config_path = config_path or os.environ.get("MCP_CONFIG_PATH", "./mcp.yml")
        self._config: Dict[str, Any] = {}

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load MCP configuration from YAML file."""
        try:
            if not os.path.exists(self._config_path):
                self._logger.warning(f"MCP configuration file not found: {self._config_path}")
                return

            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f)

            self._logger.info(f"Loaded MCP configuration from {self._config_path}")

            # Validate configuration
            self._validate_config()
        except Exception as e:
            self._logger.error(f"Error loading MCP configuration: {e}", exc_info=True)
            raise ConfigError(f"Failed to load MCP configuration: {e}") from e

    def _validate_config(self) -> None:
        """Validate MCP configuration structure."""
        if not isinstance(self._config, dict):
            raise ConfigError("Invalid MCP configuration format. Expected a dictionary.")

        if "tools" not in self._config:
            raise ConfigError("Missing 'tools' section in MCP configuration.")

        if not isinstance(self._config["tools"], dict):
            raise ConfigError("Invalid 'tools' section in MCP configuration. Expected a dictionary.")

        for tool_name, tool_config in self._config["tools"].items():
            if not isinstance(tool_config, dict):
                raise ConfigError(f"Invalid configuration for tool '{tool_name}'. Expected a dictionary.")

            if "url" not in tool_config:
                raise ConfigError(f"Missing 'url' for tool '{tool_name}'.")

    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific MCP tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool configuration dictionary or None if not found
        """
        return self._config.get("tools", {}).get(tool_name)

    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all MCP tools.

        Returns:
            Dictionary mapping tool names to their configurations
        """
        return self._config.get("tools", {})

    def get_global_config(self) -> Dict[str, Any]:
        """
        Get global MCP configuration.

        Returns:
            Global configuration dictionary
        """
        return {k: v for k, v in self._config.items() if k != "tools"}
```

```python
# src/mcp/mcp_tool.py
from typing import Dict, Any, Optional, List, Union
import aiohttp
import json
from ..tools.models import ToolDefinition, ToolResult
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import ToolExecutionError
from .mcp_config import MCPConfig

class MCPTool:
    """Integration with external MCP tool endpoints."""

    def __init__(self,
                 tool_name: str,
                 config: Optional[MCPConfig] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize MCP tool integration.

        Args:
            tool_name: Name of the MCP tool
            config: MCP configuration instance
            logger: Logger instance
        """
        self._tool_name = tool_name
        self._logger = logger or LoggerFactory.create(f"mcp_tool_{tool_name}")
        self._config = config or MCPConfig()

        # Get tool configuration
        self._tool_config = self._config.get_tool_config(tool_name)
        if not self._tool_config:
            raise ValueError(f"Configuration for MCP tool '{tool_name}' not found")

        self._url = self._tool_config["url"]
        self._headers = self._tool_config.get("headers", {})
        self._timeout = self._tool_config.get("timeout", 30)  # Default timeout: 30 seconds

        self._logger.info(f"Initialized MCP tool '{tool_name}' with URL: {self._url}")

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the MCP tool.

        Args:
            **kwargs: Parameters for the tool execution

        Returns:
            Tool execution result
        """
        self._logger.info(f"Executing MCP tool '{self._tool_name}' with parameters: {kwargs}")

        try:
            # Prepare request payload
            payload = {
                "params": kwargs
            }

            # Add request_id if provided
            if "request_id" in kwargs:
                payload["request_id"] = kwargs.pop("request_id")

            # Execute HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout
                ) as response:
                    response_data = await response.json()

                    # Check for errors in response
                    if response.status >= 400:
                        error_message = response_data.get("error", f"HTTP error {response.status}")
                        self._logger.error(f"MCP tool '{self._tool_name}' returned error: {error_message}")
                        return ToolResult(
                            success=False,
                            error=error_message,
                            tool_name=self._tool_name
                        )

                    # Process successful response
                    result = response_data.get("result")
                    return ToolResult(
                        success=True,
                        result=result,
                        tool_name=self._tool_name
                    )
        except aiohttp.ClientError as e:
            error_message = f"HTTP client error: {str(e)}"
            self._logger.error(error_message, exc_info=True)
            return ToolResult(
                success=False,
                error=error_message,
                tool_name=self._tool_name
            )
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON response: {str(e)}"
            self._logger.error(error_message, exc_info=True)
            return ToolResult(
                success=False,
                error=error_message,
                tool_name=self._tool_name
            )
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            self._logger.error(error_message, exc_info=True)
            return ToolResult(
                success=False,
                error=error_message,
                tool_name=self._tool_name
            )

    @classmethod
    def get_tool_definition(cls,
                           tool_name: str,
                           config: Optional[MCPConfig] = None) -> Optional[ToolDefinition]:
        """
        Get tool definition for MCP tool.

        Args:
            tool_name: Name of the MCP tool
            config: MCP configuration instance

        Returns:
            Tool definition or None if not found
        """
        logger = LoggerFactory.create(f"mcp_tool_def_{tool_name}")
        config = config or MCPConfig()

        tool_config = config.get_tool_config(tool_name)
        if not tool_config:
            logger.warning(f"Configuration for MCP tool '{tool_name}' not found")
            return None

        return ToolDefinition(
            name=tool_name,
            description=tool_config.get("description", f"MCP tool: {tool_name}"),
            parameters=tool_config.get("parameters", {}),
            source="mcp"  # Mark as MCP tool
        )
```

## Phase 3: Deployment & Operations (VM Focus)

### 1. Dockerfile for Core Service

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/conversations && \
    chown -R appuser:appuser /app/conversations
USER appuser

# Set entrypoint
ENTRYPOINT ["python", "telegram_bot.py"]
```

### 2. Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  agentic-ai:
```
