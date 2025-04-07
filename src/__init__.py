"""
AI Framework package.

Re-exports key components for easier access.
Example:
    from src import ToolEnabledAI, AIBase, ModelSelector, UseCase, AIInterface, ProviderInterface
"""

# Core components
from .core import ToolEnabledAI, AIBase
from .core import ModelSelector # Uncommented - Now uses ConfigFactory
from .core.model_selector import UseCase
from .core.interfaces import AIInterface, ProviderInterface

# Configuration
from .config import UnifiedConfig
# Assuming Model enum will be fixed in src/config/models.py
from .config.dynamic_models import Model, Quality, Speed, Privacy

# Conversation management
from .conversation import ConversationManager, Message

# Prompt management
from .prompts import PromptManager, PromptTemplate

# Tool management
from .tools import ToolManager
from .tools.models import ToolDefinition, ToolResult, ToolCall

# Utils
from .utils import LoggerFactory

# Exceptions
from .exceptions import AIError, AIConfigError, AIProviderError, AIToolError, ConversationError

__all__ = [
    # Core
    "ToolEnabledAI",
    "AIBase",
    "ModelSelector", # Uncommented - Now uses ConfigFactory
    "UseCase",
    "AIInterface",
    "ProviderInterface",

    # Config
    "UnifiedConfig",
    "Model", # Uncomment when src/config/models.py is fixed
    "Quality",
    "Speed",
    "Privacy",

    # Conversation
    "ConversationManager",
    "Message",

    # Prompts
    "PromptManager",
    "PromptTemplate",

    # Tools
    "ToolManager",
    "ToolDefinition",
    "ToolResult",
    "ToolCall",

    # Utils
    "LoggerFactory",

    # Exceptions
    "AIError",
    "AIConfigError",
    "AIProviderError",
    "AIToolError",
    "ConversationError",
]

# The following were commented out in the original file, keeping for reference if needed
# from src.ai.AI import AI
# from src.ModelSelector import ModelSelector, UseCase, Quality, Speed
# from src.website import Website
