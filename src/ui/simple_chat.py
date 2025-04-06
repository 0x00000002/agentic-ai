"""
Simple Chat UI for the Agentic AI Framework

This module provides a simple chat interface using Gradio for interacting with the orchestrator.
"""
import gradio as gr
import uuid
from typing import List, Tuple, Dict, Any, Optional
import logging
import time

from ..agents.coordinator import Coordinator
from src.config.user_config import UserConfig
from src.utils.logger import LoggerFactory

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger("simple_chat_ui")

# --- Enable Real Logging for Debugging ---
LoggerFactory.enable_real_loggers()
# -----------------------------------------

class SimpleChatUI:
    """
    A simple chat interface for interacting with the orchestrator.
    Uses Gradio to create a web-based chat interface.
    """
    
    def __init__(self, coordinator, title: str = "Agentic AI Chat"):
        """
        Initialize the chat UI.
        
        Args:
            orchestrator: The orchestrator instance to use for processing requests
            title: The title of the chat interface
        """
        self.coordinator = coordinator
        self.title = title
        self.logger = logger
        
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Process a user message and return the updated chat history.
        
        Args:
            message: The user message
            history: The chat history
            
        Returns:
            Updated chat history
        """
        if not message:
            return history
            
        self.logger.info(f"Processing message: {message[:50]}...")
        
        # Create a request for the orchestrator
        request = {
            "prompt": message,
            "request_id": str(uuid.uuid4()),
            "conversation_history": history
        }
        
        # Process the request
        try:
            response = self.coordinator.process_request(request)
            
            # Extract the content from the response
            if isinstance(response, dict):
                content = response.get("content", str(response))
            else:
                content = str(response)
                
            # Add the response to the history
            history.append((message, content))
            
            self.logger.info("Response processed successfully")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            history.append((message, f"Error: {str(e)}"))
            
        return history
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.
        
        Returns:
            The Gradio interface
        """
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=3
                )
                submit = gr.Button("Send")
                
            clear = gr.Button("Clear Chat")
            
            # Set up event handlers
            submit.click(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                None,
                msg
            )
            
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                None,
                msg
            )
            
            clear.click(lambda: [], None, chatbot)
            
        return interface
    
    def launch(self, **kwargs) -> None:
        """
        Launch the chat interface.
        
        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        interface = self.build_interface()
        interface.launch(**kwargs)

# --- Add the following execution block ---
if __name__ == "__main__":
    from ..agents.coordinator import Coordinator
    from ..agents.agent_factory import AgentFactory # Assuming AgentFactory exists and is needed
    from ..agents.agent_registry import AgentRegistry # Assuming AgentRegistry exists
    from ..config.unified_config import UnifiedConfig # Assuming UnifiedConfig is used
    from ..utils.logger import LoggerFactory # Assuming LoggerFactory is used

    logger.info("Initializing Agentic AI Framework components...")
    
    try:
        # Basic initialization (adjust paths/config as needed)
        # Load configuration
        config = UnifiedConfig.get_instance() # Or load from a specific path if needed
        
        # Create a logger factory
        logger_factory = LoggerFactory()
        
        # Create an agent registry
        agent_registry = AgentRegistry()
        
        # Create an agent factory (assuming default implementation)
        agent_factory = AgentFactory(registry=agent_registry, unified_config=config, logger=logger_factory.create("agent_factory"))
        
        # --- Register any specific agents your framework uses ---
        # Example: 
        # from ..agents.coding_agent import CodingAgent 
        # agent_factory.register_agent("coding_agent", CodingAgent)
        # ---------------------------------------------------------

        # Create the Coordinator 
        # It will internally create PromptTemplate instances which load YAMLs
        coordinator = Coordinator(
            agent_factory=agent_factory,
            unified_config=config,
            logger=logger_factory.create("coordinator")
            # Removed prompt_manager=... No longer needed here
            # Add other components like tool_finder, model_selector if needed
            # Note: Default ToolFinderAgent/RequestAnalyzer/ResponseAggregator created
            # inside Orchestrator will now correctly use PromptTemplate
        )
        
        logger.info("Coordinator initialized successfully.")
        
        # Create and launch the UI
        ui = SimpleChatUI(coordinator=coordinator)
        logger.info("Launching Simple Chat UI...")
        ui.launch(server_name="0.0.0.0") # Launch on all interfaces

    except ImportError as ie:
         logger.error(f"Import error during initialization: {ie}. Please ensure all necessary components exist and paths are correct.")
    except Exception as e:
        logger.error(f"Failed to initialize or launch the application: {e}", exc_info=True) 