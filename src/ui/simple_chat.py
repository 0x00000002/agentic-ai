"""
Simple Chat UI for the Agentic AI Framework

This module provides a simple chat interface using Gradio for interacting with the coordinator.
"""
import os
import sys
import gradio as gr
import uuid
from typing import List, Tuple, Dict, Any, Optional
import logging
import numpy as np

# Add the parent directory to sys.path to import the package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.agents.coordinator import Coordinator
from src.config.user_config import UserConfig
from src.utils.logger import LoggerFactory
from src.config import configure, get_config, UseCasePreset
from src.agents.agent_factory import AgentFactory
from src.agents.agent_registry import AgentRegistry
from src.core.tool_enabled_ai import ToolEnabledAI

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
    A simple chat interface for interacting with the coordinator.
    Uses Gradio to create a web-based chat interface.
    """
    
    def __init__(self, coordinator, title: str = "Agentic AI Chat"):
        """
        Initialize the chat UI.
        
        Args:
            coordinator: The coordinator instance to use for processing requests
            title: The title of the chat interface
        """
        self.coordinator = coordinator
        self.title = title
        self.logger = logger
        self.is_transcribing = False
        
    async def process_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Process a user message asynchronously and return the updated chat history 
        and an empty string to clear the input field.

        Args:
            message: The user message
            history: The chat history

        Returns:
            A tuple containing:
             - Updated chat history
             - An empty string to clear the message input box
        """
        if not message:
            return history, "" # Return unchanged history and empty string for clearing
            
        self.logger.info(f"Processing message: {message[:50]}...")
        
        # Create a request for the coordinator
        request = {
            "prompt": message,
            "request_id": str(uuid.uuid4()),
            "conversation_history": history
        }
        
        # Process the request asynchronously
        try:
            # Await the coordinator's process_request method
            response = await self.coordinator.process_request(request)
            
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
            
        # Return updated history and empty string to clear the input
        return history, ""
    
    async def process_audio(self, language: str, audio_input: Optional[str], history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str, None]:
        """
        Process an audio input asynchronously, send it for transcription, update the message box,
        and clear the audio input component.

        Args:
            language: The language selected in the dropdown ("English", "Русский").
            audio_input: The file path to the recorded audio (from gr.Audio).
            history: The current chat history (unused in transcription).

        Returns:
            A tuple containing:
            - Updated message text (transcribed audio).
            - Unchanged history.
            - Status message (e.g., "Transcribing...").
            - None to clear the audio input component.
        """
        lang_code_map = {"English": "en", "Русский": "ru"}
        lang_code = lang_code_map.get(language, "en") # Default to English if mapping fails
        self.logger.info(f"Selected language: {language}, using code: {lang_code}")

        if self.is_transcribing or audio_input is None:
            # Prevent processing if already transcribing or input is None (e.g., initial load)
            # Return current state or default empty state
            return "", history, "Ready", None 

        self.is_transcribing = True
        self.logger.info(f"Processing audio file: {audio_input}")
        status_update = "Transcribing..."

        # Create a transcription request for the coordinator
        request = {
            "type": "audio_transcription", # Specific type for coordinator routing
            "audio_path": audio_input,
            "language": lang_code,
            "request_id": str(uuid.uuid4()),
            "conversation_history": [] # History not needed for transcription
        }

        transcribed_text = ""
        try:
            # Send request to coordinator asynchronously
            response = await self.coordinator.process_request(request)

            # Extract transcribed text
            if isinstance(response, dict):
                transcribed_text = response.get("content", "")
                if response.get("status") == "error":
                    status_update = f"Error: {response.get('error', 'Transcription failed')}"
                    self.logger.error(f"Transcription error: {response.get('error', 'Unknown')}")
                else:
                    status_update = "Transcription complete. Press Send."
                    self.logger.info(f"Transcription received: {transcribed_text[:50]}...")
            else:
                # Fallback if response format is unexpected
                transcribed_text = str(response)
                status_update = "Transcription complete (unexpected format). Press Send."
                self.logger.warning(f"Unexpected transcription response format: {type(response)}")

        except Exception as e:
            self.logger.error(f"Error during transcription request: {str(e)}", exc_info=True)
            status_update = f"Error: {str(e)}"
        
        finally:
            self.is_transcribing = False # Reset transcription lock

        # Return the transcribed text to populate the message box,
        # the unchanged history, the status update, and None to clear audio.
        return transcribed_text, history, status_update, None
    
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
                lang_dropdown = gr.Dropdown(
                    label="Language", 
                    choices=["English", "Русский"], 
                    value="English" # Default value
                )
                audio_in = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", # Get file path for ListenerAgent
                    label="Record Audio Message" 
                )
                audio_status = gr.Textbox(label="Audio Status", value="Ready", interactive=False)
            
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
                outputs=[chatbot, msg] # Update outputs to clear msg
            )
            
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg] # Update outputs to clear msg
            )
            
            clear.click(lambda: [], None, chatbot)
            
            audio_in.stop_recording(
                self.process_audio,
                inputs=[lang_dropdown, audio_in, chatbot], 
                outputs=[msg, chatbot, audio_status, audio_in] 
            )
            
        return interface
    
    def launch(self, **kwargs) -> None:
        """
        Launch the chat interface.
        
        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        interface = self.build_interface()
        interface.launch(**kwargs)

def run_simple_chat():
    """Run a simple chat UI with default configuration."""
    # Configure the framework
    configure(
        model="claude-3-haiku",  # Use a smaller model by default
        use_case=UseCasePreset.CHAT,
        temperature=0.7,
        show_thinking=True
    )
    
    # Get configuration
    config = get_config()
    
    # Create a logger
    chat_logger = LoggerFactory.create("simple_chat")
    
    # Create an AI instance
    ai_instance = ToolEnabledAI(
        model=config.get_default_model(),
        system_prompt="You are a helpful assistant.",
        logger=chat_logger
    )
    
    # Create an agent registry
    registry = AgentRegistry()
    
    # Create an agent factory
    agent_factory = AgentFactory(registry=registry, unified_config=config, logger=chat_logger)
    
    # Create a coordinator agent
    coordinator = agent_factory.create("coordinator", ai_instance=ai_instance)
    
    # Create and launch the chat UI
    chat_ui = SimpleChatUI(coordinator=coordinator, title="Agentic AI Chat")
    chat_ui.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True              # Create a public URL
    )

# Direct execution
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Simple Chat UI application...")
    run_simple_chat() 