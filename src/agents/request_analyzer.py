"""
Request Analyzer V2 component.
Analyzes user requests to classify intent primarily using keyword matching.
"""
from typing import Dict, Any, Optional, Literal
import re
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIAgentError, ErrorHandler # Keep for potential future use?


class RequestAnalyzer:
    """
    Analyzes user requests to classify the primary intent:
    - META: Request for system information (agents, tools, config).
    - TASK: Any other request requiring processing or action.
    Uses simple keyword matching for META detection.
    Version: 2.0 (Refactored for simplicity and robustness)
    """

    # Keywords/patterns indicating a META request (case-insensitive)
    # Added word boundaries (\b) to avoid partial matches (e.g., 'tool' in 'stool')
    _META_PATTERNS = re.compile(
        r"\b(what tools|list tools|available tools|tool list|"
        r"what agents|list agents|available agents|agent list|"
        r"system info|configuration|config|capabilities|tell me about yourself)\b",
        re.IGNORECASE
    )

    def __init__(self,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the RequestAnalyzer.

        Args:
            unified_config: UnifiedConfig instance
            logger: Logger instance
        """
        self._config = unified_config or UnifiedConfig.get_instance()
        self._logger = logger or LoggerFactory.create(name="request_analyzer_v2")
        self._agent_config = self._config.get_agent_config("request_analyzer") # Still useful for potential future config
        self._logger.info("RequestAnalyzer V2 initialized.")


    def classify_request_intent(self, request: Dict[str, Any]) -> Literal["META", "TASK", "UNKNOWN"]:
        """
        Classify the user's request intent using simple keyword matching.

        Args:
            request: The request object containing the prompt.

        Returns:
            The classified intent: "META", "TASK", or "UNKNOWN" if prompt is missing.
        """
        prompt_text = request.get("prompt", "").strip()
        if not prompt_text:
            self._logger.warning("Request missing prompt text. Classifying intent as UNKNOWN.")
            return "UNKNOWN"

        self._logger.debug(f"Classifying intent for prompt: {prompt_text[:100]}...")

        try:
            # Check for META patterns
            if self._META_PATTERNS.search(prompt_text):
                self._logger.info(f"Prompt matched META pattern. Intent: META")
                return "META"
            else:
                # If not META, classify as TASK
                self._logger.info(f"Prompt did not match META pattern. Intent: TASK")
                return "TASK"

        except Exception as e:
            # Log full error with traceback
            err_logger = LoggerFactory.create("request_analyzer_intent_error")
            err_logger.error(f"Exception during intent classification: {str(e)}", exc_info=True)
            ErrorHandler.handle_error(
                AIAgentError(f"Failed to classify request intent: {str(e)}", agent_id="request_analyzer_v2"),
                self._logger
            )
            return "UNKNOWN"

    # --- Methods removed in V2 ---
    # - classify_use_case (responsibility moved or handled differently)
    # - get_agent_assignments (responsibility moved or handled differently)
    # - _load_system_prompt (no longer uses dedicated AI)
    # - analyze_tools (responsibility moved or handled differently)
    # - _format_agent_list (responsibility moved or handled differently)
    # - _format_tool_list (responsibility moved or handled differently)
    # - _parse_agent_assignments (responsibility moved or handled differently)
    # - _parse_tool_assignments (responsibility moved or handled differently) 