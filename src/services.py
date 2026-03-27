# src/services.py
"""
Shared state and service initialization.

All route modules import shared state from here to avoid circular imports.
"""

import logging
import os
from typing import Dict, Optional

from lib.oss.storage_factory import get_storage_client
from lib.llm.MemoriesAiManager import MemoriesAiManager, create_memories_manager
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import get_storage_mode, ensure_local_directories
from src.pipelines.v2.agent.agent_session import AgentSession

logger = logging.getLogger(__name__)

# --- Initialize Storage client (local or cloud based on config) ---
storage_client = get_storage_client()
logger.info(f"Storage mode: {get_storage_mode()}")
ensure_local_directories()

# Alias for backward compatibility with existing code
gcp_oss = storage_client

# --- Initialize Memories.ai client (optional - only if API key is configured) ---
memories_manager: Optional[MemoriesAiManager] = None

try:
    if os.environ.get("MEMORIES_API_KEY"):
        memories_manager = create_memories_manager(debug=True)
        logger.info("Memories.ai client initialized")
    else:
        logger.info("Memories.ai API key not configured - video understanding will use Gemini")
except Exception as e:
    logger.warning(f"Failed to initialize Memories.ai client: {e}")

# --- Initialize LLM client (OpenRouter or Vertex AI Gemini) ---
gemini_manager = None  # type: ignore[assignment]
_llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

if _openrouter_key and _llm_provider != "vertex":
    try:
        from lib.llm.OpenRouterManager import OpenRouterManager
        _or_model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        gemini_manager = OpenRouterManager(model=_or_model, api_key=_openrouter_key)  # type: ignore[assignment]
        logger.info(f"OpenRouter client initialized (model={_or_model})")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenRouter client: {e}")

if gemini_manager is None:
    try:
        gemini_manager = GeminiGenaiManager()
        logger.info("Gemini Vertex AI client initialized")
    except Exception as e:
        gemini_manager = None
        logger.warning(f"Failed to initialize Gemini client: {e}")

# --- Active planning sessions (project_name -> asyncio state) ---
# Each entry: {event_queue, pause_event, inject_queue, task}
_planning_sessions: Dict[str, Dict] = {}

# --- Active agent sessions (project_name -> AgentSession) ---
_agent_sessions: Dict[str, AgentSession] = {}
