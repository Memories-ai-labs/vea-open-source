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

# --- Initialize Gemini client ---
gemini_manager: Optional[GeminiGenaiManager] = None
try:
    gemini_manager = GeminiGenaiManager()
    logger.info("Gemini client initialized")
except Exception as e:
    gemini_manager = None
    logger.warning(f"Failed to initialize Gemini client: {e}")

# --- Active planning sessions (project_name -> asyncio state) ---
# Each entry: {event_queue, pause_event, inject_queue, task}
_planning_sessions: Dict[str, Dict] = {}

# --- Active agent sessions (project_name -> AgentSession) ---
_agent_sessions: Dict[str, AgentSession] = {}
