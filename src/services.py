# src/services.py
"""
Shared state and service initialization.

All route modules import shared state from here to avoid circular imports.
"""

import asyncio
import logging
import os
from typing import Dict, Optional

from lib.oss.storage_factory import get_storage_client
from lib.llm.MemoriesAiManager import MemoriesAiManager, create_memories_manager
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import get_storage_mode, ensure_local_directories, WORKSPACES_DIR
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
_memories_pending_callbacks: Dict[str, asyncio.Future] = {}

try:
    if os.environ.get("MEMORIES_API_KEY"):
        memories_manager = create_memories_manager(debug=True)  # Set debug=False to reduce output
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

# Caption API callback URL - loaded from config.json api_keys section
# Set to your public ngrok/server URL, e.g., "https://xxx.ngrok-free.app/webhooks/memories/caption"
_callback_url = os.environ.get("MEMORIES_CAPTION_CALLBACK_URL", "")
CAPTION_CALLBACK_URL = _callback_url if _callback_url else None  # Treat empty string as None
if CAPTION_CALLBACK_URL:
    logger.info(f"Video Caption API callback URL: {CAPTION_CALLBACK_URL}")
else:
    logger.info("MEMORIES_CAPTION_CALLBACK_URL not set - will use Chat API instead of Caption API")


def register_memories_callback(task_id: str) -> asyncio.Future:
    """
    Register a pending callback for a Memories.ai async operation.

    Returns a Future that will be resolved when the webhook is called.

    Usage:
        future = register_memories_callback(task_id)
        result = await asyncio.wait_for(future, timeout=300)
    """
    loop = asyncio.get_running_loop()  # Use running loop, not default event loop
    future = loop.create_future()
    _memories_pending_callbacks[task_id] = future
    logger.info(f"[CALLBACK] Registered callback for task: {task_id} (pending: {len(_memories_pending_callbacks)})")
    return future


def cleanup_orphaned_callbacks(max_age_seconds: int = 3600):
    """
    Remove callbacks that have been pending for too long.

    Call this periodically to prevent memory leaks from failed requests.
    """
    # For now, just log the count - in production you'd track timestamps
    if _memories_pending_callbacks:
        logger.warning(f"[CALLBACK] {len(_memories_pending_callbacks)} pending callbacks: {list(_memories_pending_callbacks.keys())[:5]}...")
