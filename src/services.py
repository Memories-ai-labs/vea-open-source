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

# --- Initialize LLM clients ---
# The system uses two LLMs:
#
#   main_llm   — text + tool-calling workhorse for the agent loop. Any frontier
#                model works (Gemini, Claude, GPT) because all calls go through
#                the OpenRouterManager shim. Defaults to Claude Opus 4.6 when
#                OPENROUTER_API_KEY is set; falls back to Vertex AI Gemini.
#
#   video_llm  — used ONLY for tasks that need native video input
#                (refine_clip_timestamps, verify_preview). Defaults to
#                Gemini 2.5 Pro because it accepts video frames + audio as
#                a single input, which Claude/GPT can't match today.
#
# Override model IDs via env:
#   MAIN_LLM_MODEL      e.g. anthropic/claude-opus-4.6 | openai/gpt-5
#                       Falls back to OPENROUTER_MODEL, then to Claude Opus 4.6.
#   VIDEO_LLM_MODEL     e.g. google/gemini-2.5-pro | google/gemini-2.5-flash
#                       Falls back to Gemini 2.5 Flash via Vertex.
#
# ``gemini_manager`` is kept as a backwards-compat alias pointing at main_llm so
# existing call sites (and the shim's ``genai_client`` interface) continue to
# work unchanged.
# Main-LLM catalog the dashboard switcher exposes. Video-capable models are
# pinned to ``video_llm`` and excluded here — this list is only for the
# text+tool-calling agent loop.
AVAILABLE_MAIN_MODELS = [
    {"id": "anthropic/claude-opus-4.6",         "name": "Claude Opus 4.6",     "hint": "Highest quality, slowest"},
    {"id": "anthropic/claude-sonnet-4.6",       "name": "Claude Sonnet 4.6",   "hint": "Balanced reasoning + speed"},
    {"id": "openai/gpt-5.4",                    "name": "GPT-5.4",             "hint": "OpenAI frontier"},
    {"id": "minimax/minimax-m2.7",              "name": "MiniMax M2.7",        "hint": "Agentic multi-agent workflows"},
    {"id": "qwen/qwen3.6-plus",                 "name": "Qwen 3.6 Plus",       "hint": "Hybrid linear-attention + sparse MoE"},
    {"id": "google/gemini-3-flash-preview",     "name": "Gemini 3 Flash",      "hint": "Fast, low cost"},
    {"id": "google/gemini-3.1-pro-preview",     "name": "Gemini 3.1 Pro",      "hint": "Google frontier"},
]

main_llm = None      # type: ignore[assignment]
video_llm = None     # type: ignore[assignment]

_llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

# ── main_llm: prefer OpenRouter with Claude Opus 4.6, fall back to Vertex
_main_model = (
    os.environ.get("MAIN_LLM_MODEL")
    or os.environ.get("OPENROUTER_MODEL")
    or "anthropic/claude-opus-4.6"
)

if _openrouter_key and _llm_provider != "vertex":
    try:
        from lib.llm.OpenRouterManager import OpenRouterManager
        main_llm = OpenRouterManager(model=_main_model, api_key=_openrouter_key)  # type: ignore[assignment]
        logger.info(f"main_llm initialized via OpenRouter (model={_main_model})")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenRouter main_llm: {e}")

if main_llm is None:
    try:
        main_llm = GeminiGenaiManager()
        logger.info("main_llm initialized via Vertex AI Gemini (fallback)")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini main_llm: {e}")

# ── video_llm: prefer Vertex AI Gemini (strongest native video support).
# Routing rule:
#   - If VIDEO_LLM_MODEL contains "/" (e.g. "google/gemini-2.5-pro") → OpenRouter
#   - Else (bare "gemini-2.5-flash") → Vertex AI Gemini
# This lets operators choose either path explicitly while keeping the default
# (bare Gemini model name) on Vertex where video frames work best.
_video_model = os.environ.get("VIDEO_LLM_MODEL", "gemini-2.5-flash")
_video_via_openrouter = "/" in _video_model

try:
    if _video_via_openrouter and _openrouter_key:
        from lib.llm.OpenRouterManager import OpenRouterManager
        video_llm = OpenRouterManager(model=_video_model, api_key=_openrouter_key)  # type: ignore[assignment]
        logger.info(f"video_llm initialized via OpenRouter (model={_video_model})")
    else:
        video_llm = GeminiGenaiManager(model=_video_model)
        logger.info(f"video_llm initialized via Vertex AI ({_video_model})")
except Exception as e:
    logger.warning(f"Failed to initialize video_llm: {e}. Falling back to main_llm.")
    video_llm = main_llm

# Backwards-compat alias — existing code that imports `gemini_manager` keeps
# working. New code should prefer main_llm / video_llm.
gemini_manager = main_llm

# --- Active planning sessions (project_name -> asyncio state) ---
# Each entry: {event_queue, pause_event, inject_queue, task}
_planning_sessions: Dict[str, Dict] = {}

# --- Active agent sessions (project_name -> AgentSession) ---
_agent_sessions: Dict[str, AgentSession] = {}


def set_main_llm(model_id: str) -> str:
    """Swap ``main_llm`` (and every live AgentSession's reference to it) to
    ``model_id``. Returns the id that ended up wired in.

    Only models listed in ``AVAILABLE_MAIN_MODELS`` are accepted. ``video_llm``
    is untouched — native-video tasks keep their pinned model.
    """
    global main_llm, gemini_manager
    allowed = {m["id"] for m in AVAILABLE_MAIN_MODELS}
    if model_id not in allowed:
        raise ValueError(f"Unsupported main_llm model: {model_id!r}")

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot switch model")

    from lib.llm.OpenRouterManager import OpenRouterManager
    new_llm = OpenRouterManager(model=model_id, api_key=key)

    main_llm = new_llm
    gemini_manager = new_llm
    for sess in _agent_sessions.values():
        sess.gemini = new_llm
    logger.info(f"main_llm switched to {model_id}")
    return model_id

# --- Indexing broadcast state ---
# project_name -> list of async emit callables (one per connected WS client)
_indexing_emitters: Dict[str, list] = {}
# project_name -> latest progress dict (status, percent, message, video_count, etc.)
_indexing_progress: Dict[str, dict] = {}
