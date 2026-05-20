# src/services.py
"""
Shared state and service initialization.

All route modules import shared state from here to avoid circular imports.

PORT NOTE (2026-05-19): Migrated from memories.ai-hosted indexing+chat to
lvmm-core's local pipeline. Where this module used to expose
``memories_manager`` (an HTTP client for memories.ai), it now exposes:

  * ``lvmm_ctx``       — lvmm-core PipelineContext (adapters + DB + storage)
  * ``lvmm_lifecycle`` — handle for clean shutdown of lvmm-core's DB
  * ``searcher``       — luci_memory.Searcher over the local vector DB
  * ``mavi_agent``     — MaviAgent for RAG-style chat over the local index

VEA's LLM call sites also moved to lvmm-core: previously this module
constructed VEA-local GeminiGenaiManager / OpenRouterManager singletons;
now ``gemini_manager`` is a lvmm-core ILLM adapter (GeminiAdapter or
OpenRouterAdapter). Per-site refactor: ``manager.LLM_request(...)``
became ``await llm.generate(...)`` / ``await llm.generate_structured(...)``.
The only Gemini-SDK-specific call site (agent_session.py's tool-use loop)
reaches the raw client via the adapter's ``.client`` escape hatch.
"""

import asyncio
import logging
import os
from typing import Dict, Optional

from lib.oss.storage_factory import get_storage_client
from src.config import get_storage_mode, ensure_local_directories
from src.pipelines.v2.agent.agent_session import AgentSession

logger = logging.getLogger(__name__)

# --- Initialize Storage client (local or cloud based on config) ---
storage_client = get_storage_client()
logger.info(f"Storage mode: {get_storage_mode()}")
ensure_local_directories()

# Alias for backward compatibility with existing code
gcp_oss = storage_client

# ---------------------------------------------------------------------------
# lvmm-core wiring — local indexing, retrieval, and RAG chat
# ---------------------------------------------------------------------------
#
# Replaces the previous memories.ai HTTP client. Three module-level
# singletons:
#   - lvmm_ctx        : lvmm-core PipelineContext (carries LLM, embedding,
#                       storage, database, vector_db adapter instances).
#   - searcher        : core/retrieval/luci_memory/Searcher — vector search
#                       facade. Independent of any LLM. Used by Phase 2
#                       planning and AgentSession's search_footage tool.
#   - mavi_agent      : agents/MaviAgent — fixed-pipeline RAG agent.
#                       query-rewrite → search → rerank → answer.
#                       Used by Phase 1 gist + Phase 2 chat + AgentSession's
#                       ask_memories tool.
#
# We initialise these lazily through an asyncio event loop because
# build_local_context is async. Import-time eager construction would
# require nest_asyncio or a sync wrapper; lazy init keeps the import
# graph clean and the cost only paid when the app actually uses these
# capabilities.

lvmm_ctx = None  # type: ignore[assignment]
lvmm_lifecycle = None  # type: ignore[assignment]
searcher = None  # type: ignore[assignment]
mavi_agent = None  # type: ignore[assignment]


async def init_lvmm() -> None:
    """Construct lvmm-core context + Searcher + MaviAgent (idempotent).

    Reads OPENROUTER_API_KEY / GEMINI_API_KEY etc. from the env (already
    populated by config.py at startup). MobileCLIP defaults to the local
    PyTorch adapter — fastest path with no VPN, no Ray dependency.
    """
    global lvmm_ctx, lvmm_lifecycle, searcher, mavi_agent
    if lvmm_ctx is not None:
        return

    try:
        from lvmm_core.services.local_dev import build_local_context
        from lvmm_core.core.retrieval.luci_memory.searcher import Searcher
        from lvmm_core.agents.mavi_agent import MaviAgent
    except ImportError as e:
        logger.error(
            "lvmm-core not installed. From the vea-open-source repo root run: "
            "pip install -e ../lvmm-core  (or whatever the relative path is). "
            "See requirements.txt."
        )
        raise

    # Provider/embedding choices match the team's standard local dev:
    # Gemini for the LLM (via lvmm-core's GeminiAdapter, distinct from VEA's
    # own gemini_manager); MobileCLIP-PyTorch for embeddings (in-process,
    # no Ray endpoint required).
    lvmm_ctx, lvmm_lifecycle = await build_local_context(
        provider="gemini",
        embedding="mobileclip",
        face="none",
        asr="none",
        diarization="none",
    )
    searcher = Searcher(
        vector_db=lvmm_ctx.vector_db,
        text_embedding=lvmm_ctx.text_embedding,
    )
    mavi_agent = MaviAgent(llm=lvmm_ctx.llm, searcher=searcher)
    logger.info("lvmm-core initialised (Gemini LLM + MobileCLIP embeddings + SQLite)")


async def close_lvmm() -> None:
    """Tear down the lvmm-core lifecycle (closes DB connection).

    Safe to call from FastAPI's shutdown hook or any async exit path.
    """
    global lvmm_ctx, lvmm_lifecycle, searcher, mavi_agent
    if lvmm_lifecycle is not None:
        try:
            await lvmm_lifecycle.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"lvmm-core shutdown raised: {e}")
        lvmm_ctx = None
        lvmm_lifecycle = None
        searcher = None
        mavi_agent = None


# ---------------------------------------------------------------------------
# LLM client for VEA's own non-RAG calls (planning, narration, decision)
# ---------------------------------------------------------------------------
# Note: this is SEPARATE from lvmm-core's GeminiAdapter (which mavi_agent
# uses internally). VEA's planning_prompts / narration / etc. drive Gemini
# directly through this manager; that path is untouched by the port.

# gemini_manager: lvmm-core ILLM-implementing adapter.
#
# PORT NOTE (2026-05-19): Replaces the in-VEA GeminiGenaiManager + OpenRouterManager
# pair with lvmm-core's GeminiAdapter / OpenRouterAdapter. ``LLM_request``
# was VEA-flavored and is gone; consumer code calls ILLM.generate /
# generate_structured directly. agent_session.py reaches the raw Gemini SDK
# via the adapter's ``.client`` escape hatch.
#
# Selection logic:
#   * OPENROUTER_API_KEY set (and LLM_PROVIDER != "vertex") → OpenRouterAdapter
#   * Otherwise → lvmm-core GeminiAdapter (API-key mode).
#
# CAVEAT: lvmm-core's GeminiAdapter uses API-key auth (AIstudio / public
# Gemini API), not Vertex AI's project/location auth. Teams running on
# Vertex AI need to set OPENROUTER_API_KEY or extend GeminiAdapter — the
# old GeminiGenaiManager(vertexai=True, project=...) path is not preserved
# in this port.

gemini_manager = None  # type: ignore[assignment]
_llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
_gemini_key = os.environ.get("GEMINI_API_KEY", "")

if _openrouter_key and _llm_provider != "vertex":
    try:
        from lvmm_core.adapters.llm.openai_compat import OpenRouterAdapter
        _or_model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        gemini_manager = OpenRouterAdapter(api_key=_openrouter_key, model=_or_model)
        logger.info(f"OpenRouter client initialized (model={_or_model})")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenRouter client: {e}")

if gemini_manager is None and _gemini_key:
    try:
        from lvmm_core.adapters.llm.gemini import GeminiAdapter
        _gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_manager = GeminiAdapter(api_key=_gemini_key, model=_gemini_model)
        logger.info(f"Gemini client initialized (model={_gemini_model})")
    except Exception as e:
        gemini_manager = None
        logger.warning(f"Failed to initialize Gemini client: {e}")

if gemini_manager is None:
    logger.warning(
        "No LLM client initialized — set OPENROUTER_API_KEY (preferred) or "
        "GEMINI_API_KEY. Vertex AI auth requires extending lvmm-core's "
        "GeminiAdapter; not supported in this port."
    )

# --- Active planning sessions (project_name -> asyncio state) ---
# Each entry: {event_queue, pause_event, inject_queue, task}
_planning_sessions: Dict[str, Dict] = {}

# --- Active agent sessions (project_name -> AgentSession) ---
_agent_sessions: Dict[str, AgentSession] = {}

# --- Indexing broadcast state ---
# project_name -> list of async emit callables (one per connected WS client)
_indexing_emitters: Dict[str, list] = {}
# project_name -> latest progress dict (status, percent, message, video_count, etc.)
_indexing_progress: Dict[str, dict] = {}
