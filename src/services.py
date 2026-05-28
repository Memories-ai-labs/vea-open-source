# src/services.py
"""
Shared state and service initialization.

All route modules import shared state from here to avoid circular imports.

VIDEO UNDERSTANDING (2026-05): Migrated from memories.ai cloud API to
lvmm-core's local stack. Three module-level handles replace what used
to be ``memories_manager``:

  * ``lvmm_ctx``       — lvmm-core PipelineContext (adapters + DB + storage)
  * ``querier``        — luci_memory.Querier for vector clip search
                         (renamed upstream from Searcher in lvmm-core
                         commit 330a34c — kept this alias name in VEA so
                         call sites read naturally)
  * ``mavi_agent``     — MaviAgent for RAG-style chat (rewrite → search →
                         rerank → answer)

These are lazy-initialised through :func:`init_lvmm` because lvmm-core's
``build_local_context`` is async. AgentSession + the agent tools take
these handles via constructor; routes get them from this module.

LLM clients (``main_llm`` + ``video_llm``) are unchanged — VEA's
``OpenRouterManager`` + ``GeminiGenaiManager`` stay as-is. They serve a
different purpose than lvmm-core's ILLM adapters (which back the
MaviAgent's internal LLM calls).
"""

import logging
import os
from typing import Dict

from lib.oss.storage_factory import get_storage_client
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
# listed separately in AVAILABLE_VIDEO_MODELS — this list is only for the
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

# Video-LLM catalog for tasks that need native video input (refine_clip_timestamps,
# verify_preview). Bare model names (no "/") route via Vertex; slash-prefixed IDs
# route via OpenRouter. See the _video_via_openrouter branch below.
AVAILABLE_VIDEO_MODELS = [
    {"id": "gemini-2.5-flash",                  "name": "Gemini 2.5 Flash",    "hint": "Vertex; cheap, proven default"},
    {"id": "gemini-2.5-pro",                    "name": "Gemini 2.5 Pro",      "hint": "Vertex; higher quality"},
    {"id": "google/gemini-3-flash-preview",     "name": "Gemini 3 Flash",      "hint": "OpenRouter; newer than 2.5 Flash"},
    {"id": "google/gemini-3.1-pro-preview",     "name": "Gemini 3.1 Pro",      "hint": "OpenRouter; frontier video"},
    {"id": "qwen/qwen3.6-plus",                 "name": "Qwen 3.6 Plus",       "hint": "OpenRouter; 1M context, video-capable"},
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


def set_video_llm(model_id: str) -> str:
    """Swap ``video_llm`` (and every live AgentSession's ``.video_llm``) to
    ``model_id``. Returns the id that ended up wired in.

    Only models listed in ``AVAILABLE_VIDEO_MODELS`` are accepted. Routing:
    bare names ("gemini-2.5-flash") go via Vertex; slash-prefixed IDs
    ("google/gemini-3-flash-preview") go via OpenRouter.
    """
    global video_llm
    allowed = {m["id"] for m in AVAILABLE_VIDEO_MODELS}
    if model_id not in allowed:
        raise ValueError(f"Unsupported video_llm model: {model_id!r}")

    via_openrouter = "/" in model_id
    if via_openrouter:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set — cannot switch model")
        from lib.llm.OpenRouterManager import OpenRouterManager
        new_llm = OpenRouterManager(model=model_id, api_key=key)
    else:
        new_llm = GeminiGenaiManager(model=model_id)

    video_llm = new_llm
    for sess in _agent_sessions.values():
        sess.video_llm = new_llm
    logger.info(f"video_llm switched to {model_id}")
    return model_id

# --- Indexing broadcast state ---
# project_name -> list of async emit callables (one per connected WS client)
_indexing_emitters: Dict[str, list] = {}
# project_name -> latest progress dict (status, percent, message, video_count, etc.)
_indexing_progress: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# lvmm-core wiring — local indexing, retrieval, and RAG chat
# ---------------------------------------------------------------------------
#
# Replaces the previous memories.ai HTTP client. Three module-level
# singletons:
#   - lvmm_ctx        : lvmm-core PipelineContext (carries LLM, embedding,
#                       storage, database, vector_db adapter instances).
#   - querier         : core/retrieval/luci_memory/Querier — vector clip
#                       search facade. Independent of any LLM. Used by the
#                       planning loop and AgentSession's search_footage tool.
#                       (Was named ``Searcher`` until lvmm-core commit
#                       330a34c renamed it; ctor now also takes ``database``.)
#   - mavi_agent      : agents/MaviAgent — fixed-pipeline RAG agent.
#                       query-rewrite → parallel search → rerank → answer.
#                       Used by Phase 1 gist + AgentSession's ask_memories tool.
#                       (Ctor now takes ``querier=`` not ``searcher=``;
#                       ``ask`` no longer accepts a video_ids list — only a
#                       single ``video_id``. VEA passes the first video_id
#                       when there is more than one — see ``ToolExecutor.
#                       _ask_memories``.)
#
# Lazy-initialised via ``init_lvmm()`` (called from app.py's lifespan
# startup hook) because lvmm-core's ``build_local_context`` is async.
# Routes + AgentSession + the agent ToolExecutor take these handles by
# importing them from this module.

lvmm_ctx = None  # type: ignore[assignment]
lvmm_lifecycle = None  # type: ignore[assignment]
querier = None  # type: ignore[assignment]
mavi_agent = None  # type: ignore[assignment]


async def init_lvmm() -> None:
    """Construct lvmm-core context + Searcher + MaviAgent (idempotent).

    Reads ``OPENROUTER_API_KEY`` / ``GEMINI_API_KEY`` from env (already
    populated by ``src.config`` at startup). MobileCLIP defaults to the
    local PyTorch adapter — no VPN, no Ray dependency. The weight file
    (~325 MB) auto-downloads to ``~/lvmm-data/models/mobileclip_s1.pt``
    on first call.

    Also configures lvmm-core's structured logging (idempotent). After
    this runs, every log line emitted from the ``lvmm_core.*`` logger
    tree carries auto-injected ``[pipeline=… stage=… run_id=…]`` context
    tags when inside a ``Pipeline.execute()`` call. Complements VEA's
    per-project ``logging_setup.py`` bundle.
    """
    global lvmm_ctx, lvmm_lifecycle, querier, mavi_agent
    if lvmm_ctx is not None:
        return

    try:
        from lvmm_core.services.local_dev import build_local_context
        from lvmm_core.core.retrieval.luci_memory.querier import Querier
        from lvmm_core.agents.mavi_agent import MaviAgent
        from lvmm_core.utils.logging import setup_logging
    except ImportError:
        logger.error(
            "lvmm-core not installed. From the vea-open-source repo root run: "
            "pip install -e ../lvmm-core  (or whatever the relative path is). "
            "See pyproject.toml."
        )
        raise

    _log_level = getattr(logging, os.environ.get("LVMM_LOG_LEVEL", "INFO").upper(), logging.INFO)
    setup_logging(level=_log_level)

    # provider auto-pick: prefer OpenRouter (VEA's standard env from
    # config.json), fall back to direct Gemini API if only GEMINI_API_KEY
    # is set. Note: this is the LLM that MaviAgent uses INTERNALLY for
    # query rewrite + reranking — separate from VEA's main_llm / video_llm.
    _lvmm_provider = "openrouter" if _openrouter_key else "gemini"
    lvmm_ctx, lvmm_lifecycle = await build_local_context(
        provider=_lvmm_provider,
        embedding="mobileclip-pytorch",
        face="none",   # VEA doesn't use lvmm-core's face pipeline
        asr="none",    # nor its ASR pipeline
        diarization="none",
    )
    querier = Querier(
        lvmm_ctx.vector_db,
        lvmm_ctx.text_embedding,
        lvmm_ctx.database,
    )
    mavi_agent = MaviAgent(llm=lvmm_ctx.llm, querier=querier)
    logger.info(
        f"lvmm-core initialised (provider={_lvmm_provider}, "
        "embedding=mobileclip-pytorch, SQLite local DB)"
    )


async def close_lvmm() -> None:
    """Tear down the lvmm-core lifecycle (closes DB connection).

    Safe to call from FastAPI's shutdown hook or any async exit path.
    """
    global lvmm_ctx, lvmm_lifecycle, querier, mavi_agent
    if lvmm_lifecycle is not None:
        try:
            await lvmm_lifecycle.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"lvmm-core shutdown raised: {e}")
        lvmm_ctx = None
        lvmm_lifecycle = None
        querier = None
        mavi_agent = None
