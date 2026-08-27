# src/services.py
"""
Shared state and service initialization.

All route modules import shared state from here to avoid circular imports.

VIDEO UNDERSTANDING: served by the hosted Memories.ai Video Datalake over
HTTP (``src/datalake.py``). Three module-level handles carry it:

  * ``retrieval_ctx``       — datalake context; its one live method is
                         ``ctx.database.query("transcript", ...)``
  * ``querier``        — ``DatalakeQuerier``: semantic moment search
  * ``mavi_agent``     — ``DatalakeAgent``: rewrite → search → rerank → answer

They are lazy-initialised through :func:`init_retrieval` (name kept for the
FastAPI lifespan hook and the CLI). Those three are unscoped; per-project
handles bound to a project's collection come from :func:`project_handles`,
which is what AgentSession is given.

LLM clients (``main_llm`` + ``video_llm``) are VEA's own
``OpenRouterManager`` / ``GeminiGenaiManager`` and are unrelated to the
retrieval backend.
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
# Retrieval wiring — hosted Video Datalake
# ---------------------------------------------------------------------------
#
# Three module-level singletons, all served by ``src/datalake.py``:
#   - retrieval_ctx        : datalake context. Only ``ctx.database.query(
#                       "transcript", ...)`` is live on it (audio transcript
#                       for sentence-boundary snapping).
#   - querier         : ``DatalakeQuerier`` — semantic moment search, no LLM.
#                       Used by the planning loop and the ``search_footage``
#                       tool.
#   - mavi_agent      : ``DatalakeAgent`` — rewrite → search (reranked) →
#                       answer, over VEA's own main_llm. Used by Phase 1
#                       gists and the ``ask_memories`` tool. ``ask`` takes a
#                       single ``video_id``; VEA calls it once per project
#                       video when it needs multi-video coverage.
#
# These three are unscoped. Per-project handles — bound to that project's
# collection and its filename -> ``vid_...`` map — come from
# ``project_handles(session)``, which is what AgentSession is given.
#
# Lazy-initialised via ``init_retrieval()`` (called from app.py's lifespan
# startup hook) because building the datalake context is async.

# Retrieval handles. ``retrieval_ctx`` is kept as the name every call site already
# uses; on this backend it is the datalake context (its one live method is
# ``ctx.database.query("transcript", ...)``).
retrieval_ctx = None  # type: ignore[assignment]
datalake_client = None  # type: ignore[assignment]
retrieval_lifecycle = None  # type: ignore[assignment]
querier = None  # type: ignore[assignment]
mavi_agent = None  # type: ignore[assignment]


async def init_retrieval() -> None:
    """Construct the datalake client + the process-wide retrieval handles.

    Idempotent. ``querier`` / ``mavi_agent`` here are unscoped — they answer
    against whatever collection the client is bound to. Per-project handles
    (bound to that project's collection and its filename -> vid_... map) come
    from :func:`project_handles`, which is what AgentSession is given.

    The name is kept for call-site compatibility with the FastAPI lifespan
    hook and the CLI.
    """
    global retrieval_ctx, retrieval_lifecycle, querier, mavi_agent, datalake_client
    if retrieval_ctx is not None:
        return

    from src.datalake import build_datalake_context

    retrieval_ctx, retrieval_lifecycle, querier, mavi_agent = await build_datalake_context(main_llm)
    datalake_client = retrieval_ctx.client
    logger.info("video backend: Memories.ai Video Datalake")


def project_handles(session) -> tuple:
    """Return ``(querier, agent)`` scoped to one project's session.

    Retrieval is per project: the collection recorded on the session, and the
    filename -> ``vid_...`` map its entries carry. Falls back to the unscoped
    handles when a session has neither (a project indexed before those fields
    existed).
    """
    from src.datalake import DatalakeAgent, DatalakeClient, DatalakeQuerier

    collection_id = getattr(session, "datalake_collection_id", "") or ""
    name_map = {
        v.video_name: v.datalake_video_id
        for v in getattr(session, "videos", [])
        if getattr(v, "datalake_video_id", "")
    }
    if not collection_id and not name_map:
        return querier, mavi_agent

    client = DatalakeClient(collection_id=collection_id, name_map=name_map)
    scoped_querier = DatalakeQuerier(client)
    scoped_agent = DatalakeAgent(client, main_llm, video_ids=list(name_map.keys()))
    logger.info(
        f"[DATALAKE] project handles bound (collection={collection_id or 'unset'}, "
        f"{len(name_map)} videos)"
    )
    return scoped_querier, scoped_agent


async def close_retrieval() -> None:
    """Tear down the retrieval lifecycle (closes the datalake HTTP session).

    Safe to call from FastAPI's shutdown hook or any async exit path.
    """
    global retrieval_ctx, retrieval_lifecycle, querier, mavi_agent
    if retrieval_lifecycle is not None:
        try:
            await retrieval_lifecycle.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"retrieval backend shutdown raised: {e}")
        retrieval_ctx = None
        retrieval_lifecycle = None
        querier = None
        mavi_agent = None
