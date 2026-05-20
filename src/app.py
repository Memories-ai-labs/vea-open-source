# app.py

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import WORKSPACES_DIR  # noqa: F401 - re-exported for test monkeypatching
from src.routes import v2_projects, v2_pipelines
from src.routes.v2_websockets import register_websocket_routes

# Re-export shared state so existing tests can monkeypatch via "src.app.<name>".
# PORT NOTE (2026-05-19): ``memories_manager`` removed in favour of the
# lvmm-core trio (lvmm_ctx / searcher / mavi_agent). Re-exported here so
# the same monkeypatch pattern works.
from src.services import (  # noqa: F401
    storage_client,
    gcp_oss,
    gemini_manager,
    lvmm_ctx,
    lvmm_lifecycle,
    searcher,
    mavi_agent,
    init_lvmm,
    close_lvmm,
    _planning_sessions,
    _agent_sessions,
)
# Re-export generate_fcpxml so tests can patch "src.app.generate_fcpxml"
from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml  # noqa: F401


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Lifespan: eagerly init lvmm-core at startup; tear down at shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler.

    Eagerly initialises lvmm-core at startup so the first incoming request
    doesn't pay the (one-time) cost of building the PipelineContext +
    loading adapters. The init function is itself idempotent — routes
    still call ``await services.init_lvmm()`` defensively so they work
    in tests / scripts that don't go through the full lifespan.
    """
    try:
        await init_lvmm()
        logger.info("lvmm-core ready (lifespan startup).")
    except Exception as e:
        # Don't crash app startup — let individual route calls surface the
        # error to clients with a clear message instead.
        logger.warning(f"lvmm-core init failed at startup (will retry per-request): {e}")
    try:
        yield
    finally:
        try:
            await close_lvmm()
            logger.info("lvmm-core closed (lifespan shutdown).")
        except Exception as e:
            logger.warning(f"lvmm-core shutdown raised: {e}")


# --- Initialize FastAPI app ---
app = FastAPI(lifespan=lifespan)

# --- Serve React dashboard ---
_DASHBOARD_DIST = Path(__file__).parent.parent / "dashboard" / "dist"
if _DASHBOARD_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(_DASHBOARD_DIST), html=True), name="dashboard")
    logger.info(f"Dashboard served from {_DASHBOARD_DIST}")


@app.get("/")
async def root():
    return {"message": "FastAPI inference service is running."}


# --- Include route modules ---
app.include_router(v2_projects.router)
app.include_router(v2_pipelines.router)

# --- Register WebSocket routes (cannot use APIRouter) ---
register_websocket_routes(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
