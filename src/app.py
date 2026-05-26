# app.py

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import WORKSPACES_DIR  # noqa: F401 - re-exported for test monkeypatching
from src.routes import v2_projects, v2_pipelines
from src.routes.v2_websockets import register_websocket_routes

# Re-export shared state so existing tests can monkeypatch via "src.app.<name>"
from src import services  # noqa: F401
from src.services import (  # noqa: F401
    storage_client,
    gcp_oss,
    gemini_manager,
    _planning_sessions,
    _agent_sessions,
)
# Re-export generate_fcpxml so tests can patch "src.app.generate_fcpxml"
from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml  # noqa: F401


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Lifespan: bring lvmm-core up on startup, down on shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook — init lvmm-core context + tear it down cleanly.

    lvmm-core's ``build_local_context`` is async, so we can't do it at
    module import time. The lifespan handler is the standard FastAPI
    place for async startup/shutdown.
    """
    try:
        await services.init_lvmm()
    except Exception:
        # init_lvmm logs the error itself; let the app start anyway so
        # routes can return a clean 503 instead of the server failing to
        # boot. Routes that need lvmm-core check ``services.mavi_agent``
        # before using it.
        logger.exception("lvmm-core initialization failed at startup")

    # Tool-level dependency check — surfaces silently-degraded tools at
    # startup instead of mid-edit. ``select_music`` losing beat detection
    # because librosa is missing, or ``refine_clip_timestamps`` losing
    # PySceneDetect cut hints, used to be discoverable only when a user
    # ran the affected tool. Now those gaps print on stderr at boot.
    from src.pipelines.v2.tool_prereqs import log_check_results
    log_check_results()

    yield

    try:
        await services.close_lvmm()
    except Exception:
        logger.exception("lvmm-core shutdown failed")


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
