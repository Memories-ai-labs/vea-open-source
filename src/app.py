# app.py

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import WORKSPACES_DIR  # noqa: F401 - re-exported for test monkeypatching
from src.routes import v2_projects, v2_pipelines
from src.routes.v2_websockets import register_websocket_routes

# Re-export shared state so existing tests can monkeypatch via "src.app.<name>"
from src.services import (  # noqa: F401
    storage_client,
    gcp_oss,
    memories_manager,
    gemini_manager,
    _planning_sessions,
    _agent_sessions,
)
# Re-export generate_fcpxml so tests can patch "src.app.generate_fcpxml"
from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml  # noqa: F401


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI()

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
