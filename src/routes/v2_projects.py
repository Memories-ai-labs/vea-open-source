# src/routes/v2_projects.py
"""V2 project management routes: CRUD and clear endpoints."""

import logging
from fastapi import APIRouter, HTTPException

from src import config as _config
from src.pipelines.v2.workspace import WorkspaceManager
from src.routes._route_utils import safe_workspace as _workspace
from src import services

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(f"{_config.V2_API_PREFIX}/projects")
async def v2_list_projects():
    """
    List all project workspaces, including ones with footage but not yet indexed.
    The dashboard uses this to show the project browser.
    """
    projects = WorkspaceManager.list_projects(_config.WORKSPACES_DIR)
    return {"projects": projects}


@router.get(f"{_config.V2_API_PREFIX}/system/info")
async def v2_system_info():
    """
    Model IDs currently wired into `services.main_llm` and `services.video_llm`,
    plus the catalogs the dashboard switcher can select from.
    """
    def _name(llm) -> str:
        if llm is None:
            return "none"
        return getattr(llm, "model", None) or llm.__class__.__name__

    return {
        "main_llm": _name(services.main_llm),
        "video_llm": _name(services.video_llm),
        "available_main_models": services.AVAILABLE_MAIN_MODELS,
        "available_video_models": services.AVAILABLE_VIDEO_MODELS,
    }


@router.post(f"{_config.V2_API_PREFIX}/system/model")
async def v2_set_main_model(payload: dict):
    """
    Swap the main_llm to a different model from ``AVAILABLE_MAIN_MODELS``.
    Active agent sessions pick up the new model on their next request.
    video_llm is untouched.
    """
    model_id = (payload or {}).get("model")
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(status_code=400, detail="`model` is required")
    try:
        applied = services.set_main_llm(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "main_llm": applied}


@router.post(f"{_config.V2_API_PREFIX}/system/video_model")
async def v2_set_video_model(payload: dict):
    """
    Swap the video_llm to a different model from ``AVAILABLE_VIDEO_MODELS``.
    Bare IDs route via Vertex; "org/model" IDs route via OpenRouter.
    Active agent sessions pick up the new model on their next video-tool call.
    """
    model_id = (payload or {}).get("model")
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(status_code=400, detail="`model` is required")
    try:
        applied = services.set_video_llm(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "video_llm": applied}


@router.post(f"{_config.V2_API_PREFIX}/projects/create")
async def v2_create_project(project_name: str):
    """
    Create an empty project workspace directory with the standard layout.
    After creation, drop footage into  data/workspaces/{project_name}/footage/
    and then call /v2/index.
    """
    if not project_name or not project_name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="project_name must contain only letters, numbers, hyphens, and underscores."
        )
    workspace = _workspace(project_name)
    if workspace.dir_exists():
        return {"status": "exists", "project_name": project_name, "path": str(workspace.root)}
    workspace.create()
    return {
        "status": "created",
        "project_name": project_name,
        "path": str(workspace.root),
        "next_step": f"Drop footage into {workspace.get_footage_dir()}, then call /v2/index",
    }


@router.post(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/clear/gists")
async def v2_clear_gists(project_name: str):
    """Clear per-video gists and session gist. Re-index to regenerate."""
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    session = workspace.load_session()
    for v in session.videos:
        v.gist = ""
    session.gist = ""
    workspace.save_session(session)
    # Clear context.md gist section
    ctx_path = workspace.root / "context.md"
    if ctx_path.exists():
        ctx_path.unlink()
    return {"status": "cleared", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/clear/planning")
async def v2_clear_planning(project_name: str):
    """Clear planning context: storyboard, clips, iterations, context.md, agent chat, scratchpads."""
    import shutil
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    for fname in ["storyboard.json", "clips.json", "context.md", "chat_history.json", "event_log.json"]:
        p = workspace.root / fname
        if p.exists():
            p.unlink()

    for subdir in ["iterations", "scratchpads", "fcpxml", "renders"]:
        d = workspace.root / subdir
        if d.exists():
            shutil.rmtree(d)
            d.mkdir()

    # Clear in-memory agent session
    services._agent_sessions.pop(project_name, None)

    session = workspace.load_session()
    from src.pipelines.v2.schemas import PlanningState
    session.planning = PlanningState()
    if session.status == "planning":
        session.status = "indexed"
    workspace.save_session(session)
    return {"status": "cleared", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/clear/session")
async def v2_clear_session(project_name: str):
    """Full local reset -- deletes session.json, keeps footage."""
    import shutil
    workspace = _workspace(project_name)
    if not workspace.dir_exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    for fname in ["session.json", "storyboard.json", "clips.json", "context.md"]:
        p = workspace.root / fname
        if p.exists():
            p.unlink()

    for subdir in ["iterations", "narration", "music", "fcpxml", "renders", "logs"]:
        d = workspace.root / subdir
        if d.exists():
            shutil.rmtree(d)
            d.mkdir()

    return {"status": "reset", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/clear/memories")
async def v2_clear_memories(project_name: str):
    """Delete uploaded videos from Memories.ai cloud. Irreversible."""
    if not services.memories_manager:
        raise HTTPException(status_code=500, detail="Memories.ai not configured.")
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    session = workspace.load_session()
    deleted = []
    errors = []
    for v in session.videos:
        if v.video_no:
            try:
                await services.memories_manager.delete_video(v.video_no)
                deleted.append(v.video_name)
                v.video_no = ""
            except Exception as e:
                errors.append(f"{v.video_name}: {e}")

    session.status = "new"
    session.gist = ""
    for v in session.videos:
        v.gist = ""
    workspace.save_session(session)
    return {"status": "cleared", "deleted": deleted, "errors": errors, "project_name": project_name}
