# src/routes/v2_pipelines.py
"""V2 pipeline routes: index, plan, fcpxml, narration, music, crop, render, resolve."""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.schema import (
    V2IndexRequest,
    V2IndexResponse,
    V2PlanRequest,
    V2GenerateFcpxmlRequest,
    V2GenerateFcpxmlResponse,
    V2NarrationRequest,
    V2MusicRequest,
    V2CropRequest,
    V2RenderRequest,
    V2RenderResponse,
    V2ResolveStatusResponse,
)
from src import config as _config
from src.pipelines.v2.workspace import WorkspaceManager
from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension
from src.pipelines.v2.planning.iterative_planning_loop import IterativePlanningLoop
from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml
from src import services

logger = logging.getLogger(__name__)

router = APIRouter()


def _workspace(project_name: str) -> WorkspaceManager:
    """Construct a WorkspaceManager, converting ValueError into HTTP 400."""
    try:
        return WorkspaceManager(project_name, _config.WORKSPACES_DIR)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _safe_child(directory: Path, name: str) -> Path:
    """Resolve ``directory / name`` and verify the result is inside ``directory``.

    Rejects path traversal via ``..`` or absolute paths in ``name``.
    """
    candidate = (directory / name).resolve()
    root = directory.resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"Invalid path: {name!r}")
    return candidate


@router.post(f"{_config.V2_API_PREFIX}/index", response_model=V2IndexResponse)
async def v2_index(request: V2IndexRequest):
    """
    V2: Lightweight video comprehension.
    Uploads to Memories.ai (or reuses cached video_no), gets a broad gist.
    Much faster than v1 /index -- no scene-by-scene analysis.

    source_dir is optional. If omitted, footage is read from the workspace's
    footage/ subdirectory (data/workspaces/{project_name}/footage/).
    """
    try:
        workspace = _workspace(request.project_name)

        # Auto-detect source_dir from workspace footage/ when not explicitly provided
        source_dir = request.source_dir
        if not source_dir:
            footage_files = workspace.scan_footage()
            if footage_files:
                source_dir = str(workspace.get_footage_dir())
                logger.info(f"[V2 INDEX] Auto-detected footage dir: {source_dir} ({len(footage_files)} files)")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No source_dir provided and no footage found in {workspace.get_footage_dir()}. "
                           f"Drop video files there or pass source_dir explicitly."
                )

        logger.info(f"[V2 INDEX] project={request.project_name} source={source_dir} fresh={request.start_fresh}")

        if not services.mavi_agent or not services.lvmm_ctx:
            raise HTTPException(
                status_code=503,
                detail="lvmm-core not initialised. Check server startup logs.",
            )

        pipeline = LightweightComprehension(
            project_name=request.project_name,
            source_dir=source_dir,
            lvmm_ctx=services.lvmm_ctx,
            mavi_agent=services.mavi_agent,
            workspace=workspace,
        )
        session = await pipeline.run(start_fresh=request.start_fresh)

        return V2IndexResponse(
            project_name=request.project_name,
            video_nos=[v.video_no for v in session.videos],
            gist=session.gist,
            status=session.status,
        )
    except HTTPException:
        # Already-shaped HTTP errors (e.g. 503 lvmm-core unavailable) — let
        # them propagate with their own status, don't re-wrap as 500.
        raise
    except Exception as e:
        logger.error(f"[V2 INDEX] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post(f"{_config.V2_API_PREFIX}/plan")
async def v2_plan(request: V2PlanRequest):
    """
    V2: Start the iterative planning loop for a project.

    Launches planning as a background task. Connect to the WebSocket endpoint
    WS /video-edit/v2/session/{project_name} to receive live events.

    Returns immediately with {"status": "started"} or {"status": "already_running"}.
    """
    if not services.mavi_agent or not services.querier:
        raise HTTPException(
            status_code=503,
            detail="lvmm-core not initialised. Check server startup logs.",
        )
    if not services.gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured. Check GCP credentials.")

    project_name = request.project_name

    # Check for existing running session
    session_state = services._planning_sessions.get(project_name)
    if session_state and not session_state["task"].done():
        return {"status": "already_running", "project_name": project_name}

    # Load workspace -- must have been indexed first
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found. Run /v2/index first.")

    try:
        session = workspace.load_session()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")

    if not session.videos:
        raise HTTPException(status_code=400, detail="No videos indexed for this project.")

    # Set up async coordination primitives
    raw_queue: asyncio.Queue = asyncio.Queue()   # planning loop -> broadcaster
    subscribers: List[asyncio.Queue] = []        # one queue per connected WebSocket client
    pause_event: asyncio.Event = asyncio.Event()
    inject_queue: asyncio.Queue = asyncio.Queue()

    video_nos = [v.video_no for v in session.videos]

    loop_obj = IterativePlanningLoop(
        project_name=project_name,
        user_prompt=request.prompt,
        workspace=workspace,
        querier=services.querier,
        mavi_agent=services.mavi_agent,
        gemini=services.gemini_manager,
        video_nos=video_nos,
        video_entries=session.videos,
        max_iterations=request.max_iterations,
        target_duration_seconds=request.target_duration_seconds,
        event_queue=raw_queue,
        pause_event=pause_event,
        inject_prompt_queue=inject_queue,
    )

    async def _broadcast():
        """Fan events from the planning loop out to all connected WebSocket clients."""
        while True:
            event = await raw_queue.get()
            if hasattr(event, "to_dict"):
                payload = event.to_dict()
            elif isinstance(event, dict):
                payload = event
            else:
                payload = {"event_type": "unknown", "data": str(event)}
            for q in list(subscribers):
                await q.put(payload)
            if payload.get("event_type") in ("done", "error", "session_ended"):
                break

    async def _run_planning():
        try:
            workspace.update_status("planning")
            await loop_obj.run()
        except Exception as e:
            logger.error(f"[V2 PLAN] Planning loop error for {project_name}: {e}")
            logger.error(traceback.format_exc())
            await raw_queue.put({"event_type": "error", "data": {"message": str(e)}})

    task = asyncio.create_task(_run_planning())
    asyncio.create_task(_broadcast())

    services._planning_sessions[project_name] = {
        "task": task,
        "subscribers": subscribers,
        "pause_event": pause_event,
        "inject_queue": inject_queue,
    }

    logger.info(f"[V2 PLAN] Started planning for project={project_name}")
    return {"status": "started", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/plan/pause")
async def v2_plan_pause(project_name: str):
    """Pause the running planning loop for a project."""
    state = services._planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    state["pause_event"].set()
    return {"status": "paused", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/plan/resume")
async def v2_plan_resume(project_name: str):
    """Resume a paused planning loop."""
    state = services._planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    state["pause_event"].clear()
    return {"status": "resumed", "project_name": project_name}


@router.post(f"{_config.V2_API_PREFIX}/plan/inject")
async def v2_plan_inject(project_name: str, prompt: str):
    """Inject a user prompt into the running planning loop."""
    state = services._planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    await state["inject_queue"].put(prompt)
    return {"status": "injected", "project_name": project_name, "prompt": prompt}


@router.get(f"{_config.V2_API_PREFIX}/plan/status")
async def v2_plan_status(project_name: str):
    """Get current planning status for a project."""
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    session = workspace.load_session()
    storyboard = workspace.load_storyboard()
    clips = workspace.load_clips()

    state = services._planning_sessions.get(project_name)
    running = bool(state and not state["task"].done())

    return {
        "project_name": project_name,
        "status": session.status,
        "running": running,
        "iteration": session.planning.iteration_count,
        "shots": len(storyboard.shots) if storyboard else 0,
        "clips": len(clips),
    }


@router.post(f"{_config.V2_API_PREFIX}/generate_fcpxml", response_model=V2GenerateFcpxmlResponse)
async def v2_generate_fcpxml(request: V2GenerateFcpxmlRequest):
    """
    V2: Generate FCPXML for a project that has completed planning.

    Runs the scaffold -> LLM enhance -> validate -> correct loop.
    Uses narration and music from workspace if present.
    Returns the path to the generated .fcpxml file.
    """
    if not services.gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    project_name = request.project_name
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        storyboard = workspace.load_storyboard()
        if not storyboard:
            raise HTTPException(status_code=400, detail="No storyboard — run /v2/plan first.")

        # Optional narration / music (set by on-demand tools later; skip if absent)
        narration_path_obj = workspace.get_narration_path()
        narration_path = str(narration_path_obj) if narration_path_obj.exists() else None

        music_path_obj = workspace.get_music_path()
        music_path = str(music_path_obj) if music_path_obj.exists() else None

        narration_duration = 0.0
        music_duration = 0.0

        if narration_path:
            from lib.utils.media import get_video_duration
            try:
                narration_duration = get_video_duration(narration_path)
            except Exception:
                narration_duration = 0.0

        if music_path:
            from lib.utils.media import get_video_duration
            try:
                music_duration = get_video_duration(music_path)
            except Exception:
                music_duration = 0.0

        fcpxml_path = await generate_fcpxml(
            services.gemini_manager,
            workspace,
            narration_path=narration_path,
            narration_duration=narration_duration,
            music_path=music_path,
            music_duration=music_duration,
        )

        return V2GenerateFcpxmlResponse(project_name=project_name, fcpxml_path=fcpxml_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V2 FCPXML] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post(f"{_config.V2_API_PREFIX}/narration")
async def v2_narration(request: V2NarrationRequest):
    """
    V2 On-demand: Generate narration script + audio for a project.

    Requires a completed storyboard (run /v2/plan first).
    """
    if not services.gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    workspace = _workspace(request.project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    try:
        from src.pipelines.v2.narration.narration_pipeline import NarrationPipeline
        session = workspace.load_session()
        user_prompt = " ".join(session.planning.user_prompts) or "Create an engaging edit"
        pipeline = NarrationPipeline(services.gemini_manager, workspace)
        audio_path = await pipeline.run(
            user_prompt=user_prompt,
            override_script=request.override_script,
        )
        return {"status": "ok", "project_name": request.project_name, "audio_path": audio_path}
    except Exception as e:
        logger.error(f"[V2 NARRATION] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post(f"{_config.V2_API_PREFIX}/music")
async def v2_music(request: V2MusicRequest):
    """
    V2 On-demand: Select and download background music for a project.
    """
    if not services.gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    workspace = _workspace(request.project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    try:
        from src.pipelines.v2.music.music_pipeline import MusicPipeline
        session = workspace.load_session()
        user_prompt = " ".join(session.planning.user_prompts) or ""
        pipeline = MusicPipeline(services.gemini_manager, workspace)
        music_path = await pipeline.run(
            user_prompt=user_prompt,
            mood=request.mood,
            prompt=request.prompt,
        )
        if music_path:
            return {"status": "ok", "project_name": request.project_name, "music_path": music_path}
        else:
            return {"status": "skipped", "project_name": request.project_name, "music_path": None}
    except Exception as e:
        logger.error(f"[V2 MUSIC] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post(f"{_config.V2_API_PREFIX}/crop")
async def v2_crop(request: V2CropRequest):
    """
    V2 On-demand: Inject dynamic crop transforms into existing FCPXML.

    Runs ViNet saliency detection and injects <adjust-transform> elements.
    Returns path to new FCPXML version with transforms applied.
    """
    workspace = _workspace(request.project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    try:
        from src.pipelines.v2.cropping.crop_pipeline import CropPipeline
        pipeline = CropPipeline(workspace)
        new_fcpxml_path = await pipeline.run(aspect_ratio=request.aspect_ratio)
        return {
            "status": "ok",
            "project_name": request.project_name,
            "fcpxml_path": new_fcpxml_path,
        }
    except Exception as e:
        logger.error(f"[V2 CROP] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post(f"{_config.V2_API_PREFIX}/render", response_model=V2RenderResponse)
async def v2_render(request: V2RenderRequest):
    """
    V2: Render FCPXML via DaVinci Resolve Studio.

    Requires:
    - DaVinci Resolve Studio running as -nogui daemon
    - A valid FCPXML in the workspace (run /v2/generate_fcpxml first)
    - Media files accessible from workspace/media/

    quality: "preview" (H.264 MP4) or "final" (ProRes 422 QuickTime)
    """
    workspace = _workspace(request.project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    quality = request.quality if request.quality in ("preview", "final") else "preview"

    try:
        from lib.utils.resolve_render import ResolveRenderer

        # Find latest FCPXML
        fcpxml_dir = workspace.root / "fcpxml"
        best_ver = 0
        fcpxml_path = None
        for f in fcpxml_dir.glob("edit_v*.fcpxml"):
            try:
                ver = int(f.stem.replace("edit_v", ""))
                if ver > best_ver:
                    best_ver = ver
                    fcpxml_path = str(f)
            except ValueError:
                pass
        if not fcpxml_path:
            final = fcpxml_dir / "edit_final.fcpxml"
            if final.exists():
                fcpxml_path = str(final)
        if not fcpxml_path:
            raise HTTPException(status_code=400, detail="No FCPXML found — run /v2/generate_fcpxml first.")

        media_dir = str(workspace.get_footage_dir())
        output_path = str(workspace.get_render_path("resolve"))

        renderer = ResolveRenderer()
        rendered = await renderer.render(
            fcpxml_path=fcpxml_path,
            media_dir=media_dir,
            output_path=output_path,
            quality=quality,
        )

        workspace.update_status("rendered")
        return V2RenderResponse(project_name=request.project_name, output_path=rendered)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V2 RENDER] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/renders/{{filename}}")
async def v2_serve_render(project_name: str, filename: str):
    """Serve a rendered video file from the workspace."""
    from fastapi.responses import FileResponse
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    render_path = _safe_child(workspace.root / "renders", filename)
    if not render_path.exists():
        raise HTTPException(status_code=404, detail=f"Render '{filename}' not found.")
    return FileResponse(str(render_path), media_type="video/mp4")


@router.post(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/renders/{{filename}}/reveal")
async def v2_reveal_render(project_name: str, filename: str):
    """Open the render file's parent folder in the OS file manager.

    Only supported on macOS (``open -R``) and Linux (``xdg-open``). Scoped to
    files inside the project's renders/ directory so a malicious filename
    can't point elsewhere.
    """
    import platform as _platform
    import subprocess as _subprocess

    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    render_path = _safe_child(workspace.root / "renders", filename)
    if not render_path.exists():
        raise HTTPException(status_code=404, detail=f"Render '{filename}' not found.")

    system = _platform.system()
    try:
        if system == "Darwin":
            _subprocess.run(["open", "-R", str(render_path)], check=False, timeout=5)
        elif system == "Linux":
            _subprocess.run(["xdg-open", str(render_path.parent)], check=False, timeout=5)
        else:
            raise HTTPException(status_code=501, detail=f"Reveal not supported on {system}")
        return {"status": "ok", "path": str(render_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/footage/{{filename}}")
async def v2_serve_footage(project_name: str, filename: str):
    """Serve a source footage file from the workspace (supports range requests for <video> seeking)."""
    from fastapi.responses import FileResponse
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    video_path = _safe_child(workspace.root / "footage", filename)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Footage '{filename}' not found.")
    # Determine media type from extension
    ext = video_path.suffix.lower()
    media_types = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".m4v": "video/mp4"}
    media_type = media_types.get(ext, "video/mp4")
    return FileResponse(str(video_path), media_type=media_type)


@router.get(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/footage/{{filename}}/thumbnail")
async def v2_serve_footage_thumbnail(project_name: str, filename: str):
    """Generate and serve a JPEG thumbnail for a footage file."""
    from fastapi.responses import FileResponse
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    video_path = _safe_child(workspace.root / "footage", filename)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Footage '{filename}' not found.")
    thumb_dir = workspace.root / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = _safe_child(thumb_dir, f"{filename}.jpg")
    if not thumb_path.exists():
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", "thumbnail,scale=160:-1",
            "-frames:v", "1",
            str(thumb_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg thumbnail failed: {stderr.decode()}")
    return FileResponse(str(thumb_path), media_type="image/jpeg")


@router.get(f"{_config.V2_API_PREFIX}/projects/{{project_name}}/preview/frame")
async def get_preview_frame(project_name: str, t: float = 0.0):
    """Extract a single frame at a timeline position for scrub preview."""
    workspace = _workspace(project_name)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    ed_path = workspace.root / "edit_decision.json"
    if not ed_path.exists():
        # Also check fcpxml subdir (common location)
        ed_path = workspace.root / "fcpxml" / "edit_decision.json"
    if not ed_path.exists():
        raise HTTPException(status_code=404, detail="No edit decision found.")

    try:
        import json as _json
        from src.pipelines.v2.schemas import EditDecision
        from src.pipelines.v2.preview.ffmpeg_renderer import extract_frame

        with open(ed_path) as f:
            ed_data = _json.load(f)
        edit = EditDecision.model_validate(ed_data)

        footage_dir = workspace.get_footage_dir()
        jpeg_bytes = await extract_frame(edit, footage_dir, timeline_time=t)
        if not jpeg_bytes:
            raise HTTPException(status_code=404, detail="No frame extracted.")

        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V2 PREVIEW FRAME] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(f"{_config.V2_API_PREFIX}/resolve/status", response_model=V2ResolveStatusResponse)
async def v2_resolve_status():
    """
    V2: Check whether DaVinci Resolve Studio is running and accessible.
    """
    from lib.utils.resolve_setup import check_resolve_status
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, check_resolve_status)
    return V2ResolveStatusResponse(
        running=status.get("running", False),
        version=status.get("version"),
        studio=status.get("studio", False),
        error=status.get("error"),
    )
