# app.py

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket, WebSocketDisconnect

from lib.oss.storage_factory import get_storage_client
from lib.llm.MemoriesAiManager import MemoriesAiManager, create_memories_manager
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.schema import (
    MovieFile,
    IndexRequest,
    IndexResponse,
    FlexibleResponseRequest,
    FlexibleResponseResult,
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

from src.config import (
    API_PREFIX,
    BUCKET_NAME,
    VIDEOS_DIR,
    get_storage_mode,
    ensure_local_directories,
    V2_API_PREFIX,
    WORKSPACES_DIR,
)
from src.pipelines.v2.workspace import WorkspaceManager
from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension
from src.pipelines.v2.planning.iterative_planning_loop import IterativePlanningLoop
from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml
from src.pipelines.v2.agent.agent_session import AgentSession

from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline


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
try:
    gemini_manager = GeminiGenaiManager()
    logger.info("Gemini client initialized")
except Exception as e:
    gemini_manager = None
    logger.warning(f"Failed to initialize Gemini client: {e}")

# --- Active planning sessions (project_name → asyncio state) ---
# Each entry: {event_queue, pause_event, inject_queue, task}
_planning_sessions: Dict[str, Dict] = {}

# --- Active agent sessions (project_name → AgentSession) ---
_agent_sessions: Dict[str, AgentSession] = {}

# Caption API callback URL - loaded from config.json api_keys section
# Set to your public ngrok/server URL, e.g., "https://xxx.ngrok-free.app/webhooks/memories/caption"
_callback_url = os.environ.get("MEMORIES_CAPTION_CALLBACK_URL", "")
CAPTION_CALLBACK_URL = _callback_url if _callback_url else None  # Treat empty string as None
if CAPTION_CALLBACK_URL:
    logger.info(f"Video Caption API callback URL: {CAPTION_CALLBACK_URL}")
else:
    logger.info("MEMORIES_CAPTION_CALLBACK_URL not set - will use Chat API instead of Caption API")


@app.get("/")
async def root():
    return {"message": "FastAPI inference service is running."}


# ─── V2: Project management ───────────────────────────────────────────────────

@app.get(f"{V2_API_PREFIX}/projects")
async def v2_list_projects():
    """
    List all project workspaces, including ones with footage but not yet indexed.
    The dashboard uses this to show the project browser.
    """
    projects = WorkspaceManager.list_projects(WORKSPACES_DIR)
    return {"projects": projects}


@app.post(f"{V2_API_PREFIX}/projects/create")
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
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if workspace.dir_exists():
        return {"status": "exists", "project_name": project_name, "path": str(workspace.root)}
    workspace.create()
    return {
        "status": "created",
        "project_name": project_name,
        "path": str(workspace.root),
        "next_step": f"Drop footage into {workspace.get_footage_dir()}, then call /v2/index",
    }


@app.post(f"{V2_API_PREFIX}/projects/{{project_name}}/clear/gists")
async def v2_clear_gists(project_name: str):
    """Clear per-video gists and session gist. Re-index to regenerate."""
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
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


@app.post(f"{V2_API_PREFIX}/projects/{{project_name}}/clear/planning")
async def v2_clear_planning(project_name: str):
    """Clear planning context: storyboard, clips, iterations, context.md, agent chat, scratchpads."""
    import shutil
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    for fname in ["storyboard.json", "clips.json", "context.md", "chat_history.json"]:
        p = workspace.root / fname
        if p.exists():
            p.unlink()

    for subdir in ["iterations", "scratchpads"]:
        d = workspace.root / subdir
        if d.exists():
            shutil.rmtree(d)
            d.mkdir()

    # Clear in-memory agent session
    _agent_sessions.pop(project_name, None)

    session = workspace.load_session()
    from src.pipelines.v2.schemas import PlanningState
    session.planning = PlanningState()
    if session.status == "planning":
        session.status = "indexed"
    workspace.save_session(session)
    return {"status": "cleared", "project_name": project_name}


@app.post(f"{V2_API_PREFIX}/projects/{{project_name}}/clear/session")
async def v2_clear_session(project_name: str):
    """Full local reset — deletes session.json, keeps footage."""
    import shutil
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
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


@app.post(f"{V2_API_PREFIX}/projects/{{project_name}}/clear/memories")
async def v2_clear_memories(project_name: str):
    """Delete uploaded videos from Memories.ai cloud. Irreversible."""
    if not memories_manager:
        raise HTTPException(status_code=500, detail="Memories.ai not configured.")
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    session = workspace.load_session()
    deleted = []
    errors = []
    for v in session.videos:
        if v.video_no:
            try:
                await memories_manager.delete_video(v.video_no)
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


@app.post(f"{V2_API_PREFIX}/index", response_model=V2IndexResponse)
async def v2_index(request: V2IndexRequest):
    """
    V2: Lightweight video comprehension.
    Uploads to Memories.ai (or reuses cached video_no), gets a broad gist.
    Much faster than v1 /index — no scene-by-scene analysis.

    source_dir is optional. If omitted, footage is read from the workspace's
    footage/ subdirectory (data/workspaces/{project_name}/footage/).
    """
    try:
        workspace = WorkspaceManager(request.project_name, WORKSPACES_DIR)

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

        if not memories_manager:
            raise HTTPException(status_code=500, detail="Memories.ai not configured. Set MEMORIES_API_KEY.")

        pipeline = LightweightComprehension(
            project_name=request.project_name,
            source_dir=source_dir,
            memories=memories_manager,
            workspace=workspace,
        )
        session = await pipeline.run(start_fresh=request.start_fresh)

        return V2IndexResponse(
            project_name=request.project_name,
            video_nos=[v.video_no for v in session.videos],
            gist=session.gist,
            status=session.status,
        )
    except Exception as e:
        logger.error(f"[V2 INDEX] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{V2_API_PREFIX}/plan")
async def v2_plan(request: V2PlanRequest):
    """
    V2: Start the iterative planning loop for a project.

    Launches planning as a background task. Connect to the WebSocket endpoint
    WS /video-edit/v2/session/{project_name} to receive live events.

    Returns immediately with {"status": "started"} or {"status": "already_running"}.
    """
    if not memories_manager:
        raise HTTPException(status_code=500, detail="Memories.ai not configured. Set MEMORIES_API_KEY.")
    if not gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured. Check GCP credentials.")

    project_name = request.project_name

    # Check for existing running session
    session_state = _planning_sessions.get(project_name)
    if session_state and not session_state["task"].done():
        return {"status": "already_running", "project_name": project_name}

    # Load workspace — must have been indexed first
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found. Run /v2/index first.")

    try:
        session = workspace.load_session()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")

    if not session.videos:
        raise HTTPException(status_code=400, detail="No videos indexed for this project.")

    # Set up async coordination primitives
    raw_queue: asyncio.Queue = asyncio.Queue()   # planning loop → broadcaster
    subscribers: List[asyncio.Queue] = []        # one queue per connected WebSocket client
    pause_event: asyncio.Event = asyncio.Event()
    inject_queue: asyncio.Queue = asyncio.Queue()

    video_nos = [v.video_no for v in session.videos]

    loop_obj = IterativePlanningLoop(
        project_name=project_name,
        user_prompt=request.prompt,
        workspace=workspace,
        memories=memories_manager,
        gemini=gemini_manager,
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

    _planning_sessions[project_name] = {
        "task": task,
        "subscribers": subscribers,
        "pause_event": pause_event,
        "inject_queue": inject_queue,
    }

    logger.info(f"[V2 PLAN] Started planning for project={project_name}")
    return {"status": "started", "project_name": project_name}


@app.post(f"{V2_API_PREFIX}/plan/pause")
async def v2_plan_pause(project_name: str):
    """Pause the running planning loop for a project."""
    state = _planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    state["pause_event"].set()
    return {"status": "paused", "project_name": project_name}


@app.post(f"{V2_API_PREFIX}/plan/resume")
async def v2_plan_resume(project_name: str):
    """Resume a paused planning loop."""
    state = _planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    state["pause_event"].clear()
    return {"status": "resumed", "project_name": project_name}


@app.post(f"{V2_API_PREFIX}/plan/inject")
async def v2_plan_inject(project_name: str, prompt: str):
    """Inject a user prompt into the running planning loop."""
    state = _planning_sessions.get(project_name)
    if not state or state["task"].done():
        raise HTTPException(status_code=404, detail="No active planning session for this project.")
    await state["inject_queue"].put(prompt)
    return {"status": "injected", "project_name": project_name, "prompt": prompt}


@app.get(f"{V2_API_PREFIX}/plan/status")
async def v2_plan_status(project_name: str):
    """Get current planning status for a project."""
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    session = workspace.load_session()
    storyboard = workspace.load_storyboard()
    clips = workspace.load_clips()

    state = _planning_sessions.get(project_name)
    running = bool(state and not state["task"].done())

    return {
        "project_name": project_name,
        "status": session.status,
        "running": running,
        "iteration": session.planning.iteration_count,
        "shots": len(storyboard.shots) if storyboard else 0,
        "clips": len(clips),
    }


@app.post(f"{V2_API_PREFIX}/generate_fcpxml", response_model=V2GenerateFcpxmlResponse)
async def v2_generate_fcpxml(request: V2GenerateFcpxmlRequest):
    """
    V2: Generate FCPXML for a project that has completed planning.

    Runs the scaffold → LLM enhance → validate → correct loop.
    Uses narration and music from workspace if present.
    Returns the path to the generated .fcpxml file.
    """
    if not gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    project_name = request.project_name
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
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
            gemini_manager,
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


@app.websocket(f"{V2_API_PREFIX}/session/{{project_name}}")
async def v2_session_ws(websocket: WebSocket, project_name: str):
    """
    WebSocket endpoint for live planning dashboard updates.

    Connect here to receive a stream of PlanningEvent JSON objects while the
    planning loop is running. The connection stays open until the loop finishes
    or the client disconnects.

    Also accepts incoming messages:
        {"action": "pause"}
        {"action": "resume"}
        {"action": "inject", "prompt": "..."}
    """
    await websocket.accept()
    logger.info(f"[WS] Client connected for project={project_name}")

    state = _planning_sessions.get(project_name)
    if not state:
        await websocket.send_json({"event_type": "error", "data": {"message": "No active planning session."}})
        await websocket.close()
        return

    # Each client gets its own queue; the broadcaster task fans events out to all of them
    client_queue: asyncio.Queue = asyncio.Queue()
    state["subscribers"].append(client_queue)
    logger.info(f"[WS] Client subscribed to project={project_name} ({len(state['subscribers'])} total)")

    try:
        while True:
            # Forward queued events to this client
            while not client_queue.empty():
                try:
                    payload = client_queue.get_nowait()
                    await websocket.send_json(payload)
                    if payload.get("event_type") in ("done", "error", "session_ended"):
                        await websocket.close()
                        return
                except asyncio.QueueEmpty:
                    break

            # Handle incoming control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                action = msg.get("action")
                if action == "pause":
                    state["pause_event"].set()
                elif action == "resume":
                    state["pause_event"].clear()
                elif action == "inject":
                    prompt = msg.get("prompt", "")
                    if prompt:
                        await state["inject_queue"].put(prompt)
            except asyncio.TimeoutError:
                pass

            if state["task"].done() and client_queue.empty():
                await websocket.send_json({"event_type": "session_ended", "data": {}})
                break

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected from project={project_name}")
    except Exception as e:
        logger.error(f"[WS] Error in WebSocket handler: {e}")
    finally:
        state["subscribers"].remove(client_queue)


# ─── V2: Agent chat ──────────────────────────────────────────────────────────

@app.websocket(f"{V2_API_PREFIX}/agent/{{project_name}}/chat")
async def v2_agent_chat_ws(websocket: WebSocket, project_name: str):
    """
    WebSocket endpoint for the agentic editing chat.

    On connect: sends initial state (scratchpads + chat history).
    Receives: {"type": "user_message", "text": "..."}
    Sends: {"type": "<event_type>", "data": {...}}
      event types: agent_message, tool_call, tool_result, scratchpad_update, error
    """
    await websocket.accept()
    logger.info(f"[AGENT WS] Client connected for project={project_name}")

    # Validate dependencies
    if not memories_manager:
        await websocket.send_json({"type": "error", "data": {"message": "Memories.ai not configured."}})
        await websocket.close()
        return
    if not gemini_manager:
        await websocket.send_json({"type": "error", "data": {"message": "Gemini not configured."}})
        await websocket.close()
        return

    # Load workspace
    workspace = WorkspaceManager(project_name, WORKSPACES_DIR)
    if not workspace.exists():
        await websocket.send_json({"type": "error", "data": {"message": f"Project '{project_name}' not found."}})
        await websocket.close()
        return

    session_data = workspace.load_session()
    if not session_data.videos:
        await websocket.send_json({"type": "error", "data": {"message": "No videos indexed for this project."}})
        await websocket.close()
        return

    # Get or create AgentSession
    agent = _agent_sessions.get(project_name)
    if agent is None:
        # Event emitter that sends over this WebSocket
        # We'll replace this with a subscriber model below
        pass

    # Subscriber queue for this client
    client_queue: asyncio.Queue = asyncio.Queue()

    # Build emit function that pushes to the client queue
    async def emit(event_type: str, data: dict):
        await client_queue.put({"type": event_type, "data": data})

    # Create or reuse agent session
    if agent is None:
        try:
            agent = AgentSession(
                project_name=project_name,
                workspace=workspace,
                memories_manager=memories_manager,
                gemini_manager=gemini_manager,
                video_entries=session_data.videos,
                emit=emit,
            )
            _agent_sessions[project_name] = agent
            print(f"[AGENT WS] Created new AgentSession for project={project_name}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[AGENT WS] Failed to create AgentSession: {e}", flush=True)
            await websocket.send_json({"type": "error", "data": {"message": f"Failed to initialize: {e}"}})
            await websocket.close()
            return

    # Always point the agent's emit at the current connection's queue
    agent._emit = emit
    print(f"[AGENT WS] Bound emit to current connection for project={project_name}", flush=True)

    # Send initial state on connect
    try:
        await websocket.send_json({
            "type": "init",
            "data": {
                "scratchpads": agent.get_scratchpad_state(),
                "chat_history": agent.get_chat_history(),
                "project_name": project_name,
                "video_count": len(session_data.videos),
            },
        })
        print(f"[AGENT WS] Sent init to client for project={project_name}", flush=True)
    except Exception as e:
        print(f"[AGENT WS] Failed to send init: {e}", flush=True)
        return

    # Track whether an agent loop is running
    agent_task: Optional[asyncio.Task] = None

    async def _drain_queue():
        """Send all queued events to the WebSocket client."""
        while not client_queue.empty():
            try:
                payload = client_queue.get_nowait()
                await websocket.send_json(payload)
            except asyncio.QueueEmpty:
                break

    print(f"[AGENT WS] Entering message loop for project={project_name}", flush=True)
    try:
        while True:
            # Drain any pending events from the agent
            await _drain_queue()

            # Check if agent task finished (send completion marker)
            if agent_task and agent_task.done():
                await _drain_queue()  # flush remaining events
                exc = agent_task.exception() if not agent_task.cancelled() else None
                if exc:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": str(exc)},
                    })
                await websocket.send_json({"type": "done", "data": {}})
                agent_task = None

            # Listen for incoming messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                msg_type = msg.get("type")
                print(f"[AGENT WS] Received message type={msg_type}", flush=True)

                if msg_type == "user_message":
                    text = msg.get("text", "").strip()
                    if text and agent_task is None:
                        # Reassign emit in case this is a reconnection
                        agent._emit = emit
                        agent_task = asyncio.create_task(
                            agent.handle_user_message(text)
                        )
                    elif text and agent_task is not None:
                        # Agent is busy — queue the message for later
                        await websocket.send_json({
                            "type": "queued",
                            "data": {"text": text, "message": "Agent is working. Your message will be processed next."},
                        })
                        # We'll handle queued messages when the current task finishes
                        # For now, just notify the user

            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"[AGENT WS] Client disconnected from project={project_name}")
    except Exception as e:
        logger.error(f"[AGENT WS] Error: {e}", exc_info=True)
    finally:
        # Cancel running agent task if client disconnects
        if agent_task and not agent_task.done():
            agent_task.cancel()


@app.post(f"{V2_API_PREFIX}/narration")
async def v2_narration(request: V2NarrationRequest):
    """
    V2 On-demand: Generate narration script + audio for a project.

    Requires a completed storyboard (run /v2/plan first).
    """
    if not gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    workspace = WorkspaceManager(request.project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    try:
        from src.pipelines.v2.narration.narration_pipeline import NarrationPipeline
        session = workspace.load_session()
        user_prompt = " ".join(session.planning.user_prompts) or "Create an engaging edit"
        pipeline = NarrationPipeline(gemini_manager, workspace)
        audio_path = await pipeline.run(
            user_prompt=user_prompt,
            override_script=request.override_script,
        )
        return {"status": "ok", "project_name": request.project_name, "audio_path": audio_path}
    except Exception as e:
        logger.error(f"[V2 NARRATION] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{V2_API_PREFIX}/music")
async def v2_music(request: V2MusicRequest):
    """
    V2 On-demand: Select and download background music for a project.
    """
    if not gemini_manager:
        raise HTTPException(status_code=500, detail="Gemini not configured.")

    workspace = WorkspaceManager(request.project_name, WORKSPACES_DIR)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")

    try:
        from src.pipelines.v2.music.music_pipeline import MusicPipeline
        session = workspace.load_session()
        user_prompt = " ".join(session.planning.user_prompts) or ""
        pipeline = MusicPipeline(gemini_manager, workspace)
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


@app.post(f"{V2_API_PREFIX}/crop")
async def v2_crop(request: V2CropRequest):
    """
    V2 On-demand: Inject dynamic crop transforms into existing FCPXML.

    Runs ViNet saliency detection and injects <adjust-transform> elements.
    Returns path to new FCPXML version with transforms applied.
    """
    workspace = WorkspaceManager(request.project_name, WORKSPACES_DIR)
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


@app.post(f"{V2_API_PREFIX}/render", response_model=V2RenderResponse)
async def v2_render(request: V2RenderRequest):
    """
    V2: Render FCPXML via DaVinci Resolve Studio.

    Requires:
    - DaVinci Resolve Studio running as -nogui daemon
    - A valid FCPXML in the workspace (run /v2/generate_fcpxml first)
    - Media files accessible from workspace/media/

    quality: "preview" (H.264 MP4) or "final" (ProRes 422 QuickTime)
    """
    workspace = WorkspaceManager(request.project_name, WORKSPACES_DIR)
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
        output_path = str(workspace.get_render_path(quality))

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


@app.get(f"{V2_API_PREFIX}/resolve/status", response_model=V2ResolveStatusResponse)
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


@app.get(f"{API_PREFIX}/movies", response_model=List[MovieFile])
async def list_available_movies() -> List[MovieFile]:
    """
    List available movies in storage (local or cloud).
    """
    try:
        logger.info("Fetching list of available movies...")
        blobs = storage_client.list_folder(BUCKET_NAME, f"{VIDEOS_DIR}/")
        movies = [MovieFile(name=blob[0], blob_path=blob[1]) for blob in blobs]
        logger.info(f"Found {len(movies)} movies.")
        return movies
    except Exception as e:
        logger.error(f"Error fetching movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")

@app.post(f"{API_PREFIX}/index")
async def index_longform(request: IndexRequest):
    try:
        logger.info(f"Received index request for blob: {request.blob_path} | Start fresh: {request.start_fresh}")

        # Check required dependencies
        if not memories_manager:
            raise HTTPException(status_code=500, detail="Memories.ai not configured. Set MEMORIES_API_KEY in config.json")
        if not CAPTION_CALLBACK_URL:
            raise HTTPException(status_code=500, detail="Caption callback URL not set. Run with ./run.sh to set up ngrok")

        # Auto-generate debug output dir based on video name
        from pathlib import Path
        video_stem = Path(request.blob_path).stem
        debug_output_dir = f"debug_output/{video_stem}"

        pipeline = ComprehensionPipeline(
            request.blob_path,
            start_fresh=request.start_fresh,
            debug_output_dir=debug_output_dir,
            memories_manager=memories_manager,
            caption_callback_url=CAPTION_CALLBACK_URL,
            register_caption_callback=register_memories_callback,
        )
        await pipeline.run()

        return IndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    
@app.post(f"{API_PREFIX}/generate_edit", response_model=FlexibleResponseResult)
async def generate_edit(request: FlexibleResponseRequest):
    """
    V1: Generate an edited video response from a long-form source video.

    Uses the FlexibleResponsePipeline to extract relevant clips, add narration,
    music, and dynamic cropping based on the user prompt.
    """
    try:
        logger.info(f"Generate edit for: {request.blob_path} | prompt: {request.prompt}")
        pipeline = FlexibleResponsePipeline(request.blob_path)
        response = await pipeline.run(
            request.prompt, request.video_response, request.original_audio,
            request.music, request.narration, request.aspect_ratio,
            request.subtitles, request.snap_to_beat, request.output_path
        )
        return response
    except Exception as e:
        logger.error(f"Generate edit error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Edit generation failed: {str(e)}")


# --- Memories.ai Webhook Endpoints ---

@app.post("/webhooks/memories/caption")
async def memories_caption_callback(request: Request):
    """
    Webhook endpoint for Memories.ai Video Caption API callbacks.

    When using the async Video Caption API, Memories.ai will POST results
    to this endpoint when processing completes.
    """
    import json
    from pathlib import Path

    try:
        data = await request.json()
        task_id = data.get("task_id")

        logger.info(f"[MEMORIES WEBHOOK] Received caption callback for task: {task_id}")

        # Always save callback data to file for debugging/analysis
        output_dir = Path("debug_output/caption_callbacks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[MEMORIES WEBHOOK] Saved callback to {output_file}")

        # Log response preview
        response_text = (data.get("data") or {}).get("response", "") or (data.get("data") or {}).get("text", "")
        if response_text:
            preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            logger.info(f"[MEMORIES WEBHOOK] Response preview: {preview}")

        if task_id and task_id in _memories_pending_callbacks:
            future = _memories_pending_callbacks.pop(task_id)
            if not future.done():
                future.set_result(data)
            logger.info(f"[MEMORIES WEBHOOK] Task {task_id} completed and callback resolved")
        else:
            logger.warning(f"[MEMORIES WEBHOOK] Unknown task_id or no pending callback: {task_id}")

        return {"status": "ok", "task_id": task_id}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


@app.post("/webhooks/memories/upload")
async def memories_upload_callback(request: Request):
    """
    Webhook endpoint for Memories.ai Upload API callbacks.

    Called when video upload and indexing completes.
    """
    try:
        data = await request.json()
        video_no = data.get("videoNo") or data.get("video_no")
        status = data.get("status")

        logger.info(f"[MEMORIES WEBHOOK] Upload callback - videoNo: {video_no}, status: {status}")

        # Store or process the upload completion
        if video_no and video_no in _memories_pending_callbacks:
            future = _memories_pending_callbacks.pop(video_no)
            if not future.done():
                future.set_result(data)

        return {"status": "ok", "video_no": video_no}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing upload callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
