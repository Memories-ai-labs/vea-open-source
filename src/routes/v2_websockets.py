# src/routes/v2_websockets.py
"""V2 WebSocket routes: planning session and agent chat."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect

from src import config as _config
from src.pipelines.v2.workspace import WorkspaceManager, validate_project_name, _atomic_write_json
from src.routes._route_utils import safe_workspace as _workspace
from src.pipelines.v2.agent.agent_session import AgentSession
from src import services

logger = logging.getLogger(__name__)


def register_websocket_routes(app: FastAPI):
    """Register WebSocket routes on the FastAPI app instance.

    WebSocket routes cannot use APIRouter, so we register them directly on the app.
    """

    @app.websocket(f"{_config.V2_API_PREFIX}/session/{{project_name}}")
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

        state = services._planning_sessions.get(project_name)
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

    @app.websocket(f"{_config.V2_API_PREFIX}/agent/{{project_name}}/chat")
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
        if not services.memories_manager:
            await websocket.send_json({"type": "error", "data": {"message": "Memories.ai not configured."}})
            await websocket.close()
            return
        if not services.gemini_manager:
            await websocket.send_json({"type": "error", "data": {"message": "Gemini not configured."}})
            await websocket.close()
            return

        # Load workspace. Use dir_exists() rather than exists() — the latter
        # only returns True after a session.json has been written, which
        # doesn't happen until indexing completes. A freshly-created workspace
        # with footage but no session still needs a live WebSocket so the
        # Index button can kick off indexing.
        workspace = _workspace(project_name)
        if not workspace.dir_exists():
            await websocket.send_json({"type": "error", "data": {"message": f"Project '{project_name}' not found."}})
            await websocket.close()
            return

        # Try to load session — may have zero videos if not indexed yet
        try:
            session_data = workspace.load_session()
        except Exception:
            # Session file may not exist yet
            from src.pipelines.v2.schemas import SessionData
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            session_data = SessionData(
                project_name=project_name,
                created_at=now,
                updated_at=now,
                status="new",
                videos=[],
                gist="",
            )

        needs_indexing = not session_data.videos

        # Subscriber queue for this client
        client_queue: asyncio.Queue = asyncio.Queue()

        async def emit(event_type: str, data: dict):
            await client_queue.put({"type": event_type, "data": data})

        # Get or create AgentSession (skip if no videos — agent loop needs them)
        agent = services._agent_sessions.get(project_name)
        if agent is None and not needs_indexing:
            try:
                agent = AgentSession(
                    project_name=project_name,
                    workspace=workspace,
                    memories_manager=services.memories_manager,
                    gemini_manager=services.main_llm,
                    video_llm=services.video_llm,
                    video_entries=session_data.videos,
                    emit=emit,
                )
                services._agent_sessions[project_name] = agent
                print(f"[AGENT WS] Created new AgentSession for project={project_name}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[AGENT WS] Failed to create AgentSession: {e}", flush=True)
                await websocket.send_json({"type": "error", "data": {"message": f"Failed to initialize: {e}"}})
                await websocket.close()
                return

        # Subscribe this connection to the agent's event fan-out. Multiple
        # concurrent clients for the same project all receive the same events.
        if agent is not None:
            agent.add_subscriber(emit)
            print(f"[AGENT WS] Subscribed client to project={project_name}", flush=True)

        # Register this client's emit on the project-level indexing channel so the
        # client receives index_progress events even when no agent exists yet.
        services._indexing_emitters.setdefault(project_name, []).append(emit)

        # Load edit decision if it exists
        edit_decision_data = None
        edit_decision_path = workspace.root / "fcpxml" / "edit_decision.json"
        if edit_decision_path.exists():
            try:
                import json as _json
                with open(edit_decision_path) as f:
                    edit_decision_data = _json.load(f)
            except Exception:
                pass

        # Render state — the session tracks ffmpeg + resolve independently.
        # On reconnect, fall back to scanning the filesystem so the UI
        # still shows an existing render even if no agent is loaded.
        def _file_state(name: str) -> Optional[Dict]:
            p = workspace.get_render_path(name)
            return {"status": "complete", "filename": p.name} if p.exists() else None

        if agent is not None:
            ffmpeg_render_state = (
                agent._ffmpeg_render_state
                if agent._ffmpeg_render_state.get("status") != "idle"
                else _file_state("ffmpeg") or {"status": "idle", "quality": "draft"}
            )
            resolve_render_state = (
                agent._resolve_render_state
                if agent._resolve_render_state.get("status") != "idle"
                else _file_state("resolve") or {"status": "idle"}
            )
            ffmpeg_quality_pref = agent._ffmpeg_quality_pref
        else:
            ffmpeg_render_state = _file_state("ffmpeg") or {"status": "idle", "quality": "draft"}
            resolve_render_state = _file_state("resolve") or {"status": "idle"}
            ffmpeg_quality_pref = "draft"

        # Send initial state on connect
        try:
            footage_files = [f.name for f in workspace.scan_footage()] if workspace.get_footage_dir().is_dir() else []
            indexed_files = [v.video_name for v in session_data.videos if v.video_no]
            video_meta = {
                v.video_name: {
                    "video_no": v.video_no,
                    "indexed_at": v.indexed_at,
                    "duration_seconds": v.duration_seconds,
                }
                for v in session_data.videos if v.video_no
            }

            # Detect if indexing is currently in progress
            indexing_state = services._indexing_progress.get(project_name)

            init_data: dict = {
                "project_name": project_name,
                "video_count": len(footage_files),
                "footage_files": footage_files,
                "indexed_files": indexed_files,
                "video_meta": video_meta,
                "needs_indexing": needs_indexing,
            }
            if agent is not None:
                init_data["scratchpads"] = agent.get_scratchpad_state()
                init_data["scratchpad_timestamps"] = agent.get_scratchpad_timestamps()
                init_data["chat_history"] = agent.get_chat_history()
                init_data["event_log"] = agent.get_event_log()
            else:
                init_data["scratchpads"] = {}
                init_data["scratchpad_timestamps"] = {}
                init_data["chat_history"] = []
                init_data["event_log"] = []
            init_data["ffmpeg_render_state"] = ffmpeg_render_state
            init_data["resolve_render_state"] = resolve_render_state
            init_data["ffmpeg_quality_pref"] = ffmpeg_quality_pref
            if edit_decision_data:
                init_data["edit_decision"] = edit_decision_data
            if indexing_state:
                init_data["indexing_state"] = indexing_state
            await websocket.send_json({
                "type": "init",
                "data": init_data,
            })
            print(f"[AGENT WS] Sent init to client for project={project_name} (needs_indexing={needs_indexing})", flush=True)
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
                        if not text:
                            pass
                        elif agent is None:
                            await websocket.send_json({
                                "type": "error",
                                "data": {"message": "Project must be indexed before chatting with the agent."},
                            })
                        elif agent_task is None:
                            agent_task = asyncio.create_task(
                                agent.handle_user_message(text)
                            )
                        else:
                            # Agent is busy -- queue the message for later
                            await websocket.send_json({
                                "type": "queued",
                                "data": {"text": text, "message": "Agent is working. Your message will be processed next."},
                            })

                    elif msg_type == "index_now":
                        # Trigger indexing for this project. Runs in the background and
                        # broadcasts index_progress events through the project's emitter list.
                        # Optional `files: [filename, ...]` re-indexes only those files
                        # (deletes the existing memories.ai upload first).
                        only_files = msg.get("files") or None
                        if services._indexing_progress.get(project_name, {}).get("status") == "running":
                            await emit("index_progress", services._indexing_progress[project_name])
                        else:
                            asyncio.create_task(_run_indexing(project_name, workspace, only_files=only_files))

                    elif msg_type == "render_resolve":
                        if agent is None:
                            await websocket.send_json({"type": "render_error", "data": {"error": "Project must be indexed first."}})
                        else:
                            fcpxml = agent.workspace.get_latest_fcpxml()
                            if fcpxml:
                                agent._spawn_bg_task(
                                    agent._render_resolve({"fcpxml_path": str(fcpxml)})
                                )
                            else:
                                await websocket.send_json({
                                    "type": "render_error",
                                    "data": {"error": "No FCPXML found. Generate one first."},
                                })

                    elif msg_type == "render_ffmpeg":
                        if agent is None:
                            await websocket.send_json({"type": "render_error", "data": {"error": "Project must be indexed first."}})
                        else:
                            quality = msg.get("quality", agent._ffmpeg_quality_pref)
                            if quality not in ("draft", "full"):
                                quality = "draft"
                            agent._ffmpeg_quality_pref = quality
                            agent._spawn_bg_task(agent._render_ffmpeg(quality))

                    elif msg_type == "set_ffmpeg_quality":
                        # Persist the user's quality preference without
                        # kicking off a render — the next auto-render will
                        # pick it up.
                        if agent is not None:
                            quality = msg.get("quality", "draft")
                            if quality in ("draft", "full"):
                                agent._ffmpeg_quality_pref = quality
                                await websocket.send_json({
                                    "type": "ffmpeg_quality_pref",
                                    "data": {"quality": quality},
                                })

                    elif msg_type == "crop_clip":
                        clip_id = msg.get("clip_id", "")
                        if clip_id and agent is not None:
                            asyncio.create_task(
                                _handle_crop_clip(workspace, agent, emit, clip_id)
                            )

                    elif msg_type == "edit_decision_update":
                        # Persist edit decision from client, recompile FCPXML, and broadcast.
                        # Use atomic write (tmp + os.replace) so a crash mid-write can't
                        # leave the project's source-of-truth JSON corrupted.
                        ed = msg.get("edit_decision")
                        if ed:
                            ed_path = workspace.root / "fcpxml" / "edit_decision.json"
                            _atomic_write_json(ed_path, ed)
                            logger.info(f"[AGENT WS] Persisted edit_decision for project={project_name}")

                            # Recompile FCPXML so next render uses updated edit
                            try:
                                from src.pipelines.v2.schemas import EditDecision as ED
                                from src.pipelines.v2.fcpxml.edit_compiler import compile_edit_decision
                                edit_obj = ED.model_validate(ed)
                                # Resolve source paths for clips
                                footage_dir = workspace.root / "footage"
                                for clip in edit_obj.clips:
                                    if not clip.source_path:
                                        resolved = footage_dir / clip.source_file
                                        if resolved.exists():
                                            clip.source_path = str(resolved)
                                fcpxml_out = str(workspace.get_fcpxml_path(version=1))
                                compile_edit_decision(edit_obj, fcpxml_out, workspace_root=workspace.root)
                                logger.info(f"[AGENT WS] Recompiled FCPXML from UI edit: {fcpxml_out}")
                            except Exception as e:
                                logger.warning(f"[AGENT WS] FCPXML recompile failed (non-fatal): {e}")

                            # Drop a short note into the agent's planning scratchpad so
                            # its own free-form notes don't drift stale when the user
                            # edits in the UI. The full JSON is still injected into the
                            # system prompt every turn; this is a breadcrumb, not the
                            # source of truth.
                            if agent is not None:
                                try:
                                    from datetime import datetime, timezone
                                    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
                                    num_clips = len(ed.get("clips", []))
                                    num_narr = len(ed.get("narration", []))
                                    note = (
                                        f"\n\n> User edited timeline in UI at {now} — "
                                        f"{num_clips} clips, {num_narr} narration segments. "
                                        f"Always defer to the JSON in the system prompt."
                                    )
                                    agent.scratchpads.update("planning", "append", note)
                                except Exception as e:
                                    logger.warning(f"[AGENT WS] scratchpad breadcrumb failed: {e}")

                            await emit("timeline_update", {"edit_decision": ed})

                            # Debounced auto-rerender of the ffmpeg preview
                            # so the MP4 stays in sync with the edit without
                            # the user having to click "Re-render". Resolve
                            # is intentionally not auto-triggered.
                            if agent is not None:
                                agent.schedule_ffmpeg_rerender()

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
            # Unsubscribe from the agent's event fan-out
            if agent is not None:
                agent.remove_subscriber(emit)
            # Unregister this client from the indexing emitters
            try:
                services._indexing_emitters.get(project_name, []).remove(emit)
            except (ValueError, KeyError):
                pass
            # If this was the last client for the project, drop the agent session
            # so the next connect gets a fresh one (avoids leaked state).
            remaining = services._indexing_emitters.get(project_name, [])
            if not remaining:
                dropped = services._agent_sessions.pop(project_name, None)
                if dropped is not None:
                    logger.info(
                        f"[AGENT WS] Released AgentSession for {project_name} "
                        f"(no clients remain)"
                    )

    async def _broadcast_index_progress(project_name: str, payload: dict):
        """Push an index_progress event to all connected clients for this project."""
        services._indexing_progress[project_name] = payload
        for em in list(services._indexing_emitters.get(project_name, [])):
            try:
                await em("index_progress", payload)
            except Exception:
                pass

    async def _run_indexing(
        project_name: str,
        workspace: WorkspaceManager,
        only_files: Optional[list] = None,
    ):
        """Run LightweightComprehension on the project's footage and broadcast progress.

        If only_files is provided, only those filenames are re-indexed and merged
        back into the existing session.
        """
        from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension

        try:
            footage_files = workspace.scan_footage() if workspace.get_footage_dir().is_dir() else []
            total = len(only_files) if only_files else len(footage_files)

            if not services.memories_manager:
                await _broadcast_index_progress(project_name, {
                    "status": "error",
                    "percent": 0,
                    "message": "Memories.ai not configured. Set MEMORIES_API_KEY in config.json.",
                })
                return

            await _broadcast_index_progress(project_name, {
                "status": "running",
                "percent": 1,
                "message": f"Starting indexing for {total} file(s)...",
                "videos_total": total,
                "videos_done": 0,
            })

            async def _on_progress(percent: float, message: str):
                await _broadcast_index_progress(project_name, {
                    "status": "running",
                    "percent": percent,
                    "message": message,
                    "videos_total": total,
                })

            pipeline = LightweightComprehension(
                project_name=project_name,
                source_dir=str(workspace.get_footage_dir()),
                memories=services.memories_manager,
                workspace=workspace,
            )

            session = await pipeline.run(
                start_fresh=False,
                progress_callback=_on_progress,
                only_files=only_files,
            )

            indexed = sum(1 for v in session.videos if v.video_no)
            video_meta = {
                v.video_name: {
                    "video_no": v.video_no,
                    "indexed_at": v.indexed_at,
                    "duration_seconds": v.duration_seconds,
                }
                for v in session.videos if v.video_no
            }
            await _broadcast_index_progress(project_name, {
                "status": "complete",
                "percent": 100,
                "message": f"Indexing complete: {indexed} video(s).",
                "videos_total": total,
                "videos_done": indexed,
                "video_meta": video_meta,
                "indexed_files": [v.video_name for v in session.videos if v.video_no],
            })
            # Clear the in-memory progress after a few seconds so reconnects don't keep showing it
            await asyncio.sleep(2)
            services._indexing_progress.pop(project_name, None)
        except Exception as e:
            import traceback; traceback.print_exc()
            await _broadcast_index_progress(project_name, {
                "status": "error",
                "percent": 0,
                "message": f"Indexing failed: {e}",
            })

    async def _handle_crop_clip(
        workspace: WorkspaceManager,
        agent,
        emit,
        clip_id: str,
    ):
        """Run saliency-based crop on a single clip and update the edit decision."""
        import json as _json
        try:
            # Load current edit decision
            ed_path = workspace.root / "fcpxml" / "edit_decision.json"
            if not ed_path.exists():
                await emit("crop_error", {"clip_id": clip_id, "error": "No edit decision found."})
                return

            with open(ed_path) as f:
                ed_data = _json.load(f)

            # Find the clip
            clip_data = None
            for c in ed_data.get("clips", []):
                if c.get("id") == clip_id:
                    clip_data = c
                    break

            if not clip_data:
                await emit("crop_error", {"clip_id": clip_id, "error": f"Clip '{clip_id}' not found."})
                return

            await emit("crop_progress", {"clip_id": clip_id, "status": "running", "step": "extracting"})

            # Resolve source path
            source_file = clip_data.get("source_file", "")
            source_path = clip_data.get("source_path", "")
            if not source_path:
                resolved = workspace.root / "footage" / source_file
                if resolved.exists():
                    source_path = str(resolved)
                else:
                    await emit("crop_error", {"clip_id": clip_id, "error": f"Source file '{source_file}' not found."})
                    return

            # Get timeline dimensions
            tl = ed_data.get("timeline", {})
            tl_w = tl.get("width", 1920)
            tl_h = tl.get("height", 1080)
            src_w = clip_data.get("source_width", 1920)
            src_h = clip_data.get("source_height", 1080)

            await emit("crop_progress", {"clip_id": clip_id, "status": "running", "step": "running_saliency"})

            # Run saliency crop (returns MultiShotCropResult)
            from src.pipelines.v2.cropping.crop_pipeline import crop_single_clip
            crop_result = await crop_single_clip(
                source_path=source_path,
                start_sec=clip_data.get("source_start", 0),
                end_sec=clip_data.get("source_end", 0),
                tl_w=tl_w,
                tl_h=tl_h,
                src_w=src_w,
                src_h=src_h,
            )

            # Use first shot's transform as the clip-level transform
            transform_dict = crop_result.shots[0].transform.model_dump()
            clip_data["transform"] = transform_dict
            clip_data["transform_mode"] = "saliency"

            if len(crop_result.shots) > 1:
                # Multi-shot — store per-shot transforms on the clip
                # (FCPXML compiler will emit N asset-clips at compile time)
                clip_data["shot_transforms"] = [s.model_dump() for s in crop_result.shots]
                logger.info(f"[CROP] {clip_id}: {len(crop_result.shots)} shots with different transforms")
            else:
                clip_data["shot_transforms"] = None

            # Save updated edit decision
            with open(ed_path, "w") as f:
                _json.dump(ed_data, f, indent=2)

            # Recompile FCPXML
            try:
                from src.pipelines.v2.schemas import EditDecision as ED
                from src.pipelines.v2.fcpxml.edit_compiler import compile_edit_decision
                edit_obj = ED.model_validate(ed_data)
                footage_dir = workspace.root / "footage"
                for clip in edit_obj.clips:
                    if not clip.source_path:
                        resolved = footage_dir / clip.source_file
                        if resolved.exists():
                            clip.source_path = str(resolved)
                fcpxml_out = str(workspace.get_fcpxml_path(version=1))
                compile_edit_decision(edit_obj, fcpxml_out, workspace_root=workspace.root)
            except Exception as e:
                logger.warning(f"[CROP] FCPXML recompile failed (non-fatal): {e}")

            await emit("crop_complete", {
                "clip_id": clip_id,
                "transform": transform_dict,
                "edit_decision": ed_data,
            })
            # Also send timeline_update so dashboard refreshes the full edit state
            await emit("timeline_update", {"edit_decision": ed_data})
            # Crop changed the clip's transform — auto-rerender so the
            # ffmpeg preview reflects the new framing.
            agent.schedule_ffmpeg_rerender()

        except Exception as e:
            logger.error(f"[CROP] Error cropping clip {clip_id}: {e}", exc_info=True)
            await emit("crop_error", {"clip_id": clip_id, "error": str(e)})
