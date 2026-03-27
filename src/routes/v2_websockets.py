# src/routes/v2_websockets.py
"""V2 WebSocket routes: planning session and agent chat."""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect

from src import config as _config
from src.pipelines.v2.workspace import WorkspaceManager
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

        # Load workspace
        workspace = WorkspaceManager(project_name, _config.WORKSPACES_DIR)
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
        agent = services._agent_sessions.get(project_name)
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
                    memories_manager=services.memories_manager,
                    gemini_manager=services.gemini_manager,
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

        # Always point the agent's emit at the current connection's queue
        agent._emit = emit
        print(f"[AGENT WS] Bound emit to current connection for project={project_name}", flush=True)

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

        # Get render state from agent (survives reconnects) or fall back to file check
        # Suppress Resolve-not-available errors — they're not actionable and confuse the UI
        render_state = None
        agent_rs = agent._render_state
        is_resolve_error = (
            agent_rs.get("status") == "error" and
            "resolve" in str(agent_rs.get("error", "")).lower()
        )
        if agent_rs.get("status") in ("rendering", "complete", "error") and not is_resolve_error:
            render_state = agent_rs
        else:
            render_path = workspace.get_render_path("preview")
            if render_path.exists():
                render_state = {"status": "complete", "filename": render_path.name}

        # Send initial state on connect
        try:
            # Scan footage directory for actual files
            footage_files = [f.name for f in workspace.scan_footage()] if workspace.get_footage_dir().is_dir() else []
            indexed_files = [v.video_name for v in session_data.videos if v.video_no]

            init_data: dict = {
                "scratchpads": agent.get_scratchpad_state(),
                "scratchpad_timestamps": agent.get_scratchpad_timestamps(),
                "chat_history": agent.get_chat_history(),
                "event_log": agent.get_event_log(),
                "project_name": project_name,
                "video_count": len(footage_files),
                "footage_files": footage_files,
                "indexed_files": indexed_files,
            }
            if edit_decision_data:
                init_data["edit_decision"] = edit_decision_data
            if render_state:
                init_data["render_state"] = render_state
            await websocket.send_json({
                "type": "init",
                "data": init_data,
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
                            # Agent is busy -- queue the message for later
                            await websocket.send_json({
                                "type": "queued",
                                "data": {"text": text, "message": "Agent is working. Your message will be processed next."},
                            })

                    elif msg_type == "render":
                        # Force re-render via Resolve
                        agent._emit = emit
                        fcpxml = agent.workspace.get_latest_fcpxml()
                        if fcpxml:
                            asyncio.create_task(
                                agent._auto_render_preview({"fcpxml_path": str(fcpxml)})
                            )
                        else:
                            await websocket.send_json({
                                "type": "render_error",
                                "data": {"error": "No FCPXML found. Generate one first."},
                            })

                    elif msg_type == "crop_clip":
                        clip_id = msg.get("clip_id", "")
                        if clip_id:
                            asyncio.create_task(
                                _handle_crop_clip(workspace, agent, emit, clip_id)
                            )

                    elif msg_type == "edit_decision_update":
                        # Persist edit decision from client, recompile FCPXML, and broadcast
                        import json as _json
                        ed = msg.get("edit_decision")
                        if ed:
                            ed_dir = workspace.root / "fcpxml"
                            ed_dir.mkdir(parents=True, exist_ok=True)
                            ed_path = ed_dir / "edit_decision.json"
                            with open(ed_path, "w") as f:
                                _json.dump(ed, f, indent=2)
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
                                compile_edit_decision(edit_obj, fcpxml_out)
                                logger.info(f"[AGENT WS] Recompiled FCPXML from UI edit: {fcpxml_out}")
                            except Exception as e:
                                logger.warning(f"[AGENT WS] FCPXML recompile failed (non-fatal): {e}")

                            await emit("timeline_update", {"edit_decision": ed})

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
                compile_edit_decision(edit_obj, fcpxml_out)
            except Exception as e:
                logger.warning(f"[CROP] FCPXML recompile failed (non-fatal): {e}")

            await emit("crop_complete", {
                "clip_id": clip_id,
                "transform": transform_dict,
                "edit_decision": ed_data,
            })
            # Also send timeline_update so dashboard refreshes the full edit state
            await emit("timeline_update", {"edit_decision": ed_data})

        except Exception as e:
            logger.error(f"[CROP] Error cropping clip {clip_id}: {e}", exc_info=True)
            await emit("crop_error", {"clip_id": clip_id, "error": str(e)})
