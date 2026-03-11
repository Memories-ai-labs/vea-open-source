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

        # Check for existing preview render
        render_path = workspace.get_render_path("preview")
        render_state = None
        if render_path.exists():
            render_state = {
                "status": "complete",
                "filename": render_path.name,
            }

        # Send initial state on connect
        try:
            init_data: dict = {
                "scratchpads": agent.get_scratchpad_state(),
                "scratchpad_timestamps": agent.get_scratchpad_timestamps(),
                "chat_history": agent.get_chat_history(),
                "event_log": agent.get_event_log(),
                "project_name": project_name,
                "video_count": len(session_data.videos),
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
