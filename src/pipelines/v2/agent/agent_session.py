"""AgentSession — core agentic loop for the editing chat."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from google.genai.types import Content, Part, FunctionResponse, GenerateContentConfig
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

from src.pipelines.v2.agent.scratchpad import ScratchpadManager
from src.pipelines.v2.agent.system_prompt import build_system_prompt
from src.pipelines.v2.agent.tools import TOOL_DECLARATIONS, ToolExecutor

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 80  # keep last N turns (user + model + tool rounds)
MAX_TOOL_ROUNDS = 40    # max tool call rounds per user message


@dataclass
class ChatMessage:
    """A single message in the conversation history for persistence."""
    role: str           # "user" or "model"
    text: str           # text content (or tool call summary)
    timestamp: str
    tool_calls: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)


class AgentSession:
    """
    Manages a single agentic editing conversation.

    The session owns:
    - Gemini conversation state (message history as Content objects)
    - Four scratchpads (comprehension, creative_direction, planning, fcpxml)
    - Tool execution via ToolExecutor
    - Event broadcasting to the frontend via WebSocket
    """

    def __init__(
        self,
        project_name: str,
        workspace,
        memories_manager,
        gemini_manager,
        video_entries: list,
        emit: Callable[..., Coroutine],
    ):
        self.project_name = project_name
        self.workspace = workspace
        self.memories = memories_manager
        self.gemini = gemini_manager
        self.video_entries = video_entries
        self._emit = emit

        # Build video info for system prompt and tools
        self.video_nos = [v.video_no for v in video_entries if v.video_no]

        # Build video list with footage filenames for source_file mapping
        footage_files = workspace.scan_footage() if workspace.get_footage_dir().is_dir() else []
        footage_names = [f.name for f in footage_files]
        video_lines = []
        for v in video_entries:
            # Try to match video_name to a footage filename
            matched_file = ""
            for fn in footage_names:
                if v.video_name.lower().replace(" ", "") in fn.lower().replace(" ", "").replace("_", "").replace("-", ""):
                    matched_file = fn
                    break
                if fn.lower().rsplit(".", 1)[0].replace("_", " ").replace("-", " ") in v.video_name.lower():
                    matched_file = fn
                    break
            if matched_file:
                video_lines.append(f"- {v.video_name} → filename: `{matched_file}` (video_no: {v.video_no})")
            else:
                video_lines.append(f"- {v.video_name} (video_no: {v.video_no})")
        if footage_names:
            unmatched = [fn for fn in footage_names if not any(fn in line for line in video_lines)]
            for fn in unmatched:
                video_lines.append(f"- (unmatched footage file: `{fn}`)")
        self.video_list = "\n".join(video_lines)

        # Scratchpads
        self.scratchpads = ScratchpadManager(workspace.root)

        # Seed comprehension from indexing gists, organized by file
        session_data = workspace.load_session()
        gist_parts = []
        for v in video_entries:
            if v.gist:
                gist_parts.append(f"## {v.video_name}\n{v.gist}")
        if gist_parts:
            self.scratchpads.seed_comprehension("\n\n".join(gist_parts))
        elif session_data.gist:
            self.scratchpads.seed_comprehension(session_data.gist)

        # Seed fcpxml scratchpad from existing FCPXML file if present
        if not self.scratchpads.read("fcpxml"):
            fcpxml_path = workspace.get_fcpxml_path(version=1)
            if fcpxml_path.exists():
                try:
                    self.scratchpads.update("fcpxml", "replace", fcpxml_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        # Tool executor — use a wrapper so reassigning self._emit propagates
        async def _tool_emit(event_type: str, data: dict):
            await self._emit(event_type, data)
            # Persist substep events so they survive page refresh
            if event_type == "refine_progress":
                self._event_log.append({"type": event_type, "data": data})

        self.tools = ToolExecutor(
            memories_manager=memories_manager,
            gemini_manager=gemini_manager,
            workspace=workspace,
            scratchpads=self.scratchpads,
            video_nos=self.video_nos,
            event_callback=_tool_emit,
        )

        # Conversation history (Gemini Content objects)
        self._history: List[Content] = []

        # Persistent chat log (for reloading sessions)
        self._chat_log: List[ChatMessage] = []
        self._event_log: List[Dict] = []  # persisted tool call/result events
        self._load_chat_log()
        self._load_event_log()

        # Safety settings — permissive for creative content
        self._safety = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]

    # ── Public API ────────────────────────────────────────────────────────

    async def handle_user_message(self, text: str) -> None:
        """
        Process a user message through the agentic loop.

        1. Append user message to history
        2. Build system prompt with current scratchpads
        3. Call Gemini with tools
        4. Execute tool calls, feed results back, repeat
        5. When Gemini responds with text only, emit as agent message
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"[AGENT] User message: {text[:100]}", flush=True)

            # Emit user message event
            await self._emit("user_message", {"text": text, "timestamp": timestamp})

            # Add to history
            user_content = Content(role="user", parts=[Part(text=text)])
            self._history.append(user_content)
            self._chat_log.append(ChatMessage(role="user", text=text, timestamp=timestamp))

            # Run the agent loop
            await self._agent_loop()

            # Trim and persist
            self._trim_history()
            self._save_chat_log()
            self._save_event_log()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[AGENT] handle_user_message FAILED: {e}", flush=True)
            await self._emit("error", {"message": str(e)})
            raise

    async def _agent_loop(self) -> None:
        """Run Gemini in a loop until it signals completion."""
        last_message_user_text: str | None = None
        for round_num in range(MAX_TOOL_ROUNDS):
            # Rebuild system prompt with current scratchpads
            system_instruction = build_system_prompt(
                project_name=self.project_name,
                video_list=self.video_list,
                scratchpads_text=self.scratchpads.render_all(),
            )

            config = GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[TOOL_DECLARATIONS],
                safety_settings=self._safety,
            )

            # Call Gemini
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.gemini.genai_client.models.generate_content(
                        model=self.gemini.model,
                        contents=self._history,
                        config=config,
                    ),
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[AGENT] Gemini call failed: {e}", flush=True)
                await self._emit("error", {"message": f"Gemini error: {e}"})
                return

            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("[AGENT] Empty response from Gemini")
                await self._emit("error", {"message": "Empty response from model"})
                return

            model_content = response.candidates[0].content
            self._history.append(model_content)

            # Check for function calls
            function_calls = [
                p for p in model_content.parts
                if p.function_call is not None
            ]

            if not function_calls:
                # Final text response — emit as agent message
                text_parts = [p.text for p in model_content.parts if p.text]
                full_text = "\n".join(text_parts)
                if full_text.strip():
                    is_duplicate = (
                        last_message_user_text
                        and full_text.strip() == last_message_user_text.strip()
                    )
                    if not is_duplicate:
                        await self._emit("agent_message", {
                            "text": full_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    self._chat_log.append(ChatMessage(
                        role="model",
                        text=full_text,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                return

            # Execute each function call
            function_response_parts = []
            for fc_part in function_calls:
                fc = fc_part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                # Emit tool_call event
                tool_call_data = {
                    "tool": tool_name,
                    "args": tool_args,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self._emit("tool_call", tool_call_data)
                self._event_log.append({"type": "tool_call", "data": tool_call_data})

                # Execute the tool
                result = await self.tools.execute(tool_name, tool_args)

                # Special handling for message_user — emit the message
                if tool_name == "message_user":
                    last_message_user_text = tool_args.get("message", "")
                    await self._emit("agent_message", {
                        "text": last_message_user_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                # Special handling for generate_fcpxml — emit timeline data
                if tool_name == "generate_fcpxml" and "error" not in result:
                    import json as _json
                    edit_json_path = result.get("edit_decision_path")
                    fcpxml_path = result.get("fcpxml_path")
                    if edit_json_path:
                        try:
                            with open(edit_json_path) as f:
                                edit_data = _json.load(f)
                            await self._emit("timeline_update", {
                                "edit_decision": edit_data,
                                "fcpxml_path": fcpxml_path,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                        except Exception:
                            pass  # non-critical

                    # Push FCPXML content into the fcpxml scratchpad
                    if fcpxml_path:
                        try:
                            fcpxml_content = Path(fcpxml_path).read_text(encoding="utf-8")
                            self.scratchpads.update("fcpxml", "replace", fcpxml_content)
                            await self._emit("scratchpad_update", {
                                "name": "fcpxml",
                                "operation": "replace",
                                "content": fcpxml_content,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                        except Exception:
                            pass  # non-critical

                    # Auto-trigger preview render via Resolve
                    asyncio.create_task(self._auto_render_preview(result))

                # Special handling for scratchpad updates — emit the update
                if tool_name == "update_scratchpad":
                    await self._emit("scratchpad_update", {
                        "name": tool_args.get("name"),
                        "operation": tool_args.get("operation"),
                        "content": self.scratchpads.read(tool_args.get("name", "")),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                # Emit tool_result event
                tool_result_data = {
                    "tool": tool_name,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self._emit("tool_result", tool_result_data)
                self._event_log.append({"type": "tool_result", "data": tool_result_data})

                function_response_parts.append(
                    Part(function_response=FunctionResponse(
                        name=tool_name,
                        response=result,
                    ))
                )

            # Add function responses to history
            fn_content = Content(role="tool", parts=function_response_parts)
            self._history.append(fn_content)

        # If we hit the max rounds, notify
        logger.warning(f"[AGENT] Hit max tool rounds ({MAX_TOOL_ROUNDS})")
        await self._emit("agent_message", {
            "text": "I've completed this round of work. Let me know if you'd like me to continue or adjust anything.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── Auto-render ─────────────────────────────────────────────────────

    async def _auto_render_preview(self, fcpxml_result: Dict) -> None:
        """Kick off a Resolve preview render in the background after FCPXML generation."""
        try:
            from lib.utils.resolve_render import ResolveRenderer

            fcpxml_path = fcpxml_result.get("fcpxml_path")
            if not fcpxml_path:
                return

            media_dir = str(self.workspace.get_footage_dir())
            output_path = str(self.workspace.get_render_path("preview"))

            await self._emit("render_start", {
                "quality": "preview",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            async def progress_cb(pct: float):
                await self._emit("render_progress", {
                    "percent": pct,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            renderer = ResolveRenderer()
            rendered = await renderer.render(
                fcpxml_path=fcpxml_path,
                media_dir=media_dir,
                output_path=output_path,
                quality="preview",
                progress_callback=progress_cb,
            )

            await self._emit("render_complete", {
                "output_path": rendered,
                "filename": Path(rendered).name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"[AGENT] Auto-render complete: {rendered}")

        except Exception as e:
            logger.warning(f"[AGENT] Auto-render failed (non-fatal): {e}")
            await self._emit("render_error", {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # ── History management ────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """Keep the last MAX_HISTORY_TURNS turns in the Gemini history."""
        if len(self._history) <= MAX_HISTORY_TURNS * 2:
            return

        # Keep the most recent turns
        keep = MAX_HISTORY_TURNS * 2
        dropped = len(self._history) - keep
        self._history = self._history[-keep:]
        logger.info(f"[AGENT] Trimmed {dropped} messages from history, {len(self._history)} remain")

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_chat_log(self) -> None:
        """Load chat log from workspace to rebuild history for returning users."""
        path = self.workspace.root / "chat_history.json"
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)
            for entry in data[-MAX_HISTORY_TURNS * 2:]:
                role = entry.get("role", "user")
                text = entry.get("text", "")
                if text:
                    self._history.append(
                        Content(role=role, parts=[Part(text=text)])
                    )
                    self._chat_log.append(ChatMessage(
                        role=role,
                        text=text,
                        timestamp=entry.get("timestamp", ""),
                    ))
            logger.info(f"[AGENT] Loaded {len(self._chat_log)} messages from chat history")
        except Exception as e:
            logger.warning(f"[AGENT] Could not load chat history: {e}")

    def _save_chat_log(self) -> None:
        """Persist chat log to workspace."""
        path = self.workspace.root / "chat_history.json"
        data = [
            {"role": m.role, "text": m.text, "timestamp": m.timestamp}
            for m in self._chat_log
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_event_log(self) -> None:
        """Load persisted events (tool calls, results) for replay on reconnect."""
        path = self.workspace.root / "event_log.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                self._event_log = json.load(f)
            logger.info(f"[AGENT] Loaded {len(self._event_log)} events from event log")
        except Exception as e:
            logger.warning(f"[AGENT] Could not load event log: {e}")

    def _save_event_log(self) -> None:
        """Persist event log to workspace."""
        path = self.workspace.root / "event_log.json"
        # Keep last 200 events to prevent unbounded growth
        trimmed = self._event_log[-200:]
        with open(path, "w") as f:
            json.dump(trimmed, f, indent=2)

    def get_scratchpad_state(self) -> Dict[str, str]:
        """Return current scratchpad contents for the frontend."""
        return {name: self.scratchpads.read(name) for name in self.scratchpads.pads}

    def get_scratchpad_timestamps(self) -> Dict[str, str | None]:
        """Return last_updated timestamps for each scratchpad."""
        return self.scratchpads.get_timestamps()

    def get_chat_history(self) -> List[Dict]:
        """Return chat log for the frontend to hydrate on reconnect."""
        return [
            {"role": m.role, "text": m.text, "timestamp": m.timestamp}
            for m in self._chat_log
        ]

    def get_event_log(self) -> List[Dict]:
        """Return persisted events for the frontend to hydrate on reconnect."""
        return self._event_log
