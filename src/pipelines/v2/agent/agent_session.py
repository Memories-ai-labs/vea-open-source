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

from src.pipelines.v2.agent.modes import AgentMode, MAX_TOOL_ROUNDS_BY_MODE
from src.pipelines.v2.agent.scratchpad import ScratchpadManager
from src.pipelines.v2.agent.system_prompt import build_system_prompt
from src.pipelines.v2.agent.tools import TOOL_DECLARATIONS, ToolExecutor
from src.pipelines.v2.workspace import _atomic_write_json

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 80  # keep last N turns (user + model + tool rounds)
# Per-mode cap lives in src/pipelines/v2/agent/modes.py (MAX_TOOL_ROUNDS_BY_MODE).
# Autonomous mode gets a much larger budget because there's no human to nudge it.


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
        video_llm=None,
        mode: AgentMode = AgentMode.COLLABORATIVE,
        autonomous: Optional[bool] = None,
    ):
        self.project_name = project_name
        self.workspace = workspace
        self.memories = memories_manager
        # ``mode`` selects the agent's temperament (collaborative = deferential
        # to a user in the loop; autonomous = perfectionist, non-interactive,
        # used by the CLI/MCP). ``autonomous: bool`` is a legacy kwarg kept
        # for back-compat with older callers; when set it overrides ``mode``.
        if autonomous is not None:
            mode = AgentMode.AUTONOMOUS if autonomous else AgentMode.COLLABORATIVE
        self.mode = mode
        # Legacy attribute for anyone reading ``session.autonomous`` directly.
        self.autonomous = mode != AgentMode.COLLABORATIVE
        # ``gemini`` is the main text + tool-calling LLM (may be Claude/GPT via
        # OpenRouter despite the historical name). ``video_llm`` is the
        # Gemini-family manager used for tasks that need native video input
        # (refine_clip_timestamps, verify_preview). If the caller doesn't
        # pass one, fall back to gemini for both — preserves old behavior.
        self.gemini = gemini_manager
        self.video_llm = video_llm or gemini_manager
        self.video_entries = video_entries
        # Multi-subscriber fan-out: the agent owns a set of emit callables.
        # Each WebSocket connection adds/removes itself via add_subscriber /
        # remove_subscriber. ``self._emit`` remains callable as ``await
        # self._emit(type, data)`` — it fans out to every subscriber.
        self._subscribers: "set[Callable[..., Coroutine]]" = set()
        if emit is not None:
            self._subscribers.add(emit)
        self._emit = self._broadcast_emit

        # Build video info for system prompt and tools
        self.video_nos = [v.video_no for v in video_entries if v.video_no]

        # Build video list with footage filenames AND actual durations for source_file mapping
        footage_files = workspace.scan_footage() if workspace.get_footage_dir().is_dir() else []
        footage_names = [f.name for f in footage_files]

        # Probe each footage file's actual duration via ffprobe so the agent
        # never invents source_start/source_end values past the file end.
        import subprocess
        footage_durations: Dict[str, float] = {}
        for fp in footage_files:
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(fp)],
                    capture_output=True, text=True, timeout=10,
                )
                dur = float(out.stdout.strip()) if out.stdout.strip() else 0.0
                footage_durations[fp.name] = round(dur, 2)
            except Exception:
                footage_durations[fp.name] = 0.0
        # Cache for use elsewhere (e.g. _generate_fcpxml validation)
        self.footage_durations = footage_durations

        def _dur_label(name: str) -> str:
            d = footage_durations.get(name, 0)
            return f"DURATION: {d}s — clip source_start MUST be < {d}, source_end ≤ {d}" if d > 0 else "duration unknown"

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
                video_lines.append(
                    f"- {v.video_name} → filename: `{matched_file}` "
                    f"(video_no: {v.video_no}, {_dur_label(matched_file)})"
                )
            else:
                video_lines.append(f"- {v.video_name} (video_no: {v.video_no})")
        if footage_names:
            unmatched = [fn for fn in footage_names if not any(fn in line for line in video_lines)]
            for fn in unmatched:
                video_lines.append(f"- (unmatched footage file: `{fn}`, {_dur_label(fn)})")
        self.video_list = "\n".join(video_lines)

        # Render state — persists across WebSocket reconnects
        self._render_state: Dict = {"status": "idle"}
        self._draft_render_state: Dict = {"status": "idle"}

        # Background tasks spawned from the agent loop (auto-render, etc.)
        # We track them so the session can await them before being released —
        # otherwise create_task() returns a task that nobody holds a reference
        # to, and Python's GC can kill it mid-run (see RUF006 / PYI053).
        self._bg_tasks: "set[asyncio.Task]" = set()
        # Warnings from background work (draft render, Resolve render, etc.)
        # that complete AFTER a finish_turn. Prepended to the next
        # handle_user_message so the main LLM sees them and relays to the
        # user instead of the failure being silently swallowed.
        self._pending_background_warnings: list = []

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
            video_llm=self.video_llm,
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

            # If background work (renders, etc.) failed after the last turn,
            # prepend a NOT-FROM-USER system notice so the LLM relays it to
            # the user instead of the failure being silently swallowed.
            effective_text = text
            if self._pending_background_warnings:
                notice_lines = "\n".join(
                    f"- {w}" for w in self._pending_background_warnings
                )
                effective_text = (
                    "[Background events since last turn — relay any relevant "
                    "ones to the user in your response]\n"
                    f"{notice_lines}\n\n"
                    f"---\n\nUser message: {text}"
                )
                self._pending_background_warnings = []

            # Add to history
            user_content = Content(role="user", parts=[Part(text=effective_text)])
            self._history.append(user_content)
            # Chat log stores the user's actual words, not the synthetic prefix.
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
        max_rounds = MAX_TOOL_ROUNDS_BY_MODE[self.mode]
        # Watchdog state — detects the same tool being called with identical
        # args 3× in a row, which usually means the model is stuck. We only
        # nudge in autonomous mode; in collaborative mode the user nudges.
        recent_calls: list = []
        watchdog_fired_this_turn = False
        for round_num in range(max_rounds):
            message_user_this_round = False  # reset each round
            # Load current edit decision from disk (may have been modified by user in UI)
            current_edit_json = ""
            edit_decision_path = self.workspace.root / "fcpxml" / "edit_decision.json"
            if edit_decision_path.exists():
                try:
                    import json as _json
                    with open(edit_decision_path) as f:
                        current_edit_json = _json.dumps(_json.load(f), indent=2)
                except Exception:
                    pass

            # Rebuild system prompt with current scratchpads + edit decision
            system_instruction = build_system_prompt(
                project_name=self.project_name,
                video_list=self.video_list,
                scratchpads_text=self.scratchpads.render_all(),
                current_edit_decision=current_edit_json,
                mode=self.mode,
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
            # Capture any accompanying text so ``finish_turn`` with an empty
            # ``final_message`` can fall back to it. Some models (Sonnet via
            # OpenRouter in particular) put an info-query answer in the text
            # channel alongside the finish_turn call, and the user would
            # otherwise see nothing.
            inline_text_parts = [p.text for p in model_content.parts if p.text]
            inline_text = "\n".join(inline_text_parts).strip()

            if not function_calls:
                # Final text response — emit as agent message
                text_parts = [p.text for p in model_content.parts if p.text]
                full_text = "\n".join(text_parts)
                if full_text.strip():
                    # Suppress if message_user was already called this round
                    # (Gemini tends to rephrase the same plan as follow-up text)
                    if not message_user_this_round:
                        await self._emit("agent_message", {
                            "text": full_text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    else:
                        logger.debug("[AGENT] Suppressed duplicate follow-up text after message_user")
                    self._chat_log.append(ChatMessage(
                        role="model",
                        text=full_text,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                return

            # Execute each function call
            function_response_parts = []
            finish_after_batch = False
            finish_final_message = ""
            for fc_part in function_calls:
                fc = fc_part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                # Watchdog tracking: rolling buffer of (tool, args) for the
                # last three calls across the whole turn. Used below to
                # detect stuck loops in autonomous mode.
                try:
                    call_key = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
                except Exception:
                    call_key = (tool_name, str(tool_args))
                recent_calls.append(call_key)
                if len(recent_calls) > 3:
                    recent_calls.pop(0)

                # Emit tool_call event
                tool_call_data = {
                    "tool": tool_name,
                    "args": tool_args,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self._emit("tool_call", tool_call_data)
                self._event_log.append({"type": "tool_call", "data": tool_call_data})

                # Special handling for finish_turn — exit the loop after this batch
                if tool_name == "finish_turn":
                    finish_after_batch = True
                    finish_final_message = tool_args.get("final_message", "") or ""
                    logger.info(f"[AGENT] finish_turn called (final_message={len(finish_final_message)} chars)")
                    # Synthesize a result so the model history stays consistent
                    result = {"status": "turn_ended"}
                    function_response_parts.append(
                        Part(function_response=FunctionResponse(name=tool_name, response=result))
                    )
                    self._event_log.append({"type": "tool_result", "data": {
                        "tool": tool_name, "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }})
                    continue

                # Execute the tool
                result = await self.tools.execute(tool_name, tool_args)

                # Special handling for message_user — emit the message
                if tool_name == "message_user":
                    last_message_user_text = tool_args.get("message", "")
                    message_user_this_round = True
                    logger.info(f"[AGENT] message_user: {last_message_user_text[:120]}")
                    await self._emit("agent_message", {
                        "text": last_message_user_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    self._chat_log.append(ChatMessage(
                        role="model",
                        text=last_message_user_text,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))

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

                    # Auto-trigger preview render via Resolve.
                    # Track the task so it isn't collected mid-run.
                    self._spawn_bg_task(self._auto_render_preview(result))

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

            # Watchdog: if the last 3 tool calls were identical and we're in
            # autonomous mode, the model is probably stuck. Inject a synthetic
            # user nudge so the next LLM turn sees it. Fire at most once per
            # user message to avoid cascading nudges.
            if (
                self.mode == AgentMode.AUTONOMOUS
                and not watchdog_fired_this_turn
                and len(recent_calls) == 3
                and recent_calls[0] == recent_calls[1] == recent_calls[2]
            ):
                stuck_tool = recent_calls[0][0]
                nudge = (
                    f"⚠️ Watchdog: you called `{stuck_tool}` 3× in a row with "
                    "identical arguments. Something is stuck. Commit with what "
                    "you have now (call `generate_fcpxml` if you haven't yet, "
                    "then `finish_turn`), or explain the blocker in a "
                    "`finish_turn(final_message=...)` call. Do not repeat the "
                    "same tool call again."
                )
                logger.warning(f"[AGENT] Watchdog fired on `{stuck_tool}`")
                self._history.append(Content(role="user", parts=[Part(text=nudge)]))
                watchdog_fired_this_turn = True

            # Exit the loop if finish_turn was called in this batch
            if finish_after_batch:
                # Fall back to any inline text when final_message is empty —
                # otherwise answers that the model wrote alongside the tool
                # call would be silently dropped.
                effective_final = finish_final_message.strip() or inline_text
                if effective_final and not message_user_this_round:
                    await self._emit("agent_message", {
                        "text": effective_final,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    self._chat_log.append(ChatMessage(
                        role="model",
                        text=effective_final,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                    if not finish_final_message.strip():
                        logger.info(
                            f"[AGENT] finish_turn had empty final_message; "
                            f"used inline text ({len(effective_final)} chars) as fallback"
                        )
                logger.info(f"[AGENT] Turn ended via finish_turn after {round_num + 1} rounds")
                return

        # If we hit the max rounds, notify. Structured event so the CLI /
        # orchestrator can detect exhaustion separately from a clean finish_turn.
        logger.warning(f"[AGENT] Hit max tool rounds ({max_rounds}) in {self.mode.value} mode")
        await self._emit("turn_exhausted", {
            "mode": self.mode.value,
            "rounds": max_rounds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if self.mode == AgentMode.COLLABORATIVE:
            text = (
                "I've completed this round of work. Let me know if you'd like "
                "me to continue or adjust anything."
            )
        else:
            text = (
                f"Hit the per-turn tool cap ({max_rounds} rounds) without a "
                "`finish_turn`. Stopping here. Inspect the scratchpads and "
                "edit_decision.json to see how far the agent got."
            )
        await self._emit("agent_message", {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── Auto-render ─────────────────────────────────────────────────────

    async def _render_ffmpeg_draft(self) -> None:
        """Render a draft 480p preview via FFmpeg."""
        try:
            import json as _json
            from src.pipelines.v2.schemas import EditDecision
            from src.pipelines.v2.preview.ffmpeg_renderer import render_ffmpeg_preview

            ed_path = self.workspace.root / "fcpxml" / "edit_decision.json"
            if not ed_path.exists():
                ed_path = self.workspace.root / "edit_decision.json"
            if not ed_path.exists():
                logger.warning("[AGENT] No edit_decision.json for FFmpeg draft render")
                return

            with open(ed_path) as f:
                ed_data = _json.load(f)
            edit = EditDecision.model_validate(ed_data)

            footage_dir = self.workspace.get_footage_dir()
            output_path = self.workspace.get_render_path("draft")

            self._draft_render_state = {"status": "rendering", "progress": 0}
            try:
                await self._emit("render_start", {
                    "renderer": "ffmpeg",
                    "quality": "draft",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected

            async def progress_cb(pct: float):
                self._draft_render_state["progress"] = pct
                try:
                    await self._emit("render_progress", {
                        "renderer": "ffmpeg",
                        "percent": pct,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass  # client may be disconnected

            rendered = await render_ffmpeg_preview(
                edit=edit,
                footage_dir=footage_dir,
                output_path=output_path,
                quality="draft",
                progress_callback=progress_cb,
            )

            # Measure loudness of each clip/narration/music and save back
            try:
                from src.pipelines.v2.audio.loudness import measure_edit_loudness
                loudness_summary = await asyncio.get_event_loop().run_in_executor(
                    None, measure_edit_loudness, edit, footage_dir,
                )
                # Save updated edit decision with loudness measurements
                _atomic_write_json(ed_path, edit.model_dump())
                # Emit updated edit decision to dashboard
                try:
                    await self._emit("timeline_update", {
                        "edit_decision": edit.model_dump(),
                        "loudness": loudness_summary,
                    })
                except Exception:
                    pass
                logger.info(f"[AGENT] Loudness measured: {len(loudness_summary.get('clips', []))} clips")
            except Exception as e:
                logger.warning(f"[AGENT] Loudness measurement failed (non-fatal): {e}")

            self._draft_render_state = {"status": "complete", "filename": Path(rendered).name}
            try:
                await self._emit("render_complete", {
                    "renderer": "ffmpeg",
                    "filename": Path(rendered).name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected
            logger.info(f"[AGENT] FFmpeg draft render complete: {rendered}")

        except Exception as e:
            logger.warning(f"[AGENT] FFmpeg draft render failed (non-fatal): {e}")
            self._draft_render_state = {"status": "error", "error": str(e)}
            raw = str(e)
            brief = raw if len(raw) < 240 else raw[:240] + "…"
            self._pending_background_warnings.append(
                f"FFmpeg draft render FAILED after the last generate_fcpxml "
                f"call — there is no draft.mp4 for the user to preview. "
                f"Underlying error: {brief}"
            )
            try:
                await self._emit("render_error", {
                    "renderer": "ffmpeg",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected

    def _spawn_bg_task(self, coro) -> asyncio.Task:
        """Create a background task and keep a strong reference to it.

        Without this, ``asyncio.create_task(coro)`` alone can be garbage
        collected mid-run and cancelled silently.
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def add_subscriber(self, emit: Callable[..., Coroutine]) -> None:
        """Register a WebSocket emit callable to receive agent events."""
        self._subscribers.add(emit)

    def remove_subscriber(self, emit: Callable[..., Coroutine]) -> None:
        """Stop sending events to this WebSocket emit callable."""
        self._subscribers.discard(emit)

    async def _broadcast_emit(self, event_type: str, data: dict) -> None:
        """Fan ``(event_type, data)`` out to every connected subscriber.

        A single misbehaving client is isolated: its exception is swallowed
        (with a log entry) so it cannot stop events reaching the others.
        """
        # Snapshot to avoid 'set changed during iteration' if a client
        # disconnects while we're fanning out.
        for sub in list(self._subscribers):
            try:
                await sub(event_type, data)
            except Exception as e:
                logger.warning(
                    f"[AGENT] Dropping subscriber after emit error: {e}"
                )
                self._subscribers.discard(sub)

    async def _render_ffmpeg_final(
        self,
        resolve_missing: bool = False,
        resolve_error: Optional[str] = None,
    ) -> None:
        """Render a timeline-native final MP4 via FFmpeg.

        Used as a fallback when DaVinci Resolve isn't available or its render
        failed. Slower than the draft but matches the timeline resolution and
        uses a higher-quality encoder preset.
        """
        try:
            import json as _json
            from src.pipelines.v2.schemas import EditDecision
            from src.pipelines.v2.preview.ffmpeg_renderer import render_ffmpeg_preview

            ed_path = self.workspace.root / "fcpxml" / "edit_decision.json"
            if not ed_path.exists():
                ed_path = self.workspace.root / "edit_decision.json"
            if not ed_path.exists():
                logger.warning("[AGENT] No edit_decision.json for FFmpeg final render")
                return

            with open(ed_path) as f:
                ed_data = _json.load(f)
            edit = EditDecision.model_validate(ed_data)

            footage_dir = self.workspace.get_footage_dir()
            output_path = self.workspace.get_render_path("final")

            self._render_state = {
                "status": "rendering", "progress": 0,
                "renderer": "ffmpeg", "quality": "final",
            }
            try:
                await self._emit("render_start", {
                    "renderer": "ffmpeg",
                    "quality": "final",
                    "reason": "resolve_missing" if resolve_missing else "resolve_failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass

            async def progress_cb(pct: float):
                self._render_state["progress"] = pct
                try:
                    await self._emit("render_progress", {
                        "renderer": "ffmpeg",
                        "quality": "final",
                        "percent": pct,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass

            rendered = await render_ffmpeg_preview(
                edit=edit,
                footage_dir=footage_dir,
                output_path=output_path,
                quality="final",
                progress_callback=progress_cb,
            )
            self._render_state = {
                "status": "complete", "filename": Path(rendered).name,
                "renderer": "ffmpeg", "quality": "final",
            }
            try:
                await self._emit("render_complete", {
                    "renderer": "ffmpeg",
                    "quality": "final",
                    "output_path": rendered,
                    "filename": Path(rendered).name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass
            logger.info(f"[AGENT] FFmpeg final render complete: {rendered}")

        except Exception as e:
            logger.warning(f"[AGENT] FFmpeg final render failed (non-fatal): {e}")
            self._render_state = {"status": "error", "error": str(e)}
            brief = str(e) if len(str(e)) < 240 else str(e)[:240] + "…"
            # Only surface when BOTH paths failed — Resolve-missing + ffmpeg
            # failure is the actually-broken case the agent should know about.
            if not resolve_missing or resolve_error:
                self._pending_background_warnings.append(
                    f"Both DaVinci Resolve and FFmpeg final renders failed — "
                    f"only the draft is available. FFmpeg error: {brief}"
                )
            try:
                await self._emit("render_error", {
                    "renderer": "ffmpeg",
                    "quality": "final",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass

    async def _auto_render_preview(self, fcpxml_result: Dict) -> None:
        """Kick off a Resolve preview render in the background after FCPXML generation."""
        # Start FFmpeg draft render first (fast, no Resolve dependency)
        self._spawn_bg_task(self._render_ffmpeg_draft())

        # Then attempt Resolve render
        try:
            from lib.utils.resolve_render import ResolveRenderer

            fcpxml_path = fcpxml_result.get("fcpxml_path")
            if not fcpxml_path:
                return

            media_dir = str(self.workspace.get_footage_dir())
            output_path = str(self.workspace.get_render_path("final"))

            self._render_state = {"status": "rendering", "progress": 0}
            try:
                await self._emit("render_start", {
                    "renderer": "resolve",
                    "quality": "final",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected

            async def progress_cb(pct: float):
                self._render_state["progress"] = pct
                try:
                    await self._emit("render_progress", {
                        "renderer": "resolve",
                        "percent": pct,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass  # client may be disconnected

            renderer = ResolveRenderer()
            rendered = await renderer.render(
                fcpxml_path=fcpxml_path,
                media_dir=media_dir,
                output_path=output_path,
                quality="final",
                progress_callback=progress_cb,
            )

            self._render_state = {"status": "complete", "filename": Path(rendered).name}
            try:
                await self._emit("render_complete", {
                    "renderer": "resolve",
                    "output_path": rendered,
                    "filename": Path(rendered).name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected
            logger.info(f"[AGENT] Auto-render complete: {rendered}")

        except Exception as e:
            logger.warning(f"[AGENT] Resolve final render failed (non-fatal): {e}")
            self._render_state = {"status": "error", "error": str(e)}
            raw = str(e)
            brief = raw if len(raw) < 240 else raw[:240] + "…"
            lower = raw.lower()
            is_resolve_missing = (
                "davinci" in lower or "resolve" in lower
            ) and (
                "not running" in lower or "not found" in lower
                or "connection" in lower or "no module" in lower
            )
            # Fall back to an FFmpeg final render so users without DaVinci
            # still get a timeline-native MP4 without having to install it.
            self._spawn_bg_task(self._render_ffmpeg_final(
                resolve_missing=is_resolve_missing,
                resolve_error=brief if not is_resolve_missing else None,
            ))
            try:
                await self._emit("render_error", {
                    "renderer": "resolve",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                pass  # client may be disconnected

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
        data = [
            {"role": m.role, "text": m.text, "timestamp": m.timestamp}
            for m in self._chat_log
        ]
        _atomic_write_json(self.workspace.root / "chat_history.json", data)

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
        # Keep last 200 events to prevent unbounded growth
        trimmed = self._event_log[-200:]
        _atomic_write_json(self.workspace.root / "event_log.json", trimmed)

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
