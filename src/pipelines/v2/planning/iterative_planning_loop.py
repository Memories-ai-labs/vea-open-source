"""
Iterative Planning Loop — Phase 2 of the VEA v2 pipeline.

Architecture (unchanged from pre-port):
  - MAX_ITERATIONS passes through the planning loop (configurable)
  - Each iteration:
      Call A (Gemini): Given current state, decide which retrieval tools to call → ToolCallPlan
      Execute:         Run chat() and search() calls concurrently → update context + clips
      Call B (Gemini): Given full context + clips, produce updated Storyboard
  - Stops early if should_stop=True or all shots have clips and no open_questions remain
  - Streams progress events via asyncio.Queue for WebSocket dashboard

PORT NOTE (2026-05-19): Backend swapped from memories.ai HTTP client to
lvmm-core's local MaviAgent (chat) + Searcher (search). Loop semantics
unchanged. Tool calls now hit the local SQLite index instead of crossing
the network.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from lvmm_core.interfaces.llm import ILLM
from lvmm_core.utils.llm import messages_from_prompts
from src.pipelines.v2.planning.clip_postprocess import (
    postprocess_clips,
    deduplicate_against_existing,
    sort_by_timeline,
)
from src.pipelines.v2.planning.planning_prompts import (
    DECIDE_TOOL_CALLS_SYSTEM,
    DECIDE_TOOL_CALLS_USER,
    LAST_ITERATION_NOTE,
    NO_ITERATION_NOTE,
    UPDATE_STORYBOARD_SYSTEM,
    UPDATE_STORYBOARD_USER,
)
from src.pipelines.v2.schemas import (
    RetrievedClip,
    Shot,
    Storyboard,
    ToolCallPlan,
    VideoEntry,
)
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types — emitted to dashboard via asyncio.Queue
# ---------------------------------------------------------------------------

@dataclass
class PlanningEvent:
    """A live event streamed to the WebSocket dashboard."""
    event_type: str   # "iteration_start" | "tool_call" | "tool_result" | "storyboard_update" | "done" | "error"
    data: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {"event_type": self.event_type, "data": self.data, "timestamp": self.timestamp}


# ---------------------------------------------------------------------------
# Main loop class
# ---------------------------------------------------------------------------

class IterativePlanningLoop:
    """
    Orchestrates the agentic planning loop for one v2 editing session.

    Usage:
        loop = IterativePlanningLoop(
            project_name="my_project",
            user_prompt="Create a 2-min highlight reel of the keynote",
            workspace=workspace_manager,
            searcher=searcher,          # lvmm-core luci_memory.Searcher
            mavi_agent=mavi_agent,      # lvmm-core MaviAgent
            gemini=gemini_manager,
            video_nos=[...],            # lvmm-core video_ids to query
            video_entries=[...],        # Full VideoEntry objects (for source paths)
            event_queue=queue,          # Optional — live updates to dashboard
            pause_event=event,          # Optional — caller sets to pause loop
        )
        storyboard = await loop.run()
    """

    def __init__(
        self,
        project_name: str,
        user_prompt: str,
        workspace: WorkspaceManager,
        searcher,                       # lvmm_core.core.retrieval.luci_memory.Searcher
        mavi_agent,                     # lvmm_core.agents.mavi_agent.MaviAgent
        gemini: ILLM,
        video_nos: List[str],
        video_entries: List[VideoEntry],
        max_iterations: int = 5,
        target_duration_seconds: float = 120.0,
        event_queue: Optional[asyncio.Queue] = None,
        pause_event: Optional[asyncio.Event] = None,
        inject_prompt_queue: Optional[asyncio.Queue] = None,
    ):
        self.project_name = project_name
        self.user_prompt = user_prompt
        self.workspace = workspace
        self.searcher = searcher
        self.mavi_agent = mavi_agent
        self.gemini = gemini
        self.video_nos = video_nos
        self.video_entries = video_entries
        self.max_iterations = max_iterations
        self.target_duration_seconds = target_duration_seconds
        self.event_queue = event_queue
        self.pause_event = pause_event
        self.inject_prompt_queue = inject_prompt_queue

        # Map video_no → VideoEntry for fast source-path lookup at clip-parse time.
        self._video_map: Dict[str, VideoEntry] = {v.video_no: v for v in video_entries}

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    async def run(self) -> Storyboard:
        """Run the full planning loop. Returns the final Storyboard."""
        # Load existing state (resume) or start fresh
        gist = self._load_gist()
        storyboard = self.workspace.load_storyboard() or Storyboard(
            iteration=0,
            target_duration_seconds=self.target_duration_seconds,
            theme="",
            narrative_arc="",
            shots=[],
            open_questions=[self.user_prompt],
        )
        accumulated_clips = self.workspace.load_clips()
        accumulated_context = self.workspace.load_context()

        start_iteration = storyboard.iteration
        logger.info(
            f"[PLAN] Starting planning loop from iteration {start_iteration} "
            f"(max {self.max_iterations}) for project '{self.project_name}'"
        )

        for iteration in range(start_iteration, self.max_iterations):
            # --- Check for pause/inject ---
            await self._handle_pause_and_inject()

            is_last = iteration == self.max_iterations - 1
            await self._emit("iteration_start", {"iteration": iteration, "is_last": is_last})

            # ---- CALL A: Decide tool calls ----
            tool_plan = await self._call_a_decide_tools(
                iteration=iteration,
                storyboard=storyboard,
                accumulated_clips=accumulated_clips,
                accumulated_context=accumulated_context,
                is_last=is_last,
            )
            logger.info(
                f"[PLAN] Iter {iteration}: {len(tool_plan.chat_calls)} chats, "
                f"{len(tool_plan.search_calls)} searches, stop={tool_plan.should_stop}"
            )

            await self._emit("tool_call_plan", {
                "iteration": iteration,
                "reasoning": tool_plan.reasoning,
                "chat_calls": [c.model_dump() for c in tool_plan.chat_calls],
                "search_calls": [s.model_dump() for s in tool_plan.search_calls],
                "should_stop": tool_plan.should_stop,
            })

            # Persist Call A result
            self.workspace.save_iteration_snapshot(iteration, tool_plan, storyboard)

            if tool_plan.should_stop and iteration >= 1:
                logger.info(f"[PLAN] should_stop=True at iteration {iteration} — finalizing.")
                await self._emit("stopped_early", {"iteration": iteration, "reason": "should_stop"})
                break

            # ---- EXECUTE tool calls ----
            new_context, new_clips = await self._execute_tools(tool_plan, iteration)

            # Append context
            if new_context:
                self.workspace.append_context(new_context)
                accumulated_context += new_context

            # Merge clips
            deduped = deduplicate_against_existing(new_clips, accumulated_clips)
            processed = postprocess_clips(deduped)
            accumulated_clips.extend(processed)
            accumulated_clips = postprocess_clips(accumulated_clips, max_clips=60)
            self.workspace.save_clips(accumulated_clips)

            logger.info(
                f"[PLAN] Iter {iteration}: +{len(new_clips)} raw clips → "
                f"+{len(processed)} new → {len(accumulated_clips)} total"
            )

            # ---- CALL B: Update storyboard ----
            storyboard = await self._call_b_update_storyboard(
                iteration=iteration,
                storyboard=storyboard,
                accumulated_clips=accumulated_clips,
                accumulated_context=accumulated_context,
            )
            storyboard.iteration = iteration + 1

            self.workspace.save_storyboard(storyboard)
            self.workspace.save_iteration_snapshot(iteration, tool_plan, storyboard)

            await self._emit("storyboard_update", {
                "iteration": iteration + 1,
                "shots": len(storyboard.shots),
                "open_questions": storyboard.open_questions,
                "total_duration": sum(s.duration_seconds for s in storyboard.shots),
            })

            # Update session iteration count
            session = self.workspace.load_session()
            session.planning.iteration_count = iteration + 1
            self.workspace.save_session(session)

        # Final state
        self.workspace.update_status("plan_ready")
        sorted_clips = sort_by_timeline(accumulated_clips)
        self.workspace.save_clips(sorted_clips)

        await self._emit("done", {
            "iteration": storyboard.iteration,
            "shots": len(storyboard.shots),
            "clips": len(sorted_clips),
        })

        logger.info(
            f"[PLAN] Done. {storyboard.iteration} iterations, "
            f"{len(storyboard.shots)} shots, {len(sorted_clips)} clips."
        )
        return storyboard

    # -------------------------------------------------------------------------
    # Call A
    # -------------------------------------------------------------------------

    async def _call_a_decide_tools(
        self,
        iteration: int,
        storyboard: Storyboard,
        accumulated_clips: List[RetrievedClip],
        accumulated_context: str,
        is_last: bool,
    ) -> ToolCallPlan:
        gist = self._load_gist()
        clips_summary = self._format_clips_summary(accumulated_clips)
        storyboard_json = storyboard.model_dump_json(indent=2)

        user_content = DECIDE_TOOL_CALLS_USER.format(
            user_prompt=self.user_prompt,
            gist=gist,
            accumulated_context=accumulated_context or "(none yet)",
            iteration=iteration,
            max_iterations=self.max_iterations,
            storyboard_json=storyboard_json,
            clip_count=len(accumulated_clips),
            clips_summary=clips_summary,
            last_iteration_note=LAST_ITERATION_NOTE if is_last else NO_ITERATION_NOTE,
        )

        prompt_contents = [DECIDE_TOOL_CALLS_SYSTEM, user_content]

        try:
            # PORT NOTE: was self.gemini.LLM_request wrapped in run_in_executor
            # (sync VEA manager). Now using lvmm-core ILLM.generate_structured
            # directly — natively async, returns (parsed, usage) tuple.
            result, _usage = await self.gemini.generate_structured(
                messages_from_prompts(prompt_contents),
                ToolCallPlan,
            )
            return result
        except Exception as e:
            logger.error(f"[PLAN] Call A failed at iteration {iteration}: {e}")
            return ToolCallPlan(
                reasoning=f"Call A error: {e}",
                chat_calls=[],
                search_calls=[],
                should_stop=is_last,
            )

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------

    async def _execute_tools(
        self, tool_plan: ToolCallPlan, iteration: int
    ) -> tuple[str, List[RetrievedClip]]:
        """Run all chat and search calls concurrently. Returns (new_context_str, new_clips)."""
        context_parts: List[str] = []
        new_clips: List[RetrievedClip] = []

        # Build task list
        tasks = []

        for chat_tool in tool_plan.chat_calls:
            tasks.append(self._run_chat(chat_tool.question, chat_tool.purpose, iteration))

        for search_tool in tool_plan.search_calls:
            tasks.append(self._run_search(
                search_tool.query,
                search_tool.purpose,
                search_tool.target_duration_sec,
                iteration,
            ))

        if not tasks:
            return "", []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"[PLAN] Tool call failed: {res}")
                continue
            kind, payload = res
            if kind == "chat":
                context_parts.append(payload)
            elif kind == "clips":
                new_clips.extend(payload)

        new_context = "\n\n".join(context_parts)
        return new_context, new_clips

    async def _run_chat(self, question: str, purpose: str, iteration: int) -> tuple[str, Any]:
        """Ask MaviAgent and return ("chat", context_text).

        PORT NOTE: MaviAgent takes a single ``video_id`` (or None) rather than
        memories.ai's video_nos list. If the project has exactly one video, we
        scope to it; otherwise we pass None and let MaviAgent search across
        all indexed videos (the RAG layer will pull hits from whichever video
        is most relevant).
        """
        await self._emit("tool_call", {
            "iteration": iteration,
            "type": "chat",
            "question": question,
            "purpose": purpose,
        })
        try:
            video_id = self.video_nos[0] if len(self.video_nos) == 1 else None
            trace = await self.mavi_agent.ask(question, video_id=video_id)
            text = trace.answer or ""

            context_block = (
                f"\n--- Chat Q [{iteration}]: {question} ---\n"
                f"Purpose: {purpose}\n"
                f"Answer: {text}\n"
            )
            await self._emit("tool_result", {
                "iteration": iteration,
                "type": "chat",
                "question": question,
                "answer_preview": text[:300],
            })
            return "chat", context_block
        except Exception as e:
            logger.error(f"[PLAN] chat failed: {question!r}: {e}")
            await self._emit("tool_error", {"iteration": iteration, "type": "chat", "error": str(e)})
            return "chat", ""

    async def _run_search(
        self, query: str, purpose: str, target_duration: float, iteration: int
    ) -> tuple[str, List[RetrievedClip]]:
        """Run a Searcher vector search and return ("clips", List[RetrievedClip]).

        PORT NOTE: memories.ai's BY_CLIP mode mapped onto lvmm-core's
        VIDEO_TRANSCRIPT + TRANSCRIPT collections — both carry time-windowed
        text content (visual captions and spoken dialogue respectively), which
        is what the storyboard planning prompt expects to compose with.
        """
        await self._emit("tool_call", {
            "iteration": iteration,
            "type": "search",
            "query": query,
            "purpose": purpose,
        })
        try:
            from lvmm_core.core.retrieval.luci_memory.types import (
                SearchRequest, Collection,
            )
            response = await self.searcher.search(SearchRequest(
                query=query,
                collections=[Collection.VIDEO_TRANSCRIPT, Collection.TRANSCRIPT],
                video_ids=self.video_nos or None,
                top_k=20,
            ))
            clips = self._parse_search_results(response.hits, query, purpose)
            await self._emit("tool_result", {
                "iteration": iteration,
                "type": "search",
                "query": query,
                "clip_count": len(clips),
                "clips": [c.model_dump() for c in clips[:5]],
            })
            return "clips", clips
        except Exception as e:
            logger.error(f"[PLAN] search failed: {query!r}: {e}")
            await self._emit("tool_error", {"iteration": iteration, "type": "search", "error": str(e)})
            return "clips", []

    def _parse_search_results(
        self, hits: List[Any], query: str, purpose: str
    ) -> List[RetrievedClip]:
        """Convert lvmm-core Hit objects into RetrievedClip records.

        ``hits`` is a list of ``lvmm_core.core.retrieval.luci_memory.types.Hit``
        with fields: video_id, start_time, end_time, text, score, collection.
        """
        clips: List[RetrievedClip] = []
        for h in hits or []:
            try:
                video_no = getattr(h, "video_id", "") or ""
                entry = self._video_map.get(video_no)
                start = float(getattr(h, "start_time", None) or 0)
                end = float(getattr(h, "end_time", None) or 0)
                score = float(getattr(h, "score", 0) or 0)
                desc = getattr(h, "text", "") or ""

                clip = RetrievedClip(
                    video_no=video_no,
                    video_name=entry.video_name if entry else video_no,
                    source_path=entry.source_path if entry else "",
                    start_seconds=start,
                    end_seconds=end,
                    score=score,
                    description=desc,
                    shot_query=query,
                )
                clips.append(clip)
            except Exception as e:
                logger.warning(f"[PLAN] Failed to parse hit {h}: {e}")
        return clips

    # -------------------------------------------------------------------------
    # Call B
    # -------------------------------------------------------------------------

    async def _call_b_update_storyboard(
        self,
        iteration: int,
        storyboard: Storyboard,
        accumulated_clips: List[RetrievedClip],
        accumulated_context: str,
    ) -> Storyboard:
        gist = self._load_gist()
        clips_detail = self._format_clips_detail(accumulated_clips)
        storyboard_json = storyboard.model_dump_json(indent=2)

        user_content = UPDATE_STORYBOARD_USER.format(
            user_prompt=self.user_prompt,
            gist=gist,
            accumulated_context=accumulated_context or "(none yet)",
            iteration=iteration,
            storyboard_json=storyboard_json,
            clip_count=len(accumulated_clips),
            clips_detail=clips_detail,
        )

        prompt_contents = [UPDATE_STORYBOARD_SYSTEM, user_content]

        try:
            result, _usage = await self.gemini.generate_structured(
                messages_from_prompts(prompt_contents),
                Storyboard,
            )
            return result
        except Exception as e:
            logger.error(f"[PLAN] Call B failed at iteration {iteration}: {e}")
            # Return existing storyboard unchanged
            return storyboard

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _load_gist(self) -> str:
        try:
            session = self.workspace.load_session()
            return session.gist or "(no gist available)"
        except Exception:
            return "(no gist available)"

    def _format_clips_summary(self, clips: List[RetrievedClip]) -> str:
        if not clips:
            return "(none yet)"
        lines = []
        for c in clips[:20]:
            lines.append(
                f"  [{c.video_name} {c.start_seconds:.1f}s–{c.end_seconds:.1f}s "
                f"score={c.score:.2f}] {c.description[:80]}"
            )
        if len(clips) > 20:
            lines.append(f"  ... and {len(clips) - 20} more")
        return "\n".join(lines)

    def _format_clips_detail(self, clips: List[RetrievedClip]) -> str:
        if not clips:
            return "(none)"
        lines = []
        for i, c in enumerate(clips):
            lines.append(
                f"[{i}] video={c.video_name} query={c.shot_query!r}\n"
                f"    time={c.start_seconds:.2f}s–{c.end_seconds:.2f}s "
                f"dur={c.duration_seconds:.1f}s score={c.score:.3f}\n"
                f"    {c.description[:120]}"
            )
        return "\n".join(lines)

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_queue is not None:
            event = PlanningEvent(event_type=event_type, data=data)
            await self.event_queue.put(event)

    async def _handle_pause_and_inject(self) -> None:
        """Wait if pause_event is set; consume any injected prompts."""
        if self.pause_event and self.pause_event.is_set():
            logger.info("[PLAN] Paused — waiting for resume...")
            await self._emit("paused", {})
            await asyncio.sleep(0.5)
            while self.pause_event.is_set():
                await asyncio.sleep(0.5)
            await self._emit("resumed", {})

        if self.inject_prompt_queue:
            injected: List[str] = []
            while not self.inject_prompt_queue.empty():
                try:
                    prompt = self.inject_prompt_queue.get_nowait()
                    injected.append(prompt)
                except asyncio.QueueEmpty:
                    break
            if injected:
                extra = " Also: " + " ".join(injected)
                self.user_prompt += extra
                logger.info(f"[PLAN] Injected {len(injected)} user prompt(s)")
                await self._emit("prompt_injected", {"added": injected})
