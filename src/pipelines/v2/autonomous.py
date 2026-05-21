"""
Single-source-of-truth orchestrator for V2's autonomous mode.

Three things call ``run_autonomous()``: the pytest smoke tests
(``tests/v2/test_smoke_autonomous.py``), the ad-hoc CLI shell
(``scripts/run_autonomous.py``), and any future FastAPI route that
needs headless invocation. Keeping the orchestration in one function
avoids drift between "how the smoke runs the pipeline" and "how a
researcher / production caller does."

Phases driven (in order):

* **Phase 1** — :class:`LightweightComprehension`. Indexes each video in
  the workspace footage dir via lvmm-core's master_indexing pipeline,
  populates the vector DB with transcript embeddings, then asks
  MaviAgent for a per-video gist. Saves ``session.json``.

* **Phase 2** — :class:`IterativePlanningLoop`. Up to ``max_iterations``
  passes of (decide tool calls → execute searches/chats → update
  storyboard). Saves ``storyboard.json``, ``clips.json``, ``context.md``.

* **Phase 3** — deterministic post-processing (opt-in via
  ``include_phase3``): FCPXML compile, narration TTS, music selection,
  draft MP4 render. Each step is best-effort: missing API keys skip
  cleanly with a warning rather than failing the whole run.

Observability: every intermediate model output (gist text, planning
Call A reasoning, per-iteration chat answers, retrieved clip
descriptions, storyboard summaries) is logged at INFO. The optional
``recorder`` callback / JSONL sink captures structured records of each
LLM interaction so runs are reproducible and diff-able after the fact.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from src import services
from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension
from src.pipelines.v2.planning.iterative_planning_loop import (
    IterativePlanningLoop,
    PlanningEvent,
)
from src.pipelines.v2.schemas import SessionData, Storyboard
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)
SMOKE_LOG = "[AUTO]"  # log-prefix so grep "[AUTO]" filters orchestrator output


# ---------------------------------------------------------------------------
# Structured per-run record (for the JSONL recorder + test assertions)
# ---------------------------------------------------------------------------


@dataclass
class InteractionRecord:
    """One observable event during an autonomous run.

    Phase 1 emits: ``phase1_index_start``, ``phase1_index_done``,
    ``phase1_gist_start``, ``phase1_gist_done``.

    Phase 2 emits whatever IterativePlanningLoop sends through its
    event_queue (``iteration_start``, ``tool_call_plan``, ``tool_call``,
    ``tool_result``, ``storyboard_update``, ``done``, etc.) plus our own
    ``phase2_start`` / ``phase2_done`` bookends.

    Phase 3 emits ``phase3_fcpxml_*`` / ``phase3_narration_*`` etc.
    """

    timestamp: float
    event: str
    data: dict[str, Any] = field(default_factory=dict)


class JsonlRecorder:
    """Append-only JSONL writer for InteractionRecord objects.

    Also keeps an in-memory list of records so tests can assert on what
    happened without re-parsing the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[InteractionRecord] = []

    def record(self, event: str, data: dict[str, Any] | None = None) -> None:
        rec = InteractionRecord(timestamp=time.time(), event=event, data=data or {})
        self.records.append(rec)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            # Never let recording break the actual run.
            logger.warning(f"{SMOKE_LOG} JSONL recorder write failed: {e}")


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class AutonomousResult:
    """Artifacts + observations produced by one autonomous run."""

    workspace_root: Path
    session: Optional[SessionData] = None
    storyboard: Optional[Storyboard] = None
    fcpxml_path: Optional[Path] = None
    narration_audio_path: Optional[Path] = None
    music_path: Optional[Path] = None
    draft_mp4_path: Optional[Path] = None
    interactions_jsonl: Optional[Path] = None
    interactions: list[InteractionRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # e.g. ["narration: no key"]


# ---------------------------------------------------------------------------
# Public API — what callers (pytest / CLI / future FastAPI) invoke
# ---------------------------------------------------------------------------


async def run_autonomous(
    *,
    project_name: str,
    user_prompt: str,
    source_dir: Optional[Path] = None,
    workspaces_dir: Optional[Path] = None,
    max_iterations: int = 3,
    target_duration_seconds: float = 60.0,
    include_phase3: bool = True,
    stop_after_phase: int = 3,
    record_to: Optional[Path] = None,
    start_fresh: bool = False,
) -> AutonomousResult:
    """Drive Phase 1 + 2 (+ optional 3) for one workspace, end to end.

    Parameters
    ----------
    project_name:
        Logical project name (= workspace dir under ``workspaces_dir``).
    user_prompt:
        Editing brief passed to the planning loop.
    source_dir:
        Directory holding source videos. Defaults to the workspace's
        ``footage/`` subdir (standard VEA layout).
    workspaces_dir:
        Root for workspaces. Defaults to ``src.config.WORKSPACES_DIR``.
    max_iterations:
        Cap on Phase 2 planning iterations.
    target_duration_seconds:
        Target edit duration (passed to storyboard prompt).
    include_phase3:
        Run Phase 3 (FCPXML compile + narration + music + draft render)
        after planning. Set ``False`` to stop at storyboard. Ignored if
        ``stop_after_phase`` is < 3.
    stop_after_phase:
        Hard cap on phases run. 1 = comprehension only (Smoke 1 path);
        2 = comprehension + planning; 3 = full pipeline (default).
        Mostly for tests that want to verify one phase in isolation.
    record_to:
        Optional path for a JSONL log of every observable event. If
        ``None``, records go to ``{workspace}/artifacts/run_{ts}.jsonl``.
    start_fresh:
        Re-index even if a cached session exists (Phase 1 pass-through).

    Returns
    -------
    AutonomousResult
        Artifact paths + in-memory event log. Tests assert on this.
    """
    # Late import to keep module-load cost down for ``--help`` paths
    from src import config as _config

    workspaces_dir = workspaces_dir or _config.WORKSPACES_DIR
    workspace = WorkspaceManager(project_name, workspaces_dir)
    workspace.create()

    # Recorder lives in the workspace by default; tests can pin a fixed path.
    if record_to is None:
        record_to = workspace.root / "artifacts" / f"run_{int(time.time())}.jsonl"
    recorder = JsonlRecorder(record_to)

    result = AutonomousResult(
        workspace_root=workspace.root,
        interactions_jsonl=record_to,
        interactions=recorder.records,  # live ref — pop from same list
    )

    if source_dir is None:
        source_dir = workspace.get_footage_dir()
    source_dir = Path(source_dir)

    logger.info(f"{SMOKE_LOG} ─── Autonomous run starting ───")
    logger.info(f"{SMOKE_LOG} project       = {project_name}")
    logger.info(f"{SMOKE_LOG} workspace     = {workspace.root}")
    logger.info(f"{SMOKE_LOG} source_dir    = {source_dir}")
    logger.info(f"{SMOKE_LOG} prompt        = {user_prompt[:200]}")
    logger.info(f"{SMOKE_LOG} max_iters     = {max_iterations}")
    logger.info(f"{SMOKE_LOG} target_dur    = {target_duration_seconds}s")
    logger.info(f"{SMOKE_LOG} include_phase3= {include_phase3}")
    logger.info(f"{SMOKE_LOG} jsonl_log     = {record_to}")
    recorder.record("run_start", {
        "project_name": project_name,
        "user_prompt": user_prompt,
        "max_iterations": max_iterations,
        "target_duration_seconds": target_duration_seconds,
        "include_phase3": include_phase3,
    })

    # --- Initialise lvmm-core (once per run) ---
    await services.init_lvmm()
    if services.mavi_agent is None or services.searcher is None:
        raise RuntimeError(
            "lvmm-core not initialised. Set OPENROUTER_API_KEY (preferred) "
            "or GEMINI_API_KEY before invoking run_autonomous()."
        )
    if services.gemini_manager is None:
        raise RuntimeError(
            "VEA's LLM client (gemini_manager) not initialised. Same env "
            "vars as above."
        )

    try:
        # --- Phase 1 -----------------------------------------------------
        session = await _run_phase1(
            workspace=workspace,
            source_dir=source_dir,
            start_fresh=start_fresh,
            recorder=recorder,
        )
        result.session = session

        if not session.videos:
            raise RuntimeError(f"{SMOKE_LOG} Phase 1 produced no indexed videos.")

        if stop_after_phase < 2:
            logger.info(f"{SMOKE_LOG} stop_after_phase=1 → returning after Phase 1")
            recorder.record("run_done", {"stopped_after": "phase1", "videos": len(session.videos)})
            return result

        # --- Phase 2 -----------------------------------------------------
        storyboard = await _run_phase2(
            workspace=workspace,
            user_prompt=user_prompt,
            session=session,
            max_iterations=max_iterations,
            target_duration_seconds=target_duration_seconds,
            recorder=recorder,
        )
        result.storyboard = storyboard

        if stop_after_phase < 3:
            logger.info(f"{SMOKE_LOG} stop_after_phase=2 → returning after Phase 2")
            recorder.record("run_done", {
                "stopped_after": "phase2",
                "videos": len(session.videos),
                "shots": len(storyboard.shots),
            })
            return result

        # --- Phase 3 (best-effort, opt-in) ------------------------------
        if include_phase3:
            await _run_phase3(workspace=workspace, result=result, recorder=recorder)

        recorder.record("run_done", {
            "videos": len(session.videos),
            "shots": len(storyboard.shots) if storyboard else 0,
            "fcpxml": str(result.fcpxml_path) if result.fcpxml_path else None,
            "draft_mp4": str(result.draft_mp4_path) if result.draft_mp4_path else None,
            "skipped": result.skipped,
        })
        logger.info(f"{SMOKE_LOG} ─── Autonomous run done ───")
        return result
    finally:
        # Note: do NOT close lvmm here — the FastAPI lifespan owns it.
        # Tests that drive run_autonomous directly should manage lifecycle.
        pass


# ---------------------------------------------------------------------------
# Phase 1 driver — wraps LightweightComprehension with INFO-level logging
# ---------------------------------------------------------------------------


async def _run_phase1(
    *,
    workspace: WorkspaceManager,
    source_dir: Path,
    start_fresh: bool,
    recorder: JsonlRecorder,
) -> SessionData:
    logger.info(f"{SMOKE_LOG} === Phase 1: Comprehension ===")
    recorder.record("phase1_start", {"source_dir": str(source_dir)})

    async def _progress(percent: float, message: str) -> None:
        logger.info(f"{SMOKE_LOG} [P1 {percent:5.1f}%] {message}")
        recorder.record("phase1_progress", {"percent": percent, "message": message})

    comprehension = LightweightComprehension(
        project_name=workspace.root.name,
        source_dir=str(source_dir),
        lvmm_ctx=services.lvmm_ctx,
        mavi_agent=services.mavi_agent,
        workspace=workspace,
    )
    session = await comprehension.run(
        start_fresh=start_fresh, progress_callback=_progress,
    )

    # Per-video summary so the model output is visible, not just counts.
    for v in session.videos:
        gist_preview = (v.gist or "")[:500].replace("\n", " ")
        logger.info(
            f"{SMOKE_LOG} Indexed {v.video_name} → video_id={v.video_no} "
            f"duration={v.duration_seconds}s gist={len(v.gist)} chars"
        )
        logger.info(f"{SMOKE_LOG}   gist preview: {gist_preview}…")
        recorder.record("phase1_video_done", {
            "video_id": v.video_no,
            "video_name": v.video_name,
            "duration_seconds": v.duration_seconds,
            "gist_chars": len(v.gist),
            "gist_text": v.gist,  # full text in the JSONL artifact
        })

    logger.info(
        f"{SMOKE_LOG} Phase 1 done: {len(session.videos)} videos indexed; "
        f"combined gist={len(session.gist)} chars",
    )
    recorder.record("phase1_done", {"video_count": len(session.videos)})
    return session


# ---------------------------------------------------------------------------
# Phase 2 driver — subscribes to IterativePlanningLoop's event_queue
# ---------------------------------------------------------------------------


async def _run_phase2(
    *,
    workspace: WorkspaceManager,
    user_prompt: str,
    session: SessionData,
    max_iterations: int,
    target_duration_seconds: float,
    recorder: JsonlRecorder,
) -> Storyboard:
    logger.info(f"{SMOKE_LOG} === Phase 2: Iterative planning ===")
    recorder.record("phase2_start", {
        "user_prompt": user_prompt,
        "max_iterations": max_iterations,
        "target_duration_seconds": target_duration_seconds,
    })

    event_queue: asyncio.Queue = asyncio.Queue()
    loop_obj = IterativePlanningLoop(
        project_name=workspace.root.name,
        user_prompt=user_prompt,
        workspace=workspace,
        searcher=services.searcher,
        mavi_agent=services.mavi_agent,
        gemini=services.gemini_manager,
        video_nos=[v.video_no for v in session.videos],
        video_entries=session.videos,
        max_iterations=max_iterations,
        target_duration_seconds=target_duration_seconds,
        event_queue=event_queue,
        pause_event=None,
        inject_prompt_queue=None,
    )

    # Subscriber that logs every event at INFO + writes to the JSONL.
    # IterativePlanningLoop emits "done" / "stopped_early" — we use that
    # as the exit signal for the subscriber.
    async def _subscribe() -> None:
        while True:
            ev: PlanningEvent = await event_queue.get()
            _log_phase2_event(ev)
            recorder.record(f"phase2_{ev.event_type}", ev.data)
            if ev.event_type in ("done", "stopped_early", "error"):
                break

    sub_task = asyncio.create_task(_subscribe())
    try:
        storyboard = await loop_obj.run()
    finally:
        # Drain any pending events the loop produced after subscription ended
        sub_task.cancel()
        try:
            await sub_task
        except (asyncio.CancelledError, Exception):
            pass

    total = sum(s.duration_seconds for s in storyboard.shots)
    logger.info(
        f"{SMOKE_LOG} Phase 2 done: {storyboard.iteration} iteration(s), "
        f"{len(storyboard.shots)} shots, {total:.1f}s total"
    )
    if storyboard.theme:
        logger.info(f"{SMOKE_LOG}   theme: {storyboard.theme}")
    if storyboard.narrative_arc:
        logger.info(f"{SMOKE_LOG}   arc:   {storyboard.narrative_arc}")
    recorder.record("phase2_done", {
        "iteration": storyboard.iteration,
        "shots": len(storyboard.shots),
        "total_duration": total,
        "theme": storyboard.theme,
        "narrative_arc": storyboard.narrative_arc,
        "open_questions": storyboard.open_questions,
    })
    return storyboard


def _log_phase2_event(ev: PlanningEvent) -> None:
    """One PlanningEvent → human-readable INFO line(s).

    IterativePlanningLoop already truncates answer text to 300 chars and
    cliper top-5 in tool_result events, so we can log the full data dict
    safely.
    """
    et, d = ev.event_type, ev.data
    if et == "iteration_start":
        logger.info(f"{SMOKE_LOG} ── Iteration {d.get('iteration')}/N ──")
    elif et == "tool_call_plan":
        logger.info(
            f"{SMOKE_LOG} Call A: {len(d.get('chat_calls', []))} chats, "
            f"{len(d.get('search_calls', []))} searches "
            f"(stop={d.get('should_stop', False)})"
        )
        reasoning = (d.get("reasoning") or "")[:500]
        logger.info(f"{SMOKE_LOG}   reasoning: {reasoning}")
        for c in d.get("chat_calls", []):
            logger.info(f"{SMOKE_LOG}   → chat Q: {c.get('question', '')[:200]}")
            logger.info(f"{SMOKE_LOG}            purpose: {c.get('purpose', '')[:120]}")
        for s in d.get("search_calls", []):
            logger.info(f"{SMOKE_LOG}   → search Q: {s.get('query', '')[:200]}")
            logger.info(f"{SMOKE_LOG}              purpose: {s.get('purpose', '')[:120]}")
    elif et == "tool_call":
        # Issued just before execution — not logging here; tool_result has more info.
        pass
    elif et == "tool_result":
        if d.get("type") == "chat":
            logger.info(f"{SMOKE_LOG} ← chat A: {d.get('answer_preview', '')[:300]}")
        elif d.get("type") == "search":
            logger.info(
                f"{SMOKE_LOG} ← search returned {d.get('clip_count', 0)} clips for "
                f"{d.get('query', '')[:80]!r}"
            )
            for i, c in enumerate(d.get("clips", []), 1):
                logger.info(
                    f"{SMOKE_LOG}   [{i}] {c.get('video_name', '?')} "
                    f"{c.get('start_seconds', 0):.1f}-{c.get('end_seconds', 0):.1f}s "
                    f"score={c.get('score', 0):.2f} "
                    f"\"{(c.get('description', '') or '')[:100]}\""
                )
    elif et == "tool_error":
        logger.warning(f"{SMOKE_LOG} ← tool error ({d.get('type')}): {d.get('error', '')[:300]}")
    elif et == "storyboard_update":
        logger.info(
            f"{SMOKE_LOG} Storyboard after iter {d.get('iteration')}: "
            f"{d.get('shots')} shots, {d.get('total_duration', 0):.1f}s, "
            f"open_q={len(d.get('open_questions', []))}"
        )
    elif et == "stopped_early":
        logger.info(f"{SMOKE_LOG} Stopped early at iter {d.get('iteration')}: {d.get('reason')}")
    elif et == "done":
        logger.info(
            f"{SMOKE_LOG} Phase 2 loop done at iter {d.get('iteration')} — "
            f"{d.get('shots')} shots, {d.get('clips')} clips"
        )


# ---------------------------------------------------------------------------
# Phase 3 driver — best-effort post-processing
# ---------------------------------------------------------------------------


async def _run_phase3(
    *,
    workspace: WorkspaceManager,
    result: AutonomousResult,
    recorder: JsonlRecorder,
) -> None:
    """Run Phase 3 steps best-effort: skip cleanly on missing keys.

    Sequence:
      1. FCPXML compile (requires gemini_manager — always available here)
      2. Narration TTS (requires ELEVENLABS_API_KEY)
      3. Music selection (requires OPENROUTER_API_KEY for Lyria 3)
      4. Draft MP4 render (requires ffmpeg on PATH)

    Each step writes its artifact path onto ``result`` and records what
    happened. Failures of optional steps are warnings, not exceptions.
    """
    logger.info(f"{SMOKE_LOG} === Phase 3: Post-processing ===")

    # 1) FCPXML compile — relies on Gemini + the storyboard from Phase 2
    try:
        from src.pipelines.v2.fcpxml.fcpxml_agent import generate_fcpxml
        recorder.record("phase3_fcpxml_start", {})
        fcpxml_path = await generate_fcpxml(services.gemini_manager, workspace)
        result.fcpxml_path = Path(fcpxml_path)
        logger.info(f"{SMOKE_LOG} FCPXML compiled → {result.fcpxml_path}")
        recorder.record("phase3_fcpxml_done", {"path": str(result.fcpxml_path)})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{SMOKE_LOG} FCPXML compile failed: {e}")
        recorder.record("phase3_fcpxml_failed", {"error": str(e)})

    # 2) Narration TTS — optional; needs ELEVENLABS_API_KEY
    if not os.environ.get("ELEVENLABS_API_KEY"):
        logger.info(f"{SMOKE_LOG} Narration skipped (no ELEVENLABS_API_KEY).")
        result.skipped.append("narration: no ELEVENLABS_API_KEY")
        recorder.record("phase3_narration_skipped", {"reason": "no_key"})
    else:
        try:
            from src.pipelines.v2.narration.narration_pipeline import NarrationPipeline
            recorder.record("phase3_narration_start", {})
            session = workspace.load_session()
            user_prompt = " ".join(session.planning.user_prompts) or "Create an engaging edit"
            pipeline = NarrationPipeline(services.gemini_manager, workspace)
            audio_path = await pipeline.run(user_prompt=user_prompt)
            result.narration_audio_path = Path(audio_path) if audio_path else None
            logger.info(f"{SMOKE_LOG} Narration → {result.narration_audio_path}")
            recorder.record("phase3_narration_done", {"path": str(result.narration_audio_path)})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{SMOKE_LOG} Narration failed: {e}")
            result.skipped.append(f"narration: {e}")
            recorder.record("phase3_narration_failed", {"error": str(e)})

    # 3) Music selection — best-effort; needs OPENROUTER_API_KEY for Lyria 3
    try:
        from src.pipelines.v2.music.music_pipeline import MusicPipeline
        recorder.record("phase3_music_start", {})
        session = workspace.load_session()
        user_prompt = " ".join(session.planning.user_prompts) or ""
        pipeline = MusicPipeline(services.gemini_manager, workspace)
        music_path = await pipeline.run(user_prompt=user_prompt)
        result.music_path = Path(music_path) if music_path else None
        if result.music_path:
            logger.info(f"{SMOKE_LOG} Music → {result.music_path}")
            recorder.record("phase3_music_done", {"path": str(result.music_path)})
        else:
            logger.info(f"{SMOKE_LOG} Music skipped by pipeline (likely no key).")
            result.skipped.append("music: pipeline returned None")
            recorder.record("phase3_music_skipped", {"reason": "pipeline_returned_none"})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{SMOKE_LOG} Music failed: {e}")
        result.skipped.append(f"music: {e}")
        recorder.record("phase3_music_failed", {"error": str(e)})

    # 4) Draft render via ffmpeg
    #
    # The renderer takes an EditDecision struct. The agent-tool path
    # (src/pipelines/v2/agent/tools.py) persists ``fcpxml/edit_decision.json``
    # as a side effect of compiling, because the LLM constructs the
    # EditDecision and the tool needs to keep it. The autonomous path
    # (generate_fcpxml → FcpxmlAgent → build_scaffold → FCPXML) goes
    # directly from storyboard/clips to FCPXML without an EditDecision
    # in between, so there's no JSON file to read.
    #
    # Build the EditDecision here from the same storyboard + clips data
    # FcpxmlAgent used. Persist a copy alongside the FCPXML for parity
    # with the agent flow (dashboard / debugging benefit from it too).
    try:
        from src.pipelines.v2.preview.ffmpeg_renderer import render_ffmpeg_preview
        recorder.record("phase3_render_start", {})

        edit = _build_edit_decision_from_workspace(workspace)
        if not edit.clips:
            raise RuntimeError(
                "No clips resolved from storyboard — cannot render. "
                "Storyboard shots have no retrieved_clip and no matching "
                "clip was found in workspace clips.json."
            )

        # Persist for parity with the agent flow (dashboard reads this).
        ed_path = workspace.root / "fcpxml" / "edit_decision.json"
        ed_path.parent.mkdir(parents=True, exist_ok=True)
        ed_path.write_text(edit.model_dump_json(indent=2), encoding="utf-8")

        draft_path = workspace.get_render_path("draft")
        await render_ffmpeg_preview(edit, workspace.get_footage_dir(), Path(draft_path))
        result.draft_mp4_path = Path(draft_path)
        logger.info(f"{SMOKE_LOG} Draft render → {result.draft_mp4_path}")
        recorder.record("phase3_render_done", {"path": str(result.draft_mp4_path)})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{SMOKE_LOG} Draft render failed: {e}")
        result.skipped.append(f"render: {e}")
        recorder.record("phase3_render_failed", {"error": str(e)})

    logger.info(f"{SMOKE_LOG} Phase 3 done. Skipped: {result.skipped or 'none'}")


# ---------------------------------------------------------------------------
# EditDecision construction helper (for the autonomous render path).
# ---------------------------------------------------------------------------

def _build_edit_decision_from_workspace(workspace: WorkspaceManager) -> "EditDecision":
    """Construct an EditDecision from the workspace's storyboard + clips.

    The agent-tool path constructs the EditDecision from LLM tool args and
    persists it to ``fcpxml/edit_decision.json`` before compiling FCPXML.
    The autonomous path (storyboard → FcpxmlAgent → build_scaffold) never
    builds an EditDecision — it goes straight to FCPXML. The draft renderer
    needs an EditDecision, so we reconstruct one here from the same source
    data ``FcpxmlAgent`` used.

    Maps each ``Shot`` to a ``ClipDecision`` using its ``retrieved_clip``
    (or the best-scoring match for its ``search_query`` from the workspace
    clips pool, mirroring ``fcpxml_agent._find_best_clip``). Clip duration
    is capped at the retrieved-clip window so we don't try to play past the
    end of the source.

    Returns a minimal EditDecision: clips on track 1, default timeline
    (1920×1080 @ 24fps), no narration / music / titles. The renderer is
    happy with this; the agent flow wires up narration + music later.
    """
    from src.pipelines.v2.schemas import (
        ClipDecision,
        EditDecision,
        TimelineSettings,
    )
    from src.pipelines.v2.fcpxml.fcpxml_agent import _find_best_clip

    storyboard = workspace.load_storyboard()
    if storyboard is None or not storyboard.shots:
        return EditDecision()

    clips_pool = workspace.load_clips() or []
    clips: list = []
    for shot in storyboard.shots:
        rc = shot.retrieved_clip or _find_best_clip(shot.search_query, clips_pool)
        if rc is None:
            continue
        available = max(0.0, rc.end_seconds - rc.start_seconds)
        duration = min(shot.duration_seconds, available) if available else shot.duration_seconds
        clips.append(ClipDecision(
            id=shot.id,                       # required; one clip per shot
            source_file=Path(rc.source_path).name,
            source_path=rc.source_path,
            source_start=rc.start_seconds,
            source_end=rc.start_seconds + duration,
            track=1,
        ))

    return EditDecision(
        timeline=TimelineSettings(
            name=workspace.project_name,
            fps=24.0,
            width=1920,
            height=1080,
        ),
        clips=clips,
    )
