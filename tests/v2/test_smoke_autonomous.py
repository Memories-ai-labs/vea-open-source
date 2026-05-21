"""End-to-end smoke tests for the VEA → lvmm-core port.

Two tests, both gated on ``RUN_REAL_SMOKE=1``:

1. ``test_phase1_comprehension_end_to_end`` — uploads (well, indexes
   locally) a real test video and confirms the comprehension phase
   produced a video_id, populated the vector DB, and yielded a non-
   empty gist via MaviAgent.

2. ``test_full_autonomous_pipeline`` — runs the same Phase 1 + the
   planning loop + the deterministic post-processing all the way to a
   draft MP4. Asserts at each phase that the expected artifacts landed.

Both invoke ``run_autonomous()`` from ``src/pipelines/v2/autonomous.py``
— the SAME production function the CLI shell and any future FastAPI
``/v2/autonomous`` endpoint would use. The smoke is the in-repo
canonical "does the port work end-to-end" check.

Observability: every model interaction (gist text, planner reasoning,
each chat answer, each retrieval hit's text + timestamps + score,
storyboard summaries) streams to stdout via the ``[AUTO]``-prefixed
logger lines AND lands in a JSONL artifact file in the workspace's
``artifacts/`` dir. After the run you can ``jq`` through the JSONL to
see exactly what the LLMs produced.

Run with::

    RUN_REAL_SMOKE=1 uv run pytest tests/v2/test_smoke_autonomous.py -v -s --log-cli-level=INFO

Requires (asserted by skipif decorators per-test):
  * OPENROUTER_API_KEY or GEMINI_API_KEY in env (for lvmm-core + VEA LLM)
  * ffmpeg on PATH (for master_indexing + draft renderer)
  * lvmm-core installed as a path dep (uv sync handles this)
  * Tears of Steel 720p on disk at ~/lvmm-data/test_videos/
    (auto-downloaded by lvmm-core's own smoke; this conftest skips if
    absent and prints a curl one-liner)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pytest

from src.pipelines.v2.autonomous import AutonomousResult, run_autonomous
from tests.v2.conftest import RUN_REAL_SMOKE, SMOKE_SKIP_REASON


logger = logging.getLogger("smoke")


# ---------------------------------------------------------------------------
# Skipif helpers — each test has its own gate so partial setups skip cleanly
# ---------------------------------------------------------------------------


def _missing_llm_key() -> str | None:
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return "Set OPENROUTER_API_KEY (preferred) or GEMINI_API_KEY."
    return None


def _missing_ffmpeg() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not on PATH. brew install ffmpeg (mac) / apt install ffmpeg (linux)."
    return None


# ---------------------------------------------------------------------------
# Smoke 1 — Phase 1 only (comprehension)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason=SMOKE_SKIP_REASON)
async def test_phase1_comprehension_end_to_end(smoke_workspace, smoke_video_path):
    """Upload footage (local indexing) and verify the comprehension phase.

    What this asserts:
      * ``run_autonomous(stop_after_phase=1)`` returns successfully
      * Exactly 1 video was indexed (we seeded one)
      * Each video has a non-empty ``video_id`` (= lvmm-core's derived id)
      * Each video has a non-empty ``gist`` text (MaviAgent.ask succeeded)
      * The ``session.json`` artifact exists on disk
      * The interaction JSONL contains a ``phase1_done`` event

    What's in the live log output:
      * Per-stage timing from lvmm-core's master_indexing DAG
      * The full gist text per video
      * HTTP requests to OpenRouter / Gemini
    """
    skip = _missing_llm_key() or _missing_ffmpeg()
    if skip:
        pytest.skip(skip)

    workspaces_dir = smoke_workspace
    logger.info("=" * 70)
    logger.info("SMOKE 1: Phase 1 comprehension only")
    logger.info(f"  workspace: {workspaces_dir / 'smoke_test'}")
    logger.info(f"  video:     {smoke_video_path.name}")
    logger.info("=" * 70)

    result: AutonomousResult = await run_autonomous(
        project_name="smoke_test",
        user_prompt="(comprehension only — no planning)",
        workspaces_dir=workspaces_dir,
        max_iterations=1,                  # ignored at stop_after=1 but kept honest
        target_duration_seconds=30.0,
        include_phase3=False,
        stop_after_phase=1,                # ← key: comprehension only
        start_fresh=True,                  # don't reuse any stray cached session
    )

    # --- Assertions ------------------------------------------------------
    assert result.session is not None, "Phase 1 should have produced a session"
    assert len(result.session.videos) == 1, (
        f"Expected 1 indexed video, got {len(result.session.videos)}"
    )
    v = result.session.videos[0]
    assert v.video_no, "video_id (video_no) should be non-empty after master_indexing"
    assert v.gist, "Gist should be non-empty — MaviAgent.ask must have returned text"
    assert len(v.gist) > 50, (
        f"Gist looks suspiciously short ({len(v.gist)} chars). MaviAgent may have "
        f"refused or returned a stub. First 200 chars: {v.gist[:200]!r}"
    )

    # session.json on disk
    session_path = workspaces_dir / "smoke_test" / "session.json"
    assert session_path.is_file(), f"session.json missing at {session_path}"
    assert session_path.stat().st_size > 0

    # interaction JSONL
    assert result.interactions_jsonl is not None
    assert result.interactions_jsonl.is_file()
    events = [r.event for r in result.interactions]
    assert "phase1_start" in events
    assert "phase1_done" in events
    assert "phase1_video_done" in events

    # Phase 2 / 3 must NOT have run
    assert "phase2_start" not in events, "stop_after_phase=1 leaked into Phase 2"
    assert result.storyboard is None
    assert result.fcpxml_path is None

    logger.info(
        "SMOKE 1 PASS — video_id=%s, gist_chars=%d, interactions=%d",
        v.video_no, len(v.gist), len(result.interactions),
    )


# ---------------------------------------------------------------------------
# Smoke 2 — Full autonomous pipeline (Phase 1 + Phase 2 + Phase 3 best-effort)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason=SMOKE_SKIP_REASON)
async def test_full_autonomous_pipeline(smoke_workspace, smoke_video_path):
    """Run the entire VEA V2 autonomous editing flow on a real video.

    A simple editing request ("30-second highlight reel of the most
    dramatic moments") is run end-to-end through Phase 1 (index +
    gist), Phase 2 (planning loop, 2 iterations), and Phase 3 (FCPXML
    compile, narration if ELEVENLABS_API_KEY set, music if available,
    draft MP4 render via ffmpeg).

    What this asserts:
      * Phase 1: session with 1 video + gist (same as Smoke 1)
      * Phase 2: storyboard.json exists, has ≥1 shot
      * Phase 3 FCPXML: file exists at workspace/fcpxml/edit_v*.fcpxml
      * Phase 3 draft render: draft.mp4 exists in workspace/renders/
      * Narration & music: asserted only if their respective keys / deps
        were present (otherwise recorded in result.skipped)

    What's in the live log output:
      * Everything Smoke 1 emits, plus:
      * Per-iteration: planner Call A reasoning (truncated to 500 chars)
      * Per-tool-call: chat question + answer preview, search query +
        top-5 clips with timestamps + scores + text
      * Per-iteration: storyboard delta (shot count, total duration)
      * Phase 3 step starts/ends
    """
    skip = _missing_llm_key() or _missing_ffmpeg()
    if skip:
        pytest.skip(skip)

    workspaces_dir = smoke_workspace
    user_prompt = (
        "Create a 30-second highlight reel of the most dramatic moments. "
        "Prefer scenes with clear character emotion."
    )
    logger.info("=" * 70)
    logger.info("SMOKE 2: Full autonomous pipeline")
    logger.info(f"  workspace: {workspaces_dir / 'smoke_test'}")
    logger.info(f"  video:     {smoke_video_path.name}")
    logger.info(f"  prompt:    {user_prompt}")
    logger.info("=" * 70)

    result: AutonomousResult = await run_autonomous(
        project_name="smoke_test",
        user_prompt=user_prompt,
        workspaces_dir=workspaces_dir,
        max_iterations=2,                   # keep planning short for test runtime
        target_duration_seconds=30.0,
        include_phase3=True,
        stop_after_phase=3,                 # full pipeline
        start_fresh=True,
    )

    # --- Phase 1 assertions ---------------------------------------------
    assert result.session is not None, "Phase 1 must produce a session"
    assert len(result.session.videos) == 1
    v = result.session.videos[0]
    assert v.video_no and v.gist and len(v.gist) > 50

    # --- Phase 2 assertions ---------------------------------------------
    assert result.storyboard is not None, "Phase 2 must produce a storyboard"
    assert len(result.storyboard.shots) >= 1, (
        f"Storyboard has zero shots — planning loop produced no usable plan. "
        f"theme={result.storyboard.theme!r} open_qs={result.storyboard.open_questions}"
    )
    storyboard_path = workspaces_dir / "smoke_test" / "storyboard.json"
    assert storyboard_path.is_file()
    clips_path = workspaces_dir / "smoke_test" / "clips.json"
    assert clips_path.is_file(), "clips.json should have been written by planning loop"

    # --- Phase 3 FCPXML compile (required: only Gemini needed) ----------
    assert result.fcpxml_path is not None and result.fcpxml_path.is_file(), (
        f"FCPXML must compile in Phase 3. Skipped/failed: {result.skipped}"
    )
    assert result.fcpxml_path.stat().st_size > 0

    # --- Phase 3 draft MP4 (required: only ffmpeg needed) ---------------
    assert result.draft_mp4_path is not None and result.draft_mp4_path.is_file(), (
        f"Draft MP4 must render. Skipped/failed: {result.skipped}"
    )
    assert result.draft_mp4_path.stat().st_size > 1024, "Draft MP4 looks empty"

    # --- Phase 3 narration & music — only if their keys were available --
    if os.environ.get("ELEVENLABS_API_KEY"):
        assert result.narration_audio_path is not None, (
            "ELEVENLABS_API_KEY set but narration audio not produced. "
            f"Skipped: {result.skipped}"
        )
    else:
        assert any("narration" in s for s in result.skipped), (
            "Narration should be in result.skipped when no ELEVENLABS_API_KEY"
        )
    # Music is best-effort either way — don't hard-fail on it. Just log.

    # --- Interaction recording ------------------------------------------
    events = [r.event for r in result.interactions]
    for milestone in (
        "phase1_start", "phase1_done",
        "phase2_start", "phase2_done",
        "phase3_fcpxml_done", "phase3_render_done",
    ):
        assert milestone in events, (
            f"Missing milestone event {milestone!r}. Got: {events[:30]}..."
        )

    logger.info(
        "SMOKE 2 PASS — video_id=%s shots=%d total_dur=%.1fs fcpxml=%s draft=%s "
        "interactions=%d skipped=%s",
        v.video_no,
        len(result.storyboard.shots),
        sum(s.duration_seconds for s in result.storyboard.shots),
        result.fcpxml_path.name,
        result.draft_mp4_path.name,
        len(result.interactions),
        result.skipped or "none",
    )
    logger.info("Full event JSONL: %s", result.interactions_jsonl)
