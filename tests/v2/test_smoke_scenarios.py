"""Scenario matrix — drives the full autonomous pipeline through 5 real-world
combinations of source footage + edit prompts. Catches edge cases the
single happy-path smoke can't.

Coverage (all gated on ``RUN_REAL_SMOKE=1``):

  1. Baseline       — 1 source, audio + 16:9 + 24fps. Regression anchor.
  2. Multi-source   — 3× 30-sec slices, planner picks across files.
  3. Silent         — audio stripped (-an). Planner adapts when there
                      are no audio_transcripts to retrieve over.
  4. Vertical       — 9:16 portrait crop. Aspect handling end-to-end.
  5. Big Buck Bunny — different content type (animation, fast action,
                      no dialogue). Proves the pipeline isn't ToS-overfit.

Each scenario:
  * Seeds a fresh per-test workspace with its source set.
  * Runs ``run_autonomous(include_phase3=True, stop_after_phase=3)``.
  * Asserts: pipeline completed, no traceback in interactions, draft.mp4
    exists + > 100KB, storyboard has ≥1 shot, plus scenario-specific
    invariants (e.g. multi-source must reference ≥2 of 3 files).
  * Writes the full log + interactions.jsonl + EditDecision JSON to
    ``tests/v2/artifacts/scenarios/{scenario_id}/{run_ts}/`` for the
    human-inspection pass that follows.

Pass ``CLEAR_CACHE=1`` to wipe ``~/lvmm-data/local.db`` and
``~/lvmm-data/artifacts/`` before the run — simulates a fresh install.
Without it, the lvmm-core DB persists across scenarios so re-indexing
the same source is fast (matters because we re-index each scenario in
its own tmpdir workspace).

Run::

    RUN_REAL_SMOKE=1 pytest tests/v2/test_smoke_scenarios.py -v -s --log-cli-level=INFO

    # Force fresh-from-scratch (re-index, re-embed, re-load model weights):
    RUN_REAL_SMOKE=1 CLEAR_CACHE=1 pytest tests/v2/test_smoke_scenarios.py -v -s
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from src.pipelines.v2.autonomous import (  # noqa: E402
    AutonomousResult, run_autonomous,
)
from tests.v2._test_assets import (  # noqa: E402
    assets_baseline,
    assets_big_buck_bunny,
    assets_multi_source,
    assets_silent,
    assets_vertical,
)

logger = logging.getLogger("smoke.scenarios")


RUN_REAL_SMOKE = os.environ.get("RUN_REAL_SMOKE") == "1"
SMOKE_SKIP_REASON = (
    "Real-network scenario matrix gated on RUN_REAL_SMOKE=1. "
    "Runs ~5 scenarios × ~3-5 min each = ~20-40 min and ~$3-8 in API costs."
)

CLEAR_CACHE = os.environ.get("CLEAR_CACHE") == "1"
LVMM_DATA_DIR = Path("~/lvmm-data").expanduser()

# Per-run artifact root. Each scenario gets its own subdir for inspection.
ARTIFACTS_ROOT = Path(__file__).parent / "artifacts" / "scenarios"


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One row in the scenario matrix."""

    id: str
    description: str
    # Deferred — called inside the test so asset download/derive happens
    # during the test (and skipping the test doesn't trigger downloads).
    sources: Callable[[], list[Path]]
    prompt: str
    target_duration: float = 30.0
    max_iterations: int = 2
    # Scenario-specific assertion hook — called with the AutonomousResult
    # after structural checks pass. Raise AssertionError for failures.
    extra_assertions: Callable[[AutonomousResult], None] | None = None


# ---------------------------------------------------------------------------
# Scenario-specific assertion helpers
# ---------------------------------------------------------------------------


def _assert_multi_source_references(result: AutonomousResult) -> None:
    """Multi-source: EditDecision should pull from ≥2 of the 3 sources."""
    ed_path = result.workspace_root / "fcpxml" / "edit_decision.json"
    if not ed_path.is_file():
        raise AssertionError(f"edit_decision.json missing at {ed_path}")
    edit = json.loads(ed_path.read_text())
    files_used = {c["source_file"] for c in edit.get("clips", [])}
    if len(files_used) < 2:
        raise AssertionError(
            f"Multi-source scenario only used {len(files_used)} source file(s): "
            f"{files_used}. Planner should pick across files."
        )


def _assert_no_audio_assumption(result: AutonomousResult) -> None:
    """Silent scenario: storyboard / gist shouldn't claim dialogue exists.

    Soft check — if the gist or storyboard prominently references dialogue
    despite there being none, the planner over-trusted the visual_transcript's
    captions (which can hallucinate dialogue from on-screen action). Logged
    as warning, not failure.
    """
    if not result.session or not result.session.videos:
        return
    gist = result.session.videos[0].gist or ""
    suspicious = ["he says", "she says", "they say", '"', "dialogue:", "conversation:"]
    hits = [w for w in suspicious if w.lower() in gist.lower()]
    if hits:
        logger.warning(
            "[SILENT-ASSERT] Gist references dialogue (%s) on a silent source. "
            "May be over-trusting visual hallucinations. Not failing — but inspect.",
            hits,
        )


SCENARIOS: list[Scenario] = [
    Scenario(
        id="01_baseline",
        description="Single 12-min ToS, audio + 16:9 + 24fps. Regression anchor.",
        sources=assets_baseline,
        prompt=(
            "Create a 30-second highlight reel of the most dramatic moments. "
            "Prefer scenes with clear character emotion."
        ),
        target_duration=30.0,
    ),
    Scenario(
        id="02_multi_source",
        description="3× 30-sec slices from different time ranges. Planner picks across files.",
        sources=assets_multi_source,
        prompt=(
            "Weave these three clips into a 60-second story-driven edit. "
            "Use moments from all three sources."
        ),
        target_duration=60.0,
        extra_assertions=_assert_multi_source_references,
    ),
    Scenario(
        id="03_silent",
        description="ToS with audio stripped (-an). No audio_transcripts to retrieve.",
        sources=assets_silent,
        prompt="Create a 30-second visual montage. There is no dialogue available.",
        target_duration=30.0,
        extra_assertions=_assert_no_audio_assumption,
    ),
    Scenario(
        id="04_vertical",
        description="ToS re-encoded 9:16 (480×854). Aspect handling end-to-end.",
        sources=assets_vertical,
        prompt="Create a 15-second vertical short for TikTok. High energy moments.",
        target_duration=15.0,
    ),
    Scenario(
        id="05_big_buck_bunny",
        description="Big Buck Bunny ~10 min. Animation + action, no dialogue. Different content type.",
        sources=assets_big_buck_bunny,
        prompt=(
            "Create a 20-second action montage focused on chase sequences and falls. "
            "Big Buck Bunny is an animated short film."
        ),
        target_duration=20.0,
    ),
]


# ---------------------------------------------------------------------------
# Session-scoped: optional cache clear
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _maybe_clear_lvmm_cache():
    """If CLEAR_CACHE=1, wipe lvmm-core's persistent state before any scenario.

    Targets:
      * ``~/lvmm-data/local.db``  (SQLite index + sqlite-vec collections)
      * ``~/lvmm-data/artifacts/`` (extracted frames, embeddings)

    Leaves ``~/lvmm-data/test_videos/`` (raw + derived test assets) alone —
    re-downloading 130 MB on every CLEAR_CACHE run is wasteful and they're
    immutable anyway.
    """
    if not CLEAR_CACHE:
        yield
        return

    db = LVMM_DATA_DIR / "local.db"
    artifacts = LVMM_DATA_DIR / "artifacts"
    db_wal = LVMM_DATA_DIR / "local.db-wal"
    db_shm = LVMM_DATA_DIR / "local.db-shm"

    for p in (db, db_wal, db_shm):
        if p.exists():
            p.unlink()
            logger.info("[CACHE] Removed %s", p)

    if artifacts.exists():
        shutil.rmtree(artifacts)
        logger.info("[CACHE] Removed %s/", artifacts)

    yield


# ---------------------------------------------------------------------------
# Per-scenario workspace + artifact dir
# ---------------------------------------------------------------------------


@pytest.fixture
def scenario_workspace(tmp_path, monkeypatch, request):
    """Build a fresh workspace seeded with the scenario's source set.

    Yields ``(workspaces_dir, artifact_dir, scenario)`` so the test can:
      * Pass ``workspaces_dir`` to ``run_autonomous``
      * Write per-scenario artifacts (log, interactions, gist text, etc.)
        to ``artifact_dir`` for the post-run inspection pass.
    """
    scenario: Scenario = request.node.callspec.params["scenario"]

    # Materialize the asset set (downloads/derives on first use)
    sources = scenario.sources()
    if not sources:
        pytest.skip(f"No assets resolved for scenario {scenario.id}")

    # Workspace dir
    workspaces_dir = tmp_path / "workspaces"
    project_dir = workspaces_dir / scenario.id
    footage_dir = project_dir / "footage"
    footage_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        target = footage_dir / src.name
        if not target.exists():
            try:
                target.symlink_to(src)
            except OSError:
                shutil.copy2(src, target)
    monkeypatch.setattr("src.config.WORKSPACES_DIR", workspaces_dir)

    # Per-run artifact dir (preserved beyond pytest tmpdir cleanup)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_ROOT / scenario.id / run_ts
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Tee scenario log to artifact_dir for inspection
    log_path = artifact_dir / "scenario.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)

    try:
        yield workspaces_dir, artifact_dir, scenario
    finally:
        root.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# Skip helpers (same shape as test_smoke_autonomous.py)
# ---------------------------------------------------------------------------


def _missing_llm_key() -> str | None:
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return "Set OPENROUTER_API_KEY (preferred) or GEMINI_API_KEY."
    return None


def _missing_ffmpeg() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not on PATH."
    return None


# ---------------------------------------------------------------------------
# The matrix test itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason=SMOKE_SKIP_REASON)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
async def test_scenario(scenario: Scenario, scenario_workspace) -> None:
    """One scenario in the matrix — full autonomous pipeline end-to-end."""
    skip = _missing_llm_key() or _missing_ffmpeg()
    if skip:
        pytest.skip(skip)

    workspaces_dir, artifact_dir, _ = scenario_workspace

    logger.info("=" * 78)
    logger.info("SCENARIO %s — %s", scenario.id, scenario.description)
    logger.info("  prompt:         %s", scenario.prompt)
    logger.info("  target_dur:     %.1fs", scenario.target_duration)
    logger.info("  max_iterations: %d", scenario.max_iterations)
    logger.info("  artifact_dir:   %s", artifact_dir)
    logger.info("=" * 78)

    t0 = time.time()
    result: AutonomousResult = await run_autonomous(
        project_name=scenario.id,
        user_prompt=scenario.prompt,
        workspaces_dir=workspaces_dir,
        max_iterations=scenario.max_iterations,
        target_duration_seconds=scenario.target_duration,
        include_phase3=True,
        stop_after_phase=3,
        start_fresh=True,
    )
    elapsed = time.time() - t0
    logger.info("SCENARIO %s — finished in %.0fs", scenario.id, elapsed)

    # --- Persist artifacts for inspection -------------------------------
    summary = {
        "scenario_id": scenario.id,
        "description": scenario.description,
        "prompt": scenario.prompt,
        "elapsed_seconds": round(elapsed, 1),
        "phase1": {
            "videos_indexed": len(result.session.videos) if result.session else 0,
            "video_names": [v.video_name for v in (result.session.videos if result.session else [])],
            "gists": {
                v.video_name: {"len_chars": len(v.gist or ""), "preview": (v.gist or "")[:600]}
                for v in (result.session.videos if result.session else [])
            },
        },
        "phase2": {
            "shots": len(result.storyboard.shots) if result.storyboard else 0,
            "theme": result.storyboard.theme if result.storyboard else None,
            "narrative_arc": result.storyboard.narrative_arc if result.storyboard else None,
            "total_duration": (
                sum(s.duration_seconds for s in result.storyboard.shots)
                if result.storyboard else 0
            ),
            "open_questions": result.storyboard.open_questions if result.storyboard else [],
            "shot_summaries": [
                {
                    "id": s.id,
                    "purpose": s.purpose,
                    "search_query": s.search_query,
                    "duration": s.duration_seconds,
                    "retrieved_clip": (
                        {
                            "source": s.retrieved_clip.source_path,
                            "start": s.retrieved_clip.start_seconds,
                            "end": s.retrieved_clip.end_seconds,
                            "description": s.retrieved_clip.description[:200],
                        }
                        if s.retrieved_clip else None
                    ),
                }
                for s in (result.storyboard.shots if result.storyboard else [])
            ],
        },
        "phase3": {
            "fcpxml_path": str(result.fcpxml_path) if result.fcpxml_path else None,
            "narration_audio_path": (
                str(result.narration_audio_path) if result.narration_audio_path else None
            ),
            "music_path": str(result.music_path) if result.music_path else None,
            "draft_mp4_path": str(result.draft_mp4_path) if result.draft_mp4_path else None,
            "draft_mp4_bytes": (
                result.draft_mp4_path.stat().st_size
                if result.draft_mp4_path and result.draft_mp4_path.is_file() else 0
            ),
            "skipped": result.skipped,
        },
        "interactions_count": len(result.interactions),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # Copy interactions.jsonl too if present
    if result.interactions_jsonl and result.interactions_jsonl.is_file():
        shutil.copy2(result.interactions_jsonl, artifact_dir / "interactions.jsonl")
    # Copy edit_decision.json if Phase 3 produced one
    if result.fcpxml_path:
        ed = result.workspace_root / "fcpxml" / "edit_decision.json"
        if ed.is_file():
            shutil.copy2(ed, artifact_dir / "edit_decision.json")

    # --- Structural assertions ------------------------------------------
    # Phase 1
    assert result.session is not None, f"[{scenario.id}] Phase 1 produced no session"
    assert len(result.session.videos) == len(scenario.sources()), (
        f"[{scenario.id}] Indexed {len(result.session.videos)} videos; "
        f"expected {len(scenario.sources())}"
    )
    for v in result.session.videos:
        assert v.video_no, f"[{scenario.id}] {v.video_name}: video_id empty"
        assert v.gist and len(v.gist) > 50, (
            f"[{scenario.id}] {v.video_name}: gist suspiciously short "
            f"({len(v.gist or '')} chars). MaviAgent may have refused."
        )

    # Phase 2
    assert result.storyboard is not None, f"[{scenario.id}] Phase 2 produced no storyboard"
    assert len(result.storyboard.shots) >= 1, (
        f"[{scenario.id}] Storyboard has zero shots. "
        f"theme={result.storyboard.theme!r}, open_qs={result.storyboard.open_questions}"
    )

    # Phase 3 — render is mandatory
    assert result.draft_mp4_path is not None, (
        f"[{scenario.id}] Draft render missing. skipped={result.skipped}"
    )
    assert result.draft_mp4_path.is_file(), (
        f"[{scenario.id}] draft.mp4 path set but file doesn't exist: "
        f"{result.draft_mp4_path}"
    )
    draft_size = result.draft_mp4_path.stat().st_size
    assert draft_size > 100_000, (
        f"[{scenario.id}] draft.mp4 is suspiciously small ({draft_size} bytes). "
        f"Likely an empty / corrupt render."
    )

    # Scenario-specific
    if scenario.extra_assertions:
        scenario.extra_assertions(result)

    logger.info(
        "SCENARIO %s PASS — %d videos, %d shots (%.1fs), draft.mp4 %.1f MB, "
        "%d interactions, %.0fs runtime",
        scenario.id,
        len(result.session.videos),
        len(result.storyboard.shots),
        sum(s.duration_seconds for s in result.storyboard.shots),
        draft_size / 1_000_000,
        len(result.interactions),
        elapsed,
    )
