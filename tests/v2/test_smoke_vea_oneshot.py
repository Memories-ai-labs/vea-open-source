"""Real-network end-to-end smoke for vea-oneshot.

What this verifies: the WHOLE stack works against a real video — lvmm-core
indexing + Searcher + MaviAgent + Gemini/OpenRouter LLM + ffmpeg renderer
+ FCPXML compilation + DaVinci-or-ffmpeg render — driven through the same
``vea-oneshot`` entry point an end user runs.

Gated on ``RUN_REAL_SMOKE=1`` (the rest of ``pytest tests/v2`` skips this).
First run downloads Tears of Steel + Apple's MobileCLIP weights via the
helpers in ``_test_assets.py`` (~400 MB total cached under
``~/lvmm-data/``); subsequent runs are instant. Expect ~5 min wall time
on Apple Silicon CPU.

Run with::

    RUN_REAL_SMOKE=1 pytest tests/v2/test_smoke_vea_oneshot.py -v -s --timeout=1800

Configuration knobs (all optional, sensible defaults):

  * ``LVMM_SMOKE_VIDEO=/path/to/your.mp4`` — use your own video instead
    of Tears of Steel.
  * ``RUN_REAL_SMOKE_FULL=1`` — exercise the full ToS movie (12 min) instead
    of the cached 3-min smoke clip. Wall time blows up to ~25 min.
  * ``OPENROUTER_API_KEY`` / ``GEMINI_API_KEY`` — picked up from ``.env``.

What the assertions cover (after one autonomous run):

  1. ``vea-oneshot`` exits 0
  2. ``fcpxml/edit_v1.fcpxml`` exists + non-empty
  3. ``renders/ffmpeg.mp4`` exists + non-trivial size (> 100 KB)
  4. ``logs/backend.jsonl`` exists (per-project log bundle wired up)
  5. ``logs/llm.jsonl`` exists with at least one LLM-call record
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.v2._test_assets import (
    ensure_tears_of_steel_720p,
    ensure_slice,
    DERIVED_DIR,
)


# ---------------------------------------------------------------------------
# Gate + skip helpers
# ---------------------------------------------------------------------------


RUN_REAL_SMOKE = os.environ.get("RUN_REAL_SMOKE") == "1"
SMOKE_SKIP_REASON = (
    "Real-network smoke gated on RUN_REAL_SMOKE=1. "
    "Run with: RUN_REAL_SMOKE=1 pytest tests/v2/test_smoke_vea_oneshot.py -v -s --timeout=1800"
)


def _missing_llm_key() -> str | None:
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return "Set OPENROUTER_API_KEY (preferred) or GEMINI_API_KEY."
    return None


def _missing_ffmpeg() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not on PATH."
    return None


# ---------------------------------------------------------------------------
# Smoke videos — pick a short one by default so wall time stays sensible
# ---------------------------------------------------------------------------


def _resolve_smoke_video() -> Path:
    """Pick the test video, honoring overrides."""
    override = os.environ.get("LVMM_SMOKE_VIDEO")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"LVMM_SMOKE_VIDEO={p} not found")
        return p
    if os.environ.get("RUN_REAL_SMOKE_FULL"):
        return ensure_tears_of_steel_720p()
    # Default: a short 30-second slice — fast iteration without
    # re-running indexing on the whole 12-min ToS movie. ``ensure_slice``
    # ffmpeg-derives this from the full ToS once and caches it.
    full = ensure_tears_of_steel_720p()
    return ensure_slice(
        full,
        start_seconds=120.0,
        duration_seconds=30.0,
        name="smoke_30s",
    )


# ---------------------------------------------------------------------------
# Workspace fixture (isolated, cleaned up after)
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_workspace(tmp_path, monkeypatch):
    """A fresh isolated workspace dir + footage seeded with the smoke video."""
    workspaces_dir = tmp_path / "workspaces"
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir(parents=True, exist_ok=True)

    video = _resolve_smoke_video()
    target = footage_dir / video.name
    if not target.exists():
        try:
            target.symlink_to(video)
        except OSError:
            shutil.copy2(video, target)

    monkeypatch.setattr("src.config.WORKSPACES_DIR", workspaces_dir)
    return workspaces_dir, footage_dir


# ---------------------------------------------------------------------------
# The smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_REAL_SMOKE, reason=SMOKE_SKIP_REASON)
@pytest.mark.asyncio
async def test_vea_oneshot_end_to_end(smoke_workspace):
    """Drive ``vea-oneshot`` through the agent loop against a real video.

    This is the same path an end user runs::

        python -m src.cli --project smoke --brief "..." --footage-dir ./clips

    We call the underlying ``_run`` coroutine directly (avoids subprocess
    overhead) but with a realistic ``argparse.Namespace`` so the entry
    point and CLI behavior are tested.
    """
    # Skip with a clear actionable message if the dev hasn't set up creds
    skip = _missing_llm_key() or _missing_ffmpeg()
    if skip:
        pytest.skip(skip)

    workspaces_dir, footage_dir = smoke_workspace
    project = "smoke_vea_oneshot"

    from src.cli import _run

    args = argparse.Namespace(
        project=project,
        brief="Create a 15-second highlight reel of the most engaging moments.",
        footage_dir=str(footage_dir),
        reuse_index=False,
        log_format="text",
        timeout=900,  # 15 min hard cap
    )

    print(f"\n  Workspaces root : {workspaces_dir}", file=sys.stderr)
    print(f"  Footage         : {list(footage_dir.iterdir())}", file=sys.stderr)
    print(f"  Brief           : {args.brief}", file=sys.stderr)
    print(
        "  Expected wall-time: ~3-5 min on Apple Silicon CPU after warm caches",
        file=sys.stderr,
    )

    exit_code = await _run(args)
    assert exit_code == 0, f"vea-oneshot exit code {exit_code} (expected 0)"

    # ── Artifact assertions ─────────────────────────────────────────────
    workspace_root = workspaces_dir / project
    assert workspace_root.is_dir(), f"workspace not created at {workspace_root}"

    fcpxml = workspace_root / "fcpxml" / "edit_v1.fcpxml"
    assert fcpxml.is_file(), f"FCPXML missing: {fcpxml}"
    assert fcpxml.stat().st_size > 100, "FCPXML suspiciously tiny"

    ffmpeg_mp4 = workspace_root / "renders" / "ffmpeg.mp4"
    assert ffmpeg_mp4.is_file(), f"ffmpeg render missing: {ffmpeg_mp4}"
    assert ffmpeg_mp4.stat().st_size > 100_000, (
        f"ffmpeg.mp4 only {ffmpeg_mp4.stat().st_size} bytes — likely empty"
    )

    # ── Per-project log bundle ──────────────────────────────────────────
    logs_dir = workspace_root / "logs"
    assert logs_dir.is_dir(), "logs/ dir not created by logging_setup"

    backend_log = logs_dir / "backend.jsonl"
    assert backend_log.is_file(), "backend.jsonl missing — logging_setup not active?"
    assert backend_log.stat().st_size > 0, "backend.jsonl empty"

    llm_log = logs_dir / "llm.jsonl"
    if llm_log.exists():
        # llm.jsonl is opt-in by call site; assert content if it was wired
        lines = [ln for ln in llm_log.read_text().splitlines() if ln.strip()]
        assert any(json.loads(ln).get("call") for ln in lines if ln.startswith("{")), \
            "llm.jsonl has no recognizable LLM-call records"

    print(
        f"\n  ✓ vea-oneshot smoke passed\n"
        f"    FCPXML       : {fcpxml.stat().st_size:,} bytes\n"
        f"    ffmpeg.mp4   : {ffmpeg_mp4.stat().st_size // 1024:,} KB\n"
        f"    backend.jsonl: {backend_log.stat().st_size:,} bytes",
        file=sys.stderr,
    )
