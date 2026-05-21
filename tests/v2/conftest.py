"""V2 test fixtures.

The smoke tests (test_smoke_autonomous.py) need:

* A real workspace seeded with a real video (Tears of Steel 720p from
  ``~/lvmm-data/test_videos/`` — already on disk from lvmm-core's own
  smoke test infrastructure; no fresh download).
* Live log streaming so model outputs are visible during the run
  (the default pytest log capture buffers everything until the test
  finishes, defeating the purpose of "wire up logging").
* A ``RUN_REAL_SMOKE=1`` gate so a casual ``pytest`` doesn't fire real
  API calls — matches the pattern from lvmm-core's
  ``tests/test_real_e2e_smoke.py``.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make src/ importable from tests/.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Gating + path constants
# ---------------------------------------------------------------------------

RUN_REAL_SMOKE = os.environ.get("RUN_REAL_SMOKE") == "1"
SMOKE_SKIP_REASON = (
    "Real-network smoke gated on RUN_REAL_SMOKE=1. "
    "Hits OpenRouter + lvmm-core master_indexing; costs tokens + minutes."
)

# Tears of Steel 720p — ~12 min open-movie test asset. Already on disk
# from lvmm-core's smoke test (~/lvmm-data/test_videos/). We copy from
# here into per-test workspace footage dirs.
TEARS_OF_STEEL_720P = Path("~/lvmm-data/test_videos/tears_of_steel_720p.mp4").expanduser()


# ---------------------------------------------------------------------------
# Async marker — pytest-asyncio strict mode would warn otherwise
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-mark async test functions with @pytest.mark.asyncio.

    Avoids forcing every async smoke test to wear the decorator.
    """
    for item in items:
        if hasattr(item, "function") and __import__("inspect").iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def smoke_video_path() -> Path:
    """Path to the Tears of Steel 720p test video.

    Skips the whole smoke if the file isn't on disk. Doesn't auto-download
    — by convention lvmm-core's smoke owns the download; if you haven't
    run it yet, do so once and the file lands in the shared location.
    """
    if not TEARS_OF_STEEL_720P.is_file():
        pytest.skip(
            f"Test video not found at {TEARS_OF_STEEL_720P}. "
            f"Run lvmm-core's smoke once to auto-download, or fetch manually:\n"
            f"  curl -L -o {TEARS_OF_STEEL_720P} https://download.blender.org/"
            f"durian/tears_of_steel/tears_of_steel_720p.mp4"
        )
    return TEARS_OF_STEEL_720P


@pytest.fixture
def smoke_workspace(tmp_path, smoke_video_path, monkeypatch) -> Path:
    """Build a fresh workspace under tmp_path with the test video in place.

    Returns ``workspaces_dir`` (the parent dir). The workspace itself
    lives at ``{tmp_path}/workspaces/smoke_test/`` with the video copied
    into ``footage/``. Monkeypatches ``src.config.WORKSPACES_DIR`` so
    code that constructs WorkspaceManager picks up the temp location.
    """
    workspaces_dir = tmp_path / "workspaces"
    project_dir = workspaces_dir / "smoke_test"
    footage_dir = project_dir / "footage"
    footage_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the video so we don't waste disk on a copy (Tears of Steel
    # 720p is ~75 MB; multiplying across test runs gets old fast).
    target = footage_dir / smoke_video_path.name
    if not target.exists():
        try:
            target.symlink_to(smoke_video_path)
        except OSError:
            # Fallback to copy on systems where symlinks aren't allowed
            shutil.copy2(smoke_video_path, target)

    monkeypatch.setattr("src.config.WORKSPACES_DIR", workspaces_dir)
    return workspaces_dir


@pytest.fixture(autouse=True)
def _propagate_lvmm_logs():
    """Let lvmm-core's logs stream through pytest's log-cli capture.

    PR #11 ships a structured-logging setup that disables propagation on
    the ``lvmm_core`` logger so foreign handlers don't double-emit. That
    same setting hides lvmm-core logs from pytest's caplog / log-cli
    pipeline. For tests we want the opposite — enable propagation so
    everything flows through pytest.
    """
    target = logging.getLogger("lvmm_core")
    prev = target.propagate
    target.propagate = True
    yield
    target.propagate = prev
