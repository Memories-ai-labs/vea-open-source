"""Test-asset management for the scenario smoke matrix.

Auto-downloads + ffmpeg-derives source videos needed by
``test_smoke_scenarios.py``. Everything is cached under
``~/lvmm-data/test_videos/`` (alongside lvmm-core's own test fixtures)
and reused across runs — first run pays the download/derive cost,
subsequent runs are instant.

Public surface:

* :func:`ensure_tears_of_steel_720p` — already-present anchor (the
  baseline asset lvmm-core's smoke uses too).
* :func:`ensure_big_buck_bunny` — fetched from Blender's CDN on first
  call. ~10-minute animation, distinct content type from ToS so
  scenario 5 isn't ToS-overfit.
* :func:`ensure_silent_version` — strips audio via ``ffmpeg -an``.
* :func:`ensure_vertical_version` — re-encodes to 9:16 (480×854).
* :func:`ensure_slice` — extracts a [start, start+duration] window.

Derived files live in ``~/lvmm-data/test_videos/derived/`` so the raw
sources stay pristine and a ``rm -rf derived/`` cleanly forces
regeneration.

ffmpeg is required and assumed on PATH (lvmm-core's smoke already
asserts this).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared with lvmm-core's smoke — both repos use the same dir so we don't
# duplicate 70 MB of Tears of Steel between two locations.
TEST_VIDEOS_DIR = Path("~/lvmm-data/test_videos").expanduser()
DERIVED_DIR = TEST_VIDEOS_DIR / "derived"


# ---------------------------------------------------------------------------
# Raw source assets
# ---------------------------------------------------------------------------


TEARS_OF_STEEL_720P = TEST_VIDEOS_DIR / "tears_of_steel_720p.mp4"
TEARS_OF_STEEL_720P_URL = (
    "https://download.blender.org/durian/tears_of_steel/tears_of_steel_720p.mp4"
)
TEARS_OF_STEEL_720P_MIN_BYTES = 50_000_000  # ~73 MB — guard against truncated download

BIG_BUCK_BUNNY_480P = TEST_VIDEOS_DIR / "big_buck_bunny.mp4"
# archive.org's BigBuckBunny_124 item — ~62 MB / 10 min / 720p surround.
# Reliable mirror; both Blender's own CDN and Google's gtv-videos-bucket
# returned 403 (as of 2026-05). archive.org responds with a 302 to a
# dn*.ca.archive.org backend — urllib follows automatically.
BIG_BUCK_BUNNY_480P_URL = (
    "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"
)
BIG_BUCK_BUNNY_480P_MIN_BYTES = 40_000_000  # ~62 MB actual; guard at 40 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffmpeg_required() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not on PATH. brew install ffmpeg (mac) / apt install ffmpeg (linux)."
        )


def _download(url: str, target: Path, min_bytes: int) -> Path:
    """Download a URL to target, with a size guard against truncation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= min_bytes:
        return target

    if target.exists():
        target.unlink()  # truncated — re-download

    logger.info("[ASSETS] Downloading %s → %s", url, target)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    # Many CDNs (archive.org, blender.org, googleapis) reject urllib's default
    # ``Python-urllib/3.x`` User-Agent. Sending a normal-looking UA avoids
    # spurious 403s without making us look malicious.
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (lvmm-core test-asset fetcher)"},
    )
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if tmp.stat().st_size < min_bytes:
            raise RuntimeError(
                f"Downloaded file at {target} looks truncated "
                f"({tmp.stat().st_size} bytes < {min_bytes})."
            )
        tmp.rename(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    logger.info("[ASSETS] Downloaded %s (%d MB)", target.name, target.stat().st_size // 1_000_000)
    return target


def _run_ffmpeg(cmd: list[str], output: Path) -> Path:
    """Run an ffmpeg command and verify the output landed."""
    _ffmpeg_required()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — encode to a .partial sibling, rename on success.
    # Keep the .mp4 extension so ffmpeg can infer the muxer format —
    # use ``foo.tmp.mp4`` not ``foo.mp4.partial``.
    tmp = output.with_name(f"{output.stem}.tmp{output.suffix}")
    full_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + cmd + [str(tmp)]
    logger.debug("[ASSETS] ffmpeg cmd: %s", " ".join(full_cmd))
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}) for {output.name}:\n{result.stderr[:2000]}"
        )
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output for {output.name}")
    tmp.rename(output)
    logger.info("[ASSETS] ffmpeg → %s (%d KB)", output.name, output.stat().st_size // 1024)
    return output


# ---------------------------------------------------------------------------
# Raw-source ensurers
# ---------------------------------------------------------------------------


def ensure_tears_of_steel_720p() -> Path:
    """The 12-minute ToS 720p file. Already on disk via lvmm-core's smoke."""
    return _download(TEARS_OF_STEEL_720P_URL, TEARS_OF_STEEL_720P, TEARS_OF_STEEL_720P_MIN_BYTES)


def ensure_big_buck_bunny() -> Path:
    """Big Buck Bunny 320×180 (~10 min, animation, action-heavy, no dialogue)."""
    return _download(
        BIG_BUCK_BUNNY_480P_URL, BIG_BUCK_BUNNY_480P, BIG_BUCK_BUNNY_480P_MIN_BYTES
    )


# ---------------------------------------------------------------------------
# Derived versions
# ---------------------------------------------------------------------------


def ensure_silent_version(source: Path, *, suffix: str = "_silent") -> Path:
    """Strip audio. Output: <source-stem><suffix>.mp4 in derived/."""
    out = DERIVED_DIR / f"{source.stem}{suffix}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    # -c copy keeps video as-is (no re-encode) → fast + lossless; -an drops audio.
    return _run_ffmpeg(["-i", str(source), "-c", "copy", "-an"], out)


def ensure_vertical_version(
    source: Path,
    *,
    target_width: int = 480,
    target_height: int = 854,
    suffix: str = "_vertical",
) -> Path:
    """Crop-and-scale to 9:16 portrait. Output: <stem><suffix>.mp4 in derived/.

    The crop preserves the center of the frame — for ToS this means we keep
    the actor's face in shot. Source aspect 16:9 → output 9:16 means we
    crop the sides, not the top/bottom.
    """
    out = DERIVED_DIR / f"{source.stem}{suffix}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    # scale: fit smaller dimension to target; crop: center-crop to target W×H.
    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}"
    )
    return _run_ffmpeg(
        ["-i", str(source), "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac"],
        out,
    )


def ensure_slice(
    source: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    name: str,
) -> Path:
    """Extract a [start, start+duration] window. Output: <name>.mp4 in derived/.

    Uses -c copy where possible (fast + lossless) — but starts the slice at
    the nearest preceding keyframe, which can mean ±1s drift. Acceptable
    for smoke; if you need frame-exact, switch to re-encode.
    """
    out = DERIVED_DIR / f"{name}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    return _run_ffmpeg(
        [
            "-ss", str(start_seconds),
            "-i", str(source),
            "-t", str(duration_seconds),
            "-c", "copy",
        ],
        out,
    )


# ---------------------------------------------------------------------------
# Bundles — exactly the asset sets the 5 scenarios need.
# ---------------------------------------------------------------------------


def assets_baseline() -> list[Path]:
    """Scenario 1: single 12-min ToS."""
    return [ensure_tears_of_steel_720p()]


def assets_multi_source() -> list[Path]:
    """Scenario 2: 3× 30-second slices from different time windows.

    Picks ranges that span the film's distinct beats (early scene-setting,
    mid-conflict, late chaos) so the planner has visually different material
    to choose from, not three indistinguishable slices.
    """
    src = ensure_tears_of_steel_720p()
    return [
        ensure_slice(src, start_seconds=30.0, duration_seconds=30.0, name="tos_clip_a_early"),
        ensure_slice(src, start_seconds=240.0, duration_seconds=30.0, name="tos_clip_b_mid"),
        ensure_slice(src, start_seconds=480.0, duration_seconds=30.0, name="tos_clip_c_late"),
    ]


def assets_silent() -> list[Path]:
    """Scenario 3: ToS with audio stripped."""
    return [ensure_silent_version(ensure_tears_of_steel_720p())]


def assets_vertical() -> list[Path]:
    """Scenario 4: ToS re-cropped to 9:16 vertical."""
    return [ensure_vertical_version(ensure_tears_of_steel_720p())]


def assets_big_buck_bunny() -> list[Path]:
    """Scenario 5: BBB — different content type entirely."""
    return [ensure_big_buck_bunny()]
