"""Frame-rate-aware timing helpers.

Every source video has a fixed frame rate — 23.976, 24, 29.97, 30, 60, etc.
The timeline also has its own fps. When the two differ, every clip undergoes
frame-rate conversion, which can shift audio/video alignment by up to ±1
output frame at clip boundaries (≈42ms at 24fps). The fix at the root is to
match the timeline fps to the dominant source fps, then snap every decimal-
second timestamp to a frame boundary before it reaches ffmpeg or the FCPXML
compiler.

Both the FFmpeg renderer and the FCPXML compiler import from this module so
they agree on where each frame lives.
"""
from __future__ import annotations

import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def snap_to_frame(seconds: float, fps: float) -> float:
    """Return ``seconds`` rounded to the nearest frame boundary at ``fps``.

    ``fps`` should be the exact rational rate (use 30000/1001 for 29.97,
    not 29.97 itself) so NTSC material doesn't drift across long timelines.
    """
    if fps <= 0:
        return float(seconds)
    return round(seconds * fps) / fps


def probe_fps(path: Path) -> Optional[float]:
    """Read ``avg_frame_rate`` via ffprobe. Returns None on any failure.

    ffprobe emits rationals like "30000/1001" (NTSC) or "24/1" (24p). We
    return the float value — callers that need exact rationals should parse
    the string themselves.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        raw = (out.stdout or "").strip()
        if not raw or raw == "0/0":
            return None
        if "/" in raw:
            num_s, den_s = raw.split("/", 1)
            num, den = int(num_s), int(den_s)
            if den == 0:
                return None
            return num / den
        return float(raw)
    except Exception as e:
        logger.warning(f"[TIMING] Could not probe fps for {path}: {e}")
        return None


def dominant_fps(paths: Iterable[Path]) -> Optional[float]:
    """Pick the most common fps across ``paths``, or None if none probed.

    Values are bucketed to two decimal places so 29.97 and 29.9700029997
    (floating-point of 30000/1001) group together.
    """
    samples = []
    for p in paths:
        f = probe_fps(Path(p))
        if f and f > 0:
            samples.append(f)
    if not samples:
        return None
    buckets: Counter = Counter(round(s, 2) for s in samples)
    winner = buckets.most_common(1)[0][0]
    # Return the original (higher-precision) float that matches this bucket
    for s in samples:
        if round(s, 2) == winner:
            return s
    return winner
