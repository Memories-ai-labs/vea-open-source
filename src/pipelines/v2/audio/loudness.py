"""
Measure integrated loudness (LUFS) for audio clips using pyloudnorm.

Extracts the audio segment from a video/audio file via FFmpeg,
then measures integrated LUFS. Results are stored on the clip schema
so the agent can reason about gain adjustments from real data.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target loudness for different content types (EBU R128 / streaming norms)
TARGET_DIALOGUE_LUFS = -16.0
TARGET_MUSIC_LUFS = -18.0
TARGET_NARRATION_LUFS = -16.0


def measure_lufs(
    file_path: str,
    start: float = 0.0,
    end: Optional[float] = None,
    sr: int = 48000,
) -> Optional[float]:
    """
    Measure integrated loudness (LUFS) of an audio segment.

    Args:
        file_path: Path to video or audio file.
        start: Start time in seconds.
        end: End time in seconds (None = to end of file).
        sr: Sample rate for analysis.

    Returns:
        Integrated loudness in LUFS, or None if measurement fails.
    """
    import pyloudnorm

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"[LOUDNESS] File not found: {file_path}")
        return None

    try:
        # Extract audio segment to raw PCM via FFmpeg
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(path),
            "-ss", str(start),
        ]
        if end is not None:
            cmd += ["-t", str(end - start)]
        cmd += [
            "-vn",  # no video
            "-ac", "1",  # mono (LUFS is channel-independent for integrated)
            "-ar", str(sr),
            "-f", "f32le",  # raw 32-bit float PCM
            "pipe:1",
        ]

        proc = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            logger.warning(f"[LOUDNESS] FFmpeg failed for {path.name}: {proc.stderr.decode()[:200]}")
            return None

        # Convert raw PCM to numpy array
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(audio) < sr * 0.1:  # less than 100ms of audio
            logger.warning(f"[LOUDNESS] Audio too short for {path.name} ({len(audio)} samples)")
            return None

        # Measure integrated loudness
        meter = pyloudnorm.Meter(sr)
        # pyloudnorm expects (samples, channels) shape
        loudness = meter.integrated_loudness(audio.reshape(-1, 1))

        if np.isinf(loudness) or np.isnan(loudness):
            logger.warning(f"[LOUDNESS] Invalid measurement for {path.name}: {loudness}")
            return None

        return round(float(loudness), 1)

    except Exception as e:
        logger.warning(f"[LOUDNESS] Measurement failed for {path.name}: {e}")
        return None


def measure_clip_loudness(
    source_path: str,
    source_start: float,
    source_end: float,
) -> Optional[float]:
    """Convenience wrapper for measuring a clip's loudness."""
    return measure_lufs(source_path, start=source_start, end=source_end)


def measure_music_loudness(
    file_path: str,
    start: float = 0.0,
    duration: float = 0.0,
) -> Optional[float]:
    """Measure loudness of a music track segment."""
    end = (start + duration) if duration > 0 else None
    return measure_lufs(file_path, start=start, end=end)


def measure_edit_loudness(
    edit,
    footage_dir: Path,
) -> dict:
    """
    Measure loudness of every audio source in an EditDecision.

    Mutates the edit in place (sets measured_loudness_lufs on clips, narration, music).
    Returns a summary dict for logging/reporting.

    Args:
        edit: EditDecision instance (mutated in place).
        footage_dir: Directory containing source footage files.

    Returns:
        Dict with per-clip, narration, and music loudness measurements.
    """
    summary = {"clips": [], "narration": [], "music": None}

    for clip in edit.clips:
        source = Path(clip.source_path) if clip.source_path else footage_dir / clip.source_file
        if source.exists():
            lufs = measure_lufs(str(source), start=clip.source_start, end=clip.source_end)
            clip.measured_loudness_lufs = lufs
            summary["clips"].append({
                "id": clip.id,
                "label": clip.label,
                "lufs": lufs,
                "gain_db": clip.gain_db,
            })
            logger.info(f"[LOUDNESS] Clip {clip.id} ({clip.label}): {lufs} LUFS")
        else:
            summary["clips"].append({"id": clip.id, "label": clip.label, "lufs": None})

    for seg in edit.narration:
        p = Path(seg.file)
        if p.exists():
            lufs = measure_lufs(str(p), start=seg.start, end=seg.start + seg.duration)
            seg.measured_loudness_lufs = lufs
            summary["narration"].append({"lufs": lufs, "gain_db": seg.gain_db})
            logger.info(f"[LOUDNESS] Narration segment: {lufs} LUFS")

    if edit.music and edit.music.file:
        p = Path(edit.music.file)
        if p.exists():
            end = (edit.music.start + edit.music.duration) if edit.music.duration > 0 else None
            lufs = measure_lufs(str(p), start=edit.music.start, end=end)
            edit.music.measured_loudness_lufs = lufs
            summary["music"] = {"lufs": lufs, "gain_db": edit.music.gain_db}
            logger.info(f"[LOUDNESS] Music: {lufs} LUFS")

    return summary


def suggest_gain_db(
    measured_lufs: float,
    target_lufs: float = TARGET_DIALOGUE_LUFS,
) -> float:
    """
    Suggest a gain_db adjustment to bring measured loudness to target.

    Returns the difference (target - measured) clamped to a reasonable range.
    """
    adjustment = target_lufs - measured_lufs
    # Clamp to avoid extreme adjustments
    return round(max(-40.0, min(20.0, adjustment)), 1)
