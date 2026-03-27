"""
Beat synchronization — snap clip cut points to music beats.

V2 equivalent of v1's TimelineConstructor.snap_clips_to_music_beats().
Instead of re-encoding with speed changes, this adjusts clip source_start/source_end
in the EditDecision so FCPXML cut points align with music beats.

Usage:
    from src.pipelines.v2.music.beat_sync import snap_to_beats
    modified_clips = snap_to_beats(edit_decision.clips, music_path)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def detect_beats(music_path: str, sr: int = 22050) -> np.ndarray:
    """
    Detect beat times in a music file using librosa.

    Returns:
        numpy array of beat times in seconds.
    """
    import librosa

    y, _ = librosa.load(music_path, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    logger.info(
        f"[BEAT] Detected {len(beat_times)} beats in {Path(music_path).name} "
        f"(tempo={float(tempo):.1f} BPM)"
    )
    return beat_times


def snap_to_beats(
    clips: list,
    music_path: str,
    snap_window: float = 0.8,
    max_adjust: float = 0.5,
    sr: int = 22050,
) -> Tuple[list, dict]:
    """
    Adjust clip durations so cut points land on music beats.

    Instead of v1's speed-change approach, we extend or trim the source_end
    of each clip by up to `max_adjust` seconds to align the cut with the
    nearest beat. This preserves natural playback speed.

    Args:
        clips: List of clip dicts (from EditDecision.clips, must have source_start/source_end).
        music_path: Path to the music file.
        snap_window: Max distance (seconds) to search for a nearby beat.
        max_adjust: Max amount (seconds) to extend/trim a clip's source_end.
        sr: Sample rate for beat detection.

    Returns:
        Tuple of (modified_clips, metadata) where metadata contains beat info.
    """
    beat_times = detect_beats(music_path, sr=sr)

    if len(beat_times) == 0:
        logger.warning("[BEAT] No beats detected, skipping snap")
        return clips, {"beats_detected": 0, "snapped": 0}

    current_time = 0.0
    snapped_count = 0
    adjustments = []

    for clip in clips:
        src_start = clip.get("source_start", 0) if isinstance(clip, dict) else clip.source_start
        src_end = clip.get("source_end", 0) if isinstance(clip, dict) else clip.source_end
        duration = src_end - src_start

        if duration <= 0:
            current_time += duration
            continue

        intended_end = current_time + duration

        # Find beats within snap_window of the intended cut point
        distances = np.abs(beat_times - intended_end)
        candidates_mask = distances <= snap_window
        candidates = beat_times[candidates_mask]

        if len(candidates) > 0:
            # Pick the closest beat
            snap_target = candidates[np.argmin(np.abs(candidates - intended_end))]
            adjustment = snap_target - intended_end

            # Only adjust if within max_adjust limit
            if abs(adjustment) <= max_adjust:
                new_duration = duration + adjustment
                if new_duration > 0.5:  # Don't make clips too short
                    new_src_end = src_start + new_duration

                    if isinstance(clip, dict):
                        clip["source_end"] = round(new_src_end, 3)
                    else:
                        clip.source_end = round(new_src_end, 3)

                    adjustments.append({
                        "clip_id": clip.get("id", "?") if isinstance(clip, dict) else clip.id,
                        "original_end": round(intended_end, 3),
                        "snapped_to": round(snap_target, 3),
                        "adjustment_sec": round(adjustment, 3),
                    })
                    snapped_count += 1
                    duration = new_duration

        current_time += duration

    metadata = {
        "beats_detected": len(beat_times),
        "tempo_bpm": float(np.median(np.diff(beat_times)) ** -1 * 60) if len(beat_times) > 1 else 0,
        "snapped": snapped_count,
        "total_clips": len(clips),
        "adjustments": adjustments,
    }

    logger.info(
        f"[BEAT] Snapped {snapped_count}/{len(clips)} clips to beats "
        f"(tempo={metadata['tempo_bpm']:.1f} BPM, {len(beat_times)} beats)"
    )

    return clips, metadata
