"""
Clip post-processing utilities: anti-clustering, overlap merging, temporal diversity.

All functions are pure — no I/O, no side effects.
"""
from __future__ import annotations
from typing import List

from src.pipelines.v2.schemas import RetrievedClip


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SCORE_THRESHOLD = 0.35       # Drop clips below this confidence
MIN_CLIP_DURATION_SEC = 2.0      # Drop clips shorter than this
MAX_CLIP_DURATION_SEC = 60.0     # Cap excessively long clips
OVERLAP_MERGE_GAP_SEC = 1.5      # Merge adjacent clips within this gap
MIN_TEMPORAL_GAP_SEC = 5.0       # Min gap between non-overlapping clips from same video


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess_clips(
    clips: List[RetrievedClip],
    *,
    score_threshold: float = MIN_SCORE_THRESHOLD,
    min_duration: float = MIN_CLIP_DURATION_SEC,
    max_duration: float = MAX_CLIP_DURATION_SEC,
    merge_gap: float = OVERLAP_MERGE_GAP_SEC,
    temporal_gap: float = MIN_TEMPORAL_GAP_SEC,
    max_clips: int = 40,
) -> List[RetrievedClip]:
    """
    Full post-processing pipeline for retrieved clips:
    1. Score filter
    2. Duration filter & cap
    3. Merge overlapping/adjacent clips per video
    4. Enforce temporal diversity per video
    5. Global limit

    Returns a new list; input is not modified.
    """
    clips = list(clips)
    clips = _filter_by_score(clips, score_threshold)
    clips = _filter_and_cap_duration(clips, min_duration, max_duration)
    clips = _merge_overlapping(clips, merge_gap)
    clips = _enforce_temporal_diversity(clips, temporal_gap)
    clips = clips[:max_clips]
    return clips


def deduplicate_against_existing(
    new_clips: List[RetrievedClip],
    existing_clips: List[RetrievedClip],
    overlap_threshold: float = 0.5,
) -> List[RetrievedClip]:
    """
    Remove clips from new_clips that substantially overlap with existing_clips
    (same video, overlap ratio >= overlap_threshold).
    """
    result = []
    for clip in new_clips:
        if not _overlaps_any(clip, existing_clips, overlap_threshold):
            result.append(clip)
    return result


def sort_by_timeline(clips: List[RetrievedClip]) -> List[RetrievedClip]:
    """Sort clips chronologically (by video_no then start_seconds)."""
    return sorted(clips, key=lambda c: (c.video_no, c.start_seconds))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_by_score(
    clips: List[RetrievedClip], threshold: float
) -> List[RetrievedClip]:
    return [c for c in clips if c.score >= threshold]


def _filter_and_cap_duration(
    clips: List[RetrievedClip], min_dur: float, max_dur: float
) -> List[RetrievedClip]:
    result = []
    for c in clips:
        dur = c.end_seconds - c.start_seconds
        if dur < min_dur:
            continue
        if dur > max_dur:
            # Cap by trimming end
            c = c.model_copy(update={"end_seconds": c.start_seconds + max_dur})
        result.append(c)
    return result


def _merge_overlapping(
    clips: List[RetrievedClip], gap: float
) -> List[RetrievedClip]:
    """
    Within each video, merge clips that overlap or are within `gap` seconds of each other.
    Keeps the highest score and concatenates descriptions.
    """
    # Group by video_no
    by_video: dict[str, List[RetrievedClip]] = {}
    for c in clips:
        by_video.setdefault(c.video_no, []).append(c)

    result: List[RetrievedClip] = []
    for video_no, group in by_video.items():
        group = sorted(group, key=lambda c: c.start_seconds)
        merged = [group[0]]
        for current in group[1:]:
            prev = merged[-1]
            # Overlap or within gap
            if current.start_seconds <= prev.end_seconds + gap:
                # Extend the previous clip
                new_end = max(prev.end_seconds, current.end_seconds)
                new_score = max(prev.score, current.score)
                new_desc = prev.description
                if current.description and current.description not in new_desc:
                    new_desc = f"{new_desc}; {current.description}".strip("; ")
                merged[-1] = prev.model_copy(update={
                    "end_seconds": new_end,
                    "score": new_score,
                    "description": new_desc,
                })
            else:
                merged.append(current)
        result.extend(merged)
    return result


def _enforce_temporal_diversity(
    clips: List[RetrievedClip], min_gap: float
) -> List[RetrievedClip]:
    """
    Within each video, if two clips start within `min_gap` of each other and are not
    already merged (i.e., they don't overlap), keep the higher-scored one.
    This prevents clustering of clips from the same moment.
    """
    by_video: dict[str, List[RetrievedClip]] = {}
    for c in clips:
        by_video.setdefault(c.video_no, []).append(c)

    result: List[RetrievedClip] = []
    for _, group in by_video.items():
        group = sorted(group, key=lambda c: -c.score)  # best first
        accepted: List[RetrievedClip] = []
        for candidate in group:
            too_close = False
            for acc in accepted:
                gap = abs(candidate.start_seconds - acc.start_seconds)
                overlap = _overlap_seconds(candidate, acc)
                if overlap == 0 and gap < min_gap:
                    too_close = True
                    break
            if not too_close:
                accepted.append(candidate)
        result.extend(accepted)
    return result


def _overlap_seconds(a: RetrievedClip, b: RetrievedClip) -> float:
    if a.video_no != b.video_no:
        return 0.0
    start = max(a.start_seconds, b.start_seconds)
    end = min(a.end_seconds, b.end_seconds)
    return max(0.0, end - start)


def _overlap_ratio(a: RetrievedClip, b: RetrievedClip) -> float:
    """Intersection over union of two clip intervals."""
    intersection = _overlap_seconds(a, b)
    if intersection == 0:
        return 0.0
    union = (a.end_seconds - a.start_seconds) + (b.end_seconds - b.start_seconds) - intersection
    return intersection / union if union > 0 else 0.0


def _overlaps_any(
    clip: RetrievedClip,
    others: List[RetrievedClip],
    threshold: float,
) -> bool:
    return any(_overlap_ratio(clip, o) >= threshold for o in others)
