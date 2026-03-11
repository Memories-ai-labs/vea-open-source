"""Tests for clip_postprocess.py — all pure functions, no mocks needed."""
import pytest
from src.pipelines.v2.planning.clip_postprocess import (
    deduplicate_against_existing,
    postprocess_clips,
    sort_by_timeline,
    _filter_by_score,
    _filter_and_cap_duration,
    _merge_overlapping,
    _enforce_temporal_diversity,
    _overlap_ratio,
)
from src.pipelines.v2.schemas import RetrievedClip


def make_clip(
    video_no="v1",
    video_name="video.mp4",
    start=0.0,
    end=10.0,
    score=0.8,
    desc="",
    query="test query",
) -> RetrievedClip:
    return RetrievedClip(
        video_no=video_no,
        video_name=video_name,
        source_path=f"/media/{video_name}",
        start_seconds=start,
        end_seconds=end,
        score=score,
        description=desc,
        shot_query=query,
    )


# ---------------------------------------------------------------------------
# Score filter
# ---------------------------------------------------------------------------

def test_filter_by_score_keeps_above_threshold():
    clips = [make_clip(score=0.8), make_clip(score=0.3), make_clip(score=0.5)]
    result = _filter_by_score(clips, 0.4)
    assert len(result) == 2
    assert all(c.score >= 0.4 for c in result)


def test_filter_by_score_all_pass():
    clips = [make_clip(score=0.9), make_clip(score=1.0)]
    assert len(_filter_by_score(clips, 0.5)) == 2


def test_filter_by_score_none_pass():
    clips = [make_clip(score=0.1), make_clip(score=0.2)]
    assert _filter_by_score(clips, 0.5) == []


# ---------------------------------------------------------------------------
# Duration filter & cap
# ---------------------------------------------------------------------------

def test_filter_short_clips_removed():
    clips = [make_clip(start=0, end=1.0), make_clip(start=0, end=5.0)]
    result = _filter_and_cap_duration(clips, min_dur=2.0, max_dur=60.0)
    assert len(result) == 1
    assert result[0].end_seconds == 5.0


def test_cap_long_clips():
    clips = [make_clip(start=0, end=120.0)]
    result = _filter_and_cap_duration(clips, min_dur=2.0, max_dur=60.0)
    assert len(result) == 1
    assert result[0].end_seconds == 60.0
    assert result[0].start_seconds == 0.0


def test_cap_does_not_affect_normal_clips():
    clips = [make_clip(start=10, end=25)]
    result = _filter_and_cap_duration(clips, min_dur=2.0, max_dur=60.0)
    assert result[0].end_seconds == 25.0


# ---------------------------------------------------------------------------
# Merge overlapping
# ---------------------------------------------------------------------------

def test_merge_overlapping_adjacent():
    clips = [
        make_clip(start=0, end=10, score=0.8, desc="A"),
        make_clip(start=10.5, end=20, score=0.7, desc="B"),
    ]
    result = _merge_overlapping(clips, gap=2.0)
    assert len(result) == 1
    assert result[0].start_seconds == 0
    assert result[0].end_seconds == 20
    assert result[0].score == 0.8


def test_merge_overlapping_actual_overlap():
    clips = [
        make_clip(start=0, end=15, score=0.6),
        make_clip(start=10, end=25, score=0.9),
    ]
    result = _merge_overlapping(clips, gap=2.0)
    assert len(result) == 1
    assert result[0].end_seconds == 25
    assert result[0].score == 0.9


def test_merge_keeps_separate_clips():
    clips = [
        make_clip(start=0, end=10),
        make_clip(start=20, end=30),
    ]
    result = _merge_overlapping(clips, gap=2.0)
    assert len(result) == 2


def test_merge_different_videos_not_merged():
    clips = [
        make_clip(video_no="v1", start=0, end=10),
        make_clip(video_no="v2", start=5, end=15),
    ]
    result = _merge_overlapping(clips, gap=2.0)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Temporal diversity
# ---------------------------------------------------------------------------

def test_temporal_diversity_removes_close_clips():
    # Non-overlapping clips whose start times are < min_gap apart — keep higher-scored
    # clip1: 0-3, clip2: 3.5-8 → no overlap, gap = 3.5 < 5.0 → clip2 rejected
    clips = [
        make_clip(start=0, end=3, score=0.9),
        make_clip(start=3.5, end=8, score=0.6),
    ]
    result = _enforce_temporal_diversity(clips, min_gap=5.0)
    assert len(result) == 1
    assert result[0].score == 0.9


def test_temporal_diversity_keeps_spread_clips():
    clips = [
        make_clip(start=0, end=10, score=0.9),
        make_clip(start=30, end=40, score=0.6),
    ]
    result = _enforce_temporal_diversity(clips, min_gap=5.0)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Overlap ratio
# ---------------------------------------------------------------------------

def test_overlap_ratio_full_overlap():
    a = make_clip(start=0, end=10)
    b = make_clip(start=0, end=10)
    assert _overlap_ratio(a, b) == pytest.approx(1.0)


def test_overlap_ratio_no_overlap():
    a = make_clip(start=0, end=10)
    b = make_clip(start=20, end=30)
    assert _overlap_ratio(a, b) == pytest.approx(0.0)


def test_overlap_ratio_partial():
    a = make_clip(start=0, end=10)
    b = make_clip(start=5, end=15)
    # intersection = 5, union = 15
    assert _overlap_ratio(a, b) == pytest.approx(5.0 / 15.0)


def test_overlap_ratio_different_videos():
    a = make_clip(video_no="v1", start=0, end=10)
    b = make_clip(video_no="v2", start=0, end=10)
    assert _overlap_ratio(a, b) == 0.0


# ---------------------------------------------------------------------------
# Dedup against existing
# ---------------------------------------------------------------------------

def test_dedup_removes_substantial_overlap():
    existing = [make_clip(start=0, end=10, score=0.9)]
    new = [make_clip(start=0, end=10, score=0.7)]  # 100% overlap
    result = deduplicate_against_existing(new, existing, overlap_threshold=0.5)
    assert result == []


def test_dedup_keeps_non_overlapping():
    existing = [make_clip(start=0, end=10)]
    new = [make_clip(start=20, end=30)]
    result = deduplicate_against_existing(new, existing, overlap_threshold=0.5)
    assert len(result) == 1


def test_dedup_keeps_partial_below_threshold():
    existing = [make_clip(start=0, end=20)]
    # Overlaps for 2 seconds out of 20+18=38 union → ~5% overlap
    new = [make_clip(start=18, end=36)]
    result = deduplicate_against_existing(new, existing, overlap_threshold=0.5)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Sort by timeline
# ---------------------------------------------------------------------------

def test_sort_by_timeline():
    clips = [
        make_clip(video_no="v2", start=5),
        make_clip(video_no="v1", start=20),
        make_clip(video_no="v1", start=5),
    ]
    result = sort_by_timeline(clips)
    assert result[0].video_no == "v1" and result[0].start_seconds == 5
    assert result[1].video_no == "v1" and result[1].start_seconds == 20
    assert result[2].video_no == "v2"


# ---------------------------------------------------------------------------
# Full postprocess pipeline
# ---------------------------------------------------------------------------

def test_postprocess_full_pipeline():
    clips = [
        make_clip(start=0, end=10, score=0.9),     # keep
        make_clip(start=0.5, end=10.5, score=0.7), # close → merged or diversity removed
        make_clip(start=30, end=40, score=0.2),    # below score threshold
        make_clip(start=50, end=51, score=0.8),    # below min duration → removed
        make_clip(start=60, end=200, score=0.8),   # capped to 60s
    ]
    result = postprocess_clips(
        clips,
        score_threshold=0.35,
        min_duration=2.0,
        max_duration=60.0,
        merge_gap=2.0,
        temporal_gap=5.0,
    )
    # Score < 0.35 removed, short clip removed
    assert all(c.score >= 0.35 for c in result)
    assert all(c.end_seconds - c.start_seconds >= 2.0 for c in result)
    # Capped clip
    capped = [c for c in result if c.start_seconds == 60.0]
    if capped:
        assert capped[0].end_seconds == 120.0  # 60+60


def test_postprocess_empty():
    assert postprocess_clips([]) == []


def test_postprocess_respects_max_clips():
    clips = [make_clip(start=i * 30, end=i * 30 + 10, score=0.9) for i in range(50)]
    result = postprocess_clips(clips, max_clips=10)
    assert len(result) <= 10
