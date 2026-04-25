"""Pydantic schema contract tests.

These lock down the shape of the contract between the LLM, the dashboard, and
the deterministic FCPXML compiler. A refactor that renames a field, flips a
default, or tightens a constraint should trip one of these.
"""
from __future__ import annotations
import json
import pytest
from pydantic import ValidationError

from src.pipelines.v2.schemas import (
    ClipDecision,
    EditDecision,
    MusicTrack,
    NarrationSegment,
    RefinedTimestamps,
    SpeedChange,
    TextOverlay,
    TimelineSettings,
    TransformSettings,
)


# ─── ClipDecision ────────────────────────────────────────────────────────────

def test_clip_decision_minimum_fields():
    c = ClipDecision(id="c1", source_file="v.mp4", source_start=0.0, source_end=5.0)
    assert c.id == "c1"
    assert c.source_path == ""
    assert c.gain_db is None
    assert c.transform_mode == "fit"
    assert c.track == 1
    assert c.timeline_offset is None
    assert c.source_width == 1920
    assert c.source_height == 1080


def test_clip_decision_rejects_unknown_transform_mode():
    with pytest.raises(ValidationError):
        ClipDecision(
            id="c1", source_file="v.mp4", source_start=0, source_end=1,
            transform_mode="warp",  # not in {"fit","custom","saliency"}
        )


def test_clip_decision_roundtrip_preserves_all_fields():
    original = ClipDecision(
        id="c1", source_file="v.mp4", source_start=10.5, source_end=14.2,
        label="Opening", description="establishing shot",
        gain_db=-6.0, measured_loudness_lufs=-20.1,
        speed=SpeedChange(rate=1.5),
        transform=TransformSettings(scale_x=1.2, position_x=0.1),
        transform_mode="custom",
        source_width=3840, source_height=2160,
        track=2, timeline_offset=7.5,
    )
    dumped = original.model_dump()
    rehydrated = ClipDecision.model_validate(dumped)
    assert rehydrated.model_dump() == dumped


# ─── NarrationSegment ────────────────────────────────────────────────────────

def test_narration_segment_minimum_fields():
    n = NarrationSegment(file="narration.mp3", timeline_offset=0.0, duration=3.5)
    assert n.start == 0.0
    assert n.gain_db == 0.0
    assert n.track == 1
    assert n.measured_loudness_lufs is None


def test_narration_segment_track_field_survives_roundtrip():
    # The dashboard adds track={1,2,3} for visual stacking; the field must
    # survive JSON serialization even though the compiler ignores it.
    n = NarrationSegment(
        file="x.mp3", timeline_offset=10.0, start=2.5, duration=4.0,
        gain_db=-2.0, measured_loudness_lufs=-18.5, track=3,
    )
    j = json.loads(json.dumps(n.model_dump()))
    rehydrated = NarrationSegment.model_validate(j)
    assert rehydrated.track == 3
    assert rehydrated.gain_db == -2.0
    assert rehydrated.measured_loudness_lufs == -18.5


def test_narration_segment_requires_duration():
    with pytest.raises(ValidationError):
        NarrationSegment(file="x.mp3", timeline_offset=0.0)  # type: ignore


# ─── MusicTrack ──────────────────────────────────────────────────────────────

def test_music_track_defaults():
    m = MusicTrack(file="track.mp3")
    assert m.start == 0.0
    assert m.duration == 0.0       # "use full timeline"
    assert m.gain_db == 0.0
    assert m.track == 2            # defaults to A2 so it sits below narration


def test_music_track_has_no_timeline_offset_field_yet():
    # Audit finding #5: the schema uses `start` as in-point, but the UI has
    # been writing timeline placement into `start`. If/when a separate
    # `timeline_offset` is added, this test should flip — but until then
    # we pin the current shape explicitly so a refactor doesn't silently
    # change the meaning of `start`.
    fields = MusicTrack.model_fields
    assert "start" in fields
    assert "timeline_offset" not in fields, (
        "MusicTrack gained a timeline_offset field — update ffmpeg_renderer.py "
        "and edit_compiler.py to use it and drop this assertion."
    )


# ─── TextOverlay ─────────────────────────────────────────────────────────────

def test_text_overlay_defaults():
    t = TextOverlay(text="hello", timeline_offset=0.0, duration=2.0)
    assert t.lane == 1
    assert t.font_size == 72
    assert t.style == "title"
    assert t.position == "center"


# ─── EditDecision ────────────────────────────────────────────────────────────

def test_edit_decision_empty_is_valid():
    # An empty edit decision should validate — the agent starts with nothing
    # and builds up. A refactor that makes clips/timeline required would
    # break mid-session agent state.
    ed = EditDecision()
    assert ed.clips == []
    assert ed.narration == []
    assert ed.music is None
    assert ed.titles == []
    assert ed.timeline.width == 1920
    assert ed.timeline.height == 1080
    assert ed.timeline.fps == 24.0


def test_edit_decision_full_roundtrip():
    payload = {
        "timeline": {"name": "test", "fps": 30.0, "width": 1080, "height": 1920},
        "clips": [
            {"id": "c1", "source_file": "v.mp4", "source_start": 0, "source_end": 5},
            {"id": "c2", "source_file": "v.mp4", "source_start": 10, "source_end": 15, "track": 2, "timeline_offset": 3.0},
        ],
        "narration": [
            {"file": "n.mp3", "timeline_offset": 0, "duration": 4.0, "gain_db": -1.5},
        ],
        "music": {"file": "m.mp3", "gain_db": -3.0},
        "titles": [{"text": "Title", "timeline_offset": 0, "duration": 3.0, "lane": 2}],
    }
    ed = EditDecision.model_validate(payload)
    assert len(ed.clips) == 2
    assert ed.clips[1].track == 2
    assert ed.clips[1].timeline_offset == 3.0
    assert ed.narration[0].gain_db == -1.5
    assert ed.music.gain_db == -3.0
    assert ed.titles[0].lane == 2
    # Roundtrip should produce the same structural payload
    redump = ed.model_dump()
    assert redump["music"]["file"] == "m.mp3"


def test_edit_decision_ignores_extra_fields_by_default():
    # Pydantic v2 default is to ignore unknown fields. A refactor that
    # flips to strict mode would break forward compat with future LLM
    # outputs that include new fields we haven't added yet.
    ed = EditDecision.model_validate({"clips": [], "some_future_field": 123})
    assert ed.clips == []


# ─── RefinedTimestamps ───────────────────────────────────────────────────────

def test_refined_timestamps_new_start_must_be_nonnegative():
    with pytest.raises(ValidationError):
        RefinedTimestamps(new_start=-0.1, new_end=1.0, reasoning="bad")


def test_refined_timestamps_defaults():
    r = RefinedTimestamps(new_start=1.0, new_end=3.0, reasoning="ok")
    assert r.focus_type == ""
    assert r.speech_truncated_start is False
    assert r.speech_truncated_end is False
