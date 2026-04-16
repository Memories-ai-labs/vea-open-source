"""Tests for build_timeline_view — the computed table injected into the system prompt.

This is the agent's primary source of truth for temporal reasoning. A refactor
that changes the shape of the output or silently drops an overlap warning
would quietly degrade edit quality.
"""
from __future__ import annotations
import json

from src.pipelines.v2.agent.timeline_view import build_timeline_view


# ─── Empty / malformed input ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        assert build_timeline_view("") == ""

    def test_malformed_json_returns_empty(self):
        assert build_timeline_view("{not json") == ""

    def test_no_clips_no_narration_no_music_returns_empty(self):
        payload = json.dumps({"clips": [], "narration": [], "titles": []})
        assert build_timeline_view(payload) == ""


# ─── Minimal edit with V1 clips ──────────────────────────────────────────────

class TestSingleTrack:
    def test_one_clip_produces_a_table(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 5,
                "label": "opener",
            }],
        })
        out = build_timeline_view(payload)
        assert "Computed timeline view" in out
        assert "c1" in out
        assert "opener" in out

    def test_v1_clips_are_concatenated_sequentially(self):
        payload = json.dumps({
            "clips": [
                {"id": "c1", "source_file": "v.mp4", "source_start": 0, "source_end": 3},
                {"id": "c2", "source_file": "v.mp4", "source_start": 0, "source_end": 4},
            ],
        })
        out = build_timeline_view(payload)
        # c1 covers 0→3, c2 covers 3→7 (next slot after previous end)
        assert "0.00→3.00" in out
        assert "3.00→7.00" in out


# ─── Audio-issue detection ───────────────────────────────────────────────────

class TestAudioIssueDetection:
    def test_narration_overlapping_dialogue_clip_is_flagged(self):
        # Clip id contains "dialogue" so the detector treats it as dialogue.
        # Narration sits on top of it → should flag.
        payload = json.dumps({
            "clips": [{
                "id": "dialogue_hero", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10, "label": "dialogue",
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 2.0, "duration": 4.0},
            ],
        })
        out = build_timeline_view(payload)
        assert "Audio issues detected" in out
        assert "dialogue_hero" in out
        assert "Split the narration" in out

    def test_muted_clip_with_no_narration_coverage_is_flagged(self):
        # gain -96 but 0% narration overlap → the "muted but uncovered" warning.
        payload = json.dumps({
            "clips": [{
                "id": "broll", "source_file": "v.mp4",
                "source_start": 0, "source_end": 5, "gain_db": -96,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 10.0, "duration": 2.0},
            ],
        })
        out = build_timeline_view(payload)
        assert "Audio issues detected" in out
        assert "muted" in out.lower()

    def test_non_overlapping_narration_is_clean(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 0.0, "duration": 3.0},
                {"file": "n.mp3", "timeline_offset": 4.0, "duration": 3.0},
            ],
        })
        out = build_timeline_view(payload)
        assert "Audio issues detected" not in out


# ─── Overlay clips ───────────────────────────────────────────────────────────

class TestOverlayClips:
    def test_v2_clip_is_included(self):
        payload = json.dumps({
            "clips": [
                {"id": "c1", "source_file": "v.mp4", "source_start": 0, "source_end": 6},
                {
                    "id": "ov", "source_file": "v.mp4",
                    "source_start": 0, "source_end": 2,
                    "track": 2, "timeline_offset": 1.5,
                },
            ],
        })
        out = build_timeline_view(payload)
        assert "c1" in out
        assert "ov" in out


# ─── Narration overlap + spine checks (new) ──────────────────────────────────

class TestNarrationOverlapDetection:
    def test_overlapping_narration_segments_flagged(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 0.0, "duration": 5.0},
                {"file": "n.mp3", "timeline_offset": 3.0, "duration": 2.0},
            ],
        })
        out = build_timeline_view(payload)
        assert "Audio issues detected" in out
        assert "Narration segments overlap" in out

    def test_adjacent_non_overlapping_is_clean(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 0.0, "duration": 3.0},
                {"file": "n.mp3", "timeline_offset": 3.0, "duration": 2.0},
            ],
        })
        out = build_timeline_view(payload)
        # Overlap exactly 0 — not flagged
        assert "Narration segments overlap" not in out

    def test_narration_past_spine_is_flagged(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 5,
            }],
            "narration": [
                # Starts after the spine ends (at 5.0) → compiler will drop it
                {"file": "n.mp3", "timeline_offset": 6.0, "duration": 2.0},
            ],
        })
        out = build_timeline_view(payload)
        assert "beyond the spine end" in out


# ─── Music tail / orphan checks (new) ────────────────────────────────────────

class TestMusicTailChecks:
    def test_music_ends_before_spine_flagged(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "music": {"file": "m.mp3", "start": 0.0, "duration": 5.0},
        })
        out = build_timeline_view(payload)
        assert "Music ends at" in out
        assert "silence" in out

    def test_music_matches_spine_clean(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "music": {"file": "m.mp3", "start": 0.0, "duration": 20.0},
        })
        out = build_timeline_view(payload)
        assert "Music ends at" not in out

    def test_music_extends_past_spine_flagged(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "music": {"file": "m.mp3", "start": 0.0, "duration": 30.0},
        })
        out = build_timeline_view(payload)
        assert "past the spine end" in out

    def test_music_starts_late_flagged(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "music": {"file": "m.mp3", "start": 5.0, "duration": 15.0},
        })
        out = build_timeline_view(payload)
        assert "Music starts at" in out


# ─── Title lane → V-track mapping (new) ──────────────────────────────────────

class TestTitleLaneColumn:
    def test_title_with_lane_1_appears_in_v2_column(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
            }],
            "titles": [
                {"text": "HELLO WORLD", "timeline_offset": 0, "duration": 3, "lane": 1},
            ],
        })
        out = build_timeline_view(payload)
        # V2 column must exist and the title text must land there, not in a
        # stale "T1 Titles" column.
        assert "| V2 |" in out
        assert "T1 Titles" not in out
        assert "HELLO WORLD" in out

    def test_title_with_lane_2_appears_in_v3_column(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
            }],
            "titles": [
                {"text": "UPPER TITLE", "timeline_offset": 0, "duration": 3, "lane": 2},
            ],
        })
        out = build_timeline_view(payload)
        assert "V3" in out
        assert "UPPER TITLE" in out


# ─── Per-narration-track columns (new) ───────────────────────────────────────

class TestNarrationTrackColumns:
    def test_segments_on_different_tracks_get_different_columns(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 20,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 0.0, "duration": 3.0, "track": 1},
                {"file": "n.mp3", "timeline_offset": 5.0, "duration": 3.0, "track": 2},
            ],
        })
        out = build_timeline_view(payload)
        assert "A1 Narration" in out
        assert "A2 Narration" in out


# ─── Extra clip metadata (new) ───────────────────────────────────────────────

class TestClipMetadataFields:
    def test_speed_shown_when_non_unity(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
                "speed": {"rate": 0.5},
            }],
        })
        out = build_timeline_view(payload)
        assert "speed=0.5x" in out

    def test_speed_not_shown_at_1x(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
            }],
        })
        out = build_timeline_view(payload)
        assert "speed=" not in out

    def test_source_range_shown(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 214.5, "source_end": 224.8,
            }],
        })
        out = build_timeline_view(payload)
        assert "src=[214.5-224.8s]" in out

    def test_transition_after_appears(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 5,
                "transition_after": {"type": "cross-dissolve", "duration_seconds": 0.5},
            }],
        })
        out = build_timeline_view(payload)
        assert "cross-dissolve" in out

    def test_narration_cell_shows_audio_range(self):
        payload = json.dumps({
            "clips": [{
                "id": "c1", "source_file": "v.mp4",
                "source_start": 0, "source_end": 10,
            }],
            "narration": [
                {"file": "n.mp3", "timeline_offset": 0.0, "start": 3.4, "duration": 2.6},
            ],
        })
        out = build_timeline_view(payload)
        assert "audio:[3.40-6.00s]" in out
