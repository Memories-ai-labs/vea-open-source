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
