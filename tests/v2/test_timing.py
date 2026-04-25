"""Tests for the frame-rate timing helpers.

These are the anchors that make clip boundaries land on exact source frames
regardless of timeline fps. Regressions here silently reintroduce the ±1-frame
A/V drift we fixed — cheap to test in isolation since they're pure functions.
"""
from __future__ import annotations

from src.pipelines.v2.timing import snap_to_frame, dominant_fps


class TestSnapToFrame:
    def test_identity_when_already_on_frame(self):
        # 24fps: 1.0s is exactly frame 24 — should be unchanged.
        assert snap_to_frame(1.0, 24.0) == 1.0

    def test_snaps_to_nearest_frame_at_24fps(self):
        # frame duration = 1/24 ≈ 0.04167s. Midpoint between frame 24 (1.0s)
        # and frame 25 (1.04167s) is ~1.02083s. 1.03 is past the midpoint,
        # so it snaps UP to frame 25.
        got = snap_to_frame(1.03, 24.0)
        assert abs(got - 25 / 24) < 1e-9

    def test_snaps_down_when_closer_to_lower_frame(self):
        # 1.02 is BEFORE the midpoint (1.02083), so it snaps DOWN to frame 24.
        got = snap_to_frame(1.02, 24.0)
        assert abs(got - 1.0) < 1e-9

    def test_snaps_at_ntsc_2997(self):
        # 29.97 NTSC exact rate = 30000/1001. Agent writes 1504.48; nearest
        # frame is round(1504.48 * 30000/1001) = frame 45089 → 45089*1001/30000.
        fps = 30000 / 1001
        got = snap_to_frame(1504.48, fps)
        frame_idx = round(1504.48 * fps)
        assert abs(got - frame_idx / fps) < 1e-9

    def test_zero_fps_is_noop(self):
        # Defensive: unknown fps shouldn't crash or produce NaN.
        assert snap_to_frame(3.14, 0) == 3.14
        assert snap_to_frame(3.14, -1) == 3.14


class TestDominantFps:
    def test_empty_returns_none(self, tmp_path):
        assert dominant_fps([]) is None

    def test_missing_files_return_none(self, tmp_path):
        # probe_fps of a nonexistent file returns None and is filtered out.
        assert dominant_fps([tmp_path / "does-not-exist.mp4"]) is None
