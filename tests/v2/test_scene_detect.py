"""Tests for the scene-cut detection helper used by refine_clip_timestamps.

The helper feeds shot-boundary hints into the Gemini refinement prompt. A
refactor that swaps detectors or changes the return shape would silently
degrade the "avoid sloppy cuts near a boundary" rule in the prompt.
"""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.pipelines.v2.agent.tools import _detect_scene_cuts


def _fake_scene(sec_start: float) -> tuple:
    """Build a ``(start_tc, end_tc)`` tuple the helper can unpack."""
    start = MagicMock()
    start.get_seconds.return_value = sec_start
    end = MagicMock()
    end.get_seconds.return_value = sec_start + 5.0
    return (start, end)


class TestDetectSceneCuts:
    def test_empty_scene_list_returns_empty(self):
        """One continuous shot → no cuts reported and no error."""
        with patch("scenedetect.detect", return_value=[]):
            cuts, error = _detect_scene_cuts("/path/to/video.mp4", 10.0, 20.0)
        assert cuts == []
        assert error is None

    def test_single_scene_has_no_cuts(self):
        """A single scene spanning the whole window is not a cut."""
        with patch("scenedetect.detect", return_value=[_fake_scene(10.0)]):
            cuts, error = _detect_scene_cuts("/path/to/video.mp4", 10.0, 20.0)
        assert cuts == []
        assert error is None

    def test_multiple_scenes_return_boundaries_relative_to_window(self):
        # Scenes starting at abs 10.0, 13.42, 17.80. Window starts at 10.0.
        # Expected relative cuts: 3.42, 7.80 (first scene start excluded).
        scenes = [_fake_scene(10.0), _fake_scene(13.42), _fake_scene(17.80)]
        with patch("scenedetect.detect", return_value=scenes):
            cuts, error = _detect_scene_cuts("/path/video.mp4", 10.0, 20.0)
        assert cuts == [3.42, 7.80]
        assert error is None

    def test_near_zero_cuts_are_discarded(self):
        # If detection reports a boundary within 0.05s of window start,
        # the helper drops it — this is the noise-floor filter.
        scenes = [_fake_scene(10.0), _fake_scene(10.02)]
        with patch("scenedetect.detect", return_value=scenes):
            cuts, error = _detect_scene_cuts("/path/video.mp4", 10.0, 20.0)
        assert cuts == []
        assert error is None

    def test_rounding_to_two_decimal_places(self):
        scenes = [_fake_scene(10.0), _fake_scene(13.4567)]
        with patch("scenedetect.detect", return_value=scenes):
            cuts, error = _detect_scene_cuts("/path/video.mp4", 10.0, 20.0)
        assert cuts == [3.46]
        assert error is None

    def test_failed_detection_returns_error_string(self):
        # PySceneDetect raises on an invalid path; the helper returns
        # ([], "...") so refine_clip_timestamps can surface the failure
        # as a warning to the main LLM.
        with patch("scenedetect.detect", side_effect=RuntimeError("bad file")):
            cuts, error = _detect_scene_cuts("/nope.mp4", 0.0, 5.0)
        assert cuts == []
        assert error is not None
        assert "bad file" in error

    def test_import_failure_returns_error_string(self):
        # If scenedetect isn't importable (dev env without it installed),
        # the helper degrades gracefully AND surfaces the error.
        with patch.dict("sys.modules", {"scenedetect": None}):
            cuts, error = _detect_scene_cuts("/any.mp4", 0.0, 5.0)
        assert cuts == []
        assert error is not None
