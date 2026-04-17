"""Tests for the FFmpeg renderer's pure helpers.

We avoid running ffmpeg from the test process — those would be flaky without
real media files. Instead we exercise the filter-graph builders and lookup
helpers that do the interesting thinking, so a refactor that silently breaks
rotation, overlay positioning, or transition mapping trips these tests.
"""
from __future__ import annotations

import inspect

import pytest

from src.pipelines.v2.preview import ffmpeg_renderer as R
from src.pipelines.v2.preview.ffmpeg_renderer import (
    QUALITY_PRESETS,
    _build_transform_filter,
    _transition_to_xfade_type,
    render_ffmpeg_preview,
)
from src.pipelines.v2.schemas import TransformSettings


# ─── Quality presets ──────────────────────────────────────────────────────


class TestQualityPresets:
    def test_both_quality_modes_defined(self):
        assert {"draft", "final"} <= set(QUALITY_PRESETS.keys())

    def test_draft_is_cheap(self):
        # Draft must remain fast enough to run in the agent loop after every
        # generate_fcpxml call. The specific values are less important than
        # the invariant that draft should be cheaper than final.
        d, f = QUALITY_PRESETS["draft"], QUALITY_PRESETS["final"]
        assert d["height"] == 480
        assert f["height"] is None  # timeline native
        assert int(d["crf"]) > int(f["crf"])  # higher CRF = lower quality = cheaper

    def test_final_uses_better_scaler(self):
        assert QUALITY_PRESETS["final"]["scale_flags"] == "lanczos"
        assert QUALITY_PRESETS["draft"]["scale_flags"] == "bilinear"


class TestRenderSignature:
    def test_quality_param_is_part_of_signature(self):
        sig = inspect.signature(render_ffmpeg_preview)
        assert "quality" in sig.parameters
        assert sig.parameters["quality"].default == "draft"

    def test_deprecated_resolution_param_still_accepted(self):
        # Keeps existing callers (dashboard, older branches) working.
        sig = inspect.signature(render_ffmpeg_preview)
        assert "resolution" in sig.parameters


# ─── Transform filter ─────────────────────────────────────────────────────


class TestTransformFilter:
    def test_fit_always_honours_sar(self):
        # Every filter chain begins with SAR normalization — critical for
        # anamorphic sources (non-square pixels) to match how Resolve
        # interprets the same FCPXML. For square-pixel content it's a no-op.
        t = TransformSettings()
        out = _build_transform_filter(t, 1920, 1080, 1920, 1080, 854, 480)
        assert out.startswith("scale=iw*sar:ih")
        assert "setsar=1" in out
        # No rotation in this trivial case.
        assert "rotate" not in out

    def test_fit_uses_aspect_preserving_scale_with_pad(self):
        # Fit path should preserve aspect and pad black bars to fill the
        # canvas — matches Resolve's default "Spatial Conform: Fit" behaviour
        # on FCPXML import. Content that doesn't match timeline aspect gets
        # letterboxed/pillarboxed; no cropping, no stretching.
        t = TransformSettings()
        out = _build_transform_filter(t, 1920, 1080, 1920, 1080, 854, 480)
        assert "force_original_aspect_ratio=decrease" in out
        assert "pad=854:480" in out

    def test_scale_less_than_one_produces_crop(self):
        # Zooming in (scale < full fit) should emit a crop before scale.
        t = TransformSettings(scale_x=0.5, scale_y=0.5)
        out = _build_transform_filter(t, 1920, 1080, 1920, 1080, 854, 480)
        assert "crop=" in out
        assert "scale=854:480" in out

    def test_rotation_is_applied(self):
        # The whole point of this fix: rotation was in the schema but not in
        # the filter chain. Regressing this would render rotated clips as
        # un-rotated, which is silent and hard to catch in end-to-end tests.
        t = TransformSettings(rotation=45.0)
        out = _build_transform_filter(t, 1920, 1080, 1920, 1080, 854, 480)
        assert "rotate=" in out
        assert "PI/180" in out  # degrees → radians conversion
        assert "ow=854" in out and "oh=480" in out

    def test_no_rotation_filter_when_rotation_is_zero(self):
        t = TransformSettings(rotation=0.0)
        out = _build_transform_filter(t, 1920, 1080, 1920, 1080, 854, 480)
        assert "rotate=" not in out

    def test_scale_flags_plumbed_through(self):
        t = TransformSettings()
        out = _build_transform_filter(
            t, 1920, 1080, 1920, 1080, 854, 480, scale_flags="lanczos"
        )
        assert "flags=lanczos" in out


# ─── Transition mapping ──────────────────────────────────────────────────


class TestTransitionMapping:
    def test_cross_dissolve_maps_to_fade(self):
        assert _transition_to_xfade_type("cross-dissolve") == "fade"

    def test_fade_in_and_fade_out_map_to_fadeblack(self):
        # Both are interpreted as 'dip to black' between clips — fadeblack
        # is the ffmpeg xfade name for that.
        assert _transition_to_xfade_type("fade-in") == "fadeblack"
        assert _transition_to_xfade_type("fade-out") == "fadeblack"

    def test_unknown_type_falls_back_to_fade(self):
        assert _transition_to_xfade_type("splat") == "fade"


# ─── System font lookup ──────────────────────────────────────────────────


class TestSystemFont:
    def test_returns_string_or_none_without_raising(self):
        # Caches the result, so call it twice; neither should raise.
        a = R._system_font()
        b = R._system_font()
        assert a == b
        assert a is None or isinstance(a, str)

    def test_returned_font_file_exists_if_not_none(self):
        import os
        f = R._system_font()
        if f is not None:
            assert os.path.exists(f)
