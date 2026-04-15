"""Tests for the workspace_root audio path resolution in edit_compiler.

Regression guard: bare narration/music filenames used to fall back to
``file:///media/<name>``, which Resolve could not reach. The fix now resolves
against ``{workspace}/narration/`` and ``{workspace}/music/``. A refactor that
drops the workspace_root plumbing would silently bring the bug back.
"""
from __future__ import annotations
import re
from pathlib import Path

import pytest

from src.pipelines.v2.fcpxml.edit_compiler import (
    _resolve_audio_path,
    compile_edit_decision,
)
from src.pipelines.v2.schemas import (
    ClipDecision,
    EditDecision,
    MusicTrack,
    NarrationSegment,
    TimelineSettings,
)


# ─── _resolve_audio_path helper ──────────────────────────────────────────────

class TestResolveAudioPath:
    def test_bare_filename_resolves_to_narration_dir(self, tmp_path):
        ws = tmp_path
        (ws / "narration").mkdir()
        target = ws / "narration" / "narration.mp3"
        target.write_bytes(b"fake")
        assert _resolve_audio_path("narration.mp3", ws) == str(target.resolve())

    def test_bare_filename_resolves_to_music_dir(self, tmp_path):
        ws = tmp_path
        (ws / "music").mkdir()
        target = ws / "music" / "track.mp3"
        target.write_bytes(b"fake")
        assert _resolve_audio_path("track.mp3", ws) == str(target.resolve())

    def test_narration_preferred_over_music_when_both_contain_name(self, tmp_path):
        ws = tmp_path
        (ws / "narration").mkdir()
        (ws / "music").mkdir()
        (ws / "narration" / "dup.mp3").write_bytes(b"n")
        (ws / "music" / "dup.mp3").write_bytes(b"m")
        # narration comes first in the subdir search order
        resolved = _resolve_audio_path("dup.mp3", ws)
        assert "narration" in resolved

    def test_absolute_existing_path_returns_unchanged(self, tmp_path):
        ws = tmp_path
        f = tmp_path / "elsewhere.mp3"
        f.write_bytes(b"x")
        assert _resolve_audio_path(str(f), ws) == str(f)

    def test_missing_file_returns_input_unchanged(self, tmp_path):
        # Returns the bare name so the compiler's downstream fallback
        # (file:///media/<name>) can fire instead of raising.
        ws = tmp_path
        assert _resolve_audio_path("nope.mp3", ws) == "nope.mp3"

    def test_workspace_root_none_returns_input(self):
        assert _resolve_audio_path("anything.mp3", None) == "anything.mp3"


# ─── compile_edit_decision — end-to-end path wiring ──────────────────────────

@pytest.fixture
def workspace_with_audio(tmp_path):
    """Build a minimal workspace with a fake narration + music file."""
    (tmp_path / "narration").mkdir()
    (tmp_path / "music").mkdir()
    (tmp_path / "footage").mkdir()
    (tmp_path / "narration" / "n.mp3").write_bytes(b"n")
    (tmp_path / "music" / "m.mp3").write_bytes(b"m")
    # A real video file path the compiler can write into the asset src.
    video_path = tmp_path / "footage" / "v.mp4"
    video_path.write_bytes(b"v")
    return tmp_path, video_path


def _make_edit(video_path: Path, narration_file: str, music_file: str | None) -> EditDecision:
    return EditDecision(
        timeline=TimelineSettings(fps=24.0, width=1920, height=1080),
        clips=[ClipDecision(
            id="c1",
            source_file=video_path.name,
            source_path=str(video_path),
            source_start=0.0,
            source_end=2.0,
        )],
        narration=[NarrationSegment(
            file=narration_file,
            timeline_offset=0.0,
            start=0.0,
            duration=1.5,
        )],
        music=MusicTrack(file=music_file) if music_file else None,
    )


class TestCompileEditDecisionAudioResolution:
    def test_bare_narration_filename_becomes_absolute(self, workspace_with_audio, tmp_path):
        ws, video_path = workspace_with_audio
        edit = _make_edit(video_path, narration_file="n.mp3", music_file=None)
        out = tmp_path / "out.fcpxml"
        compile_edit_decision(edit, str(out), workspace_root=ws)

        fcp_text = out.read_text()
        narration_src_line = [L for L in fcp_text.splitlines() if "n.mp3" in L and "src=" in L]
        assert narration_src_line, "No asset line referencing narration file"
        # Path must be absolute file:// URI pointing into the workspace.
        assert any(f"file://{ws.resolve()}" in L for L in narration_src_line), \
            f"Narration path not absolute: {narration_src_line}"
        # Should NOT have the legacy media:// fallback.
        assert "file:///media/n.mp3" not in fcp_text

    def test_bare_music_filename_becomes_absolute(self, workspace_with_audio, tmp_path):
        ws, video_path = workspace_with_audio
        edit = _make_edit(video_path, narration_file="n.mp3", music_file="m.mp3")
        out = tmp_path / "out.fcpxml"
        compile_edit_decision(edit, str(out), workspace_root=ws)

        fcp_text = out.read_text()
        assert "file:///media/m.mp3" not in fcp_text
        assert any(f"file://{ws.resolve()}/music/m.mp3" in L for L in fcp_text.splitlines())

    def test_no_workspace_root_leaves_media_fallback(self, workspace_with_audio, tmp_path):
        ws, video_path = workspace_with_audio
        edit = _make_edit(video_path, narration_file="n.mp3", music_file=None)
        out = tmp_path / "out.fcpxml"
        # Deliberately omit workspace_root — we're verifying the fallback
        # still fires for callers that don't pass it (e.g. ablation scripts).
        compile_edit_decision(edit, str(out))
        fcp_text = out.read_text()
        assert "file:///media/n.mp3" in fcp_text
