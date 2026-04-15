"""Safety tests for the workspace layer.

Path traversal, atomic writes, and project-name validation are security
boundaries. A refactor that loosens any of them would open the app to reading
or overwriting arbitrary files via a crafted project name.
"""
from __future__ import annotations
import json
import os
import pytest
from pathlib import Path

from src.pipelines.v2.workspace import (
    WorkspaceManager,
    _atomic_write_json,
    validate_project_name,
)


# ─── validate_project_name ───────────────────────────────────────────────────

class TestValidateProjectName:
    @pytest.mark.parametrize("name", [
        "movie1", "movie 1", "movie.1", "movie-1", "movie_1",
        "A", "A1", "A" * 128,                                  # max length
        "project.name-v2", "my project (2)".replace("(", "").replace(")", ""),
    ])
    def test_accepts_safe_names(self, name):
        assert validate_project_name(name) == name

    @pytest.mark.parametrize("name", [
        "",                     # empty
        "../escape",            # traversal
        "foo/bar",              # contains slash
        "foo\\bar",             # backslash (Windows traversal)
        ".hidden",              # leading dot
        "/abs/path",            # absolute path
        "A" * 129,              # too long
        "foo\x00bar",           # null byte
        "foo\nbar",             # newline
    ])
    def test_rejects_unsafe_names(self, name):
        with pytest.raises(ValueError):
            validate_project_name(name)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_project_name(None)  # type: ignore
        with pytest.raises(ValueError):
            validate_project_name(123)  # type: ignore


# ─── WorkspaceManager path resolution ─────────────────────────────────────────

class TestWorkspaceManagerPathSafety:
    def test_happy_path_creates_root_under_workspaces(self, tmp_path):
        ws = WorkspaceManager("ok-project", tmp_path)
        assert ws.root == (tmp_path / "ok-project").resolve() or \
               ws.root == tmp_path / "ok-project"

    def test_symlink_escape_is_blocked(self, tmp_path):
        # Create a symlink INSIDE workspaces/ that points OUTSIDE.
        # validate_project_name catches path-separator names, but a
        # symlinked directory with a legal name could still let a caller
        # read/write outside the workspaces root. We rely on the
        # post-resolve check in WorkspaceManager.__init__.
        outside = tmp_path.parent / "outside_target"
        outside.mkdir()
        link = tmp_path / "linked-project"
        link.symlink_to(outside)
        # The name itself is legal; the resolve check should fire.
        with pytest.raises(ValueError):
            WorkspaceManager("linked-project", tmp_path)


# ─── _atomic_write_json ──────────────────────────────────────────────────────

class TestAtomicWriteJson:
    def test_writes_the_payload(self, tmp_path):
        path = tmp_path / "out.json"
        _atomic_write_json(path, {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "file.json"
        _atomic_write_json(path, {"ok": True})
        assert path.exists()
        assert json.loads(path.read_text()) == {"ok": True}

    def test_overwrites_existing_file_atomically(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text('{"old": 1}')
        _atomic_write_json(path, {"new": 2})
        assert json.loads(path.read_text()) == {"new": 2}

    def test_leaves_no_tempfile_on_success(self, tmp_path):
        path = tmp_path / "out.json"
        _atomic_write_json(path, {"ok": True})
        leftovers = [p for p in tmp_path.iterdir() if p.suffix.startswith(".tmp")]
        assert leftovers == []

    def test_failure_to_serialize_leaves_original_intact(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text('{"original": true}')
        # An object with a non-serializable value should raise before replace.
        class NotJSON:
            pass
        with pytest.raises(TypeError):
            _atomic_write_json(path, {"bad": NotJSON()})
        # Original must still be readable — the whole point of "atomic".
        assert json.loads(path.read_text()) == {"original": True}

    def test_indent_param_controls_pretty_printing(self, tmp_path):
        path = tmp_path / "out.json"
        _atomic_write_json(path, {"a": 1, "b": 2}, indent=0)
        # indent=0 still uses separators; content is readable but not prettified
        assert '"a": 1' in path.read_text()


# ─── WorkspaceManager create/exists lifecycle ────────────────────────────────

class TestWorkspaceLifecycle:
    def test_create_makes_expected_subdirs(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        for sub in ["footage", "iterations", "narration", "music",
                    "fcpxml", "renders", "logs"]:
            assert (ws.root / sub).is_dir(), f"Missing expected subdir: {sub}"

    def test_exists_false_before_session_saved(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        assert ws.exists() is False

    def test_scan_footage_empty_when_dir_absent(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        # deliberately don't call create()
        assert ws.scan_footage() == []

    def test_scan_footage_filters_by_video_extension(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        (ws.root / "footage" / "a.mp4").touch()
        (ws.root / "footage" / "b.mov").touch()
        (ws.root / "footage" / "c.txt").touch()
        (ws.root / "footage" / "d.png").touch()
        names = {p.name for p in ws.scan_footage()}
        assert names == {"a.mp4", "b.mov"}
