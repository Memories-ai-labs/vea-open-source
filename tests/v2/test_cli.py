"""Tests for the one-shot CLI bits that don't require external services.

Covers the argument parser, the footage symlink staging, and the stdout
emitter's output shape. The full agent loop is out of scope — it needs
LLM + Memories.ai calls and is covered by the end-to-end test harness.
"""
from __future__ import annotations
import asyncio
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import (
    StdoutEmitter,
    _build_parser,
    _resolve_render_paths,
    _stage_footage,
    _summarize_tool_args,
)
from src.pipelines.v2.workspace import WorkspaceManager


# ─── Argument parser ─────────────────────────────────────────────────────────

class TestParser:
    def test_happy_path(self):
        args = _build_parser().parse_args([
            "--project", "promo",
            "--brief", "make a 30s highlight",
        ])
        assert args.project == "promo"
        assert args.brief == "make a 30s highlight"
        assert args.footage_dir is None
        assert args.reuse_index is False
        assert args.log_format == "text"
        assert args.timeout == 900

    def test_rejects_missing_required_args(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--project", "p"])  # no --brief
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--brief", "b"])   # no --project

    def test_rejects_unknown_log_format(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([
                "--project", "p", "--brief", "b", "--log-format", "yaml",
            ])

    def test_accepts_optional_flags(self):
        args = _build_parser().parse_args([
            "--project", "p", "--brief", "b",
            "--footage-dir", "/tmp/clips",
            "--reuse-index",
            "--log-format", "jsonl",
            "--timeout", "120",
        ])
        assert args.footage_dir == "/tmp/clips"
        assert args.reuse_index is True
        assert args.log_format == "jsonl"
        assert args.timeout == 120


# ─── _stage_footage (symlink in) ─────────────────────────────────────────────

class TestStageFootage:
    def test_symlinks_only_video_files(self, tmp_path):
        # Source dir has a mix of video + non-video files.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.mp4").write_bytes(b"v")
        (src / "b.mov").write_bytes(b"v")
        (src / "readme.txt").write_bytes(b"skip")
        (src / "cover.png").write_bytes(b"skip")

        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        ws = WorkspaceManager("proj", ws_root)
        ws.create()

        count = _stage_footage(ws, str(src))
        assert count == 2
        linked = {p.name for p in ws.get_footage_dir().iterdir()}
        assert linked == {"a.mp4", "b.mov"}
        # They must be symlinks, not copies (we want the original bytes unchanged
        # and the footage dir cheap to set up).
        for p in ws.get_footage_dir().iterdir():
            assert p.is_symlink()

    def test_noop_when_footage_dir_is_none(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        count = _stage_footage(ws, None)
        assert count == 0

    def test_skips_existing_symlinks(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.mp4").write_bytes(b"v")

        ws = WorkspaceManager("proj", tmp_path / "ws")
        ws.create()
        # First pass links it
        assert _stage_footage(ws, str(src)) == 1
        # Second pass sees the existing link and stays at 0
        assert _stage_footage(ws, str(src)) == 0

    def test_raises_on_missing_footage_dir(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path / "ws")
        ws.create()
        with pytest.raises(FileNotFoundError):
            _stage_footage(ws, str(tmp_path / "nope"))


# ─── _resolve_render_paths ───────────────────────────────────────────────────

class TestResolveRenderPaths:
    def test_returns_none_for_absent_artifacts(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        paths = _resolve_render_paths(ws)
        assert paths == {"fcpxml": None, "draft_mp4": None, "final_mp4": None}

    def test_returns_paths_for_present_artifacts(self, tmp_path):
        ws = WorkspaceManager("proj", tmp_path)
        ws.create()
        (ws.root / "fcpxml" / "edit_v1.fcpxml").parent.mkdir(exist_ok=True)
        (ws.root / "fcpxml" / "edit_v1.fcpxml").write_text("<fcpxml/>")
        (ws.root / "renders" / "draft.mp4").write_bytes(b"x")
        paths = _resolve_render_paths(ws)
        assert paths["fcpxml"].endswith("edit_v1.fcpxml")
        assert paths["draft_mp4"].endswith("draft.mp4")
        assert paths["final_mp4"] is None


# ─── StdoutEmitter ───────────────────────────────────────────────────────────

class TestStdoutEmitter:
    def _capture(self, emitter, event_type, data):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            asyncio.run(emitter(event_type, data))
        return buf.getvalue()

    def test_jsonl_mode_produces_valid_json_line(self):
        out = self._capture(
            StdoutEmitter("jsonl"),
            "tool_call", {"tool": "search_footage", "args": {"query": "foo"}},
        )
        line = out.strip()
        parsed = json.loads(line)
        assert parsed["event"] == "tool_call"
        assert parsed["data"]["tool"] == "search_footage"
        assert "timestamp" in parsed

    def test_text_mode_suppresses_scratchpad_update(self):
        out = self._capture(
            StdoutEmitter("text"),
            "scratchpad_update", {"name": "planning", "content": "x" * 5000},
        )
        assert out == ""  # intentionally silent

    def test_text_mode_suppresses_render_progress(self):
        out = self._capture(
            StdoutEmitter("text"),
            "render_progress", {"percent": 25},
        )
        assert out == ""

    def test_text_mode_prints_tool_call(self):
        out = self._capture(
            StdoutEmitter("text"),
            "tool_call", {"tool": "generate_fcpxml", "args": {}},
        )
        assert "generate_fcpxml" in out

    def test_text_mode_prints_agent_message(self):
        out = self._capture(
            StdoutEmitter("text"),
            "agent_message", {"text": "here is the final cut"},
        )
        assert "[agent]" in out
        assert "final cut" in out

    def test_text_mode_prints_tool_error(self):
        out = self._capture(
            StdoutEmitter("text"),
            "tool_result", {"tool": "generate_fcpxml", "result": {"error": "bad"}},
        )
        assert "ERROR" in out


# ─── _summarize_tool_args ────────────────────────────────────────────────────

class TestSummarizeToolArgs:
    def test_ask_memories_shows_prompt(self):
        s = _summarize_tool_args("ask_memories", {"prompt": "what is the plot?"})
        assert "what is the plot" in s

    def test_refine_clip_timestamps_shows_range(self):
        s = _summarize_tool_args(
            "refine_clip_timestamps",
            {"source_file": "v.mp4", "source_start": 10.5, "source_end": 14.2},
        )
        assert "v.mp4" in s
        assert "10.5" in s
        assert "14.2" in s

    def test_finish_turn_with_final_message(self):
        s = _summarize_tool_args("finish_turn", {"final_message": "shipped a 60s promo"})
        assert "shipped" in s

    def test_finish_turn_without_final_message(self):
        s = _summarize_tool_args("finish_turn", {})
        assert s == "(done)"

    def test_unknown_tool_returns_empty_string(self):
        assert _summarize_tool_args("no_such_tool", {}) == ""
