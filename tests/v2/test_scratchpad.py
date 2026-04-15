"""Tests for ScratchpadManager — the agent's persistent memory layer.

These lock in the update operations, the truncation cap, and the persistence
format. A refactor that changes operation semantics would silently corrupt
agent state across sessions.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.pipelines.v2.agent.scratchpad import (
    MAX_PAD_SIZE,
    PAD_NAMES,
    ScratchpadManager,
)


# ─── Basic construction ──────────────────────────────────────────────────────

class TestInit:
    def test_all_pads_start_empty(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        for name in PAD_NAMES:
            assert mgr.read(name) == ""
            assert mgr.last_updated[name] is None

    def test_creates_scratchpads_directory(self, tmp_path):
        ScratchpadManager(tmp_path)
        assert (tmp_path / "scratchpads").is_dir()

    def test_expected_pad_names(self):
        assert set(PAD_NAMES) == {
            "comprehension", "creative_direction", "planning", "fcpxml",
        }


# ─── Update operations ───────────────────────────────────────────────────────

class TestUpdateOperations:
    def test_replace_overwrites_content(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "first version")
        mgr.update("planning", "replace", "second version")
        assert mgr.read("planning") == "second version"

    def test_append_concatenates(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "append", "hello ")
        mgr.update("planning", "append", "world")
        assert mgr.read("planning") == "hello world"

    def test_prepend_puts_new_content_before_old(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "existing")
        mgr.update("planning", "prepend", "prefix-")
        assert mgr.read("planning") == "prefix-existing"

    def test_update_sets_last_updated_timestamp(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        assert mgr.last_updated["planning"] is None
        mgr.update("planning", "replace", "x")
        assert mgr.last_updated["planning"] is not None
        # ISO 8601 with timezone suffix
        assert "T" in mgr.last_updated["planning"]


# ─── Error paths ─────────────────────────────────────────────────────────────

class TestErrorPaths:
    def test_rejects_unknown_pad_name(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        result = mgr.update("not_a_pad", "replace", "x")
        assert "error" in result
        assert "Unknown scratchpad" in result["error"]
        for name in PAD_NAMES:
            assert mgr.read(name) == ""  # nothing was written

    def test_rejects_unknown_operation(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        result = mgr.update("planning", "clobber", "x")  # type: ignore
        assert "error" in result
        assert "Unknown operation" in result["error"]

    def test_read_unknown_pad_returns_placeholder(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        assert "unknown scratchpad" in mgr.read("nope").lower()


# ─── Truncation cap ──────────────────────────────────────────────────────────

class TestTruncation:
    def test_pad_below_cap_is_untouched(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "x" * (MAX_PAD_SIZE - 10))
        assert len(mgr.read("planning")) == MAX_PAD_SIZE - 10

    def test_pad_over_cap_is_truncated_from_start(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        long = ("A" * 100) + ("B" * (MAX_PAD_SIZE + 500))
        result = mgr.update("planning", "replace", long)
        assert len(mgr.read("planning")) == MAX_PAD_SIZE
        # The leading A's should have been dropped; the final character is a B.
        assert mgr.read("planning")[-1] == "B"
        assert mgr.read("planning")[0] == "B"
        assert result.get("truncated") is True
        assert result["truncated_chars"] == 100 + 500  # the dropped prefix
        assert "warning" in result

    def test_truncation_via_repeated_appends(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        for _ in range(3):
            mgr.update("planning", "append", "Z" * 3000)
        assert len(mgr.read("planning")) == MAX_PAD_SIZE


# ─── Persistence across instances ────────────────────────────────────────────

class TestPersistence:
    def test_content_survives_new_manager(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "durable")
        mgr.update("comprehension", "replace", "also durable")

        fresh = ScratchpadManager(tmp_path)
        assert fresh.read("planning") == "durable"
        assert fresh.read("comprehension") == "also durable"
        assert fresh.last_updated["planning"] is not None

    def test_pad_file_is_markdown(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "# heading\n- item")
        pad_file = tmp_path / "scratchpads" / "planning.md"
        assert pad_file.exists()
        assert pad_file.read_text() == "# heading\n- item"


# ─── render_all ──────────────────────────────────────────────────────────────

class TestRenderAll:
    def test_empty_pads_render_as_placeholder(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        out = mgr.render_all()
        for name in PAD_NAMES:
            assert f"SCRATCHPAD: {name}" in out
        assert "(empty)" in out

    def test_populated_pad_content_appears(self, tmp_path):
        mgr = ScratchpadManager(tmp_path)
        mgr.update("planning", "replace", "shot list goes here")
        out = mgr.render_all()
        assert "shot list goes here" in out
