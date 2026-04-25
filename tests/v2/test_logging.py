"""Tests for per-workspace structured logging.

These pure-function tests are cheap and cover the contract that matters:
- Secret values get redacted before we write anything to disk.
- Log records land in the active workspace's backend.jsonl, one per line.
- Switching scope correctly routes records to the second workspace.
- Clearing the scope drops records rather than leaking them somewhere.
- Manifest captures the runtime fingerprint we need for bug reports.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.pipelines.v2.logging_setup import (
    WorkspaceLogScope,
    append_llm_event,
    open_render_log,
    redact,
    write_manifest,
)


class TestRedact:
    def test_passthrough_safe_values(self):
        assert redact({"a": 1, "b": "hello"}) == {"a": 1, "b": "hello"}

    def test_masks_secret_keys(self):
        got = redact({"api_key": "sk-abcd", "API-KEY": "x", "nested": {"authorization": "y"}})
        assert got["api_key"] == "***"
        assert got["API-KEY"] == "***"
        assert got["nested"]["authorization"] == "***"

    def test_preserves_plural_tokens(self):
        # ``tokens`` is the LLM response usage dict (prompt/output counts),
        # not a secret. The regex was too greedy in an earlier version.
        got = redact({"tokens": {"prompt": 123, "output": 45}})
        assert got["tokens"] == {"prompt": 123, "output": 45}

    def test_still_masks_token_variants(self):
        got = redact({
            "token": "sk-x",
            "access_token": "sk-y",
            "refresh_token": "sk-z",
        })
        assert got["token"] == "***"
        assert got["access_token"] == "***"
        assert got["refresh_token"] == "***"

    def test_masks_bearer_in_free_form_string(self):
        got = redact("Authorization: Bearer sk-super-secret-token-abc123")
        assert "Bearer ***" in got
        assert "sk-super-secret-token-abc123" not in got

    def test_survives_depth_limit(self):
        # A cycle would blow Python's recursion before ours — build a deep
        # but finite dict so the depth guard is exercised.
        deep = cur = {}
        for _ in range(20):
            cur["next"] = {}
            cur = cur["next"]
        # Shouldn't raise.
        redact(deep)


class TestWorkspaceLogScope:
    def test_routes_records_to_active_workspace(self, tmp_path: Path):
        ws = tmp_path / "projA"
        ws.mkdir()
        test_logger = logging.getLogger("vea.test.scope1")
        test_logger.setLevel(logging.INFO)
        with WorkspaceLogScope(ws):
            test_logger.info("hello from scope")
        log = ws / "logs" / "backend.jsonl"
        assert log.exists(), "backend.jsonl should be created on first emit"
        lines = log.read_text().strip().splitlines()
        assert any("hello from scope" in line for line in lines)
        parsed = [json.loads(line) for line in lines]
        assert all("ts" in p and "level" in p and "msg" in p for p in parsed)

    def test_drops_records_outside_scope(self, tmp_path: Path):
        # Without a scope, logging must NOT touch any workspace on disk.
        logger = logging.getLogger("vea.test.nowhere")
        logger.info("floating record")
        # Anything under tmp_path should be untouched.
        assert not any(tmp_path.rglob("backend.jsonl"))

    def test_nested_scopes_route_correctly(self, tmp_path: Path):
        a = tmp_path / "A"; a.mkdir()
        b = tmp_path / "B"; b.mkdir()
        logger = logging.getLogger("vea.test.nested")
        with WorkspaceLogScope(a):
            logger.info("into-A")
            with WorkspaceLogScope(b):
                logger.info("into-B")
            logger.info("back-to-A")
        a_text = (a / "logs" / "backend.jsonl").read_text()
        b_text = (b / "logs" / "backend.jsonl").read_text()
        assert "into-A" in a_text and "back-to-A" in a_text
        assert "into-A" not in b_text
        assert "into-B" in b_text
        assert "into-B" not in a_text


class TestAppendLlmEvent:
    def test_writes_jsonl_and_redacts(self, tmp_path: Path):
        ws = tmp_path / "proj"
        append_llm_event(ws, {
            "role": "main_llm",
            "model": "claude-opus",
            "api_key": "sk-leak",
            "response": {"text": "ok"},
        })
        path = ws / "logs" / "llm.jsonl"
        assert path.exists()
        record = json.loads(path.read_text().strip())
        assert record["api_key"] == "***"
        assert record["role"] == "main_llm"
        assert record["response"] == {"text": "ok"}
        assert "ts" in record


class TestOpenRenderLog:
    def test_returns_fresh_path_under_logs_renders(self, tmp_path: Path):
        ws = tmp_path / "proj"
        p1 = open_render_log(ws, "ffmpeg-draft")
        assert p1.parent == ws / "logs" / "renders"
        assert p1.name.startswith("ffmpeg-draft-")
        assert p1.name.endswith(".log")
        # Parent dir is created even if the log itself hasn't been written yet.
        assert p1.parent.is_dir()

    def test_sanitizes_renderer_name(self, tmp_path: Path):
        p = open_render_log(tmp_path, "weird/name:with bad chars")
        assert "/" not in p.name
        assert ":" not in p.name


class TestWriteManifest:
    def test_captures_env_fingerprint(self, tmp_path: Path):
        ws = tmp_path / "proj"
        write_manifest(
            ws,
            project_name="demo",
            mode="autonomous",
            models={"main_llm": "claude", "video_llm": "gemini"},
        )
        m = json.loads((ws / "logs" / "manifest.json").read_text())
        assert m["project_name"] == "demo"
        assert m["mode"] == "autonomous"
        assert m["models"] == {"main_llm": "claude", "video_llm": "gemini"}
        # These three fields drive 90% of support triage.
        assert "git_sha" in m
        assert "ffmpeg_version" in m
        assert "platform" in m
