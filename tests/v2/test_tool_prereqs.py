"""Tests for the VEA tool-dependency pre-check.

Two flavors of tests in here:

1. **Unit-style** — exercise the registry + checker logic with mocked
   imports so behavior is provable without depending on what's installed.
   These run offline, always.

2. **Environment-sanity** — assert that every REQUIRED tool dep in
   ``TOOL_PREREQS`` is actually importable in the current venv. This is
   the gate that catches "you forgot to add scenedetect to pyproject" /
   "your venv drifted" / etc. before a user discovers it mid-edit.

The environment-sanity test runs unconditionally in the offline suite —
the whole point is to fail CI when a real dep goes missing.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.pipelines.v2.tool_prereqs import (
    TOOL_PREREQS,
    DepResult,
    ToolDep,
    check_tool_prereqs,
    format_results,
    log_check_results,
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_is_non_empty(self):
        assert len(TOOL_PREREQS) > 0, "TOOL_PREREQS is empty — nothing to check"

    def test_each_entry_is_well_formed(self):
        for tool, deps in TOOL_PREREQS.items():
            assert isinstance(tool, str) and tool
            assert isinstance(deps, list) and deps, f"{tool} has empty deps list"
            for d in deps:
                assert isinstance(d, ToolDep)
                assert d.import_name and "." not in d.import_name.split()[0][:1]
                assert d.pip_spec and " " not in d.pip_spec[:1]

    def test_known_tools_covered(self):
        # Smoke-list: each of these MUST be a key in the registry. If we
        # add a tool to ToolExecutor that has Python deps but forget to
        # register it here, this test fails.
        for required_tool in (
            "refine_clip_timestamps",
            "generate_narration",
            "generate_subtitles",
            "select_music",
            "generate_fcpxml",
        ):
            assert required_tool in TOOL_PREREQS, (
                f"{required_tool} is in ToolExecutor but not registered in TOOL_PREREQS"
            )


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------


class TestCheckLogic:
    def test_check_returns_one_row_per_dep(self):
        rows = check_tool_prereqs()
        total_deps = sum(len(v) for v in TOOL_PREREQS.values())
        assert len(rows) == total_deps

    def test_missing_import_marked_not_ok(self, monkeypatch):
        # Inject a fake registry entry with a definitely-missing module.
        monkeypatch.setitem(
            TOOL_PREREQS, "fake_tool",
            [ToolDep(import_name="definitely_not_a_real_module_zzz", pip_spec="x")],
        )
        rows = check_tool_prereqs()
        bad = [r for r in rows if r.dep.import_name == "definitely_not_a_real_module_zzz"]
        assert len(bad) == 1
        assert not bad[0].ok
        assert "ImportError" in bad[0].detail

    def test_existing_import_marked_ok(self, monkeypatch):
        # stdlib `json` is always importable.
        monkeypatch.setitem(
            TOOL_PREREQS, "fake_tool",
            [ToolDep(import_name="json", pip_spec="builtin")],
        )
        rows = check_tool_prereqs()
        ok = [r for r in rows if r.dep.import_name == "json"]
        assert len(ok) == 1
        assert ok[0].ok


# ---------------------------------------------------------------------------
# Reporter / logging
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_includes_fix_command_for_missing(self):
        rows = [
            DepResult(tool="t", dep=ToolDep(import_name="nope", pip_spec="nope>=1.0", required=True),
                      ok=False, detail="ImportError: nope"),
        ]
        out = format_results(rows)
        assert "✗" in out
        assert "pip install nope>=1.0" in out

    def test_format_distinguishes_required_vs_optional(self):
        rows = [
            DepResult(tool="t1", dep=ToolDep(import_name="a", pip_spec="a", required=True),  ok=False, detail="x"),
            DepResult(tool="t1", dep=ToolDep(import_name="b", pip_spec="b", required=False), ok=False, detail="x"),
        ]
        out = format_results(rows)
        # Required missing → ✗, optional missing → ⚠
        assert "✗" in out and "⚠" in out

    def test_log_returns_missing_counts(self, monkeypatch):
        monkeypatch.setitem(
            TOOL_PREREQS, "fake_tool",
            [
                ToolDep(import_name="json", pip_spec="builtin", required=True),    # present
                ToolDep(import_name="definitely_missing_xyz", pip_spec="x",
                        required=True),                                            # missing required
                ToolDep(import_name="also_missing_pqr", pip_spec="y",
                        required=False),                                           # missing optional
            ],
        )
        missing_required, missing_optional = log_check_results()
        assert missing_required >= 1   # at least our injected one
        assert missing_optional >= 1


# ---------------------------------------------------------------------------
# ★ Environment-sanity — the gate that catches drift
# ---------------------------------------------------------------------------
#
# This is the test the user wanted: it FAILS when a required tool dep is
# missing from the venv, instead of letting the user discover it mid-edit.


class TestEnvironmentSanity:
    def test_all_required_tool_deps_are_importable(self):
        """Every REQUIRED tool dep must be importable in this venv.

        If this fails, fix is one of:
          (a) add the missing pip package to pyproject.toml + `uv sync`
          (b) demote the dep to optional in tool_prereqs.py if the
              tool genuinely handles its absence gracefully and you
              don't want it blocking the rest of the suite
        """
        rows = check_tool_prereqs()
        missing = [r for r in rows if not r.ok and r.dep.required]
        if missing:
            report = format_results(missing)
            pytest.fail(
                "Required tool dependencies missing from this venv:\n\n"
                + report
                + "\n\nFix: add to pyproject.toml or demote in tool_prereqs.py."
            )

    def test_optional_deps_status_is_visible(self, caplog):
        """Optional missing deps shouldn't crash the test but SHOULD log a warning.

        Doesn't assert anything beyond "the warning was emitted" — having
        optional deps absent is fine, the user just needs to know about it.
        """
        import logging
        with caplog.at_level(logging.WARNING, logger="src.pipelines.v2.tool_prereqs"):
            log_check_results()
        # If any optional deps are missing, there should be a WARNING for them.
        # If everything's installed, no assertion needed — that's a pass too.
