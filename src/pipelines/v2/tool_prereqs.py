"""Tool-level dependency pre-check for VEA v2's agent tools.

Background
----------
Several V2 tools do their Python imports lazily inside the executor body
(e.g. ``scenedetect`` in ``refine_clip_timestamps``, ``librosa`` in
``select_music``). When one of those packages isn't installed, the tool
*degrades gracefully* with a warning instead of crashing — which is the
right behavior at runtime, but **silently leaves a worse-quality result
on the user's screen**. The user discovers it mid-edit, not at deploy.

This module is the opposite end of that contract: a registry of "which
tools depend on which Python packages", plus a checker that surfaces
every missing dep in one place. Wire the checker into:

* **App startup** (``app.py`` lifespan) — logs a banner with status per
  tool. Missing optional deps log WARNING; missing required ones log
  ERROR but don't crash the server (the user can still index + chat,
  just not use the affected tool).
* **CLI startup** (``cli.py`` _run) — same, before the agent loop
  starts, so a one-shot run prints "scenedetect missing" on stderr
  before doing 5 minutes of work that produces a degraded result.
* **Smoke test** (``tests/v2/test_tool_prereqs.py``) — hard-fails the
  test suite when any tool dep is missing from the dev venv. This is
  the gate that catches the issue at CI time instead of letting it
  reach a user.

What's NOT in here
------------------
- lvmm-core's own prereqs (``services.init_lvmm()`` handles those).
- External binaries (``ffmpeg``) — main's ``logging_setup`` /
  ``ffmpeg_renderer`` already shutils.which() at first use.
- Env vars / API keys — those are handled at the LLM-init layer.

How to add a new tool
---------------------
When you add a new V2 tool to ``ToolExecutor`` that has its own Python
deps, add an entry below. Three tuples: (tool name, [(import name,
pip spec)], required-or-optional). The check formats the output and
prints actionable pip install commands.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolDep:
    """One Python-import dependency for one VEA tool."""

    import_name: str          # e.g. "scenedetect" — what ``importlib`` resolves
    pip_spec: str             # e.g. "scenedetect>=0.6.0" — for the install hint
    required: bool = True     # True → ERROR on missing; False → WARNING
    note: str = ""            # one-line context for the install hint


# Registry. Keyed by the V2 tool name (matches what ToolExecutor.execute
# dispatches on, so the operator can grep both this file and tools.py).
TOOL_PREREQS: dict[str, list[ToolDep]] = {
    "refine_clip_timestamps": [
        ToolDep(
            import_name="scenedetect",
            pip_spec="scenedetect>=0.6.0",
            required=False,
            note="Without it, refine_clip_timestamps degrades to STT-only word boundaries (still works, less accurate at scene changes).",
        ),
        ToolDep(
            import_name="elevenlabs",
            pip_spec="elevenlabs>=1.0.0",
            required=True,
            note="Required for the ElevenLabs STT word grid that the refine tool's video-LLM pass aligns against.",
        ),
    ],
    "generate_narration": [
        ToolDep(
            import_name="elevenlabs",
            pip_spec="elevenlabs>=1.0.0",
            required=True,
            note="ElevenLabs TTS — narration tool is non-functional without it.",
        ),
    ],
    "generate_subtitles": [
        ToolDep(
            import_name="elevenlabs",
            pip_spec="elevenlabs>=1.0.0",
            required=True,
            note="ElevenLabs Scribe STT — subtitles tool is non-functional without it.",
        ),
    ],
    "select_music": [
        ToolDep(
            import_name="librosa",
            pip_spec="librosa>=0.10.0",
            required=False,
            note="librosa powers the beat-grid detection. Without it, select_music returns the track but no beats[]/tempo_bpm.",
        ),
    ],
    "generate_fcpxml": [
        ToolDep(
            import_name="pyloudnorm",
            pip_spec="pyloudnorm>=0.1.1",
            required=True,
            note="ITU-R BS.1770 LUFS measurement for narration/music auto-gain. Required by ffmpeg_renderer.",
        ),
    ],
}


@dataclass
class DepResult:
    """One row of a check pass."""

    tool: str
    dep: ToolDep
    ok: bool
    detail: str = ""   # "importable", or the ImportError message


def check_tool_prereqs() -> list[DepResult]:
    """Resolve every entry in :data:`TOOL_PREREQS`. Returns one row per dep.

    Pure function — doesn't log or raise. Callers (startup, CLI, smoke
    test) decide what to do with the results.
    """
    rows: list[DepResult] = []
    for tool, deps in TOOL_PREREQS.items():
        for dep in deps:
            try:
                importlib.import_module(dep.import_name)
                rows.append(DepResult(tool=tool, dep=dep, ok=True, detail="importable"))
            except ImportError as e:
                rows.append(DepResult(tool=tool, dep=dep, ok=False, detail=f"ImportError: {e}"))
    return rows


def format_results(rows: list[DepResult]) -> str:
    """Pretty-print a results list with ✓/✗ + per-failure pip hints."""
    if not rows:
        return "(no tool prereqs registered)"
    by_tool: dict[str, list[DepResult]] = {}
    for r in rows:
        by_tool.setdefault(r.tool, []).append(r)
    out = ["VEA tool dependency pre-check:"]
    for tool, group in by_tool.items():
        out.append(f"  {tool}")
        for r in group:
            mark = "✓" if r.ok else ("✗" if r.dep.required else "⚠")
            req_label = "required" if r.dep.required else "optional"
            out.append(f"    {mark} {r.dep.import_name:<24} ({req_label})  {r.detail}")
            if not r.ok:
                out.append(f"        fix:  pip install {r.dep.pip_spec}")
                if r.dep.note:
                    out.append(f"        note: {r.dep.note}")
    return "\n".join(out)


def log_check_results(rows: list[DepResult] | None = None) -> tuple[int, int]:
    """Log results at INFO (full table) + ERROR/WARNING per missing dep.

    Returns ``(num_missing_required, num_missing_optional)`` so a caller
    can decide whether to refuse to start.
    """
    if rows is None:
        rows = check_tool_prereqs()
    logger.info("[TOOL PREREQS]\n%s", format_results(rows))
    missing_required = 0
    missing_optional = 0
    for r in rows:
        if r.ok:
            continue
        if r.dep.required:
            missing_required += 1
            logger.error(
                "[TOOL PREREQS] %s requires %s — install with: pip install %s",
                r.tool, r.dep.import_name, r.dep.pip_spec,
            )
        else:
            missing_optional += 1
            logger.warning(
                "[TOOL PREREQS] %s has optional dep %s missing — "
                "tool will degrade. Install with: pip install %s",
                r.tool, r.dep.import_name, r.dep.pip_spec,
            )
    return missing_required, missing_optional
