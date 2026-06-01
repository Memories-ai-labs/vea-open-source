"""VEA v2 one-shot CLI.

Runs the agent loop non-interactively on a brief + footage and writes a final
JSON summary to stdout. Designed to be invoked by another agent (orchestrator)
via subprocess, MCP, or a shell pipeline.

Typical use:

    python -m src.cli \\
        --project promo \\
        --brief "make a 60-second promo with narration and music" \\
        --footage-dir ./clips

Flags:
    --project NAME           (required) project name; footage lives in
                             data/workspaces/NAME/footage/
    --brief TEXT             (required) the editing brief for the agent
    --footage-dir DIR        if set, symlink every video in DIR into the
                             project's footage/ before running
    --reuse-index            skip indexing if session.json exists
    --log-format {text,jsonl} how progress is rendered on stdout
    --timeout S              hard wall-clock limit in seconds

The agent's tool-call round cap is mode-driven (collaborative 40 / autonomous
120, see pipelines/v2/agent/modes.py); the CLI runs in autonomous mode and does
not expose a flag to override it.

The last line of stdout is always a single JSON object with the final state:

    {"status": "ok", "project": "promo",
     "fcpxml": "/abs/path", "ffmpeg_mp4": "/abs/path", "resolve_mp4": "/abs/path",
     "edit_decision": {...}}

Non-zero exit on any unrecoverable failure; the JSON still prints with
``status: "error"`` so the caller gets a structured signal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Importing src.config loads config.json into os.environ, so must happen
# before anything that reads env (services, LLMs).
from src import config as _config
from src import services
from src.pipelines.v2.agent.agent_session import AgentSession
from src.pipelines.v2.agent.modes import AgentMode
from src.pipelines.v2.comprehension.lightweight_comprehension import (
    LightweightComprehension,
)
from src.pipelines.v2.workspace import WorkspaceManager, VIDEO_EXTS

logger = logging.getLogger(__name__)

# How long we wait for background renders (draft + Resolve final) after the
# agent's finish_turn before giving up on them. Resolve can be slow; a draft
# MP4 finishes in seconds.
DEFAULT_RENDER_WAIT_SECONDS = 300


# ─── Progress emitters ───────────────────────────────────────────────────────

class StdoutEmitter:
    """Emitter callable the AgentSession hands events to.

    Two modes:
    - ``text``: human-readable single line per event.
    - ``jsonl``: each event is a JSON line, stable schema for programmatic
      consumers (piping to jq, parsing from a parent agent, etc.).

    The agent fires a LOT of events (search, refine, scratchpad updates). We
    keep the text mode terse — one line per meaningful event, suppressing
    low-signal ones.
    """

    _TEXT_SUPPRESS = {
        "scratchpad_update",     # too chatty; scratchpad contents re-rendered each turn
        "render_progress",       # polling noise
    }

    def __init__(self, fmt: str = "text") -> None:
        self.fmt = fmt

    async def __call__(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.fmt == "jsonl":
            sys.stdout.write(json.dumps({
                "event": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }) + "\n")
            sys.stdout.flush()
            return

        if event_type in self._TEXT_SUPPRESS:
            return

        line = self._format_text(event_type, data)
        if line:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    @staticmethod
    def _format_text(event_type: str, data: Dict[str, Any]) -> str:
        # Compact summaries for common event types. Unknown types fall through
        # to a minimal one-liner so we never drop signal silently.
        if event_type == "agent_message":
            msg = (data.get("text") or "").replace("\n", " ")
            return f"[agent] {msg[:200]}"
        if event_type == "tool_call":
            tool = data.get("tool", "?")
            args = data.get("args", {})
            hint = _summarize_tool_args(tool, args)
            return f"[tool] {tool} {hint}".rstrip()
        if event_type == "tool_result":
            tool = data.get("tool", "?")
            result = data.get("result", {}) or {}
            if "error" in result:
                return f"[tool] {tool} ERROR: {str(result['error'])[:200]}"
            return ""  # successful tool result is implied by the next tool_call
        if event_type == "render_start":
            return f"[render] {data.get('kind', 'render')} started"
        if event_type == "render_complete":
            p = data.get("filename") or data.get("path") or ""
            return f"[render] complete: {p}"
        if event_type == "render_error":
            return f"[render] error: {str(data.get('error', ''))[:200]}"
        if event_type == "error":
            return f"[error] {str(data.get('message', ''))[:200]}"
        if event_type == "user_message":
            # We echo the brief once at the top from the CLI, no need to repeat
            return ""
        return ""


def _summarize_tool_args(tool: str, args: Dict[str, Any]) -> str:
    """Produce a short one-line hint of what the tool call is doing."""
    if tool == "ask_memories":
        return f"\"{str(args.get('prompt', ''))[:80]}\""
    if tool == "search_footage":
        return f"\"{str(args.get('query', ''))[:80]}\""
    if tool == "refine_clip_timestamps":
        return f"{args.get('source_file', '?')} [{args.get('source_start')}-{args.get('source_end')}]"
    if tool == "update_scratchpad":
        return f"{args.get('name', '?')} ({args.get('operation', '?')})"
    if tool == "generate_fcpxml":
        return "compiling"
    if tool == "generate_narration":
        script = str(args.get("script", ""))
        return f"{len(script)} chars"
    if tool == "select_music":
        return f"\"{str(args.get('prompt', ''))[:60]}\""
    if tool == "message_user":
        return f"\"{str(args.get('message', ''))[:80]}\""
    if tool == "finish_turn":
        final = str(args.get("final_message", ""))
        return f"→ {final[:80]}" if final else "(done)"
    return ""


# ─── Footage setup ────────────────────────────────────────────────────────────

def _stage_footage(workspace: WorkspaceManager, footage_dir: Optional[str]) -> int:
    """Symlink every video file from ``footage_dir`` into the project's
    footage/ directory. Returns the count of files linked. If the link
    already exists or points at the same inode, skip.

    If ``footage_dir`` is None, do nothing — caller is expected to have
    placed footage in workspace/footage/ already.
    """
    dest_dir = workspace.get_footage_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if footage_dir is None:
        return 0

    src_dir = Path(footage_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--footage-dir does not exist or is not a directory: {src_dir}")

    linked = 0
    for p in sorted(src_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        dest = dest_dir / p.name
        if dest.exists() or dest.is_symlink():
            continue
        os.symlink(p, dest)
        linked += 1
    return linked


# ─── Indexing ─────────────────────────────────────────────────────────────────

async def _ensure_indexed(
    workspace: WorkspaceManager,
    emitter: StdoutEmitter,
    reuse: bool,
) -> None:
    """Run the v2 LightweightComprehension pipeline if needed.

    If ``reuse`` and ``session.json`` is already present, we trust the cached
    session. Otherwise (or if ``reuse=False``), run indexing from scratch.
    """
    # lvmm-core has to be initialised before indexing. The CLI runs outside
    # FastAPI's lifespan hook so we trigger it here on first need.
    # Even when --reuse-index skips indexing, the agent still needs these
    # handles for ask_memories/search_footage during the edit turn.
    await services.init_lvmm()
    if not services.mavi_agent or not services.lvmm_ctx:
        raise RuntimeError(
            "lvmm-core failed to initialise. Check OPENROUTER_API_KEY / "
            "GEMINI_API_KEY in .env and the server logs."
        )

    if reuse and workspace.exists():
        await emitter("index_skipped", {"reason": "session.json exists and --reuse-index is set"})
        return

    # Tool-level dependency check — same as app.py lifespan does for the
    # dashboard. Catches missing scenedetect/librosa/etc. before the
    # agent loop discovers them mid-run.
    from src.pipelines.v2.tool_prereqs import log_check_results
    log_check_results()

    pipeline = LightweightComprehension(
        project_name=workspace.project_name,
        source_dir=str(workspace.get_footage_dir()),
        lvmm_ctx=services.lvmm_ctx,
        mavi_agent=services.mavi_agent,
        workspace=workspace,
    )

    async def _progress(pct: float, msg: str) -> None:
        await emitter("index_progress", {"percent": pct, "message": msg})

    await emitter("index_start", {"project": workspace.project_name})
    await pipeline.run(start_fresh=not reuse, progress_callback=_progress)
    await emitter("index_complete", {"project": workspace.project_name})


# ─── Main flow ────────────────────────────────────────────────────────────────

def _resolve_render_paths(workspace: WorkspaceManager) -> Dict[str, Optional[str]]:
    """Return absolute paths for fcpxml + renders if they exist."""
    fcpxml = workspace.root / "fcpxml" / "edit_v1.fcpxml"
    ffmpeg_mp4 = workspace.root / "renders" / "ffmpeg.mp4"
    resolve_mp4 = workspace.root / "renders" / "resolve.mp4"
    return {
        "fcpxml": str(fcpxml) if fcpxml.exists() else None,
        "ffmpeg_mp4": str(ffmpeg_mp4) if ffmpeg_mp4.exists() else None,
        "resolve_mp4": str(resolve_mp4) if resolve_mp4.exists() else None,
    }


def _load_edit_decision(workspace: WorkspaceManager) -> Optional[Dict[str, Any]]:
    p = workspace.root / "fcpxml" / "edit_decision.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


async def _run(args: argparse.Namespace) -> int:
    emitter = StdoutEmitter(fmt=args.log_format)

    workspaces_root = Path(_config.WORKSPACES_DIR).resolve()
    workspace = WorkspaceManager(args.project, workspaces_root)

    try:
        # Ensure the project directory exists.
        if not workspace.dir_exists():
            workspace.create()
            await emitter("project_created", {"project": args.project, "path": str(workspace.root)})

        linked = _stage_footage(workspace, args.footage_dir)
        if linked:
            await emitter("footage_staged", {"linked": linked})

        footage = workspace.scan_footage()
        if not footage:
            _print_result({"status": "error", "error": "No footage found", "project": args.project})
            return 2

        # Indexing (cached unless --reuse-index is false or session missing).
        try:
            await _ensure_indexed(workspace, emitter, reuse=args.reuse_index)
        except Exception as e:
            logger.exception("Indexing failed")
            _print_result({"status": "error", "error": f"indexing: {e}", "project": args.project})
            return 3

        # Load the session that indexing produced (or reused).
        session_data = workspace.load_session()
        if not session_data.videos:
            _print_result({"status": "error", "error": "No videos in session after indexing", "project": args.project})
            return 4

        # Build the agent session in autonomous mode — CLI runs are non-interactive
        # by definition (no WebSocket back-channel for the agent to message).
        # lvmm-core handles came up during _ensure_indexed; if they're still None
        # something went sideways and the build below will surface a clear error.
        agent = AgentSession(
            project_name=args.project,
            workspace=workspace,
            mavi_agent=services.mavi_agent,
            querier=services.querier,
            gemini_manager=services.main_llm,
            video_llm=services.video_llm,
            video_entries=session_data.videos,
            emit=emitter,
            mode=AgentMode.AUTONOMOUS,
        )

        # One-shot: deliver the brief and wait.
        try:
            await asyncio.wait_for(
                agent.handle_user_message(args.brief),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError:
            _print_result({
                "status": "error",
                "error": f"Timed out after {args.timeout}s",
                "project": args.project,
                **_resolve_render_paths(workspace),
            })
            return 5
        except Exception as e:
            logger.exception("Agent loop failed")
            _print_result({
                "status": "error",
                "error": f"agent: {e}",
                "project": args.project,
                **_resolve_render_paths(workspace),
            })
            return 6

        # Background tasks may include the ffmpeg draft render and Resolve final
        # render. Wait for them with a cap so we don't hang forever on Resolve.
        if agent._bg_tasks:
            await emitter("render_wait", {"pending": len(agent._bg_tasks)})
            try:
                await asyncio.wait_for(
                    asyncio.gather(*agent._bg_tasks, return_exceptions=True),
                    timeout=DEFAULT_RENDER_WAIT_SECONDS,
                )
            except asyncio.TimeoutError:
                await emitter("render_wait_timeout", {"seconds": DEFAULT_RENDER_WAIT_SECONDS})

        # Final result.
        paths = _resolve_render_paths(workspace)
        result = {
            "status": "ok",
            "project": args.project,
            "edit_decision": _load_edit_decision(workspace),
            **paths,
        }
        _print_result(result)
        return 0 if paths["fcpxml"] else 7
    finally:
        await services.close_lvmm()


def _print_result(payload: Dict[str, Any]) -> None:
    """Write the single-line final-status JSON to stdout. Always the last line."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vea-oneshot",
        description="Run the VEA v2 agent non-interactively on a brief + footage.",
    )
    p.add_argument("--project", required=True, help="Project name (workspace directory)")
    p.add_argument("--brief", required=True, help="Editing brief for the agent")
    p.add_argument(
        "--footage-dir",
        default=None,
        help="Directory of video files to symlink into the project's footage/",
    )
    p.add_argument(
        "--reuse-index",
        action="store_true",
        help="Skip indexing if session.json exists. Default re-indexes.",
    )
    p.add_argument(
        "--log-format",
        choices=("text", "jsonl"),
        default="text",
        help="Progress output format on stdout (default: text)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Hard wall-clock timeout in seconds for the agent loop (default: 900)",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("VEA_CLI_LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
