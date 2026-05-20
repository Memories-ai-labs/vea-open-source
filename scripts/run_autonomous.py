#!/usr/bin/env python3
"""
Standalone CLI for VEA V2 autonomous mode.

Runs Phase 1 (lightweight comprehension — local indexing + gist) and Phase 2
(iterative planning loop — Storyboard generation) end-to-end on a workspace
without the FastAPI server or React dashboard. Useful for batch / scripted
edits, CI / eval, and the "I just want a storyboard out of a folder of clips"
researcher path.

Phase 3+ (narration / music / FCPXML / preview) is deterministic and runs
independently — invoke separately or via the existing route handlers.

PORT NOTE (2026-05-19): All retrieval / chat goes through lvmm-core (local
SQLite + MaviAgent + Searcher) — no memories.ai API calls. Gemini is still
needed for Phase 2's planning prompts (decide tool calls + update storyboard).

Usage::

    python scripts/run_autonomous.py \\
        --project my_project \\
        --prompt "Create a 60-second highlight reel of the keynote" \\
        --source-dir /path/to/my/footage \\
        --max-iterations 5 \\
        --target-duration 60

The workspace lands under ``data/workspaces/{project}/`` per VEA convention.
If ``--source-dir`` is omitted, the script reads from the workspace's
``footage/`` subdirectory.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make `src` and `lib` importable when invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run VEA V2 autonomous mode (index + plan) without the web server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project", required=True, help="Project name (= workspace dir name).")
    p.add_argument(
        "--prompt", required=True,
        help="User prompt for the planning loop (e.g. \"60-second highlight reel\").",
    )
    p.add_argument(
        "--source-dir", default=None,
        help="Directory holding source videos. Defaults to data/workspaces/{project}/footage/.",
    )
    p.add_argument("--max-iterations", type=int, default=5, help="Planning loop iteration cap.")
    p.add_argument(
        "--target-duration", type=float, default=120.0,
        help="Target edit duration in seconds (passed to storyboard prompt).",
    )
    p.add_argument(
        "--start-fresh", action="store_true",
        help="Force re-index even if a cached session exists.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return p


async def _on_progress(percent: float, message: str) -> None:
    """Simple stdout progress reporter — replaces the WebSocket dashboard."""
    print(f"  [{percent:5.1f}%] {message}", flush=True)


async def main_async(args: argparse.Namespace) -> int:
    # Imports kept inside main_async so --help works without pulling lvmm-core.
    from src import config as _config
    from src import services
    from src.pipelines.v2.workspace import WorkspaceManager
    from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension
    from src.pipelines.v2.planning.iterative_planning_loop import IterativePlanningLoop

    # Initialise lvmm-core (lvmm_ctx + searcher + mavi_agent).
    print("→ Initialising lvmm-core (loading adapters, opening local DB)…", flush=True)
    try:
        await services.init_lvmm()
    except Exception as e:
        print(f"✗ lvmm-core init failed: {e}", file=sys.stderr)
        return 1
    if services.mavi_agent is None or services.searcher is None:
        print("✗ lvmm-core init reported success but mavi_agent/searcher are None.", file=sys.stderr)
        return 1
    if services.gemini_manager is None:
        print(
            "✗ Gemini manager not configured. Set OPENROUTER_API_KEY or "
            "GOOGLE_CLOUD_PROJECT in config.json / env before running.",
            file=sys.stderr,
        )
        return 1

    workspace = WorkspaceManager(args.project, _config.WORKSPACES_DIR)
    source_dir = args.source_dir
    if not source_dir:
        if not workspace.exists():
            workspace.create()
        source_dir = str(workspace.get_footage_dir())
        if not Path(source_dir).is_dir() or not any(Path(source_dir).iterdir()):
            print(
                f"✗ No --source-dir provided and {source_dir} is empty. "
                f"Drop video files there or pass --source-dir.",
                file=sys.stderr,
            )
            return 1

    try:
        # --- Phase 1: local comprehension ---
        print(f"\n=== Phase 1: indexing videos in {source_dir} ===", flush=True)
        comprehension = LightweightComprehension(
            project_name=args.project,
            source_dir=source_dir,
            lvmm_ctx=services.lvmm_ctx,
            mavi_agent=services.mavi_agent,
            workspace=workspace,
        )
        session = await comprehension.run(
            start_fresh=args.start_fresh,
            progress_callback=_on_progress,
        )
        print(
            f"✓ Phase 1 done: {len(session.videos)} video(s) indexed, "
            f"gist={len(session.gist)} chars",
            flush=True,
        )

        if not session.videos:
            print("✗ No videos indexed — cannot proceed to planning.", file=sys.stderr)
            return 1

        # --- Phase 2: iterative planning ---
        print(
            f"\n=== Phase 2: planning (max {args.max_iterations} iterations, "
            f"target {args.target_duration:.0f}s) ===",
            flush=True,
        )
        loop_obj = IterativePlanningLoop(
            project_name=args.project,
            user_prompt=args.prompt,
            workspace=workspace,
            searcher=services.searcher,
            mavi_agent=services.mavi_agent,
            gemini=services.gemini_manager,
            video_nos=[v.video_no for v in session.videos],
            video_entries=session.videos,
            max_iterations=args.max_iterations,
            target_duration_seconds=args.target_duration,
            event_queue=None,    # no dashboard subscriber
            pause_event=None,
            inject_prompt_queue=None,
        )
        storyboard = await loop_obj.run()

        # --- Final summary ---
        total_duration = sum(s.duration_seconds for s in storyboard.shots)
        print(
            f"\n✓ Phase 2 done: {storyboard.iteration} iteration(s), "
            f"{len(storyboard.shots)} shot(s), "
            f"{total_duration:.1f}s total",
            flush=True,
        )
        if storyboard.theme:
            print(f"  Theme: {storyboard.theme}")
        if storyboard.narrative_arc:
            print(f"  Arc:   {storyboard.narrative_arc}")
        if storyboard.open_questions:
            print(f"  Open questions: {len(storyboard.open_questions)}")
            for q in storyboard.open_questions[:3]:
                print(f"    - {q}")

        out_path = workspace.root
        print(
            f"\nArtifacts written under {out_path}/:\n"
            f"  session.json          (videos + gist)\n"
            f"  storyboard.json       (final shots)\n"
            f"  clips.json            (retrieved clips, sorted by timeline)\n"
            f"  context.md            (accumulated tool-call context)\n"
            f"\nNext: run /v2/generate_fcpxml against the workspace to compile, "
            f"or invoke phase 3+ pipelines directly.",
            flush=True,
        )
        return 0
    finally:
        await services.close_lvmm()


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
