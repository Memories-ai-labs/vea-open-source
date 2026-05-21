#!/usr/bin/env python3
"""
Thin CLI shell over ``src/pipelines/v2/autonomous.py::run_autonomous``.

Same function the pytest smoke test (``tests/v2/test_smoke_autonomous.py``)
calls, and the same function any future FastAPI ``/v2/autonomous`` route
would call. This script is the researcher-convenience entry point — for
"does the system work" go run pytest instead.

Usage::

    uv run python scripts/run_autonomous.py \\
        --project my_project \\
        --prompt "Create a 30-second highlight reel" \\
        --max-iterations 3 \\
        --target-duration 30 \\
        --include-phase3
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make repo root importable when run directly without `uv run`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run VEA V2 autonomous mode (no web server, no dashboard).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project", required=True, help="Project name (= workspace dir name).")
    p.add_argument("--prompt", required=True, help="Editing brief for the planning loop.")
    p.add_argument("--source-dir", default=None, help="Source video dir. Default: workspace footage/.")
    p.add_argument("--max-iterations", type=int, default=3, help="Planning loop iteration cap.")
    p.add_argument(
        "--target-duration", type=float, default=60.0,
        help="Target edit duration in seconds (passed to storyboard prompt).",
    )
    p.add_argument(
        "--stop-after-phase", type=int, default=3, choices=[1, 2, 3],
        help="1=comprehension only, 2=+planning, 3=+post-processing (default).",
    )
    p.add_argument(
        "--include-phase3", action="store_true",
        help="Run Phase 3 (fcpxml + narration + music + render). Default on; "
             "skipped if --stop-after-phase < 3.",
    )
    p.add_argument(
        "--no-include-phase3", dest="include_phase3", action="store_false",
        help="Skip Phase 3 even if --stop-after-phase=3.",
    )
    p.set_defaults(include_phase3=True)
    p.add_argument(
        "--start-fresh", action="store_true",
        help="Re-index even if a cached session exists.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG-level logging (shows full LLM responses, all HTTP requests, "
             "every SQLite operation — noisy).",
    )
    return p


async def main_async(args: argparse.Namespace) -> int:
    # Late import: keeps --help fast and lets users run with no lvmm-core
    # configured (the helper itself errors with a clear message in that case).
    from src.pipelines.v2.autonomous import run_autonomous
    from src import services

    source_dir = Path(args.source_dir) if args.source_dir else None
    try:
        result = await run_autonomous(
            project_name=args.project,
            user_prompt=args.prompt,
            source_dir=source_dir,
            max_iterations=args.max_iterations,
            target_duration_seconds=args.target_duration,
            include_phase3=args.include_phase3,
            stop_after_phase=args.stop_after_phase,
            start_fresh=args.start_fresh,
        )
    except Exception as e:
        print(f"\n✗ Autonomous run failed: {e}", file=sys.stderr)
        return 1
    finally:
        # CLI owns the lvmm-core lifecycle (the FastAPI app does it via lifespan;
        # we mirror that here so the SQLite handle releases cleanly).
        try:
            await services.close_lvmm()
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 70)
    print("AUTONOMOUS RUN SUMMARY")
    print("=" * 70)
    print(f"workspace:   {result.workspace_root}")
    if result.session:
        print(f"videos:      {len(result.session.videos)} indexed")
    if result.storyboard:
        total = sum(s.duration_seconds for s in result.storyboard.shots)
        print(f"storyboard:  {len(result.storyboard.shots)} shots, {total:.1f}s")
        print(f"  theme:     {result.storyboard.theme or '(none)'}")
    if result.fcpxml_path:
        print(f"fcpxml:      {result.fcpxml_path}")
    if result.narration_audio_path:
        print(f"narration:   {result.narration_audio_path}")
    if result.music_path:
        print(f"music:       {result.music_path}")
    if result.draft_mp4_path:
        print(f"draft mp4:   {result.draft_mp4_path}")
    if result.skipped:
        print(f"skipped:     {', '.join(result.skipped)}")
    if result.interactions_jsonl:
        print(f"events log:  {result.interactions_jsonl}")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
