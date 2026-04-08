"""
Lightweight comprehension pipeline — Phase 1 of v2 agentic editing.

Uploads video to Memories.ai (or reuses existing video_no from session cache),
gets a broad gist via Chat API, and saves session.json.

Deliberately avoids heavy scene-by-scene analysis. All detailed understanding
happens on-demand during the iterative planning loop.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

ProgressCallback = Callable[[float, str], Awaitable[None]]

from lib.llm.MemoriesAiManager import MemoriesAiManager
from src.pipelines.v2.schemas import SessionData, VideoEntry
from src.pipelines.v2.workspace import WorkspaceManager
from src.pipelines.v2.planning.planning_prompts import GIST_PROMPT
from src.config import VIDEO_EXTS

logger = logging.getLogger(__name__)


class LightweightComprehension:
    """
    Phase 1: Get video into Memories.ai and extract a broad gist.

    Session cache: if workspace already has a valid session with PARSE video_nos,
    skip re-upload and return the existing session immediately.
    """

    def __init__(
        self,
        project_name: str,
        source_dir: str,
        memories: MemoriesAiManager,
        workspace: WorkspaceManager,
    ):
        self.project_name = project_name
        self.source_dir = Path(source_dir)
        self.memories = memories
        self.workspace = workspace

    async def run(
        self,
        start_fresh: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SessionData:
        """
        Run comprehension. Returns SessionData with video_nos and gist.

        Args:
            start_fresh: If True, re-upload all videos even if session exists.
            progress_callback: optional async fn(percent, message) for stage updates.
        """
        async def _report(percent: float, message: str):
            if progress_callback:
                try:
                    await progress_callback(percent, message)
                except Exception:
                    pass

        await _report(2, "Checking for existing session...")

        # --- Check for existing valid session ---
        if not start_fresh and self.workspace.exists():
            try:
                session = self.workspace.load_session()
                if session.videos and session.gist:
                    # Verify all videos are still PARSE on Memories.ai
                    statuses = await asyncio.gather(*[
                        self.memories.get_video_status(v.video_no)
                        for v in session.videos
                    ], return_exceptions=True)
                    all_ready = all(
                        not isinstance(s, Exception) and s.status == "PARSE"
                        for s in statuses
                    )
                    if all_ready:
                        logger.info(
                            f"[COMPREHENSION] Resuming existing session for '{self.project_name}' "
                            f"({len(session.videos)} videos, gist cached)"
                        )
                        return session
                    else:
                        logger.info("[COMPREHENSION] Some videos not PARSE — re-uploading")
            except Exception as e:
                logger.warning(f"[COMPREHENSION] Could not load existing session: {e}")

        # --- Find source video files ---
        video_files = self._find_videos()
        if not video_files:
            raise ValueError(f"No video files found in: {self.source_dir}")
        logger.info(f"[COMPREHENSION] Found {len(video_files)} video files in {self.source_dir}")
        await _report(8, f"Found {len(video_files)} video file(s)")

        # --- Create workspace ---
        self.workspace.create()

        # --- Upload all videos in parallel ---
        logger.info("[COMPREHENSION] Uploading videos to Memories.ai...")
        await _report(15, f"Uploading {len(video_files)} video(s) to Memories.ai...")
        upload_results = await asyncio.gather(*[
            self._upload_one(vf) for vf in video_files
        ], return_exceptions=True)

        video_entries: List[VideoEntry] = []
        for vf, result in zip(video_files, upload_results):
            if isinstance(result, Exception):
                logger.error(f"[COMPREHENSION] Upload failed for {vf.name}: {result}")
                continue
            video_entries.append(result)

        if not video_entries:
            raise RuntimeError("All video uploads failed")

        # --- Wait for all videos to be indexed ---
        logger.info("[COMPREHENSION] Waiting for Memories.ai indexing to complete...")
        await _report(40, f"Waiting for Memories.ai to index {len(video_entries)} video(s)...")
        await asyncio.gather(*[
            self.memories.wait_for_ready(v.video_no, timeout=3600)
            for v in video_entries
        ])
        logger.info(f"[COMPREHENSION] All {len(video_entries)} videos indexed")
        await _report(75, "Indexing complete, generating content gist...")

        # --- Save initial session (no gist yet) ---
        session = self.workspace.init_session(videos=video_entries)

        # --- Get per-video gist via Chat API (one call per video, in parallel) ---
        logger.info("[COMPREHENSION] Requesting per-video gist from Memories.ai Chat API...")

        async def _gist_one(entry: VideoEntry) -> str:
            try:
                resp = await self.memories.chat([entry.video_no], GIST_PROMPT)
                return resp.text
            except Exception as e:
                logger.warning(f"[COMPREHENSION] Gist failed for {entry.video_name}: {e}")
                return ""

        gists = await asyncio.gather(*[_gist_one(v) for v in video_entries])
        for entry, gist_text in zip(video_entries, gists):
            entry.gist = gist_text
            logger.info(f"[COMPREHENSION] Gist for {entry.video_name}: {len(gist_text)} chars")

        # Combined gist for the session (used as planning context)
        if len(video_entries) == 1:
            combined_gist = video_entries[0].gist
            memories_session_id = None
        else:
            combined_parts = "\n\n---\n\n".join(
                f"**{v.video_name}**\n\n{v.gist}" for v in video_entries if v.gist
            )
            combined_gist = combined_parts
            memories_session_id = None

        # --- Update session ---
        session.videos = video_entries
        session.gist = combined_gist
        session.memories_session_id = memories_session_id
        session.status = "indexed"
        self.workspace.save_session(session)

        # Also save gist to context.md as initial context
        self.workspace.append_context(f"# Video Gist\n\n{combined_gist}\n")

        logger.info(f"[COMPREHENSION] Session saved: {self.workspace.root / 'session.json'}")
        await _report(98, "Saving session...")
        return session

    async def _upload_one(self, video_path: Path) -> VideoEntry:
        """Upload a single video file and return a VideoEntry."""
        # Check if already uploaded under this name
        existing = await self.memories.find_video_by_name(video_path.name)
        if existing and existing.status == "PARSE":
            logger.info(f"[COMPREHENSION] Reusing existing video: {video_path.name} -> {existing.video_no}")
            return VideoEntry(
                video_no=existing.video_no,
                video_name=video_path.name,
                source_path=str(video_path.resolve()),
                duration_seconds=existing.duration,
            )

        video_no = await self.memories.upload_video_file(str(video_path))
        return VideoEntry(
            video_no=video_no,
            video_name=video_path.name,
            source_path=str(video_path.resolve()),
        )

    def _find_videos(self) -> List[Path]:
        """Find all video files in source_dir."""
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        videos = []
        for ext in VIDEO_EXTS:
            videos.extend(self.source_dir.glob(f"*{ext}"))
            videos.extend(self.source_dir.glob(f"*{ext.upper()}"))
        return sorted(set(videos))
