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
from typing import List, Optional

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

    async def run(self, start_fresh: bool = False) -> SessionData:
        """
        Run comprehension. Returns SessionData with video_nos and gist.

        Args:
            start_fresh: If True, re-upload all videos even if session exists.
        """
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

        # --- Create workspace ---
        self.workspace.create()

        # --- Upload all videos in parallel ---
        logger.info("[COMPREHENSION] Uploading videos to Memories.ai...")
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
        await asyncio.gather(*[
            self.memories.wait_for_ready(v.video_no, timeout=3600)
            for v in video_entries
        ])
        logger.info(f"[COMPREHENSION] All {len(video_entries)} videos indexed")

        # --- Save initial session (no gist yet) ---
        session = self.workspace.init_session(videos=video_entries)

        # --- Get broad gist via Chat API ---
        logger.info("[COMPREHENSION] Requesting video gist from Memories.ai Chat API...")
        video_nos = [v.video_no for v in video_entries]
        chat_response = await self.memories.chat(video_nos, GIST_PROMPT)
        gist = chat_response.text
        # Save session_id for continuity in planning loop
        memories_session_id = chat_response.session_id

        logger.info(f"[COMPREHENSION] Gist received ({len(gist)} chars)")
        if self.memories.debug:
            logger.debug(f"[COMPREHENSION] Gist preview: {gist[:500]}")

        # --- Update session with gist ---
        session.gist = gist
        session.memories_session_id = memories_session_id
        session.status = "indexed"
        self.workspace.save_session(session)

        # Also save gist to context.md as initial context
        self.workspace.append_context(f"# Video Gist\n\n{gist}\n")

        logger.info(f"[COMPREHENSION] Session saved: {self.workspace.root / 'session.json'}")
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
