"""
Lightweight comprehension pipeline — Phase 1 of v2 agentic editing.

Ingests a workspace's footage into the project's Video Datalake collection
and reads back a per-video summary as its gist:

  1. ``DatalakeIngestor.ingest([...])``   → one ``vid_...`` per file, uploads
     gated behind a semaphore because the API's in-flight ingest window is
     about five videos wide.
  2. ``client.summary(video_id)``         → the gist (the datalake produces it
     during indexing; no extra LLM call from VEA).

Deliberately avoids heavy scene-by-scene analysis. All detailed understanding
happens on demand during the agent conversation, through ``ask_memories`` and
``search_footage``.

The collection is named after the project and its id is recorded on the
session, so a project's retrieval is scoped to exactly its own footage.
``video_no`` stays the FILENAME — the agent writes it into ``source_file`` and
the FCPXML compiler resolves it against ``footage/`` — while the datalake's
own id rides on ``VideoEntry.datalake_video_id``.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

ProgressCallback = Callable[[float, str], Awaitable[None]]

from src.datalake import DatalakeClient, DatalakeIngestor
from src.metrics import metric
from src.pipelines.v2.schemas import SessionData, VideoEntry
from src.pipelines.v2.workspace import WorkspaceManager
from src.config import VIDEO_EXTS

logger = logging.getLogger(__name__)


class LightweightComprehension:
    """
    Phase 1: ingest a workspace's videos into its datalake collection and
    keep a broad gist per video.

    Session cache: if the workspace already has a session and every video in
    it is still ``ready`` in the collection, skip re-ingesting and return the
    cached session immediately.

    Parameters
    ----------
    project_name:
        Logical name for the project (workspace name).
    source_dir:
        Directory holding the input video files.
    client:
        Datalake client. Its bound collection is used when the session has
        none yet; otherwise the session's collection wins.
    workspace:
        WorkspaceManager that owns session.json + other workspace artefacts.
    """

    def __init__(
        self,
        project_name: str,
        source_dir: str,
        client: DatalakeClient,
        workspace: WorkspaceManager,
        *,
        run_id: Optional[str] = None,
        concurrency: int = 4,
    ):
        """
        Parameters
        ----------
        run_id:
            Optional identifier tagged into this phase's log lines so a run
            can be correlated after the fact.
        concurrency:
            In-flight ingests. The API's window is about five wide and answers
            429 beyond it, so 4 is the safe default.
        """
        self.project_name = project_name
        self.source_dir = Path(source_dir)
        self.client = client
        self.workspace = workspace
        self.run_id = run_id
        self.concurrency = concurrency

    async def run(
        self,
        start_fresh: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        only_files: Optional[List[str]] = None,
    ) -> SessionData:
        """
        Run comprehension. Returns SessionData with per-video video_ids and gists.

        Args:
            start_fresh: If True, re-index all videos even if a cached session
                exists. Skips the "is the cached session still valid?" check.
            progress_callback: optional async fn(percent, message) for stage updates.
            only_files: If provided, only re-index these specific filenames and
                merge the results back into the existing session (other videos
                are kept). Matching DB rows are deleted first to force re-index.
        """
        async def _report(percent: float, message: str):
            if progress_callback:
                try:
                    await progress_callback(percent, message)
                except Exception:
                    pass

        await _report(2, "Checking for existing session...")

        # --- Per-file re-index path ---
        if only_files:
            return await self._reindex_files(only_files, _report)

        # --- Check for existing valid session ---
        if not start_fresh and self.workspace.exists():
            try:
                session = self.workspace.load_session()
                if session.videos and session.gist:
                    all_present = await self._all_videos_still_indexed(session.videos)
                    if all_present:
                        logger.info(
                            f"[COMPREHENSION] Resuming existing session for '{self.project_name}' "
                            f"({len(session.videos)} videos, gist cached)"
                        )
                        return session
                    else:
                        logger.info("[COMPREHENSION] Some indexed videos missing from local DB — re-indexing")
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

        # --- Ingest into the project's datalake collection ---
        collection_id = await self._ensure_collection()
        logger.info(f"[COMPREHENSION] Ingesting {len(video_files)} videos into {collection_id}...")
        await _report(15, f"Uploading {len(video_files)} video(s) to the datalake...")

        video_entries: List[VideoEntry] = await self._ingest(video_files, collection_id, _report)
        if not video_entries:
            raise RuntimeError("All video ingestion failed")

        logger.info(f"[COMPREHENSION] Indexed {len(video_entries)}/{len(video_files)} videos")
        await _report(60, "Indexing complete, generating content gist...")

        # --- Save initial session (no gist yet) ---
        session = self.workspace.init_session(videos=video_entries)
        session.datalake_collection_id = collection_id

        # --- Per-video gist: the datalake's own summary, one read per video ---
        logger.info("[COMPREHENSION] Reading per-video summaries from the datalake...")

        for idx, entry in enumerate(video_entries):
            pct = 60 + (35 * (idx / max(len(video_entries), 1)))
            await _report(pct, f"Gist {idx+1}/{len(video_entries)}: {entry.video_name}")
            entry.gist = await self._gist_one(entry)
            logger.info(
                f"[COMPREHENSION] Gist for {entry.video_name}: {len(entry.gist)} chars"
            )

        # Combined gist for the session (used as planning context)
        if len(video_entries) == 1:
            combined_gist = video_entries[0].gist
        else:
            combined_parts = "\n\n---\n\n".join(
                f"**{v.video_name}**\n\n{v.gist}" for v in video_entries if v.gist
            )
            combined_gist = combined_parts

        # --- Update session ---
        session.videos = video_entries
        session.gist = combined_gist
        # memories_session_id is a memories.ai-only concept; left None for the
        # local port. Kept on the schema for back-compat (existing session.json
        # files in the wild may still have the field).
        session.memories_session_id = None
        session.datalake_collection_id = collection_id
        session.status = "indexed"
        self.workspace.save_session(session)

        # Also save gist to context.md as initial context
        self.workspace.append_context(f"# Video Gist\n\n{combined_gist}\n")

        logger.info(f"[COMPREHENSION] Session saved: {self.workspace.root / 'session.json'}")
        await _report(98, "Saving session...")
        return session

    async def _reindex_files(
        self,
        filenames: List[str],
        report: Callable[[float, str], Awaitable[None]],
    ) -> SessionData:
        """Re-ingest a specific subset of files.

        Deletes each target's previous video from the collection first so the
        re-ingest starts clean, then merges the new VideoEntry objects back
        into the existing session."""
        try:
            session = self.workspace.load_session()
        except Exception:
            session = self.workspace.init_session(videos=[])

        all_video_files = self._find_videos()
        targets = [vf for vf in all_video_files if vf.name in set(filenames)]
        if not targets:
            raise ValueError(f"No matching footage files found for: {filenames}")

        await report(8, f"Re-indexing {len(targets)} file(s)...")

        # Delete old DB rows for matching files so re-indexing produces a clean state.
        existing_by_name = {v.video_name: v for v in session.videos}
        for vf in targets:
            old = existing_by_name.get(vf.name)
            if old and old.video_no:
                try:
                    await self._delete_video(old.datalake_video_id or old.video_no)
                    logger.info(f"[COMPREHENSION] Removed previous ingest of {vf.name} ({old.datalake_video_id})")
                except Exception as e:
                    logger.warning(f"[COMPREHENSION] Could not delete old index for {vf.name}: {e}")

        new_entries: List[VideoEntry] = []
        for idx, vf in enumerate(targets):
            pct = 15 + (50 * (idx / max(len(targets), 1)))
            await report(pct, f"Re-indexing {idx+1}/{len(targets)}: {vf.name}")
            try:
                new_entries.append(await self._index_one(vf))
            except Exception as e:
                logger.error(f"[COMPREHENSION] Re-index failed for {vf.name}: {e}")

        if not new_entries:
            raise RuntimeError("All re-index attempts failed")

        await report(75, "Generating updated content gists...")
        for idx, entry in enumerate(new_entries):
            pct = 75 + (20 * (idx / max(len(new_entries), 1)))
            await report(pct, f"Gist {idx+1}/{len(new_entries)}: {entry.video_name}")
            entry.gist = await self._gist_one(entry)

        # Merge new entries back into session, replacing matching ones
        new_by_name = {v.video_name: v for v in new_entries}
        merged = [new_by_name.get(v.video_name, v) for v in session.videos]
        existing_names = {v.video_name for v in session.videos}
        for v in new_entries:
            if v.video_name not in existing_names:
                merged.append(v)
        session.videos = merged

        # Rebuild combined gist
        if len(session.videos) == 1:
            session.gist = session.videos[0].gist
        else:
            session.gist = "\n\n---\n\n".join(
                f"**{v.video_name}**\n\n{v.gist}" for v in session.videos if v.gist
            )

        session.status = "indexed"
        self.workspace.save_session(session)
        logger.info(f"[COMPREHENSION] Re-indexed {len(new_entries)} file(s); session saved")
        await report(98, "Saving session...")
        return session

    # ------------------------------------------------------------------
    # Per-video ingest + gist
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> str:
        """Resolve this project's collection, creating it on first index.

        Precedence: the session's recorded collection, then the client's bound
        one, then a collection named after the project. Collections are free to
        create and list, and one per project keeps every search scoped to that
        project's own footage without a filter.
        """
        try:
            if self.workspace.exists():
                recorded = getattr(self.workspace.load_session(), "datalake_collection_id", "")
                if recorded:
                    self.client.collection_id = recorded
                    return recorded
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[COMPREHENSION] Could not read session collection: {e}")

        if self.client.collection_id:
            return self.client.collection_id

        name = f"vea-{self.project_name}"
        collection_id = await self.client.ensure_collection(name)
        self.client.collection_id = collection_id
        logger.info(f"[COMPREHENSION] Collection '{name}' -> {collection_id}")
        return collection_id

    async def _ingest(
        self,
        video_files: List[Path],
        collection_id: str,
        report: Callable[[float, str], Awaitable[None]],
    ) -> List[VideoEntry]:
        """Upload + wait for every file; one VideoEntry per success.

        ``video_no`` is the filename (what the agent puts in ``source_file``);
        the datalake id rides alongside on ``datalake_video_id``.
        """
        now = datetime.now(timezone.utc).isoformat()
        total = len(video_files)

        async def on_progress(done: int, count: int, name: str, vid: str) -> None:
            await report(15 + 45 * (done / max(count, 1)), f"Indexed {done}/{count}: {name}")

        ingestor = DatalakeIngestor(self.client, concurrency=self.concurrency)
        try:
            ids = await ingestor.ingest(
                [str(v.resolve()) for v in video_files],
                collection_id=collection_id,
                on_progress=on_progress,
            )
        except Exception as e:
            logger.error(f"[COMPREHENSION] Ingest failed: {e}")
            raise

        entries: List[VideoEntry] = []
        for vf, vid in zip(video_files, ids):
            if not vid:
                logger.error(f"[COMPREHENSION] No video id for {vf.name} — skipped")
                continue
            duration = await asyncio.to_thread(_probe_duration, vf)
            if duration:
                metric("vea.comprehension.video_duration_seconds", duration, video_id=vid)
            entries.append(VideoEntry(
                video_no=vf.name,
                video_name=vf.name,
                source_path=str(vf.resolve()),
                duration_seconds=duration,
                indexed_at=now,
                datalake_video_id=vid,
            ))
            logger.info(f"[COMPREHENSION] Ingested {vf.name} -> {vid}")
        return entries

    async def _index_one(self, video_path: Path) -> VideoEntry:
        """Ingest a single file. Thin wrapper over :meth:`_ingest`."""
        async def _noop(pct: float, msg: str) -> None:
            return None

        entries = await self._ingest([video_path], await self._ensure_collection(), _noop)
        if not entries:
            raise RuntimeError(f"Ingest produced no entry for {video_path.name}")
        return entries[0]

    async def _gist_one(self, entry: VideoEntry) -> str:
        """The datalake's own summary for this video. Returns "" on failure.

        Cheaper and more consistent than asking an LLM for a gist: the summary
        is produced during indexing, so this is one $0.001 derived read.
        """
        vid = entry.datalake_video_id or entry.video_no
        try:
            gist = await self.client.summary(vid)
            metric("vea.comprehension.gist_chars", len(gist or ""), video_id=vid)
            return gist
        except Exception as e:
            logger.warning(f"[COMPREHENSION] Gist failed for {entry.video_name}: {e}")
            return ""

    async def _all_videos_still_indexed(self, videos: List[VideoEntry]) -> bool:
        """True if every entry is still ``ready`` in the collection.

        A video someone deleted from the collection (or one that never finished
        indexing) must force a re-ingest rather than leave the agent searching
        for footage the datalake no longer holds.
        """
        try:
            present = {
                v.get("id") or v.get("video_id"): v.get("status")
                for v in await self.client.list_videos()
            }
        except Exception as e:
            logger.warning(f"[COMPREHENSION] Collection check failed: {e}")
            return False

        for v in videos:
            vid = v.datalake_video_id
            if not vid or present.get(vid) != "ready":
                return False
        return True

    async def _delete_video(self, video_id: str) -> None:
        """Remove a video from the collection so a re-index starts clean."""
        await purge_video_index(self.client, video_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_videos(self) -> List[Path]:
        """Find all video files in source_dir."""
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        videos = []
        for ext in VIDEO_EXTS:
            videos.extend(self.source_dir.glob(f"*{ext}"))
            videos.extend(self.source_dir.glob(f"*{ext.upper()}"))
        return sorted(set(videos))


async def purge_video_index(client, video_id: str) -> None:
    """Delete a video from its datalake collection.

    Used by the per-file re-index path and by the dashboard's clear-memories
    endpoint. This is a real deletion of indexed data — the frames, captions,
    transcription and vectors that indexing was billed for go away, and a
    re-index pays again. Storage stops accruing for that video.
    """
    if client is None or not video_id:
        return
    try:
        await client.delete_video(video_id)
        logger.info(f"[COMPREHENSION] Deleted {video_id} from the collection")
    except Exception as e:  # noqa: BLE001
        # Already gone, or never ingested — nothing to clean up.
        logger.warning(f"[COMPREHENSION] Could not delete {video_id}: {e}")


def _probe_duration(video_path: Path) -> Optional[float]:
    """ffprobe fallback when master_indexing's result dict didn't include duration."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        s = out.stdout.strip()
        return round(float(s), 2) if s else None
    except Exception:
        return None
