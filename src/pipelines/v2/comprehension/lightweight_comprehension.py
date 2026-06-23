"""
Lightweight comprehension pipeline — Phase 1 of v2 agentic editing.

Indexes a workspace's videos LOCALLY via lvmm-core's master_indexing
pipeline and asks lvmm-core's MaviAgent for a per-video gist. No upload
to memories.ai, no polling for "ready" status — indexing is synchronous
and runs in-process.

Deliberately avoids heavy scene-by-scene analysis. All detailed
understanding happens on-demand during the iterative planning loop.

PORT NOTE (2026-05-19): Migrated from memories.ai-hosted indexing. Where
this module used to call ``MemoriesAiManager.upload_video_url`` +
``wait_for_ready`` + ``chat(GIST_PROMPT)`` per video, it now runs:
  1. ``build_indexing_pipeline("classic").execute({"video_path": ...})``
     → fills SQLite **and** vector_db (StoreVisualStage handles both).
  2. ``mavi_agent.ask(GIST_PROMPT, video_id=...)``      → produces gist

Sequential per video for now; concurrent indexing is a follow-up (see
the workspace-level notes file).
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

ProgressCallback = Callable[[float, str], Awaitable[None]]

from lvmm_core.utils.logging import metric
from src.pipelines.v2.schemas import SessionData, VideoEntry
from src.pipelines.v2.workspace import WorkspaceManager
from src.pipelines.v2.planning.planning_prompts import GIST_PROMPT
from src.config import VIDEO_EXTS

logger = logging.getLogger(__name__)


class LightweightComprehension:
    """
    Phase 1: Index a workspace's videos locally (via lvmm-core) and
    extract a broad gist per video.

    Session cache: if the workspace already has a session and all videos
    in it are still present in the local DB, skip re-indexing and return
    the cached session immediately.

    Parameters
    ----------
    project_name:
        Logical name for the project (workspace name).
    source_dir:
        Directory holding the input video files.
    lvmm_ctx:
        lvmm-core PipelineContext (carries adapters + DB + storage).
    mavi_agent:
        lvmm-core MaviAgent used to generate per-video gists.
    workspace:
        WorkspaceManager that owns session.json + other workspace artefacts.
    """

    def __init__(
        self,
        project_name: str,
        source_dir: str,
        lvmm_ctx,                       # lvmm_core.pipelines.base.PipelineContext
        mavi_agent,                     # lvmm_core.agents.mavi_agent.MaviAgent
        workspace: WorkspaceManager,
        *,
        run_id: Optional[str] = None,
        pipeline_hooks: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        run_id:
            Optional UUID-like identifier propagated to every
            ``Pipeline.execute()`` call we drive in Phase 1. Threads into
            lvmm-core's structured-logging ContextVars so every log line
            emitted by lvmm-core (pipeline stage starts/ends, MaviAgent
            query rewrites, etc.) carries this run_id. Lets us correlate
            "which run produced this log line" after the fact.
        pipeline_hooks:
            Optional list of ``lvmm_core._internal.hooks.PipelineHooks``
            passed straight through to ``Pipeline.execute(hooks=...)``.
            Common pair: ``[ConsoleHooks(), JSONFileHooks(path=...)]`` —
            the JSONL sink writes one structured record per L1 stage
            start/end/error, sitting alongside VEA's own L3 interactions
            JSONL for full top-to-bottom observability.
        """
        self.project_name = project_name
        self.source_dir = Path(source_dir)
        self.lvmm_ctx = lvmm_ctx
        self.mavi_agent = mavi_agent
        self.workspace = workspace
        self.run_id = run_id
        self.pipeline_hooks = pipeline_hooks or []

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

        # --- Index videos sequentially via lvmm-core ---
        # NOTE: sequential for the first cut. Concurrent indexing is a follow-up;
        # SQLite write contention will need the _Sequenced* stage-wrapper pattern
        # from master_indexing (see workspace local-notes for tracking).
        logger.info("[COMPREHENSION] Indexing videos locally via lvmm-core...")
        await _report(15, f"Indexing {len(video_files)} video(s) locally...")

        video_entries: List[VideoEntry] = []
        for idx, vf in enumerate(video_files):
            pct = 15 + (40 * (idx / max(len(video_files), 1)))
            await _report(pct, f"Indexing {idx+1}/{len(video_files)}: {vf.name}")
            try:
                entry = await self._index_one(vf)
                video_entries.append(entry)
            except Exception as e:
                logger.error(f"[COMPREHENSION] Indexing failed for {vf.name}: {e}")
                # Continue with the rest — partial success is better than total failure

        if not video_entries:
            raise RuntimeError("All video indexing failed")

        logger.info(f"[COMPREHENSION] Indexed {len(video_entries)}/{len(video_files)} videos")
        await _report(60, "Indexing complete, generating content gist...")

        # --- Save initial session (no gist yet) ---
        session = self.workspace.init_session(videos=video_entries)

        # --- Get per-video gist via MaviAgent (one call per video, sequential) ---
        logger.info("[COMPREHENSION] Requesting per-video gist via lvmm-core MaviAgent...")

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
        """Re-index a specific subset of files. Deletes existing rows for those
        videos from the local DB first so master_indexing re-runs cleanly, then
        merges the new VideoEntry objects back into the existing session."""
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
                    await self._delete_video(old.video_no)
                    logger.info(f"[COMPREHENSION] Deleted old local index for {vf.name} ({old.video_no})")
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
    # Per-video index + gist (the lvmm-core swap-in)
    # ------------------------------------------------------------------

    async def _index_one(self, video_path: Path) -> VideoEntry:
        """Index a single video locally and return a VideoEntry.

        Runs the lvmm-core ``classic`` indexing DAG, which includes
        ``StoreVisualStage`` — that stage writes both the SQLite rows
        (videos / video_transcripts / segments / summary) and the
        sqlite-vec collections (vec_video_transcript / vec_keyframe /
        vec_transcript / summary). One pipeline.execute() does it all.
        """
        # PORT NOTE (2026-05-19): use the visual-only indexing pipeline
        # (build_indexing_pipeline) instead of build_master_indexing_pipeline.
        # Master adds portrait + multimodal_asr which need ctx.face_detector
        # and ctx.asr. VEA's services.py builds ctx with face/asr/diar="none"
        # — so master fails. Visual-only is sufficient for what VEA needs
        # downstream (the planning loop searches video_transcripts text).
        # Adding audio + face indexing is a follow-up if/when VEA wants to
        # search dialogue or filter by person.
        from lvmm_core.pipelines.indexing.video_indexing import build_indexing_pipeline

        now = datetime.now(timezone.utc).isoformat()
        pipeline = build_indexing_pipeline("classic")
        # Thread run_id + hooks into the L1 pipeline. The run_id binds
        # lvmm-core's logging ContextVars for the duration of execute(),
        # so every visual_transcriber / embedding / store_visual stage's
        # log line carries the same run_id tag as our L3 events. The
        # hooks (typically ConsoleHooks + JSONFileHooks) capture each
        # stage's start / end / cost / errors as structured JSONL records.
        result = await pipeline.execute(
            {"video_path": str(video_path), "user_id": "vea_local"},
            self.lvmm_ctx,
            run_id=self.run_id,
            sample_id=video_path.name,
            hooks=self.pipeline_hooks,
        )

        # DeriveVideoIDStage produces the video_id; downstream stages all
        # reference it. It lands in the result dict.
        video_id = (
            result.get("video_id")
            or result.get("derived_video_id")
            or video_path.stem  # fallback — should never fire in practice
        )

        # Best-effort duration from the result; ffprobe fallback if missing.
        duration = result.get("duration_seconds") or result.get("video_duration")
        if duration is None:
            duration = await asyncio.to_thread(_probe_duration, video_path)

        logger.info(
            f"[COMPREHENSION] Indexed {video_path.name} → video_id={video_id} "
            f"({duration:.1f}s)" if duration else
            f"[COMPREHENSION] Indexed {video_path.name} → video_id={video_id}"
        )
        # Aggregatable per-video index metric. Tagged with video_id so we can
        # diff "video X took N seconds and indexed M transcript chunks" across
        # runs without parsing log strings.
        if duration:
            metric("vea.comprehension.video_duration_seconds", duration, video_id=video_id)

        return VideoEntry(
            video_no=video_id,  # kept as video_no for back-compat with existing session.json files
            video_name=video_path.name,
            source_path=str(video_path.resolve()),
            duration_seconds=duration,
            indexed_at=now,
        )

    async def _gist_one(self, entry: VideoEntry) -> str:
        """Ask MaviAgent for a per-video gist. Returns the answer text or ""."""
        try:
            trace = await self.mavi_agent.ask(GIST_PROMPT, video_id=entry.video_no)
            # Per-gist aggregatables. ``rewrite_reason`` and ``answer`` are
            # logged at INFO by MaviAgent already; what's useful to aggregate
            # is the cost (tokens) and the size of the resulting gist.
            metric("vea.comprehension.gist_chars", len(trace.answer or ""), video_id=entry.video_no)
            metric("vea.comprehension.gist_input_tokens", trace.total_input_tokens, video_id=entry.video_no)
            metric("vea.comprehension.gist_output_tokens", trace.total_output_tokens, video_id=entry.video_no)
            return trace.answer
        except Exception as e:
            logger.warning(f"[COMPREHENSION] Gist failed for {entry.video_name}: {e}")
            return ""

    async def _all_videos_still_indexed(self, videos: List[VideoEntry]) -> bool:
        """Return True if every VideoEntry's video_id is still present in the local DB.

        Checks the ``summary`` table — lvmm-core's classic IndexingPipeline
        (StoreVisualStage) writes one summary row per indexed video. (Earlier
        revisions used a ``videos`` table; the current schema is ``summary``.
        Querying the wrong name made this always return False → re-index on
        every call.)
        """
        try:
            for v in videos:
                rows = await self.lvmm_ctx.database.query(
                    "summary", {"video_id": v.video_no}
                )
                if not rows:
                    return False
            return True
        except Exception as e:
            logger.warning(f"[COMPREHENSION] DB check failed: {e}")
            return False

    async def _delete_video(self, video_id: str) -> None:
        """Best-effort: drop all of a video's indexed data so re-indexing is clean.

        Delegates to :func:`purge_video_index`, which clears both the
        relational rows and the sqlite-vec vectors lvmm-core writes per video.
        """
        await purge_video_index(self.lvmm_ctx, video_id)

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


# Collections (relational table name == vector collection name) that
# lvmm-core's classic IndexingPipeline (StoreVisualStage) populates per
# video. The collection name doubles as the sqlite-vec ``vec_<name>`` and
# ``<name>_meta`` prefix. Keep this in sync with StoreVisualStage in
# lvmm-core ``pipelines/indexing/video_indexing.py``.
#
# (Earlier lvmm-core revisions used plural/legacy names — videos / keyframes
# / video_transcripts. The current schema is singular. Deleting the old names
# silently no-ops and leaves the video indexed, so DO NOT revert these.)
_INDEXED_COLLECTIONS = ("summary", "keyframe", "video_transcript", "transcript")

# Relational-only tables lvmm-core writes per video that have NO vector
# collection / ``_meta`` sidecar (so they're cleared in the relational pass
# only, never the vector pass). ``keyclip`` carries the FINCH segment grid
# written by StoreVisualStage; lvmm-core added it in the indexer update that
# also taught db_cleanup.clear_video about it. Keep this in sync with
# StoreVisualStage in lvmm-core ``pipelines/indexing/video_indexing.py``.
_INDEXED_RELATIONAL_ONLY = ("keyclip",)


async def purge_video_index(lvmm_ctx, video_id: str) -> None:
    """Remove ALL of a video's indexed data from lvmm-core's local stores.

    Clears both the relational rows (``summary`` / ``keyframe`` /
    ``video_transcript`` / ``transcript`` / ``keyclip``) AND the sqlite-vec
    vectors for ``video_id``. Vectors are enumerated from each
    ``<collection>_meta`` sidecar (which carries a ``video_id`` column) and
    deleted by id — the adapter's ``delete(collection, id)`` drops both the
    ``vec_<collection>`` row and the ``<collection>_meta`` row.

    Best-effort: missing tables/rows are ignored so a partially-indexed
    video still cleans up. Used by both per-file re-index (delete-then-
    reindex) and the dashboard's clear-memories endpoint.
    """
    db = getattr(lvmm_ctx, "database", None)
    vec = getattr(lvmm_ctx, "vector_db", None)
    if db is None:
        return

    # 1. Vectors (+ their _meta sidecar rows): enumerate ids per collection
    #    from the meta table, then delete each via the vector adapter.
    if vec is not None:
        for col in _INDEXED_COLLECTIONS:
            try:
                rows = await db.query(f"{col}_meta", {"video_id": video_id})
            except Exception:
                rows = []
            for r in (rows or []):
                vid = r.get("id")
                if vid is None:
                    continue
                try:
                    await vec.delete(col, str(vid))
                except Exception:
                    pass

    # 2. Relational rows (vector-backed collections + relational-only tables
    #    like ``keyclip`` that have no vectors to enumerate above).
    for tbl in (*_INDEXED_COLLECTIONS, *_INDEXED_RELATIONAL_ONLY):
        try:
            await db.delete(tbl, {"video_id": video_id})
        except Exception:
            # Table may not exist, or schema may differ — ignore.
            pass


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
