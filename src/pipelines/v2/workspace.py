"""WorkspaceManager — owns all file I/O for a v2 editing session."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.v2.schemas import (
    SessionData, Storyboard, RetrievedClip, ToolCallPlan, VideoEntry, PlanningState
)

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".m4v", ".ts"}


class WorkspaceManager:
    """
    Manages the workspace directory for a single v2 editing session.

    Directory layout:
        data/workspaces/{project_name}/
        ├── footage/        ← drop raw video files here
        ├── session.json
        ├── context.md
        ├── storyboard.json
        ├── clips.json
        ├── iterations/
        │   ├── iter_0_tool_plan.json
        │   ├── iter_0_storyboard.json
        │   └── ...
        ├── narration/
        ├── music/
        ├── fcpxml/
        ├── renders/
        └── logs/
    """

    def __init__(self, project_name: str, workspaces_dir: Path):
        self.project_name = project_name
        self.root = workspaces_dir / project_name

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def exists(self) -> bool:
        return (self.root / "session.json").exists()

    def dir_exists(self) -> bool:
        """True if the project directory itself exists (even if not yet indexed)."""
        return self.root.is_dir()

    def create(self) -> None:
        """Create workspace directory structure."""
        for subdir in ["footage", "iterations", "narration", "music", "fcpxml", "renders", "logs"]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"[WORKSPACE] Created workspace: {self.root}")

    # -------------------------------------------------------------------------
    # Footage discovery
    # -------------------------------------------------------------------------

    def get_footage_dir(self) -> Path:
        """Return the footage/ subdirectory (may not exist yet)."""
        return self.root / "footage"

    def scan_footage(self) -> List[Path]:
        """Return all video files inside footage/. Returns [] if directory is absent."""
        footage_dir = self.root / "footage"
        if not footage_dir.is_dir():
            return []
        return sorted(
            p for p in footage_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    # -------------------------------------------------------------------------
    # Project summary (used by the list endpoint)
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary dict for the project browser."""
        summary: Dict[str, Any] = {
            "project_name": self.project_name,
            "status": "new",
            "video_count": 0,
            "clip_count": 0,
            "iteration_count": 0,
            "footage_files": [],
            "indexed_files": [],   # filenames that have a video_no in session.json
            "video_gists": {},     # {video_name: gist_text}
            "gist": "",
            "has_storyboard": False,
            "has_fcpxml": False,
            "has_renders": False,
            "last_updated": None,
        }

        # Footage
        footage = self.scan_footage()
        summary["footage_files"] = [f.name for f in footage]
        summary["video_count"] = len(footage)

        # Session info
        if self.exists():
            try:
                session = self.load_session()
                summary["status"] = session.status
                summary["video_count"] = len(session.videos) or len(footage)
                summary["iteration_count"] = session.planning.iteration_count if session.planning else 0
                summary["last_updated"] = session.updated_at
                summary["indexed_files"] = [v.video_name for v in session.videos if v.video_no]
                summary["gist"] = session.gist or ""
                summary["video_gists"] = {
                    v.video_name: v.gist for v in session.videos if v.gist
                }
            except Exception:
                pass

        # Clips
        clips = self.load_clips()
        summary["clip_count"] = len(clips)

        # FCPXML
        fcpxml_dir = self.root / "fcpxml"
        if fcpxml_dir.is_dir():
            summary["has_fcpxml"] = any(fcpxml_dir.iterdir())
            if summary["has_fcpxml"]:
                summary["has_storyboard"] = (self.root / "storyboard.json").exists()

        # Renders
        renders_dir = self.root / "renders"
        if renders_dir.is_dir():
            summary["has_renders"] = any(renders_dir.iterdir())

        # Last updated fallback
        if not summary["last_updated"]:
            try:
                mtime = self.root.stat().st_mtime
                summary["last_updated"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except Exception:
                pass

        return summary

    @classmethod
    def list_projects(cls, workspaces_dir: Path) -> List[Dict[str, Any]]:
        """Scan workspaces_dir and return summaries for all project directories."""
        if not workspaces_dir.is_dir():
            return []
        projects = []
        for entry in sorted(workspaces_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                mgr = cls(entry.name, workspaces_dir)
                projects.append(mgr.get_summary())
        # Sort: active first, then by last_updated desc
        def _sort_key(p: Dict[str, Any]):
            status_order = {"planning": 0, "indexed": 1, "new": 2, "done": 3, "error": 4}
            return (status_order.get(p["status"], 9), -(p["last_updated"] or "").count(""))
        projects.sort(key=_sort_key)
        return projects

    # -------------------------------------------------------------------------
    # Session
    # -------------------------------------------------------------------------

    def init_session(self, videos: List[VideoEntry], gist: str = "") -> SessionData:
        """Create and save a new session."""
        now = datetime.now(timezone.utc).isoformat()
        session = SessionData(
            project_name=self.project_name,
            created_at=now,
            updated_at=now,
            status="indexed",
            videos=videos,
            gist=gist,
        )
        self.save_session(session)
        return session

    def load_session(self) -> SessionData:
        path = self.root / "session.json"
        with open(path) as f:
            return SessionData.from_dict(json.load(f))

    def save_session(self, session: SessionData) -> None:
        session.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.root / "session.json"
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def update_status(self, status: str) -> None:
        session = self.load_session()
        session.status = status
        self.save_session(session)

    # -------------------------------------------------------------------------
    # Planning state
    # -------------------------------------------------------------------------

    def load_storyboard(self) -> Optional[Storyboard]:
        path = self.root / "storyboard.json"
        if not path.exists():
            return None
        with open(path) as f:
            return Storyboard.model_validate(json.load(f))

    def save_storyboard(self, sb: Storyboard) -> None:
        path = self.root / "storyboard.json"
        with open(path, "w") as f:
            json.dump(sb.model_dump(), f, indent=2)

    def load_clips(self) -> List[RetrievedClip]:
        path = self.root / "clips.json"
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return [RetrievedClip.model_validate(c) for c in data]

    def save_clips(self, clips: List[RetrievedClip]) -> None:
        path = self.root / "clips.json"
        with open(path, "w") as f:
            json.dump([c.model_dump() for c in clips], f, indent=2)

    def load_context(self) -> str:
        path = self.root / "context.md"
        if not path.exists():
            return ""
        return path.read_text()

    def append_context(self, text: str) -> None:
        path = self.root / "context.md"
        with open(path, "a") as f:
            f.write(text)

    def save_iteration_snapshot(
        self, iteration: int, tool_plan: ToolCallPlan, storyboard: Storyboard
    ) -> None:
        idir = self.root / "iterations"
        with open(idir / f"iter_{iteration}_tool_plan.json", "w") as f:
            json.dump(tool_plan.model_dump(), f, indent=2)
        with open(idir / f"iter_{iteration}_storyboard.json", "w") as f:
            json.dump(storyboard.model_dump(), f, indent=2)

    # -------------------------------------------------------------------------
    # Media
    # -------------------------------------------------------------------------

    def get_log_path(self) -> Path:
        return self.root / "logs" / "run.log"

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    def get_fcpxml_path(self, version) -> Path:
        return self.root / "fcpxml" / f"edit_v{version}.fcpxml"

    def get_final_fcpxml_path(self) -> Path:
        return self.root / "fcpxml" / "edit_final.fcpxml"

    def get_latest_fcpxml(self) -> Path | None:
        """Return the most recently modified .fcpxml file, or None."""
        fcpxml_dir = self.root / "fcpxml"
        if not fcpxml_dir.is_dir():
            return None
        files = sorted(fcpxml_dir.glob("*.fcpxml"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def get_narration_path(self) -> Path:
        return self.root / "narration" / "narration.mp3"

    def get_narration_script_path(self) -> Path:
        return self.root / "narration" / "narration_script.txt"

    def get_music_path(self) -> Path:
        return self.root / "music" / "track.mp3"

    def get_render_path(self, quality: str) -> Path:
        return self.root / "renders" / f"{quality}.mp4"
