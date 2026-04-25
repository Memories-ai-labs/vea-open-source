"""Shared helpers for route modules."""
from pathlib import Path

from fastapi import HTTPException

from src import config as _config
from src.pipelines.v2.workspace import WorkspaceManager


def safe_workspace(project_name: str) -> WorkspaceManager:
    """Construct a WorkspaceManager, converting bad input into HTTP 400."""
    try:
        return WorkspaceManager(project_name, _config.WORKSPACES_DIR)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def safe_child(directory: Path, name: str) -> Path:
    """Resolve ``directory / name`` and verify the result is inside ``directory``.

    Rejects path traversal via ``..`` or absolute paths in ``name``.
    """
    candidate = (directory / name).resolve()
    root = directory.resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"Invalid path: {name!r}")
    return candidate
