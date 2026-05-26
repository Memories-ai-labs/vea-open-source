"""Tests for project-management REST routes.

Covers create / list / clear/planning / clear/session. These endpoints back
the dashboard project browser and the "Manage" menu's reset actions.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client_and_workspaces(tmp_path, monkeypatch):
    """Build a TestClient with WORKSPACES_DIR pointed at a fresh tmp dir."""
    from unittest.mock import AsyncMock
    with (
        patch("lib.oss.storage_factory.get_storage_client") as mock_storage,
        patch("lib.llm.GeminiGenaiManager.GeminiGenaiManager") as mock_gemini,
        patch("lib.llm.OpenRouterManager.OpenRouterManager") as mock_or,
        # Stub the async lvmm-core init so app startup doesn't spin up a
        # real PipelineContext just to test route behaviour.
        patch("src.services.init_lvmm", new=AsyncMock(return_value=None)),
        patch("src.services.close_lvmm", new=AsyncMock(return_value=None)),
        patch.dict("os.environ", {
            "GOOGLE_CLOUD_PROJECT": "p",
            "OPENROUTER_API_KEY": "or",
        }),
    ):
        mock_storage.return_value = MagicMock()
        mock_gemini.return_value = MagicMock()
        mock_or.return_value = MagicMock()

        from src.app import app
        monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
        monkeypatch.setattr("src.routes.v2_projects._config.WORKSPACES_DIR", tmp_path)
        monkeypatch.setattr("src.routes._route_utils._config.WORKSPACES_DIR", tmp_path)

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, tmp_path


# ─── GET /projects (list) ────────────────────────────────────────────────────

class TestListProjects:
    def test_empty_root_returns_empty_list(self, client_and_workspaces):
        client, _ = client_and_workspaces
        resp = client.get("/video-edit/v2/projects")
        assert resp.status_code == 200
        assert resp.json() == {"projects": []}

    def test_pre_existing_project_dir_appears(self, client_and_workspaces):
        client, ws_root = client_and_workspaces
        (ws_root / "alpha" / "footage").mkdir(parents=True)
        resp = client.get("/video-edit/v2/projects")
        assert resp.status_code == 200
        names = [p["project_name"] for p in resp.json()["projects"]]
        assert "alpha" in names


# ─── POST /projects/create ───────────────────────────────────────────────────

class TestCreateProject:
    def test_creates_new_workspace(self, client_and_workspaces):
        client, ws_root = client_and_workspaces
        resp = client.post(
            "/video-edit/v2/projects/create",
            params={"project_name": "my-project"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "created"
        assert body["project_name"] == "my-project"
        assert (ws_root / "my-project" / "footage").is_dir()

    def test_existing_workspace_returns_exists(self, client_and_workspaces):
        client, ws_root = client_and_workspaces
        (ws_root / "already-here").mkdir()
        resp = client.post(
            "/video-edit/v2/projects/create",
            params={"project_name": "already-here"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "exists"

    def test_rejects_unsafe_project_name(self, client_and_workspaces):
        client, _ = client_and_workspaces
        for bad in ["../escape", "has space", "has.dot"]:
            resp = client.post(
                "/video-edit/v2/projects/create",
                params={"project_name": bad},
            )
            assert resp.status_code == 400, f"unexpected 200 for {bad!r}"


# ─── POST /projects/{name}/clear/planning ────────────────────────────────────

class TestClearPlanning:
    def test_unknown_project_returns_404(self, client_and_workspaces):
        client, _ = client_and_workspaces
        resp = client.post("/video-edit/v2/projects/ghost/clear/planning")
        assert resp.status_code == 404

    def test_clears_expected_subdirs_and_files(self, client_and_workspaces, monkeypatch):
        client, ws_root = client_and_workspaces

        # Create a project with all the files clear/planning is meant to purge.
        from src.pipelines.v2.workspace import WorkspaceManager
        from src.pipelines.v2.schemas import VideoEntry
        ws = WorkspaceManager("proj", ws_root)
        ws.create()
        ws.init_session([VideoEntry("v1", "v.mp4", "/v.mp4", 10.0)], gist="g")

        # Seed files + dirs that should be purged
        (ws.root / "chat_history.json").write_text("[]")
        (ws.root / "event_log.json").write_text("[]")
        (ws.root / "context.md").write_text("x")
        (ws.root / "fcpxml").mkdir(exist_ok=True)
        (ws.root / "fcpxml" / "edit_v1.fcpxml").write_text("<fcpxml/>")
        (ws.root / "scratchpads").mkdir(exist_ok=True)
        (ws.root / "scratchpads" / "planning.md").write_text("old plan")

        resp = client.post("/video-edit/v2/projects/proj/clear/planning")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared", "project_name": "proj"}

        # Files should be gone, directories emptied.
        assert not (ws.root / "chat_history.json").exists()
        assert not (ws.root / "event_log.json").exists()
        assert not (ws.root / "context.md").exists()
        assert (ws.root / "fcpxml").is_dir()
        assert list((ws.root / "fcpxml").iterdir()) == []
        assert (ws.root / "scratchpads").is_dir()
        assert list((ws.root / "scratchpads").iterdir()) == []

    def test_evicts_in_memory_agent_session(self, client_and_workspaces):
        client, ws_root = client_and_workspaces
        from src.pipelines.v2.workspace import WorkspaceManager
        from src.pipelines.v2.schemas import VideoEntry
        from src import services

        ws = WorkspaceManager("proj", ws_root)
        ws.create()
        ws.init_session([VideoEntry("v1", "v.mp4", "/v.mp4", 10.0)], gist="")

        services._agent_sessions["proj"] = object()  # sentinel
        client.post("/video-edit/v2/projects/proj/clear/planning")
        assert "proj" not in services._agent_sessions
