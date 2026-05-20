"""Integration tests for v2 API endpoints using FastAPI TestClient."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.pipelines.v2.schemas import VideoEntry, Storyboard, Shot, RetrievedClip


# We need to patch external clients before importing app.
#
# PORT NOTE (2026-05-19): Replaced the MemoriesAiManager patch with a no-op
# init_lvmm patch — lvmm-core's build_local_context is the new "external
# dependency" to mock at app startup. The lvmm_ctx / searcher / mavi_agent
# singletons are left None here; individual tests monkeypatch them as
# needed.
@pytest.fixture(scope="module")
def client():
    """Create a TestClient with mocked external dependencies."""
    async def _noop_init_lvmm():
        return None

    # PORT NOTE: previously patched lib.llm.GeminiGenaiManager — that file
    # has been deleted and VEA now uses lvmm_core.adapters.llm.gemini.GeminiAdapter
    # (or OpenRouterAdapter). We patch the GeminiAdapter constructor in
    # services so the FastAPI app starts without a real Gemini key.
    with (
        patch("lib.oss.storage_factory.get_storage_client") as mock_storage,
        patch("src.services.init_lvmm", _noop_init_lvmm),
        patch("lvmm_core.adapters.llm.gemini.GeminiAdapter") as mock_gemini,
        patch.dict("os.environ", {
            "GEMINI_API_KEY": "test-key",
        }),
    ):
        mock_storage.return_value = MagicMock()
        mock_gemini.return_value = MagicMock()

        from src.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def project_workspace(tmp_path, monkeypatch):
    """Create a real workspace and point WORKSPACES_DIR to it."""
    from src.pipelines.v2.workspace import WorkspaceManager
    ws = WorkspaceManager("test_proj", tmp_path)
    ws.create()
    entries = [VideoEntry("v1", "clip.mp4", "/media/clip.mp4", 120.0)]
    ws.init_session(entries, gist="Test gist for keynote")

    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    return ws, tmp_path


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "running" in resp.json().get("message", "").lower()


# ---------------------------------------------------------------------------
# V2 Index — requires Memories.ai mock
# ---------------------------------------------------------------------------

def test_v2_index_missing_memories(client, tmp_path, monkeypatch):
    """Should return 500 if lvmm-core's mavi_agent isn't initialised.

    PORT NOTE: previously asserted on the memories.ai API-key error path.
    Now asserts on the equivalent lvmm-core init failure path.
    """
    monkeypatch.setattr("src.services.mavi_agent", None)
    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    resp = client.post("/video-edit/v2/index", json={
        "project_name": "p1",
        "source_dir": str(tmp_path),
        "start_fresh": False,
    })
    assert resp.status_code == 500
    assert "lvmm-core" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# V2 Plan
# ---------------------------------------------------------------------------

def test_v2_plan_project_not_found(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    monkeypatch.setattr("src.services.mavi_agent", MagicMock())
    monkeypatch.setattr("src.services.searcher", MagicMock())
    monkeypatch.setattr("src.services.gemini_manager", MagicMock())
    resp = client.post("/video-edit/v2/plan", json={
        "project_name": "doesnotexist",
        "prompt": "Make a highlights reel",
    })
    assert resp.status_code == 404


def test_v2_plan_starts_for_existing_project(client, project_workspace, monkeypatch):
    ws, tmp_path = project_workspace
    monkeypatch.setattr("src.services.mavi_agent", MagicMock())
    monkeypatch.setattr("src.services.searcher", MagicMock())
    monkeypatch.setattr("src.services.gemini_manager", MagicMock())

    # Patch IterativePlanningLoop.run to return immediately
    async def fake_run(self):
        return Storyboard(iteration=1, shots=[])

    with patch("src.pipelines.v2.planning.iterative_planning_loop.IterativePlanningLoop.run", fake_run):
        resp = client.post("/video-edit/v2/plan", json={
            "project_name": "test_proj",
            "prompt": "Make a 60s recap",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("started", "already_running")
    assert data["project_name"] == "test_proj"


# ---------------------------------------------------------------------------
# V2 Plan Status
# ---------------------------------------------------------------------------

def test_v2_plan_status(client, project_workspace, monkeypatch):
    ws, tmp_path = project_workspace
    resp = client.get("/video-edit/v2/plan/status", params={"project_name": "test_proj"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_name"] == "test_proj"
    assert data["status"] in ("indexed", "planning", "plan_ready", "fcpxml_ready", "rendered")


def test_v2_plan_status_not_found(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    resp = client.get("/video-edit/v2/plan/status", params={"project_name": "ghost"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# V2 Generate FCPXML
# ---------------------------------------------------------------------------

def test_v2_generate_fcpxml_no_storyboard(client, project_workspace, monkeypatch):
    """Should return 400 if no storyboard exists."""
    monkeypatch.setattr("src.services.gemini_manager", MagicMock())
    resp = client.post("/video-edit/v2/generate_fcpxml", json={"project_name": "test_proj"})
    # No storyboard -> 400
    assert resp.status_code == 400
    assert "storyboard" in resp.json()["detail"].lower()


def test_v2_generate_fcpxml_with_storyboard(client, project_workspace, tmp_path, monkeypatch):
    ws, _ = project_workspace
    monkeypatch.setattr("src.services.gemini_manager", MagicMock())

    # Create a storyboard
    clip = RetrievedClip(
        video_no="v1", video_name="clip.mp4", source_path="/tmp/clip.mp4",
        start_seconds=0, end_seconds=10, score=0.9,
    )
    shot = Shot(id="s1", purpose="intro", search_query="opener", retrieved_clip=clip)
    sb = Storyboard(iteration=1, shots=[shot])
    ws.save_storyboard(sb)

    # Mock the generate_fcpxml function
    fake_path = str(ws.get_fcpxml_path(1))
    Path(fake_path).parent.mkdir(parents=True, exist_ok=True)
    Path(fake_path).write_text("<fcpxml/>")

    async def fake_generate(*args, **kwargs):
        return fake_path

    with patch("src.routes.v2_pipelines.generate_fcpxml", fake_generate):
        resp = client.post("/video-edit/v2/generate_fcpxml", json={"project_name": "test_proj"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["project_name"] == "test_proj"
    assert "fcpxml_path" in data


# ---------------------------------------------------------------------------
# V2 Resolve Status
# ---------------------------------------------------------------------------

def test_v2_resolve_status(client):
    with patch("lib.utils.resolve_setup.check_resolve_status", return_value={
        "running": False, "version": None, "studio": False, "error": "Resolve not found"
    }):
        resp = client.get("/video-edit/v2/resolve/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["studio"] is False


# ---------------------------------------------------------------------------
# V2 Narration — missing project
# ---------------------------------------------------------------------------

def test_v2_narration_not_found(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    monkeypatch.setattr("src.services.gemini_manager", MagicMock())
    resp = client.post("/video-edit/v2/narration", json={"project_name": "ghost"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# V2 Crop — missing project
# ---------------------------------------------------------------------------

def test_v2_crop_not_found(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.WORKSPACES_DIR", tmp_path)
    resp = client.post("/video-edit/v2/crop", json={"project_name": "ghost", "aspect_ratio": 0.5625})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# V2 Pause / Resume / Inject — no active session
# ---------------------------------------------------------------------------

def test_v2_plan_pause_no_session(client):
    resp = client.post("/video-edit/v2/plan/pause", params={"project_name": "ghost"})
    assert resp.status_code == 404


def test_v2_plan_resume_no_session(client):
    resp = client.post("/video-edit/v2/plan/resume", params={"project_name": "ghost"})
    assert resp.status_code == 404


def test_v2_plan_inject_no_session(client):
    resp = client.post("/video-edit/v2/plan/inject", params={"project_name": "ghost", "prompt": "add more action"})
    assert resp.status_code == 404
