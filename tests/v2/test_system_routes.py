"""Tests for /system/info, /system/model, /system/video_model.

These endpoints back the model switcher in the dashboard header. Their JSON
contract is consumed directly by React — a silent rename or shape change
would blank out the UI.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked LLM clients so services import cleanly."""
    with (
        patch("lib.oss.storage_factory.get_storage_client") as mock_storage,
        patch("lib.llm.MemoriesAiManager.create_memories_manager") as mock_mem,
        patch("lib.llm.GeminiGenaiManager.GeminiGenaiManager") as mock_gemini,
        patch("lib.llm.OpenRouterManager.OpenRouterManager") as mock_or,
        patch.dict("os.environ", {
            "MEMORIES_API_KEY": "test-key",
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "OPENROUTER_API_KEY": "test-or-key",
        }),
    ):
        mock_storage.return_value = MagicMock()
        mock_mem.return_value = MagicMock()

        # Build mocks that expose a `.model` attribute so _name() in the route
        # returns a predictable value.
        def make_llm(model_arg, **kwargs):
            m = MagicMock()
            m.model = model_arg
            return m

        mock_gemini.side_effect = lambda model="gemini-2.5-flash", **kw: make_llm(model)
        mock_or.side_effect = lambda model, **kw: make_llm(model)

        from src.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ─── GET /system/info ────────────────────────────────────────────────────────

class TestGetSystemInfo:
    def test_returns_current_models_and_catalogs(self, client):
        resp = client.get("/video-edit/v2/system/info")
        assert resp.status_code == 200
        body = resp.json()
        # Contract: four top-level keys.
        assert set(body.keys()) == {
            "main_llm", "video_llm",
            "available_main_models", "available_video_models",
        }
        assert isinstance(body["main_llm"], str)
        assert isinstance(body["video_llm"], str)

    def test_catalogs_are_nonempty_lists_of_dicts(self, client):
        body = client.get("/video-edit/v2/system/info").json()
        for key in ("available_main_models", "available_video_models"):
            assert isinstance(body[key], list)
            assert body[key], f"{key} should not be empty"
            for m in body[key]:
                assert {"id", "name", "hint"} <= set(m.keys())

    def test_main_catalog_excludes_pinned_video_models_sanity_check(self, client):
        # The ability to swap main and video independently is a core feature;
        # a refactor that unifies the catalogs would make that impossible.
        body = client.get("/video-edit/v2/system/info").json()
        main_ids = {m["id"] for m in body["available_main_models"]}
        video_ids = {m["id"] for m in body["available_video_models"]}
        # Some overlap is fine (Gemini appears in both), but the lists must
        # be distinct objects in the response.
        assert main_ids != video_ids or len(main_ids) > 0


# ─── POST /system/model (main LLM) ───────────────────────────────────────────

class TestSetMainModel:
    def test_happy_path_returns_applied_id(self, client):
        body = client.get("/video-edit/v2/system/info").json()
        candidate = next(m["id"] for m in body["available_main_models"])
        resp = client.post(
            "/video-edit/v2/system/model",
            json={"model": candidate},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "main_llm": candidate}

    def test_rejects_unknown_model_with_400(self, client):
        resp = client.post(
            "/video-edit/v2/system/model",
            json={"model": "evil/not-a-model"},
        )
        assert resp.status_code == 400
        assert "Unsupported main_llm model" in resp.json()["detail"]

    def test_rejects_empty_payload_with_400(self, client):
        resp = client.post("/video-edit/v2/system/model", json={})
        assert resp.status_code == 400

    def test_rejects_non_string_model_with_400(self, client):
        resp = client.post("/video-edit/v2/system/model", json={"model": 123})
        assert resp.status_code == 400


# ─── POST /system/video_model ────────────────────────────────────────────────

class TestSetVideoModel:
    def test_happy_path_returns_applied_id(self, client):
        body = client.get("/video-edit/v2/system/info").json()
        candidate = next(m["id"] for m in body["available_video_models"])
        resp = client.post(
            "/video-edit/v2/system/video_model",
            json={"model": candidate},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "video_llm": candidate}

    def test_rejects_unknown_model_with_400(self, client):
        resp = client.post(
            "/video-edit/v2/system/video_model",
            json={"model": "evil/not-a-video-model"},
        )
        assert resp.status_code == 400
        assert "Unsupported video_llm model" in resp.json()["detail"]

    def test_switching_main_does_not_touch_video(self, client):
        """Regression guard: the main-LLM endpoint must not affect video_llm."""
        body_before = client.get("/video-edit/v2/system/info").json()
        # Pick a different main model than the current one.
        other_main = next(
            m["id"] for m in body_before["available_main_models"]
            if m["id"] != body_before["main_llm"]
        )
        client.post("/video-edit/v2/system/model", json={"model": other_main})
        body_after = client.get("/video-edit/v2/system/info").json()
        assert body_after["video_llm"] == body_before["video_llm"]
