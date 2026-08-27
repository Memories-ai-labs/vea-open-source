"""Tests for the Video Datalake retrieval backend (``src/datalake.py``).

The backend's whole job is to be indistinguishable from lvmm-core's
``Querier`` / ``MaviAgent`` to everything downstream, so these lock in the
places where a refactor would silently break the swap:

- **Field mapping.** The datalake says ``start`` / ``end`` / ``score``; VEA
  reads ``start_time`` / ``end_time`` / ``similarity``. A rename on either
  side must fail here, not in the FCPXML compiler.
- **id <-> filename translation.** VEA's ``video_no`` is a filename (the agent
  puts it in ``source_file``); the datalake's is ``vid_...``. If the
  translation drops, every edit decision references a file that isn't there.
- **Retry policy.** 429 / 5xx retry, other 4xx do not — a retried 400 just
  burns the round, and an un-retried 429 aborts the agent's turn.
- **Cost accounting.** Reranked searches are billed x3; the tally is what makes
  a run's spend visible in the logs.

No network: a fake ``aiohttp`` session is injected into the client.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src import datalake as dl


# ─── fakes ───────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self._body = body

    async def json(self, content_type=None):  # noqa: ANN001 - aiohttp signature
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Replays a queued list of (status, body) responses and records calls."""

    closed = False

    def __init__(self, responses: list[tuple[int, dict]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method: str, url: str, **kw):
        self.calls.append((method, url, kw.get("json") or {}))
        status, body = self._responses.pop(0) if self._responses else (200, {})
        return _FakeResponse(status, body)

    async def close(self) -> None:
        self.closed = True


class FakeLLM:
    """Stands in for OpenRouterManager / GeminiGenaiManager."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.contexts: list[str] = []

    def LLM_request(self, prompt_contents, schema=None, retry_delay=60,
                    max_retries=3, context=None):
        self.contexts.append(context or "")
        if context in self.replies:
            return self.replies[context]
        if isinstance(self.replies.get("_raise"), Exception) and context == "datalake_rewrite":
            raise self.replies["_raise"]
        return f"answer for {context}"


def client(responses, **kw) -> dl.DatalakeClient:
    return dl.DatalakeClient(
        api_key="sk-mai-test",
        collection_id="col_test",
        session=FakeSession(responses),
        **kw,
    )


def moment(video_id="vid_abc", start=1.0, end=4.0, score=0.5, snippet="hands"):
    return {
        "ref": f"{video_id}@{start}-{end}",
        "video_id": video_id,
        "target": "caption",
        "score": score,
        "start": start,
        "end": end,
        "snippet": snippet,
    }


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry tests must not actually wait out the backoff."""
    async def _instant(_seconds, result=None):
        return result
    monkeypatch.setattr(dl.asyncio, "sleep", _instant)


# ─── sidecar + translation ───────────────────────────────────────────────────

class TestSidecarMap:
    def test_sidecar_supplies_collection_id_and_both_directions(self, tmp_path):
        p = tmp_path / "datalake.json"
        p.write_text(json.dumps({
            "collection_id": "col_from_sidecar",
            "videos": {"clip_a.mp4": "vid_aaa", "clip_b.mp4": "vid_bbb"},
        }))
        c = dl.DatalakeClient(api_key="k", map_path=str(p), session=FakeSession([]))

        assert c.collection_id == "col_from_sidecar"
        assert c.to_ids(["clip_a.mp4", "clip_b.mp4"]) == ["vid_aaa", "vid_bbb"]
        assert c.to_name("vid_bbb") == "clip_b.mp4"

    def test_explicit_collection_id_wins_over_sidecar(self, tmp_path):
        p = tmp_path / "datalake.json"
        p.write_text(json.dumps({"collection_id": "col_sidecar", "videos": {}}))
        c = dl.DatalakeClient(api_key="k", collection_id="col_explicit",
                              map_path=str(p), session=FakeSession([]))
        assert c.collection_id == "col_explicit"

    def test_unknown_names_and_ids_pass_through_untouched(self):
        c = client([])
        # No sidecar: a filename the map doesn't know must still reach the API
        # rather than becoming None and widening the search to the collection.
        assert c.to_ids(["mystery.mp4"]) == ["mystery.mp4"]
        assert c.to_name("vid_unmapped") == "vid_unmapped"

    def test_no_collection_id_anywhere_is_a_construction_error(self):
        with pytest.raises(ValueError, match="collection id required"):
            dl.DatalakeClient(api_key="k", session=FakeSession([]))

    def test_missing_api_key_is_a_construction_error(self, monkeypatch):
        monkeypatch.delenv("MEMORIES_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MEMORIES_API_KEY"):
            dl.DatalakeClient(collection_id="col_test", session=FakeSession([]))

    async def test_search_translates_ids_in_and_names_out(self, tmp_path):
        p = tmp_path / "datalake.json"
        p.write_text(json.dumps({
            "collection_id": "col_test",
            "videos": {"clip_a.mp4": "vid_aaa"},
        }))
        sess = FakeSession([(200, {"results": [moment(video_id="vid_aaa")]})])
        c = dl.DatalakeClient(api_key="k", map_path=str(p), session=sess)

        out = await c.search("q", targets=["caption"], video_ids=["clip_a.mp4"])

        assert sess.calls[0][2]["filter"] == {"video_ids": ["vid_aaa"]}   # in
        assert out[0]["video_id"] == "clip_a.mp4"                          # out


# ─── querier contract ────────────────────────────────────────────────────────

class TestQuerierContract:
    def test_lvmm_collection_names_map_to_datalake_targets(self):
        assert dl.DatalakeQuerier.targets_for(["video_transcript", "transcript"]) == [
            "caption", "transcription"
        ]
        assert dl.DatalakeQuerier.targets_for(None) == ["caption", "transcription"]
        # An unknown collection must not produce an empty `targets`, which the
        # API rejects ("Non-empty search targets").
        assert dl.DatalakeQuerier.targets_for(["nonsense"]) == ["caption"]

    async def test_hit_shape_matches_what_search_footage_reads(self):
        c = client([(200, {"results": [moment(start=41.008, end=51.02, score=0.66)]})])
        hits = await dl.DatalakeQuerier(c).search("screwdriver", top_k=5)

        assert hits[0]["start_time"] == pytest.approx(41.008)
        assert hits[0]["end_time"] == pytest.approx(51.02)
        assert hits[0]["similarity"] == pytest.approx(0.66)
        assert hits[0]["snippet"] == "hands"

    async def test_zero_end_becomes_none_so_callers_apply_target_duration(self):
        # tools._search_footage falls back to start + target_duration when
        # end_time is None; a literal 0.0 would produce a zero-length clip.
        c = client([(200, {"results": [moment(start=5.0, end=0)]})])
        hits = await dl.DatalakeQuerier(c).search("q")
        assert hits[0]["end_time"] is None

    async def test_querier_search_does_not_pay_for_rerank(self):
        c = client([(200, {"results": []})])
        await dl.DatalakeQuerier(c).search("q")
        assert "rerank" not in c._session.calls[0][2]
        assert c.cost.reranked_searches == 0


# ─── retry policy ────────────────────────────────────────────────────────────

class TestRetryPolicy:
    async def test_429_is_retried_and_then_succeeds(self):
        c = client([
            (429, {"error": {"code": "rate_limited", "retry_after": 1}}),
            (200, {"results": [moment()]}),
        ])
        out = await c.search("q", targets=["caption"])
        assert len(out) == 1
        assert len(c._session.calls) == 2

    async def test_5xx_is_retried(self):
        c = client([(503, {"error": "upstream"}), (200, {"results": []})])
        await c.search("q", targets=["caption"])
        assert len(c._session.calls) == 2

    async def test_other_4xx_fails_immediately(self):
        c = client([(400, {"error": {"message": "bad targets"}}), (200, {"results": []})])
        with pytest.raises(RuntimeError, match="-> 400"):
            await c.search("q", targets=[])
        assert len(c._session.calls) == 1   # no pointless retry

    async def test_exhausted_attempts_raise_with_the_last_reason(self):
        c = client([(429, {"error": {"retry_after": 1}})] * 3, max_attempts=3)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            await c.search("q", targets=["caption"])
        assert len(c._session.calls) == 3

    async def test_transport_errors_are_retried(self):
        class FlakySession(FakeSession):
            def request(self, method, url, **kw):
                self.calls.append((method, url, kw.get("json") or {}))
                if len(self.calls) == 1:
                    raise dl.aiohttp.ClientOSError("connection reset")
                return _FakeResponse(200, {"results": []})

        c = dl.DatalakeClient(api_key="k", collection_id="col_test",
                              session=FlakySession([]))
        await c.search("q", targets=["caption"])
        assert len(c._session.calls) == 2


# ─── pagination ──────────────────────────────────────────────────────────────

class TestPagination:
    async def test_follows_next_cursor_until_max_results(self):
        c = client([
            (200, {"results": [moment(start=1), moment(start=2)], "next_cursor": "CUR1"}),
            (200, {"results": [moment(start=3), moment(start=4)], "next_cursor": None}),
        ])
        out = await c.search("q", targets=["caption"], top_k=2, max_results=4)

        assert [h["start"] for h in out] == [1, 2, 3, 4]
        assert c._session.calls[1][2]["cursor"] == "CUR1"
        # "all other fields must match the first page exactly"
        first, second = c._session.calls[0][2], dict(c._session.calls[1][2])
        second.pop("cursor")
        assert first == second

    async def test_stops_at_one_page_by_default(self):
        c = client([
            (200, {"results": [moment()], "next_cursor": "CUR1"}),
            (200, {"results": [moment()]}),
        ])
        out = await c.search("q", targets=["caption"], top_k=1)
        assert len(out) == 1
        assert len(c._session.calls) == 1

    async def test_hybrid_mode_never_paginates(self):
        c = client([
            (200, {"results": [moment()], "next_cursor": "CUR1"}),
            (200, {"results": [moment()]}),
        ])
        await c.search("q", targets=["caption"], top_k=5, max_results=20, mode="hybrid")
        assert len(c._session.calls) == 1


# ─── cost accounting ─────────────────────────────────────────────────────────

class TestCost:
    def test_rerank_is_billed_three_times_a_plain_search(self):
        cost = dl.Cost(searches=2, reranked_searches=1)
        assert cost.usd == pytest.approx(
            dl.PRICE_SEARCH_USD + dl.PRICE_SEARCH_USD * dl.PRICE_RERANK_MULTIPLIER
        )

    async def test_searches_and_derived_reads_are_counted(self):
        c = client([
            (200, {"results": []}),
            (200, {"summary": "a summary"}),
            (200, {"segments": []}),
        ])
        await c.search("q", targets=["caption"], rerank=True)
        await c.summary("vid_abc")
        await c.transcription("vid_abc")

        assert (c.cost.searches, c.cost.reranked_searches) == (1, 1)
        assert c.cost.derived_reads == 2
        assert c.cost.usd == pytest.approx(
            dl.PRICE_SEARCH_USD * dl.PRICE_RERANK_MULTIPLIER + 2 * dl.PRICE_DERIVED_READ_USD
        )

    async def test_retries_are_not_billed(self):
        # "Failed requests are never billed" — only the successful page counts.
        c = client([(429, {"error": {"retry_after": 1}}), (200, {"results": []})])
        await c.search("q", targets=["caption"])
        assert c.cost.searches == 1


# ─── transcript shim ─────────────────────────────────────────────────────────

class TestDatabaseShim:
    async def test_transcript_rows_use_lvmm_column_names(self):
        c = client([(200, {"segments": [{"text": "using a screwdriver",
                                        "start": 4.0, "end": 6.5}]})])
        rows = await dl._DatabaseShim(c).query("transcript", {"video_id": "clip_a.mp4"})
        assert rows == [{"text": "using a screwdriver", "start_time": 4.0, "end_time": 6.5}]

    async def test_other_tables_return_empty_rather_than_raising(self):
        c = client([])
        assert await dl._DatabaseShim(c).query("keyframe", {"video_id": "x"}) == []
        assert await dl._DatabaseShim(c).query("transcript", {}) == []

    async def test_api_failure_degrades_to_no_transcript(self):
        # tools._get_transcript_segments treats [] as "no enrichment", so a
        # transcription outage must not break the tool.
        c = client([(400, {"error": "no transcription for this video"})])
        assert await dl._DatabaseShim(c).query("transcript", {"video_id": "x"}) == []


# ─── agent (rewrite -> retrieve -> answer) ───────────────────────────────────

class TestDatalakeAgent:
    async def test_ask_rewrites_then_searches_with_the_rewritten_query(self):
        c = client([
            (200, {"results": [moment()]}),        # caption pass
            (200, {"results": []}),                # transcription pass
        ])
        llm = FakeLLM({"datalake_rewrite": "bicycle freehub screwdriver"})
        trace = await dl.DatalakeAgent(c, llm, rerank=False).ask("What tasks appear here?")

        assert trace.search_query == "bicycle freehub screwdriver"
        assert c._session.calls[0][2]["query"] == "bicycle freehub screwdriver"
        assert llm.contexts == ["datalake_rewrite", "datalake_ask"]

    async def test_rewrite_failure_falls_back_to_the_raw_question(self):
        c = client([(200, {"results": []}), (200, {"results": []})])
        llm = FakeLLM({"_raise": RuntimeError("model down")})
        trace = await dl.DatalakeAgent(c, llm, rerank=False).ask("What tasks appear here?")
        assert trace.search_query == "What tasks appear here?"

    async def test_trace_counts_map_onto_the_fields_vea_sums(self):
        # tools._ask_memories sums reranked_* into one "reference_count".
        c = client([
            (200, {"results": [moment(), moment()]}),
            (200, {"results": [moment()]}),
            (200, {"summary": "s"}),
        ])
        llm = FakeLLM({"datalake_rewrite": "q"})
        trace = await dl.DatalakeAgent(c, llm, rerank=False).ask("q?", video_id="clip_a.mp4")

        assert trace.reranked_video_ts == 2
        assert trace.reranked_audio_ts == 1
        assert trace.reranked_videos == 1
        assert len(trace.evidence) == 3
        assert "answer for datalake_ask" in trace.answer

    async def test_rerank_defaults_on_and_only_on_the_caption_pass(self):
        c = client([(200, {"results": []}), (200, {"results": []})])
        agent = dl.DatalakeAgent(c, FakeLLM({"datalake_rewrite": "q"}))
        assert agent.rerank is True

        await agent.ask("q?")
        caption_call, transcription_call = c._session.calls[0][2], c._session.calls[1][2]
        assert caption_call.get("rerank") is True
        assert "rerank" not in transcription_call

    def test_rerank_can_be_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("DATALAKE_RERANK", "0")
        assert dl.DatalakeAgent(client([]), FakeLLM()).rerank is False


# ─── backend selection + indexing guards ─────────────────────────────────────

class TestBackendGuards:
    """``VIDEO_BACKEND=datalake`` serves retrieval only.

    Indexing (and index-purging) must refuse with the actual next step. Left
    unguarded, ``/v2/index`` drives lvmm-core's pipeline against a context
    with no ``storage`` / ``text_embedding`` / ``vector_db`` and dies deep
    inside a stage, and ``clear/memories`` — best-effort by design — quietly
    strips every ``video_no`` from the session while the collection keeps its
    data.
    """

    def test_video_backend_reports_the_selected_backend(self, monkeypatch):
        from src import services

        monkeypatch.delenv("VIDEO_BACKEND", raising=False)
        assert services.video_backend() == "lvmm"

        monkeypatch.setenv("VIDEO_BACKEND", "DataLake")   # case-insensitive
        assert services.video_backend() == "datalake"

        monkeypatch.setenv("VIDEO_BACKEND", "something-else")
        assert services.video_backend() == "lvmm"

    def test_index_route_refuses_on_the_datalake_backend(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi.testclient import TestClient

        with (
            patch("lib.oss.storage_factory.get_storage_client", return_value=MagicMock()),
            patch("src.services.init_lvmm", new=AsyncMock(return_value=None)),
            patch("src.services.close_lvmm", new=AsyncMock(return_value=None)),
            patch.dict("os.environ", {"VIDEO_BACKEND": "datalake", "OPENROUTER_API_KEY": "or"}),
        ):
            from src import services
            from src.app import app

            monkeypatch.setattr("src.routes.v2_pipelines._config.WORKSPACES_DIR", tmp_path)
            monkeypatch.setattr("src.routes._route_utils._config.WORKSPACES_DIR", tmp_path)
            footage = tmp_path / "proj" / "footage"
            footage.mkdir(parents=True)
            (footage / "a.mp4").write_bytes(b"not really a video")
            monkeypatch.setattr(services, "mavi_agent", MagicMock(), raising=False)
            monkeypatch.setattr(services, "lvmm_ctx", MagicMock(), raising=False)

            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/video-edit/v2/index", json={"project_name": "proj"})

            assert resp.status_code == 409
            assert "scripts.datalake_ingest" in resp.json()["detail"]

    def test_clear_memories_refuses_on_the_datalake_backend(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi.testclient import TestClient

        with (
            patch("lib.oss.storage_factory.get_storage_client", return_value=MagicMock()),
            patch("src.services.init_lvmm", new=AsyncMock(return_value=None)),
            patch("src.services.close_lvmm", new=AsyncMock(return_value=None)),
            patch.dict("os.environ", {"VIDEO_BACKEND": "datalake", "OPENROUTER_API_KEY": "or"}),
        ):
            from src import services
            from src.app import app

            monkeypatch.setattr("src.routes.v2_projects._config.WORKSPACES_DIR", tmp_path)
            monkeypatch.setattr("src.routes._route_utils._config.WORKSPACES_DIR", tmp_path)
            monkeypatch.setattr(services, "lvmm_ctx", MagicMock(), raising=False)

            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/video-edit/v2/projects/proj/clear/memories")

            assert resp.status_code == 409
            assert "DELETE /datalake/v1/videos" in resp.json()["detail"]
