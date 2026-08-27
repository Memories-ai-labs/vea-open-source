"""Memories.ai Video Datalake backend for VEA's V2 retrieval handles.

VEA V2 reaches its video-understanding layer through exactly two objects:

    mavi_agent.ask(question, video_id=...)  -> trace with .answer
    querier.search(query, video_ids=, top_k=, collections=) -> list[dict]

This module implements both against the hosted **Video Datalake**
(``https://api.memories.ai/datalake/v1``), plus the ingest side, so the agent
loop, the tool executor, the FCPXML compiler and the renderer need no
knowledge of where understanding comes from.

Anything else that satisfies those two signatures can be dropped in the same
way — see ``src/services.init_retrieval``.

Mapping
-------
=========================  ==================================================
What VEA asks for          Datalake
=========================  ==================================================
``Querier.search``         ``POST /search`` (semantic, targets caption +
                           transcription, ``filter.video_ids`` scoping)
``MaviAgent.ask``          rewrite -> ``POST /search`` (reranked) + ``GET
                           /videos/{id}/summary``, answered by VEA's own
                           ``main_llm``
``ctx.database.query(      ``GET /videos/{id}/transcription``
"transcript", ...)``
``video_no``               filename; the datalake's ``vid_...`` ids live on
                           ``VideoEntry.datalake_video_id`` in session.json
=========================  ==================================================

Environment
-----------
=============================  ==============================================
``MEMORIES_API_KEY``           required
``DATALAKE_COLLECTION_ID``     default collection for unscoped clients; each
                               project normally supplies its own from the
                               session (see ``services.project_handles``)
``DATALAKE_RERANK``            ``0`` disables the cross-encoder pass in
                               ``ask`` (billed x3); default on
``DATALAKE_MAX_ATTEMPTS``      HTTP attempts per request (default 5)
=============================  ==============================================
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, TypedDict

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://api.memories.ai"
API_PREFIX = "/datalake/v1"

# VEA's collection names -> datalake search targets
_TARGET_MAP = {
    "video_transcript": "caption",       # visual captions
    "transcript": "transcription",       # spoken audio
    "summary": "summary",
    "keyframe": "frame_embedding",
}

# Published per-call prices (datalake/pricing) — used only for the running
# cost line in the logs, so a run's spend is visible without opening billing.
PRICE_SEARCH_USD = 0.008
PRICE_RERANK_MULTIPLIER = 3      # "billed x3 when actually run"
PRICE_DERIVED_READ_USD = 0.001   # frame / clip / caption / transcription / summary


class LLMLike(Protocol):
    """The slice of VEA's LLM managers this module uses (sync, off-thread)."""

    def LLM_request(
        self,
        prompt_contents: list,
        schema: Any = None,
        retry_delay: int = 60,
        max_retries: int = 3,
        context: Optional[str] = None,
    ) -> Any: ...


class Hit(TypedDict, total=False):
    """One retrieved moment, in the shape ``tools._search_footage`` expects."""

    video_id: str
    start_time: float
    end_time: Optional[float]
    similarity: float
    snippet: str
    target: Optional[str]
    ref: Optional[str]


@dataclass
class Cost:
    """Running tally of priced datalake calls for one process."""

    searches: int = 0
    reranked_searches: int = 0
    derived_reads: int = 0

    @property
    def usd(self) -> float:
        plain = self.searches - self.reranked_searches
        return (
            plain * PRICE_SEARCH_USD
            + self.reranked_searches * PRICE_SEARCH_USD * PRICE_RERANK_MULTIPLIER
            + self.derived_reads * PRICE_DERIVED_READ_USD
        )

    def line(self) -> str:
        return (
            f"searches={self.searches} (reranked={self.reranked_searches}) "
            f"derived_reads={self.derived_reads} ~${self.usd:.3f}"
        )


class DatalakeClient:
    """Thin async client over the datalake REST surface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
        host: str = DEFAULT_HOST,
        map_path: Optional[str] = None,
        name_map: Optional[Dict[str, str]] = None,
        max_attempts: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("MEMORIES_API_KEY", "")
        if not self.api_key:
            raise ValueError("MEMORIES_API_KEY required for the datalake backend")

        # A sidecar file ({"collection_id": ..., "videos": {filename: vid_...}})
        # is still accepted via ``map_path`` / ``DATALAKE_MAP`` for scripted
        # runs; the normal path is ``name_map`` straight from the session.
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}
        sidecar = map_path or os.environ.get("DATALAKE_MAP", "")
        mapped_collection = ""
        if sidecar and os.path.exists(sidecar):
            with open(sidecar) as fh:
                blob = json.load(fh)
            self.name_to_id = dict(blob.get("videos") or {})
            self.id_to_name = {v: k for k, v in self.name_to_id.items()}
            mapped_collection = blob.get("collection_id") or ""

        # A caller that already holds the project's session (filename -> vid_...)
        # passes it directly; the sidecar file is the CLI/script path.
        if name_map:
            self.name_to_id.update(name_map)
            self.id_to_name.update({v: k for k, v in name_map.items()})

        self.collection_id = (
            collection_id or os.environ.get("DATALAKE_COLLECTION_ID", "") or mapped_collection
        )
        self.host = host.rstrip("/")
        self.max_attempts = max_attempts or int(os.environ.get("DATALAKE_MAX_ATTEMPTS", "5"))
        self.cost = Cost()
        self._session: Optional[aiohttp.ClientSession] = session

    # ── id <-> filename translation ───────────────────────────────────
    #
    # VEA identifies footage by filename (``video_no`` in session.json); the
    # datalake identifies it by ``vid_...``. Translate on the way in and out so
    # every id the agent ever sees is a filename it can put in an edit decision.

    def to_ids(self, names: Optional[List[str]]) -> Optional[List[str]]:
        if not names:
            return None
        return [self.name_to_id.get(n, n) for n in names]

    def to_name(self, video_id: str) -> str:
        return self.id_to_name.get(video_id, video_id)

    # ── transport ─────────────────────────────────────────────────────

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Authorization only: a session-level Content-Type would override
            # the multipart boundary on uploads (the API answers 400 "request
            # body is missing or not valid JSON"). aiohttp sets the right type
            # per request — json= for JSON, FormData for uploads.
            self._session = aiohttp.ClientSession(
                headers={"Authorization": self.api_key},
                timeout=aiohttp.ClientTimeout(total=1800),
            )
        return self._session

    async def close(self) -> None:
        if self.cost.searches or self.cost.derived_reads:
            logger.info(f"[DATALAKE COST] {self.cost.line()}")
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, path: str, **kw: Any) -> Dict[str, Any]:
        """One datalake call, retried on 429 / 5xx / transport errors.

        The datalake rate-limits ingest and search separately and answers 429
        with a ``retry_after`` hint; a mid-run 429 used to abort the agent's
        whole tool round, which is a bad trade for a one-second wait.
        """
        sess = await self._sess()
        url = f"{self.host}{API_PREFIX}{path}"
        last = ""

        for attempt in range(self.max_attempts):
            try:
                async with sess.request(method, url, **kw) as r:
                    body = await r.json(content_type=None)
                    if r.status == 429:
                        hint = float((body.get("error") or {}).get("retry_after") or 1)
                        wait = hint + 2 * attempt
                        last = f"429 rate_limited (retry_after={hint})"
                    elif r.status >= 500:
                        wait = 1 + 2 * attempt
                        last = f"{r.status} {str(body)[:120]}"
                    elif r.status >= 400:
                        # 4xx other than 429 is our bug or a bad id — no retry.
                        raise RuntimeError(f"datalake {method} {path} -> {r.status}: {body}")
                    else:
                        return body
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = 1 + 2 * attempt
                last = f"{type(e).__name__}: {e}"

            if attempt + 1 < self.max_attempts:
                logger.warning(
                    f"[DATALAKE] {method} {path} {last} — retry "
                    f"{attempt + 2}/{self.max_attempts} in {wait:.0f}s"
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"datalake {method} {path} failed after {self.max_attempts} attempts: {last}"
        )

    async def _get(self, path: str) -> Dict[str, Any]:
        return await self._request("GET", path)

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", path, json=payload)

    # ── priced reads ──────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        targets: List[str],
        top_k: int = 15,
        video_ids: Optional[List[str]] = None,
        mode: str = "semantic",
        rerank: bool = False,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search one collection, following ``next_cursor`` up to *max_results*.

        ``max_results`` defaults to *top_k* (one page). Hybrid mode does not
        paginate server-side, so raise *top_k* there instead.
        """
        if not self.collection_id:
            raise ValueError(
                "no collection bound to this client — index the project first "
                "(POST /v2/index) or pass collection_id"
            )
        payload: Dict[str, Any] = {
            "collection_id": self.collection_id,
            "query": query,
            "targets": targets,
            "top_k": top_k,
            "mode": mode,
        }
        if rerank:
            payload["rerank"] = True
        scoped = self.to_ids(video_ids)
        if scoped:
            payload["filter"] = {"video_ids": scoped}

        want = max_results or top_k
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            # "cursor: all other fields must match the first page exactly"
            page_payload = dict(payload)
            if cursor:
                page_payload["cursor"] = cursor
            body = await self._post("/search", page_payload)
            self.cost.searches += 1
            if rerank:
                self.cost.reranked_searches += 1

            page = body.get("results") or []
            results.extend(page)
            cursor = body.get("next_cursor")
            if not cursor or not page or len(results) >= want or mode == "hybrid":
                break

        for r in results:
            r["video_id"] = self.to_name(r.get("video_id", ""))
        return results[:want]

    # ── collection + video management (free per pricing) ──────────────

    async def create_collection(self, name: str) -> str:
        body = await self._post("/collections", {"name": name})
        return body.get("id") or (body.get("collection") or {}).get("id", "")

    async def list_collections(self) -> List[Dict[str, Any]]:
        body = await self._get("/collections")
        return body.get("collections") or []

    async def ensure_collection(self, name: str) -> str:
        """Return the id of the collection called *name*, creating it if absent."""
        for c in await self.list_collections():
            if c.get("name") == name:
                return c.get("id", "")
        return await self.create_collection(name)

    async def list_videos(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        cid = collection_id or self.collection_id
        out: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            q = f"?collection_id={cid}&page_size=100" + (f"&cursor={cursor}" if cursor else "")
            body = await self._get(f"/videos{q}")
            out.extend(body.get("videos") or [])
            cursor = body.get("next_cursor")
            if not cursor:
                return out

    async def videos_by_title(self, collection_id: Optional[str] = None) -> Dict[str, str]:
        """``{metadata.title: video_id}`` for everything already ingested."""
        by_title: Dict[str, str] = {}
        for v in await self.list_videos(collection_id):
            title = (v.get("metadata") or {}).get("title") or ""
            vid = v.get("id") or v.get("video_id") or ""
            if title and vid:
                by_title[title] = vid
        return by_title

    async def get_video(self, video_id: str) -> Dict[str, Any]:
        video_id = self.name_to_id.get(video_id, video_id)
        return await self._get(f"/videos/{video_id}")

    async def delete_video(self, video_id: str) -> None:
        video_id = self.name_to_id.get(video_id, video_id)
        await self._request("DELETE", f"/videos/{video_id}")

    async def upload_file(
        self,
        path: str,
        collection_id: Optional[str] = None,
        title: Optional[str] = None,
        fps: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Ingest one local file; returns its ``vid_...``.

        Indexing is priced per video-minute. Concurrency matters: the API
        accepts roughly five in-flight ingests and answers 429 after that, and
        its ``retry_after`` hint does not describe that limit — callers should
        gate uploads (see ``DatalakeIngestor``) rather than lean on retries.
        ``idempotency_key`` makes a replay return the existing video unbilled.
        """
        import os.path as _osp

        cid = collection_id or self.collection_id
        name = title or _osp.basename(path)
        meta = {
            "collection_id": cid,
            "fps": fps,
            "metadata": {"title": name, "tags": tags or ["vea"]},
            "idempotency_key": f"vea-{cid[-8:]}-{name}",
        }
        sess = await self._sess()
        url = f"{self.host}{API_PREFIX}/videos"
        last = ""

        for attempt in range(self.max_attempts):
            # FormData is single-use; rebuild it (and reopen the file) per try.
            form = aiohttp.FormData()
            form.add_field("json", json.dumps(meta), content_type="application/json")
            with open(path, "rb") as fh:
                form.add_field("file", fh, filename=name, content_type="video/mp4")
                async with sess.post(url, data=form) as r:
                    body = await r.json(content_type=None)
                    if r.status == 429:
                        hint = float((body.get("error") or {}).get("retry_after") or 1)
                        last = f"429 (ingest window full, retry_after={hint})"
                    elif r.status >= 400:
                        raise RuntimeError(f"datalake upload {name} -> {r.status}: {body}")
                    else:
                        vid = body.get("video_id") or body.get("id") or ""
                        self.name_to_id[name] = vid
                        self.id_to_name[vid] = name
                        return vid
            wait = 2 + 5 * attempt
            logger.warning(f"[DATALAKE] upload {name} {last} — retrying in {wait}s")
            await asyncio.sleep(wait)

        raise RuntimeError(f"datalake upload {name} failed after {self.max_attempts} attempts: {last}")

    async def wait_ready(self, video_id: str, timeout_s: int = 1800, poll_s: int = 10) -> str:
        """Block until a video reports ``ready``; raises on failure/timeout."""
        waited = 0
        last = ""
        while waited < timeout_s:
            body = await self.get_video(video_id)
            status = body.get("status", "")
            if status != last:
                logger.info(f"[DATALAKE] {video_id} {status}")
                last = status
            if status == "ready":
                return status
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"{video_id} ended as {status}: {body.get('error')}")
            await asyncio.sleep(poll_s)
            waited += poll_s
        raise TimeoutError(f"{video_id} not ready after {timeout_s}s")

    # ── priced reads ──────────────────────────────────────────────────

    async def summary(self, video_id: str) -> str:
        video_id = self.name_to_id.get(video_id, video_id)
        body = await self._get(f"/videos/{video_id}/summary")
        self.cost.derived_reads += 1
        return body.get("summary") or body.get("text") or ""

    async def transcription(self, video_id: str) -> List[Dict[str, Any]]:
        video_id = self.name_to_id.get(video_id, video_id)
        body = await self._get(f"/videos/{video_id}/transcription")
        self.cost.derived_reads += 1
        segs = body.get("segments") or body.get("transcription") or []
        return segs if isinstance(segs, list) else []

    async def caption(self, video_id: str) -> List[Dict[str, Any]]:
        video_id = self.name_to_id.get(video_id, video_id)
        body = await self._get(f"/videos/{video_id}/caption")
        self.cost.derived_reads += 1
        caps = body.get("captions") or body.get("caption") or []
        return caps if isinstance(caps, list) else []


class DatalakeIngestor:
    """Ingest local files into a collection, then read back their summaries.

    The API accepts about five in-flight ingests and answers 429 once that
    window is full, so upload+wait pairs run behind a semaphore instead of
    firing everything at once. Files already in the collection under the same
    title are reused, so a re-run neither re-uploads nor re-bills.
    """

    def __init__(self, client: DatalakeClient, concurrency: int = 4) -> None:
        self.client = client
        self.sem = asyncio.Semaphore(concurrency)

    async def ingest(
        self,
        paths: List[str],
        collection_id: Optional[str] = None,
        on_progress: Optional[Any] = None,
    ) -> List[str]:
        """Return one ``vid_...`` per input path, in the same order."""
        cid = collection_id or self.client.collection_id
        existing = await self.client.videos_by_title(cid)
        if existing:
            logger.info(f"[DATALAKE] {len(existing)} videos already in {cid}, reusing by title")

        done = 0
        total = len(paths)

        async def one(path: str) -> str:
            nonlocal done
            import os.path as _osp
            name = _osp.basename(path)
            async with self.sem:
                vid = existing.get(name)
                if vid:
                    self.client.name_to_id[name] = vid
                    self.client.id_to_name[vid] = name
                else:
                    vid = await self.client.upload_file(path, cid)
                await self.client.wait_ready(vid)
            done += 1
            if on_progress:
                await on_progress(done, total, name, vid)
            return vid

        return list(await asyncio.gather(*[one(p) for p in paths]))


# ---------------------------------------------------------------- querier


class DatalakeQuerier:
    """Semantic moment search: ``POST /search`` in the hit shape VEA reads."""

    def __init__(self, client: DatalakeClient) -> None:
        self.client = client

    @staticmethod
    def targets_for(collections: Optional[List[str]]) -> List[str]:
        targets = [
            _TARGET_MAP[c] for c in (collections or ["video_transcript", "transcript"])
            if c in _TARGET_MAP
        ]
        return targets or ["caption"]

    async def search(
        self,
        query: str,
        video_ids: Optional[List[str]] = None,
        top_k: int = 15,
        collections: Optional[List[str]] = None,
        **_: Any,
    ) -> List[Hit]:
        targets = self.targets_for(collections)
        # No rerank here: search_footage runs many times per turn and rerank is
        # billed x3. ``ask`` opts in, where the call count is small.
        results = await self.client.search(
            query, targets=targets, top_k=top_k, video_ids=video_ids
        )
        hits: List[Hit] = [
            {
                "video_id": r.get("video_id", ""),
                "start_time": float(r.get("start") or 0),
                "end_time": float(r.get("end") or 0) or None,
                "similarity": float(r.get("score") or 0),
                # the datalake's own caption/transcription text for this moment
                "snippet": (r.get("snippet") or "").strip(),
                "target": r.get("target"),
                "ref": r.get("ref"),
            }
            for r in results
        ]
        logger.info(
            f"[DATALAKE] search '{query[:60]}' targets={targets} -> {len(hits)} moments "
            f"| {self.client.cost.line()}"
        )
        return hits


# ------------------------------------------------------------------ agent


@dataclass
class DatalakeTrace:
    """``MaviTrace`` work-alike — only the fields VEA reads."""

    answer: str = ""
    reranked_videos: int = 0
    reranked_video_ts: int = 0
    reranked_audio_ts: int = 0
    reranked_keyframes: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    search_query: str = ""


_REWRITE_PROMPT = """Rewrite this question into ONE short retrieval query for a
video search index (visual captions + speech transcription). Keep the concrete
nouns, actions and objects; drop meta-words like "footage", "video", "clip",
"summarise", "editorial". Answer with the query only, no punctuation, no quotes,
at most 18 words.

QUESTION
{question}
"""

_ANSWER_PROMPT = """You are answering a question about video footage, using ONLY the evidence below.
The evidence comes from a video datalake: a whole-video summary plus timestamped
caption and transcription moments retrieved for this question.

Rules:
- Ground every claim in the evidence. Never invent a shot that is not there.
- Always cite timestamps as seconds, e.g. (12.0-18.5s), so an editor can cut on them.
- If the evidence does not answer the question, say exactly what is missing.

QUESTION
{question}

EVIDENCE
{evidence}
"""


async def _no_summary() -> str:
    """Whole-collection questions have no single video to summarise."""
    return ""


class DatalakeAgent:
    """``MaviAgent`` work-alike: rewrite -> retrieve -> answer.

    Four stages: the rewrite is one cheap LLM call, retrieval is a reranked
    caption search plus a transcription search, the summary read covers the
    whole video, and the answer comes from VEA's own main_llm.
    """

    def __init__(
        self,
        client: DatalakeClient,
        llm: LLMLike,
        video_ids: Optional[List[str]] = None,
        rerank: Optional[bool] = None,
    ) -> None:
        self.client = client
        self.llm = llm
        self.video_ids = video_ids or []
        self.rerank = (
            rerank if rerank is not None
            else os.environ.get("DATALAKE_RERANK", "1").lower() not in ("0", "false", "no")
        )

    async def _llm(self, prompt: str, context: str) -> str:
        out = await asyncio.to_thread(self.llm.LLM_request, [prompt], None, 30, 2, context)
        return out if isinstance(out, str) else str(out)

    async def _rewrite(self, question: str) -> str:
        """Question -> retrieval query. Falls back to the question itself."""
        try:
            q = (await self._llm(_REWRITE_PROMPT.format(question=question), "datalake_rewrite"))
            q = q.strip().strip('"').splitlines()[0].strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[DATALAKE] rewrite failed, using raw question: {e}")
            return question
        if not q or len(q) > 200:
            return question
        return q

    async def ask(self, question: str, video_id: Optional[str] = None, **_: Any) -> DatalakeTrace:
        scope = [video_id] if video_id else (self.video_ids or None)
        query = await self._rewrite(question)

        summary_task = self.client.summary(video_id) if video_id else _no_summary()
        caption_hits, transcript_hits, summary = await asyncio.gather(
            self.client.search(
                query, targets=["caption"], top_k=12, video_ids=scope, rerank=self.rerank
            ),
            self.client.search(query, targets=["transcription"], top_k=8, video_ids=scope),
            summary_task,
        )

        lines: List[str] = []
        if summary:
            lines.append(f"WHOLE-VIDEO SUMMARY ({video_id}):\n{summary}\n")
        if caption_hits:
            lines.append("VISUAL CAPTION MOMENTS:")
            for h in caption_hits:
                lines.append(
                    f"  [{h.get('video_id')}] {float(h.get('start') or 0):.1f}-{float(h.get('end') or 0):.1f}s "
                    f"(score {float(h.get('score') or 0):.3f}): {(h.get('snippet') or '').strip()}"
                )
        if transcript_hits:
            lines.append("SPOKEN TRANSCRIPTION MOMENTS:")
            for h in transcript_hits:
                lines.append(
                    f"  [{h.get('video_id')}] {float(h.get('start') or 0):.1f}-{float(h.get('end') or 0):.1f}s: "
                    f"{(h.get('snippet') or '').strip()}"
                )
        evidence = "\n".join(lines) if lines else "(no moments matched)"

        answer = await self._llm(
            _ANSWER_PROMPT.format(question=question, evidence=evidence), "datalake_ask"
        )

        trace = DatalakeTrace(
            answer=answer,
            reranked_video_ts=len(caption_hits),
            reranked_audio_ts=len(transcript_hits),
            reranked_videos=1 if summary else 0,
            evidence=list(caption_hits) + list(transcript_hits),
            search_query=query,
        )
        logger.info(
            f"[DATALAKE] ask '{question[:50]}' q='{query[:50]}' video={video_id} -> "
            f"{len(trace.answer)} chars from {len(trace.evidence)} moments "
            f"| {self.client.cost.line()}"
        )
        return trace


# ------------------------------------------------------------ ctx + build


class _DatabaseShim:
    """Serves the one ``ctx.database.query`` call VEA makes (audio transcript)."""

    def __init__(self, client: DatalakeClient) -> None:
        self.client = client

    async def query(self, table: str, where: Dict[str, Any]) -> List[Dict[str, Any]]:
        if table != "transcript":
            return []
        video_id = where.get("video_id")
        if not video_id:
            return []
        try:
            segs = await self.client.transcription(video_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[DATALAKE] transcription({video_id}) failed: {e}")
            return []
        return [
            {
                "text": s.get("text") or s.get("content") or "",
                "start_time": s.get("start") or s.get("start_time") or 0,
                "end_time": s.get("end") or s.get("end_time") or 0,
            }
            for s in segs
        ]


@dataclass
class DatalakeContext:
    """``PipelineContext`` stand-in. Only ``.database`` and ``.llm`` are read."""

    client: DatalakeClient
    database: _DatabaseShim
    llm: Optional[LLMLike] = None
    vector_db: None = None

    async def close(self) -> None:
        await self.client.close()


class _Lifecycle:
    def __init__(self, ctx: DatalakeContext) -> None:
        self._ctx = ctx

    async def close(self) -> None:
        await self._ctx.close()


async def build_datalake_context(
    llm: LLMLike,
    video_ids: Optional[List[str]] = None,
) -> tuple[DatalakeContext, _Lifecycle, DatalakeQuerier, DatalakeAgent]:
    """Return ``(ctx, lifecycle, querier, agent)`` for ``services.init_retrieval``."""
    client = DatalakeClient()
    ctx = DatalakeContext(client=client, database=_DatabaseShim(client), llm=llm)
    querier = DatalakeQuerier(client)
    agent = DatalakeAgent(client, llm, video_ids=video_ids)
    logger.info(
        f"[DATALAKE] backend ready (collection={client.collection_id}, host={client.host}, "
        f"rerank={agent.rerank}, max_attempts={client.max_attempts})"
    )
    return ctx, _Lifecycle(ctx), querier, agent
