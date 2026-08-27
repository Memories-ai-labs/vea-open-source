"""Ingest a VEA project's footage into a Memories.ai Video Datalake collection.

Replaces the local lvmm-core indexing step for ``VIDEO_BACKEND=datalake`` runs:
uploads every file in ``data/workspaces/<project>/footage/``, waits for the
datalake to finish indexing, pulls each video's summary as its gist, and writes
the workspace ``session.json`` with the datalake ``vid_...`` ids as ``video_no``.

    python -m scripts.datalake_ingest --project egoexo-dl [--collection-name X]

Prints the collection id on the last line; export it as
``DATALAKE_COLLECTION_ID`` before running ``python -m src.cli``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from src import config as _config  # loads config.json into os.environ
from src.pipelines.v2.schemas import VideoEntry
from src.pipelines.v2.workspace import WorkspaceManager

HOST = "https://api.memories.ai"
PREFIX = "/datalake/v1"


def _probe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return round(float(out), 3)
    except Exception:
        return None


class Ingestor:
    def __init__(self, api_key: str) -> None:
        self.key = api_key
        self.sess: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.sess = aiohttp.ClientSession(
            headers={"Authorization": self.key},
            timeout=aiohttp.ClientTimeout(total=1800),
        )
        return self

    async def __aexit__(self, *exc):
        if self.sess:
            await self.sess.close()

    async def _json(self, method: str, path: str, **kw) -> Dict[str, Any]:
        # The datalake rate-limits at a few QPS and answers 429 with
        # ``retry_after``; honour it instead of failing the whole ingest.
        for attempt in range(8):
            async with self.sess.request(method, f"{HOST}{PREFIX}{path}", **kw) as r:
                body = await r.json(content_type=None)
                if r.status == 429:
                    wait = float((body.get("error") or {}).get("retry_after") or 1) + attempt
                    await asyncio.sleep(wait)
                    continue
                if r.status >= 400:
                    raise RuntimeError(f"{method} {path} -> {r.status}: {body}")
                return body
        raise RuntimeError(f"{method} {path}: still rate-limited after 8 attempts")

    async def create_collection(self, name: str) -> str:
        body = await self._json("POST", "/collections", json={"name": name})
        return body["id"] if "id" in body else body["collection"]["id"]

    async def list_by_title(self, collection_id: str) -> Dict[str, str]:
        """{metadata.title -> video_id} for everything already in the collection."""
        out: Dict[str, str] = {}
        cursor: Optional[str] = None
        while True:
            q = f"?collection_id={collection_id}&page_size=100"
            if cursor:
                q += f"&cursor={cursor}"
            body = await self._json("GET", f"/videos{q}")
            for v in body.get("videos") or []:
                title = (v.get("metadata") or {}).get("title") or ""
                vid = v.get("id") or v.get("video_id") or ""
                if title and vid:
                    out[title] = vid
            cursor = body.get("next_cursor")
            if not cursor:
                return out

    async def upload(self, path: Path, collection_id: str) -> str:
        meta = {
            "collection_id": collection_id,
            "fps": 1.0,
            "metadata": {"title": path.name, "tags": ["vea"]},
            "idempotency_key": f"vea-{collection_id[-8:]}-{path.name}",
        }
        body: Dict[str, Any] = {}
        for attempt in range(12):
            # FormData is single-use: rebuild it (and reopen the file) per try.
            form = aiohttp.FormData()
            form.add_field("json", json.dumps(meta), content_type="application/json")
            with path.open("rb") as fh:
                form.add_field("file", fh, filename=path.name, content_type="video/mp4")
                async with self.sess.post(f"{HOST}{PREFIX}/videos", data=form) as r:
                    body = await r.json(content_type=None)
                    if r.status == 429:
                        wait = float((body.get("error") or {}).get("retry_after") or 1) + 5 * attempt
                        print(f"[RATE] {path.name} 429 (ingest window full), retrying in {wait:.0f}s", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    if r.status >= 400:
                        raise RuntimeError(f"upload {path.name} -> {r.status}: {body}")
            break
        else:
            raise RuntimeError(f"upload {path.name}: still rate-limited after 12 attempts")
        vid = body.get("video_id") or body.get("id")
        print(f"[UPLOAD] {path.name} -> {vid} ({body.get('status')})", flush=True)
        return vid

    async def wait_ready(self, video_id: str, timeout_s: int = 1800) -> str:
        deadline = asyncio.get_event_loop().time() + timeout_s
        last = ""
        while asyncio.get_event_loop().time() < deadline:
            body = await self._json("GET", f"/videos/{video_id}")
            status = body.get("status", "")
            if status != last:
                print(f"[STATUS] {video_id} {status}", flush=True)
                last = status
            if status == "ready":
                return status
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"{video_id} ended as {status}: {body.get('error')}")
            await asyncio.sleep(10)
        raise TimeoutError(f"{video_id} not ready after {timeout_s}s")

    async def summary(self, video_id: str) -> str:
        try:
            body = await self._json("GET", f"/videos/{video_id}/summary")
        except Exception as e:
            print(f"[WARN] summary({video_id}): {e}", flush=True)
            return ""
        return body.get("summary") or body.get("text") or ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--collection-name", default=None)
    ap.add_argument("--collection-id", default=None,
                    help="reuse an existing collection instead of creating one")
    args = ap.parse_args()

    key = os.environ.get("MEMORIES_API_KEY", "")
    if not key:
        print("MEMORIES_API_KEY not set (config.json api_keys or env)", file=sys.stderr)
        return 2

    ws = WorkspaceManager(args.project, _config.WORKSPACES_DIR)
    files = ws.scan_footage()
    if not files:
        print(f"no footage in {ws.get_footage_dir()}", file=sys.stderr)
        return 2
    print(f"[INGEST] {len(files)} files from {ws.get_footage_dir()}", flush=True)

    async with Ingestor(key) as ing:
        coll = args.collection_id or await ing.create_collection(
            args.collection_name or f"vea-{args.project}-{datetime.now(timezone.utc):%Y%m%d}"
        )
        print(f"[INGEST] collection {coll}", flush=True)

        # The datalake caps concurrent in-flight ingests (~5) and answers 429
        # once that window is full, so gate upload+wait pairs on a semaphore
        # instead of firing all 20 uploads at once.
        sem = asyncio.Semaphore(4)

        existing = await ing.list_by_title(coll)
        if existing:
            print(f"[INGEST] {len(existing)} videos already in collection, reusing by title", flush=True)

        async def ingest_one(f: Path) -> str:
            async with sem:
                vid = existing.get(f.name) or await ing.upload(f, coll)
                await ing.wait_ready(vid)
                return vid

        ids: List[str] = list(await asyncio.gather(*[ingest_one(f) for f in files]))
        summaries = await asyncio.gather(*[ing.summary(v) for v in ids])

    # video_no stays the FILENAME (same convention as the local lvmm-core
    # backend) so the agent's edit decisions name real files; the datalake's
    # own vid_... ids live in the sidecar map that src/datalake.py loads.
    entries = [
        VideoEntry(
            video_no=f.name,
            video_name=f.name,
            source_path=str(f.resolve()),
            duration_seconds=_probe_duration(f),
            gist=gist,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        for f, vid, gist in zip(files, ids, summaries)
    ]
    ws.create()
    sidecar = Path(ws.root) / "datalake.json"
    sidecar.write_text(json.dumps(
        {"collection_id": coll, "videos": {f.name: vid for f, vid in zip(files, ids)}},
        indent=2,
    ))
    print(f"[INGEST] wrote {sidecar}", flush=True)
    ws.init_session(entries, gist="\n\n---\n\n".join(
        f"**{e.video_name}**\n\n{e.gist}" for e in entries if e.gist
    ))
    print(f"[INGEST] session.json written with {len(entries)} videos", flush=True)
    print(coll)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
