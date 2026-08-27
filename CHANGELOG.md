# Changelog

## 2.1.0

Removes the local indexing backend. Video understanding is now one HTTP
dependency: the **Memories.ai Video Datalake**.

### Why

The local backend was a path dependency on a private sibling repository, so
`uv sync` could never work outside Memories.ai — the public repo was not
installable at all. It also pulled ~860 lines of transitive lock entries
(torch, OpenVINO, MobileCLIP, scipy, sklearn, sqlite-vec) for a stack nobody
outside could run.

### Changes

* `VIDEO_BACKEND` is gone; there is one backend and `MEMORIES_API_KEY` is now
  required.
* Indexing works through the API again — `POST /v2/index`, the dashboard's
  **Index footage** button, and the CLI all ingest into the project's
  collection. The 2.0 release refused those paths on the datalake backend.
* **One collection per project**, named `vea-{project}` and recorded on
  `SessionData.datalake_collection_id`. Each `VideoEntry` carries its
  `datalake_video_id`; `video_no` stays the filename the agent writes into
  `source_file`.
* Retrieval is **scoped per project** via `services.project_handles(session)`,
  so one project's search never reaches another's footage. `DATALAKE_MAP` and
  the sidecar file it pointed at are no longer needed.
* `POST /v2/projects/{p}/clear/memories` now deletes the project's videos from
  its collection. This is real deletion of billed indexing — re-indexing pays
  again.
* Gists come from the datalake's own per-video summary (one $0.001 read)
  instead of an extra LLM round trip.
* `/v2/plan` wraps `main_llm` in `StructuredLLM` instead of borrowing the
  retrieval backend's LLM adapter.

### Swapping the backend

The contract is still two methods — `ask(question, video_id=...)` and
`search(query, video_ids=, top_k=, collections=)`. Implement them, return them
from `init_retrieval()`, and everything downstream is unchanged.

## 2.0.0

VEA 2.0 is the agent release. Editing is now a conversation with a tool-using
agent instead of a fixed pipeline, and video understanding is a swappable
backend rather than a hard dependency on one service.

### The agent replaces the pipeline

* Editing runs through `AgentSession` — an LLM loop with ten tools
  (`ask_memories`, `search_footage`, `refine_clip_timestamps`,
  `update_scratchpad`, `generate_fcpxml`, `generate_narration`,
  `select_music`, `generate_subtitles`, `message_user`, `finish_turn`),
  four persistent scratchpads for memory across the context window, and two
  temperaments: collaborative (dashboard) and autonomous (CLI).
* `EditDecision` JSON is compiled to FCPXML 1.10 deterministically — no LLM
  in the compile step — with exact rational frame timing.
* Draft renders happen automatically through ffmpeg after every
  `generate_fcpxml`; DaVinci Resolve renders the final when it is installed.
* A React dashboard carries the chat, an NLE-style multi-track timeline, the
  scratchpads and the preview.
* `python -m src.cli` (`vea-oneshot`) runs one autonomous turn end to end and
  prints a single JSON line, for orchestration by another agent.

### Two interchangeable video-understanding backends

The agent reaches its understanding layer through exactly two contracts, so
the backend behind them is a choice:

Two backends shipped in 2.0 behind `VIDEO_BACKEND`: a local index and the
hosted **Memories.ai Video Datalake**. 2.1.0 removed the local one — see
below.

### Removed

* The V1 pipeline (`videoComprehension` → `flexibleResponse`), its
  `/video-edit/v1` routes and the `MemoriesAiManager` cloud client. The
  paper's original codebase is preserved on the
  [`legacy/v1-main`](https://github.com/Memories-ai-labs/vea-open-source/tree/legacy/v1-main)
  branch.
* `run.sh`'s unconditional ngrok setup, which existed only to receive the V1
  caption webhook. Now behind `--ngrok`.

### Notes for operators

* `MEMORIES_API_KEY` is no longer a V1 leftover — it is the datalake key, and
  only needed with `VIDEO_BACKEND=datalake`.
* Datalake ingest caps concurrent in-flight uploads: the API accepts about
  five at a time and answers `429` after that, and its `retry_after` hint does
  not describe that limit. `scripts/datalake_ingest.py` gates on a semaphore
  and reuses videos already present in the collection by title, so re-runs
  neither re-upload nor re-bill.
* Every priced datalake call is tallied into the logs:
  `[DATALAKE COST] searches=23 (reranked=4) derived_reads=6 ~$0.27`.

### Tests

341 offline tests (`.venv/bin/pytest tests/v2 -q`), no LLM, network or ffmpeg
calls. The agent turn loop, the ffmpeg renderer and the frontend are still
covered only by running them.
