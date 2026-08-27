# Changelog

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

| `VIDEO_BACKEND` | Retrieval | Indexing |
|---|---|---|
| unset (default) | local **lvmm-core** — SQLite + sqlite-vec, MobileCLIP embeddings, no infrastructure | `POST /v2/index`, on your machine |
| `datalake` | hosted **Memories.ai Video Datalake** — captions, transcription, summaries | `python -m scripts.datalake_ingest --project NAME` |

Everything downstream — agent loop, tools, compiler, renderers — is identical
on both paths. Indexing does not cross over: on the datalake backend the
local-index endpoints refuse with the command to run instead.

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
