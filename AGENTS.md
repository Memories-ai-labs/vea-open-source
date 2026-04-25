# CLAUDE.md

Orientation for AI coding assistants (Codex, Claude Code, etc.) working in this repository. For end-user docs, see `README.md`. For deeper technical detail, see `docs/architecture.md` and `docs/onboarding.md`.

## What this project is

VEA is a video editing automation service. The current product is a **conversational editing agent** that runs in a React dashboard. A user drops video files into a workspace, the system indexes them via Memories.ai, and an LLM-driven agent collaborates with the user in chat to plan, refine, and compile a Final Cut Pro XML edit. Drafts auto-render via FFmpeg; high-quality finals can render via DaVinci Resolve.

There is also a legacy V1 pipeline (videoComprehension → flexibleResponse → ...) at `src/pipelines/` that is kept for reproducibility of the paper. **Most active development happens in V2.** Don't conflate the two.

## Key directories

```
src/
├── app.py                          # FastAPI entrypoint (port 8000)
├── services.py                     # Shared singletons (main_llm, video_llm, Memories.ai, agent sessions)
├── cli.py                          # One-shot CLI (vea-oneshot) for non-interactive runs
├── routes/                         # FastAPI routers
│   ├── _route_utils.py             # Path-safety helpers (workspace resolution)
│   ├── v2_pipelines.py             # REST: plan, FCPXML, narration, music, crop
│   ├── v2_projects.py              # REST: list/create/clear + /system/info + /system/model
│   └── v2_websockets.py            # WebSocket: agent chat + indexing progress
├── pipelines/
│   ├── v2/                         # ★ Current architecture
│   │   ├── agent/
│   │   │   ├── agent_session.py    # Agent loop, history, persistence
│   │   │   ├── tools.py            # Tool executor (functions invoked by LLM)
│   │   │   ├── tool_definitions.py # Gemini FunctionDeclarations (10 tools)
│   │   │   ├── tool_helpers.py     # ElevenLabs / ffmpeg helpers
│   │   │   ├── system_prompt.py    # System prompt template + builder
│   │   │   ├── scratchpad.py       # ScratchpadManager (4 persistent .md files)
│   │   │   └── timeline_view.py    # Programmatic timeline diagram for the prompt
│   │   ├── comprehension/
│   │   │   └── lightweight_comprehension.py   # Upload + gist (V2 indexing)
│   │   ├── fcpxml/
│   │   │   └── edit_compiler.py    # Deterministic EditDecision → FCPXML 1.10
│   │   ├── preview/
│   │   │   └── ffmpeg_renderer.py  # Auto draft renderer (no DaVinci needed)
│   │   ├── audio/
│   │   │   └── loudness.py         # ITU-R BS.1770 LUFS measurement (pyloudnorm)
│   │   ├── music/
│   │   │   └── beat_sync.py        # librosa beat detection (advisory, agent reads)
│   │   ├── workspace.py            # WorkspaceManager (file I/O for projects)
│   │   └── schemas.py              # SessionData, EditDecision, ClipDecision, etc.
│   ├── common/                     # Shared (V1+V2): TimelineConstructor, dynamic crop
│   └── flexibleResponse/, videoComprehension/, ...   # V1 legacy pipelines
├── schema.py                       # FastAPI request/response models
└── config.py                       # Config loading, paths, env var population
lib/
├── llm/
│   ├── MemoriesAiManager.py        # Memories.ai client (upload, chat, search)
│   ├── GeminiGenaiManager.py       # Vertex AI Gemini client
│   └── OpenRouterManager.py        # OpenRouter (drop-in replacement for Gemini)
└── utils/
    ├── media.py                    # ffmpeg/ffprobe helpers
    ├── resolve_setup.py            # DaVinci Resolve health check / pythonpath
    └── resolve_render.py           # DaVinci Resolve render entrypoint
dashboard/                          # React + Vite + TypeScript frontend
└── src/
    ├── App.tsx
    ├── hooks/useAgentChat.ts       # WebSocket hook, manages all real-time state
    └── components/                 # AgentChat, NLETimeline, AudioInspector, ...
data/
└── workspaces/{project}/           # Per-project storage (footage, edits, renders)
docs/                               # architecture.md, onboarding.md
tests/v2/                           # pytest suite (230+ tests, offline)
```

## Setup commands

```bash
# Install everything (Python deps via uv, dashboard deps + build)
uv sync
cd dashboard && npm install && npm run build && cd ..

# Configure
cp config.example.json config.json   # then fill in api_keys

# Start backend (also handles setup if missing)
./dev.sh up

# Or run by hand:
source .venv/bin/activate
python -m src.app
```

The dashboard is served at **http://localhost:8000/app**. API docs at **http://localhost:8000/docs**.

System dependency: `ffmpeg` must be installed (`brew install ffmpeg` or distro equivalent).

## LLM providers (main_llm vs video_llm)

V2 uses **two** LLM slots, both initialized in `src/services.py`:

* **`main_llm`** — text + tool-calling workhorse for the agent loop. Any frontier model works (Claude, GPT, Gemini, MiniMax, Qwen) because all calls go through the OpenRouter shim. Defaults to `MAIN_LLM_MODEL` from `config.json` (config.example sets Claude Opus 4.6). Requires `OPENROUTER_API_KEY`.
* **`video_llm`** — used ONLY for tasks that need native video input (`refine_clip_timestamps`, `verify_preview`). Controlled by `VIDEO_LLM_MODEL`. Bare names like `gemini-2.5-flash` route via Vertex AI (needs `GOOGLE_CLOUD_PROJECT` + `gcloud auth application-default login`); slash-prefixed IDs like `google/gemini-3-flash-preview` route via OpenRouter.

Both are live-swappable at runtime:
- Dashboard: header pill opens a two-section dropdown (main + video).
- REST: `POST /video-edit/v2/system/model` / `POST /video-edit/v2/system/video_model` with `{"model": "..."}`.
- Catalogs: `GET /video-edit/v2/system/info` returns current + available models.

`gemini_manager` is kept as a backwards-compat alias pointing at `main_llm`.

## V2 agent architecture (the important part)

### Agent loop

`AgentSession.handle_user_message()` → `_agent_loop()` (max 40 rounds per user message). Each round:

1. Rebuild system prompt with current scratchpad state
2. Call LLM with `_history` + `TOOL_DECLARATIONS` (function calling)
3. If response has function calls: execute via `ToolExecutor`, append `FunctionResponse` parts, loop again
4. If response is text only or `finish_turn` is called: emit `agent_message`, end the turn

History is capped at `MAX_HISTORY_TURNS = 80` and persisted to `chat_history.json`.

### Tools (10)

Declared in `src/pipelines/v2/agent/tool_definitions.py`, executed by `ToolExecutor` in `tools.py`:

| Tool | Purpose |
|------|---------|
| `ask_memories` | Memories.ai chat (Q&A about footage) |
| `search_footage` | Memories.ai BY_CLIP search (returns clips + transcripts) |
| `refine_clip_timestamps` | ffmpeg extract → PySceneDetect boundaries + ElevenLabs STT word grid → video LLM two-pass (reasoning + structured) for in/out points. Auto-retries with wider window if speech is truncated. |
| `update_scratchpad` | Write to `comprehension` / `creative_direction` / `planning` / `fcpxml` |
| `generate_fcpxml` | Validate clips against ffprobe durations and narration word-grid, compile EditDecision → FCPXML (with `workspace_root` for absolute audio paths), kick off draft render as a background task. |
| `generate_narration` | ElevenLabs TTS with `convert_with_timestamps` (real word boundaries), persists `words.json` |
| `select_music` | Google Lyria 3 Pro music generation via OpenRouter. Returns `beats: [float]` and `tempo_bpm` — beat-aligned cutting is advisory, the agent decides when to snap |
| `generate_subtitles` | STT via ElevenLabs Scribe → subtitle text overlays |
| `message_user` | Mid-flow message to dashboard (does NOT end the turn) |
| `finish_turn` | Explicit turn-end signal with optional final message |

### Scratchpads

Four markdown files in `{workspace}/scratchpads/` give the agent durable memory across the sliding context window. They're injected into the system prompt on every call by `system_prompt.build_system_prompt()`. Operations: `replace`, `append`, `prepend`. Cap: 6000 chars per pad.

### EditDecision schema

`src/pipelines/v2/schemas.py` defines the contract between the LLM and the FCPXML compiler. Top-level: `timeline`, `clips[]`, `narration[]`, `music`, `titles[]`. Each clip has `source_file`, `source_start`, `source_end`, `gain_db`, `speed`, `transform`, `transform_mode`, `shot_transforms`, optional `track`/`timeline_offset` for overlays, and a read-only `measured_loudness_lufs` set after rendering.

### Audio gain semantics (easy to get wrong)

* **Source clips** (`clips[].gain_db`) — literal dB adjustment. `0` = original. After a render, the agent reads `measured_loudness_lufs` and computes `gain_db = target_lufs - measured`.
* **Music / narration** (`gain_db` on `MusicTrack` / `NarrationSegment`) — **offset from target**, not literal. Default `0` means "play at target loudness". Narration target is always `-16` LUFS. Music target is **`-35` LUFS when any narration segment exists** (voice-over separation), else `-18`. The renderer in `ffmpeg_renderer.py:TARGET_*` computes `(target + offset) - measured` automatically. Writing `gain_db = -18` for music under narration thinking it's the target would produce `-53` LUFS — inaudible — so the system prompt warns about the offset semantics explicitly.

### FCPXML compilation

`src/pipelines/v2/fcpxml/edit_compiler.py:compile_edit_decision(edit, output_path, workspace_root=...)` is fully deterministic — no LLM in this step. It uses `Fraction` for exact rational frame timing, registers assets for unique source files, and emits FCPXML 1.10. Pass `workspace_root` so bare narration/music filenames get resolved to absolute URIs under `{workspace}/narration/` and `{workspace}/music/` — without it, Resolve imports fail because the fallback `file:///media/<name>` path doesn't exist.

### Renders

* **FFmpeg** — `src/pipelines/v2/preview/ffmpeg_renderer.py` runs as a **background asyncio task** spawned by `AgentSession._render_ffmpeg` after `generate_fcpxml` succeeds. Outputs `{workspace}/renders/ffmpeg.mp4`. Always available. Two quality modes: `"draft"` (480p, ultrafast preset — default) and `"full"` (timeline-native, slow preset). Toggle per-session via the dashboard render tab or `set_ffmpeg_quality` WebSocket message.
* **Resolve** — DaVinci Resolve via `lib/utils/resolve_render.py`, spawned in parallel to the ffmpeg render. Outputs `{workspace}/renders/resolve.mp4`. Optional; requires DaVinci Resolve Studio.

Because renders are async, callers that need the final file (e.g. the CLI) must `await` the session's `_bg_tasks`. The dashboard's preview tabs show ffmpeg (with the 480p/full toggle) and resolve side-by-side; resolve shows an idle message when unavailable. UI edits (drag/gain/undo) call `schedule_ffmpeg_rerender()` which debounces for ~1.5s before re-rendering.

### Title lane rendering

Titles have a `lane: int` field (FCPXML overlay convention: `lane=1` is "one lane above the spine"). The dashboard `NLETimeline` maps `lane=N` to V-track `V(N+1)` so V1 stays the dedicated spine row and titles render on V2+. Don't confuse this with the old "T1" track layout — titles share the video-family space with overlay clips.

## Conventions

### File / path

* All workspace I/O goes through `WorkspaceManager` — don't hardcode paths.
* Workspaces live under `data/workspaces/{project_name}/`. Footage in `footage/`, generated audio in `narration/` and `music/`, FCPXML in `fcpxml/`, MP4s in `renders/`.
* Timestamps inside the agent and EditDecision are **float seconds**. SRT/subtitle code uses `HH:MM:SS,mmm`. FCPXML uses rational fractions like `1001/30000s`.

### Logging

Tagged prefixes make backend logs easy to grep:

* `[AGENT]` — agent loop, tool execution
* `[AGENT WS]` — WebSocket lifecycle
* `[MEMORIES]` — Memories.ai API calls
* `[COMPREHENSION]` — V2 indexing
* `[COMPILER]` — FCPXML compiler
* `[LOUDNESS]` — LUFS measurement
* `[RENDER]` — draft/final rendering
* `[MUSIC]` — Lyria 3 music generation (via OpenRouter)

### Tool result protocol

Tool executors return a `dict` that gets serialized into the LLM `FunctionResponse`. Always include either `success`/`result` or `error`. Side effects that the dashboard should react to are emitted as separate WebSocket events from `agent_session.py` (e.g. `timeline_update`, `scratchpad_update`, `render_start/progress/complete`).

### Per-project logging

`src/pipelines/v2/logging_setup.py` installs a contextvar-scoped JSONL file handler on the root logger. `AgentSession.handle_user_message`, `_render_ffmpeg`, and `_render_resolve` each wrap their work in a `WorkspaceLogScope`, which routes every `logger.info` / `logger.warning` into `{workspace}/logs/backend.jsonl`. Contextvars propagate through `asyncio.create_task`, so bg renders inherit the scope. Outside a scope, records are dropped rather than leaking between projects.

Bundle contents (all cleared on `clear/planning`, regenerated on next session):
- `manifest.json` — git SHA + ffmpeg version + platform + model IDs (written once per session)
- `backend.jsonl` — Python logger output, one JSON record per line, ISO ts
- `llm.jsonl` — every `main_llm.generate_content` call (and video_llm calls from tools): prompt turn count, response text, function_calls, token usage, duration. Secrets redacted via `redact()` in `logging_setup.py`
- `renders/ffmpeg-<quality>-<ts>.log` — full ffmpeg subprocess stderr for each render pass, commands included; written via `_render_log_path` contextvar in `ffmpeg_renderer.py`
- `refine_clips/` — PySceneDetect / STT intermediates from `refine_clip_timestamps`

When adding new LLM call sites or subprocess invocations, wire them to the bundle — `append_llm_event(workspace.root, {...})` for LLM calls, `open_render_log(workspace.root, "name")` for subprocess stderr — so support zips stay comprehensive.

### Don't

* Don't add try/except around tool executor logic to "make it more robust" — failures should surface to the LLM as `error` fields so it can recover or message the user.
* Don't hardcode `gs://` paths or assume GCS. V2 is local-storage only. The legacy V1 pipelines have GCS code; leave it alone unless touching V1 explicitly.
* Don't add new sections to the agent system prompt without checking if the existing one already covers it. The prompt has been carefully deduplicated.
* Don't reach into `_agent_sessions` or `_indexing_emitters` from random places — these are private to `services.py` / `v2_websockets.py`.

## Common tasks

### Add a new agent tool

1. Add a `FunctionDeclaration` to `tool_definitions.py`.
2. Add an executor method `_my_tool(args: Dict) -> Dict` on `ToolExecutor` in `tools.py` and dispatch from `execute()`.
3. If the tool produces user-facing side effects, emit a WebSocket event from `agent_session.py` after the tool call.
4. If it touches the workspace, add the file ops to `workspace.py`.
5. Update the system prompt only if the tool's behavior isn't obvious from its declaration.

### Modify the EditDecision schema

1. Update `src/pipelines/v2/schemas.py`.
2. Update `_EDIT_DECISION_SCHEMA` constant in `tool_definitions.py` so the LLM sees the new field.
3. Update `edit_compiler.py` to honor the new field in compilation.
4. Update `timeline_view.py` if the field is timeline-relevant.
5. If the field affects rendering, update `preview/ffmpeg_renderer.py` and `lib/utils/resolve_render.py`.
6. If the field is editable in the UI, wire it through `dashboard/src/hooks/useAgentChat.ts` and the relevant component.

### Run the dashboard with hot-reload

```bash
./dev.sh up --frontend-dev
# Backend on :8000, Vite dev on :5173 (proxies API + WS to :8000)
```

The Vite proxy is in `dashboard/vite.config.ts`; it forwards both REST (`/video-edit/*`) and WebSocket upgrades.

## Testing

Offline pytest suite lives in `tests/v2/`. Run:

```bash
.venv/bin/pytest tests/v2 -q          # ~230 tests, <2s, no LLM or ffmpeg calls
```

Coverage focuses on refactor-safety for contracts: schema round-tripping, workspace
path safety + atomic writes, FCPXML audio-path resolution, scene-detect helper,
scratchpad ops + truncation, REST `/system/*` endpoints, project CRUD + clear,
timeline-view computation, CLI parser + stdout emitter.

**Not covered by pytest** (needs real services):
- Full agent turn loop — use the CLI or dashboard for end-to-end.
- ffmpeg renderer — needs the binary and real media files.
- Frontend components — no Vitest runner configured yet.

For end-to-end: `./dev.sh up` → dashboard at :8000/app, OR the one-shot CLI:

```bash
python -m src.cli --project promo --brief "30s highlight" --footage-dir ./clips
```

Playwright MCP can drive the dashboard for UI regressions.

## One-shot CLI (non-interactive orchestration)

`src/cli.py` (also on PATH as `vea-oneshot` after `uv sync`) runs a single
agent turn without the dashboard. Useful for letting another agent drive VEA
via subprocess or MCP.

```bash
python -m src.cli \
  --project promo \
  --brief "make a 60s promo with narration and music" \
  --footage-dir ./clips \
  [--reuse-index] [--log-format text|jsonl] [--timeout 900]
```

Under the hood it:
1. Symlinks videos from `--footage-dir` into `{workspace}/footage/`.
2. Runs `LightweightComprehension` unless `--reuse-index` and a session exists.
3. Builds an `AgentSession` with `mode=AgentMode.AUTONOMOUS` — the system prompt
   gains the `_AUTONOMOUS_ADDENDUM` (`system_prompt.py`) telling the agent to
   skip clarifying questions, iterate perfectionist-style across multiple
   `generate_fcpxml` calls, and ship a decisive factual `final_message`.
4. Awaits `handle_user_message(brief)`, then waits for ffmpeg + Resolve
   background renders (cap: `DEFAULT_RENDER_WAIT_SECONDS` = 300s).
5. Prints a final JSON line on stdout: `{"status":"ok","fcpxml":"...","ffmpeg_mp4":"...","resolve_mp4":"...","edit_decision":{...}}`.
   Exit 0 on success; non-zero + structured error JSON on failure
   (2=no footage, 3=indexing, 4=empty session, 5=timeout, 6=agent error, 7=no FCPXML produced).

## Agent modes

Two temperaments, both drive the same `_agent_loop`:

| Mode | Where used | Temperament |
|------|-----------|-------------|
| `COLLABORATIVE` (default) | Dashboard / WebSocket | Deferential — proposes plans, asks questions, waits on feedback between iterations |
| `AUTONOMOUS` | CLI / MCP / batch | Perfectionist, non-interactive — commits to reasonable defaults, self-iterates (`generate_fcpxml` is a checkpoint, not a goalpost), ships a decisive `final_message` |

`src/pipelines/v2/agent/modes.py` defines the enum + per-mode `MAX_TOOL_ROUNDS`
cap (collaborative 40 / autonomous 120). The addendum text lives in
`system_prompt.py`. In autonomous mode, the loop also runs a **watchdog** —
if the same tool is called 3× in a row with identical args, a synthetic
user nudge is injected into history once per turn telling the agent to
commit or explain the blocker. On max-rounds exhaustion, the loop emits
a `turn_exhausted` event (separate from a clean `finish_turn`).

Backcompat: the old `autonomous: bool` kwarg on `AgentSession` and
`build_system_prompt` still works — `autonomous=True` maps to AUTONOMOUS.

## Things not to be confused by

* **Two README files in `dashboard/`** — the one at `dashboard/README.md` is a Vite template stub, the real docs are at the repo root.
* **`run.sh` vs `dev.sh`** — `run.sh` is the legacy V1 launcher (sets up ngrok for the V1 webhook indexer). For V2 work, always use `dev.sh`.
* **`vinet_v2/` directory** — V1 dynamic cropping uses ViNet saliency. V2 doesn't use ViNet; the agent uses LLM-based saliency analysis instead. You can ignore this directory unless touching V1.
* **`config/apiKeys.json`** — does not exist in V2. Keys live in `config.json` under `api_keys`.
* **`gs://` GCS paths** in v1 code — irrelevant to V2. Don't propagate them.
* **`docs/architecture-v2.md` and `docs/implementation-plan.md`** — historical planning docs. The authoritative architecture doc is `docs/architecture.md`.
