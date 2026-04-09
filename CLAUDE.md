# CLAUDE.md

Orientation for AI coding assistants (Codex, Claude Code, etc.) working in this repository. For end-user docs, see `README.md`. For deeper technical detail, see `docs/architecture.md` and `docs/onboarding.md`.

## What this project is

VEA is a video editing automation service. The current product is a **conversational editing agent** that runs in a React dashboard. A user drops video files into a workspace, the system indexes them via Memories.ai, and an LLM-driven agent collaborates with the user in chat to plan, refine, and compile a Final Cut Pro XML edit. Drafts auto-render via FFmpeg; high-quality finals can render via DaVinci Resolve.

There is also a legacy V1 pipeline (videoComprehension → flexibleResponse → ...) at `src/pipelines/` that is kept for reproducibility of the paper. **Most active development happens in V2.** Don't conflate the two.

## Key directories

```
src/
├── app.py                          # FastAPI entrypoint (port 8000)
├── services.py                     # Shared singletons (LLM, Memories.ai, agent sessions)
├── routes/                         # FastAPI routers
│   ├── v2_endpoints.py             # REST endpoints for the V2 API
│   └── v2_websockets.py            # WebSocket: agent chat + indexing progress
├── pipelines/
│   ├── v2/                         # ★ Current architecture
│   │   ├── agent/
│   │   │   ├── agent_session.py    # Agent loop, history, persistence
│   │   │   ├── tools.py            # Tool executor (functions invoked by LLM)
│   │   │   ├── tool_definitions.py # Gemini FunctionDeclarations (10 tools)
│   │   │   ├── tool_helpers.py     # ElevenLabs / Soundstripe / ffmpeg helpers
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

## LLM provider

V2 supports two backends, controlled by env vars (loaded from `config.json` at startup):

* **OpenRouter** — set `OPENROUTER_API_KEY`. Model defaults to `google/gemini-2.5-flash`, override with `OPENROUTER_MODEL`. This is the default if both are configured.
* **Vertex AI Gemini** — set `GOOGLE_CLOUD_PROJECT` and run `gcloud auth application-default login`. Force this path by setting `LLM_PROVIDER=vertex`.

The selection happens in `src/services.py`. Both providers expose the same interface (`gemini_manager`), so call sites are identical.

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
| `refine_clip_timestamps` | ffmpeg extract → downsample → Gemini structured output for in/out points |
| `update_scratchpad` | Write to `comprehension` / `creative_direction` / `planning` / `fcpxml` |
| `generate_fcpxml` | Validate clips against ffprobe durations, compile EditDecision → FCPXML, kick off draft render |
| `generate_narration` | ElevenLabs TTS with `convert_with_timestamps` (real word boundaries) |
| `select_music` | Soundstripe search → LLM picks track → download |
| `generate_subtitles` | STT via ElevenLabs Scribe → subtitle text overlays |
| `message_user` | Mid-flow message to dashboard (does NOT end the turn) |
| `finish_turn` | Explicit turn-end signal with optional final message |

### Scratchpads

Four markdown files in `{workspace}/scratchpads/` give the agent durable memory across the sliding context window. They're injected into the system prompt on every call by `system_prompt.build_system_prompt()`. Operations: `replace`, `append`, `prepend`. Cap: 6000 chars per pad.

### EditDecision schema

`src/pipelines/v2/schemas.py` defines the contract between the LLM and the FCPXML compiler. Top-level: `timeline`, `clips[]`, `narration[]`, `music`, `titles[]`. Each clip has `source_file`, `source_start`, `source_end`, `gain_db`, `speed`, `transform`, `transform_mode`, `shot_transforms`, optional `track`/`timeline_offset` for overlays, and a read-only `measured_loudness_lufs` set after rendering.

### Audio gain semantics (easy to get wrong)

* **Source clips** (`clips[].gain_db`) — literal dB adjustment. `0` = original. After a render, the agent reads `measured_loudness_lufs` and computes `gain_db = target_lufs - measured`.
* **Music / narration** (`gain_db` on `MusicTrack` / `NarrationSegment`) — **offset from target**, not literal. Default `0` means "play at target loudness" (-18 LUFS for music, -16 for narration). The renderer computes `(target + offset) - measured` automatically. Writing `gain_db = -18` for music makes it inaudible — that's the most common LLM mistake; the system prompt warns about it explicitly.

### FCPXML compilation

`src/pipelines/v2/fcpxml/edit_compiler.py` is fully deterministic — no LLM in this step. It uses `Fraction` for exact rational frame timing, registers assets for unique source files, and emits FCPXML 1.10. Source paths are absolute filesystem paths inside the workspace `footage/` directory.

### Renders

* **Draft** — `src/pipelines/v2/preview/ffmpeg_renderer.py` runs synchronously after `generate_fcpxml`. Outputs `{workspace}/renders/draft.mp4`. Always available.
* **Final** — DaVinci Resolve via `lib/utils/resolve_render.py`. Outputs `{workspace}/renders/final.mp4`. Optional; requires DaVinci Resolve Studio.

The dashboard's preview tabs default to draft, with Final shown when available.

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
* `[MUSIC]` — Soundstripe selection

### Tool result protocol

Tool executors return a `dict` that gets serialized into the LLM `FunctionResponse`. Always include either `success`/`result` or `error`. Side effects that the dashboard should react to are emitted as separate WebSocket events from `agent_session.py` (e.g. `timeline_update`, `scratchpad_update`, `render_start/progress/complete`).

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

## Testing notes

There's no formal test suite yet. The standard sanity check is:

1. `./dev.sh up`
2. Open dashboard, pick a small project (a few short clips)
3. Click Index, wait for completion
4. Send a brief like "make a 30 second highlight reel with narration and music"
5. Watch the chat, scratchpads, timeline, and draft preview update live
6. Verify `data/workspaces/{project}/fcpxml/edit_v1.fcpxml` and `renders/draft.mp4` exist

For end-to-end testing, the Playwright MCP server can drive the dashboard.

## Things not to be confused by

* **Two README files in `dashboard/`** — the one at `dashboard/README.md` is a Vite template stub, the real docs are at the repo root.
* **`run.sh` vs `dev.sh`** — `run.sh` is the legacy V1 launcher (sets up ngrok for the V1 webhook indexer). For V2 work, always use `dev.sh`.
* **`vinet_v2/` directory** — V1 dynamic cropping uses ViNet saliency. V2 doesn't use ViNet; the agent uses LLM-based saliency analysis instead. You can ignore this directory unless touching V1.
* **`config/apiKeys.json`** — does not exist in V2. Keys live in `config.json` under `api_keys`.
* **`gs://` GCS paths** in v1 code — irrelevant to V2. Don't propagate them.
* **`docs/architecture-v2.md` and `docs/implementation-plan.md`** — historical planning docs. The authoritative architecture doc is `docs/architecture.md`.
