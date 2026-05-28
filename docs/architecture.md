# VEA V2 Architecture

This document describes the technical architecture of VEA's V2 agentic editing system. For quickstart instructions, see [onboarding.md](onboarding.md).

---

## System Overview

```
                                    +---------------------+
                                    |   React Dashboard   |
                                    |   (Vite + TypeScript)|
                                    +----------+----------+
                                               |
                                          WebSocket
                                          JSON events
                                               |
+------------------+    local    +-------------+-------------+
|  lvmm-core       |<---------->|   FastAPI Backend (8000)   |
|  (video index,   |            |                             |
|   chat, search)  |            |   AgentSession              |
+------------------+            |     +-- ScratchpadManager   |
                                |     +-- ToolExecutor        |
+------------------+            |     +-- WorkspaceManager    |
|  LLM provider    |<---------->|                             |
|  (OpenRouter or  |            +----+---+---+---+---+-------+
|   Vertex Gemini, |                 |   |   |   |   |
|   function call) |                 |   |   |   |   +-- DaVinci Resolve
+------------------+                 |   |   |   +------ Lyria 3 (OpenRouter)
                                     |   |   +---------- ElevenLabs TTS
+------------------+                 |   +-------------- FCPXML Compiler
|  Local Filesystem|<----------------|
|  data/workspaces/|                 v
|  {project}/      |            edit_decision.json
+------------------+            edit_v1.fcpxml
```

### Key Design Decisions

1. **LLM as agent, not pipeline.** V1 uses a fixed pipeline (comprehend -> plan -> generate). V2 gives Gemini a set of tools and a system prompt; the LLM decides which tools to call and in what order, guided by the user's conversational input.

2. **Scratchpads as durable memory.** Gemini's context window is a sliding window capped at 80 turns. Scratchpads (persisted markdown files) are injected into the system prompt on every call, giving the agent persistent memory across a long editing session.

3. **EditDecision as contract.** The LLM produces a JSON EditDecision object. A deterministic compiler (no LLM involvement) converts it to valid FCPXML 1.10. This separation keeps the XML generation reliable.

4. **Workspace-per-project.** Every project gets a self-contained directory under `data/workspaces/` with footage, session state, scratchpads, generated assets, and outputs.

---

## Agent Session Lifecycle

### Connection Flow

```
Client connects to WS /video-edit/v2/agent/{project_name}/chat
    |
    v
Backend loads WorkspaceManager + SessionData
    |
    v
Creates or reuses AgentSession (one per project, cached in _agent_sessions dict)
    |
    v
Sends "init" event: scratchpads, chat_history, event_log, edit_decision, render_state
    |
    v
Enters message loop:
    - Receives {"type": "user_message", "text": "..."}
    - Creates asyncio task for agent.handle_user_message(text)
    - Drains event queue, forwarding events to WebSocket client
    - Sends "done" when agent task completes
```

### Agent Loop (per user message)

```
handle_user_message(text)
    |
    +-- Append user Content to _history
    |
    +-- _agent_loop():
        |
        +-- For each round (max 40):
            |
            +-- Rebuild system prompt (includes current scratchpads)
            +-- Call Gemini with _history + TOOL_DECLARATIONS
            +-- Append model response to _history
            |
            +-- If response contains function_calls:
            |   +-- For each function_call:
            |   |   +-- Emit "tool_call" event
            |   |   +-- Execute via ToolExecutor
            |   |   +-- Emit "tool_result" event
            |   |   +-- Special handling:
            |   |       - message_user -> emit "agent_message"
            |   |       - generate_fcpxml -> emit "timeline_update", update fcpxml scratchpad
            |   |       - update_scratchpad -> emit "scratchpad_update"
            |   +-- Append FunctionResponse parts to _history
            |   +-- Loop again (next round)
            |
            +-- If response is text only:
                +-- Emit "agent_message"
                +-- Return (agent turn complete)
```

### History Management

- `MAX_HISTORY_TURNS = 80` -- the last 160 Content objects are kept
- History is trimmed after each user message
- Chat log is persisted to `chat_history.json` for session reload
- Event log (tool calls/results) persisted to `event_log.json` (last 200 events)

---

## Tool System

The agent has 10 tools, declared as Gemini `FunctionDeclaration` objects in `tool_definitions.py` and executed by `ToolExecutor` in `tools.py`:

| Tool | Purpose | Calls |
|------|---------|-------|
| `ask_memories` | Ask natural-language questions about footage | lvmm-core MaviAgent |
| `search_footage` | Find clips by query, returns timestamps + transcripts | lvmm-core Querier |
| `refine_clip_timestamps` | Precise in/out point selection using video + audio analysis | ffmpeg extract + downsample, then Gemini structured output |
| `update_scratchpad` | Write to one of 4 persistent scratchpads | Local filesystem |
| `generate_fcpxml` | Validate clips against ffprobe durations, compile EditDecision JSON to FCPXML 1.10, kick off draft render | Deterministic compiler + FFmpeg |
| `generate_narration` | Convert script to voiceover audio with real word-level timestamps | ElevenLabs TTS (`convert_with_timestamps`) |
| `select_music` | Generate background music from text prompt | Google Lyria 3 Pro via OpenRouter |
| `generate_subtitles` | Transcribe original audio, add subtitle text overlays to the edit | ElevenLabs Scribe STT |
| `message_user` | Send visible message to user mid-flow (does NOT end the turn) | WebSocket event |
| `finish_turn` | Explicit signal that the agent's turn is complete (with optional final summary) | WebSocket event |

### refine_clip_timestamps Detail

This tool is critical for edit quality. It performs a multi-step process:

1. Extract the video segment from source using ffmpeg (`-c copy`)
2. Downsample to 480p at 2fps for Gemini analysis
3. Build a refinement prompt describing what to look for
4. Send downsampled video + prompt to Gemini with `RefinedTimestamps` schema
5. Convert Gemini's relative offsets back to absolute source timestamps
6. Return `new_start`, `new_end`, `duration`, `reasoning`, `focus_type`

The system prompt instructs the agent to call this on every clip before generating FCPXML.

---

## Scratchpad System

### Overview

Four named scratchpads provide persistent memory across the sliding conversation window:

| Scratchpad | Purpose | Updated When |
|------------|---------|--------------|
| `comprehension` | Structured knowledge about footage content | After every `ask_memories` call |
| `creative_direction` | User preferences, constraints, feedback | When user expresses intent |
| `planning` | Edit plan: narrative arc, shot list, clip assignments | During planning phase |
| `fcpxml` | Export state, generated FCPXML content, revision notes | After `generate_fcpxml` |

### Implementation

- Managed by `ScratchpadManager` (`src/pipelines/v2/agent/scratchpad.py`)
- Stored as individual `.md` files in `{workspace}/scratchpads/`
- Timestamps tracked in `scratchpads/timestamps.json`
- Max size: 6000 characters per pad (truncated from start if exceeded)
- Operations: `replace`, `append`, `prepend`
- All four pads are rendered into the system prompt on every Gemini call via `render_all()`
- Seeded from indexing gist on session creation

---

## Data Flow: User Message to FCPXML

```
1. User: "Create a 60-second highlight reel focusing on the keynote demo"
       |
2. WebSocket receives {"type": "user_message", "text": "..."}
       |
3. AgentSession.handle_user_message() starts agent loop
       |
4. Gemini decides to call tools:
   a. update_scratchpad(creative_direction, replace, "60s highlight, keynote demo focus")
   b. ask_memories("What happens during the keynote demo?")
   c. update_scratchpad(comprehension, replace, "<structured findings>")
   d. message_user("Here's what I found... Here's my proposed plan...")
       |
5. User approves plan -> agent continues:
   a. search_footage("keynote product demo reveal") -> clips with timestamps
   b. refine_clip_timestamps(source_file, start, end, 8, "Find the moment...") per clip
   c. update_scratchpad(planning, replace, "<shot list with refined timestamps>")
       |
6. Agent assembles EditDecision JSON and calls:
   generate_fcpxml(edit_decision_json)
       |
7. ToolExecutor._generate_fcpxml():
   a. Parses JSON into EditDecision Pydantic model
   b. Resolves source_path for each clip from footage directory
   c. Saves edit_decision.json to workspace/fcpxml/
   d. Calls compile_edit_decision() -> writes edit_v1.fcpxml
   e. Returns paths to both files
       |
8. AgentSession emits:
   - "timeline_update" with edit_decision data (dashboard renders NLE timeline)
   - "scratchpad_update" with FCPXML content
   - Kicks off auto-render via DaVinci Resolve (if available)
       |
9. Dashboard receives events, updates NLE Timeline + scratchpad panel
```

---

## EditDecision Schema

The `EditDecision` model (`src/pipelines/v2/schemas.py`) is the contract between the LLM and the FCPXML compiler. The LLM fills in creative decisions; deterministic code handles XML structure.

```
EditDecision
  +-- timeline: TimelineSettings
  |     name, fps (24.0), width (1920), height (1080)
  |
  +-- clips: List[ClipDecision]        # V1 spine clips are sequential; track 2+ clips are overlays
  |     id, source_file, source_path
  |     source_start, source_end        # seconds (validated against ffprobe duration)
  |     source_width, source_height
  |     label, description
  |     gain_db                         # LITERAL dB adjustment (0 = original, -96 = mute)
  |     measured_loudness_lufs          # READ-ONLY, set by renderer after measurement
  |     speed: SpeedChange              # rate (0.5 = slow-mo, 2.0 = fast)
  |     transform: TransformSettings    # static crop/reframe (scale, position, rotation)
  |     transform_mode                  # "fit" | "custom" | "saliency"
  |     shot_transforms                 # per-shot crop results from dynamic crop tool
  |     transition_after: TransitionSpec
  |     track                           # 1 = V1 spine, 2+ = overlay tracks
  |     timeline_offset                 # absolute timeline position (required for track 2+)
  |
  +-- narration: List[NarrationSegment]
  |     file, timeline_offset
  |     start, duration                 # MUST land on word boundaries from generate_narration
  |     gain_db                         # OFFSET from -16 LUFS target (default 0)
  |     measured_loudness_lufs          # READ-ONLY
  |
  +-- music: MusicTrack (optional)
  |     file
  |     start, duration                 # duration=0 means "match timeline length"
  |     gain_db                         # OFFSET from -18 LUFS target (default 0 — NOT -18!)
  |     measured_loudness_lufs          # READ-ONLY
  |
  +-- titles: List[TextOverlay]
        text, timeline_offset, duration
        font_size, lane                 # positive lane = above spine
        style                           # "title" (centered) | "subtitle" (bottom caption)
        position                        # "center" | "bottom" | "top"
```

### Audio gain semantics

The two `gain_db` rules are easy to confuse:

* **Source clips** — `gain_db` is a literal dB adjustment. `0` = play at original level, `-96` = mute. The agent computes the exact value from `measured_loudness_lufs` after the first render: `gain_db = target_lufs - measured_lufs` where dialogue target is -16 and b-roll target is -18.
* **Narration / music** — `gain_db` is an **offset from the target**, not literal. The renderer measures the actual loudness and applies `(target + agent_offset) - measured`. Default `0` means "play at default target". Writing `gain_db = -18` for music makes it inaudible — that's the most common LLM mistake. The system prompt warns about it explicitly.

---

## FCPXML Compilation

The compiler (`src/pipelines/v2/fcpxml/edit_compiler.py`) converts EditDecision to FCPXML 1.10 with no LLM involvement.

### Compilation Steps

1. Create root `<fcpxml version="1.10">` element
2. Register `<format>` resource with fps/resolution
3. Register video assets (one per unique source file)
4. Register audio assets for narration and music
5. Build `<spine>` with sequential `<asset-clip>` elements:
   - Compute timeline offsets using frame-accurate rational fractions
   - Apply speed changes via `<timeMap>`
   - Apply transforms via `<adjust-transform>`
   - Apply volume via `<adjust-volume>`
6. Place narration segments in lane -1 (attached to parent spine clips)
7. Place music in lane -2 (attached to first spine clip)
8. Place titles in positive lanes with `<text-style-def>`
9. Write XML to disk

### Frame-Accurate Timing

All timing uses Python `Fraction` for exact rational arithmetic:
- FPS is represented as a fraction (e.g., 24fps = 24/1, 29.97fps = 30000/1001)
- Durations are quantized to whole frames
- FCPXML values use rational format: `"1001/30000s"`

### Current Limitations

- Transitions are disabled (hard cuts only) due to audio overlap issues with cross-dissolves
- No keyframed transforms (static crop/position only)

---

## Workspace File Layout

Each project lives in `data/workspaces/{project_name}/`:

```
{project_name}/
+-- footage/                    # Source video files (user drops files here)
|   +-- keynote.mp4
|   +-- interview.mov
|
+-- session.json                # SessionData: video entries, indexing status
+-- chat_history.json           # Persisted chat messages for session reload
+-- event_log.json              # Persisted tool call/result events (last 200)
|
+-- scratchpads/                # Agent's persistent memory
|   +-- comprehension.md
|   +-- creative_direction.md
|   +-- planning.md
|   +-- fcpxml.md
|   +-- timestamps.json
|
+-- narration/                  # Generated voiceover
|   +-- narration_script.txt
|   +-- narration.mp3
|
+-- music/                      # Downloaded music track
|   +-- track.mp3
|
+-- fcpxml/                     # Generated FCPXML files
|   +-- edit_decision.json      # Raw EditDecision JSON (dashboard reads this)
|   +-- edit_v1.fcpxml          # Compiled FCPXML 1.10
|
+-- renders/
|   +-- ffmpeg.mp4              # FFmpeg render, always (480p preview or timeline-native)
|   +-- resolve.mp4             # DaVinci Resolve render, optional (timeline-native)
|
+-- logs/                       # Per-project debug bundle (see "Logging" below)
    +-- manifest.json           # Git SHA, ffmpeg version, platform, model IDs
    +-- backend.jsonl           # JSONL: every [AGENT]/[RENDER]/[COMPILER] log line
    +-- llm.jsonl               # JSONL: every LLM request+response (redacted)
    +-- renders/                # One ffmpeg stderr log per render pass
    +-- refine_clips/           # PySceneDetect + STT intermediates
```

> **Note:** The legacy V1 planning loop wrote additional files (`storyboard.json`, `clips.json`, `iterations/`, `context.md`). The V2 agent flow does not produce these. You may see them in older workspaces; they are safe to ignore or delete.

### Session States

The `session.json` `status` field tracks the project lifecycle:

| Status | Meaning |
|--------|---------|
| `new` | Directory exists but no indexing has been done |
| `indexed` | Videos indexed locally through lvmm-core, gist generated |
| `planning` | Iterative planning loop is running (v1 flow) |
| `fcpxml_ready` | FCPXML has been generated |
| `rendered` | DaVinci Resolve has produced a render |

---

## Dashboard Architecture

### Technology Stack

- React 19 + TypeScript
- Vite for bundling (dev server on 5173, production build served by FastAPI at `/app`)
- No state management library -- React hooks + lifted state
- WebSocket for real-time communication with backend

### Application Structure

```
App.tsx
  +-- ProjectBrowser            # List/create projects (REST: GET /v2/projects)
  +-- AgentChat                 # Main editing workspace
      +-- Header row            # Project name, footage strip, status pills, manage menu
      +-- Middle row            # NLETimeline + VideoPreview (resizable split)
      +-- Bottom row            # Chat panel + Scratchpad panel (resizable split)
```

### Key Components

**ProjectBrowser** (`ProjectBrowser.tsx`): Lists all projects from `data/workspaces/`, shows status, footage count, and allows creating new projects. Fetches data from `GET /video-edit/v2/projects`.

**AgentChat** (`AgentChat.tsx`): The main workspace view with three resizable rows:
- Top: project header with footage pills, index status, manage dropdown
- Middle: NLE timeline (left) + video preview (right), separated by a draggable column divider
- Bottom: chat messages (left) + scratchpad tabs (right)

Features a "Manage" dropdown for: re-indexing, clearing gists, clearing planning/chat, and clearing the local lvmm-core index.

**NLETimeline** (`NLETimeline.tsx`): A custom-built non-linear editing timeline visualization:
- Builds tracks from EditDecision: V1 (video spine), V2+ (overlay clips and titles via their `lane` field), A1 (narration), A2 (music). Titles with `lane=N` render on V-track `N+1` so V1 stays the dedicated spine row.
- Each clip is a colored block with label, sublabel (filename), gain badge
- Audio tracks show procedural waveform shapes
- Timecode ruler with adaptive tick intervals
- Zoom (Ctrl+scroll or slider, 25%-400%)
- Drag-to-pan horizontally
- Hover tooltips with clip details (source file, in/out points, gain, speed)
- Transition diamonds between video clips

**VideoPreview** (`VideoPreview.tsx`): Displays rendered preview video when available.

### useAgentChat Hook

`dashboard/src/hooks/useAgentChat.ts` manages the WebSocket connection and all real-time state:

**Connection management:**
- Connects to `ws://localhost:8000/video-edit/v2/agent/{projectName}/chat`
- Exponential backoff reconnection (500ms -> 5s cap)
- Cleans up on unmount or project change

**State managed:**
- `events: AgentEvent[]` -- all received events
- `messages: ChatMessage[]` -- user + model messages
- `scratchpads: ScratchpadState` -- current content of all 4 pads
- `scratchpadTimestamps` -- last-updated timestamps
- `editDecision: EditDecision | null` -- latest timeline data
- `renderState: RenderState` -- idle/rendering/complete/error
- `connected: boolean`, `busy: boolean`

**Event handling:**
- `init`: Hydrate all state from persisted data on (re)connect
- `agent_message`: Append to messages
- `scratchpad_update`: Update pad content + timestamp
- `timeline_update`: Set editDecision from new edit_decision data
- `render_start/progress/complete/error`: Update render state
- `tool_call`, `tool_result`: Append to events array
- `done`: Clear busy flag

---

## lvmm-core Integration

`src/services.py` initializes lvmm-core's local stack at backend startup:
`PipelineContext`, `Querier`, and `MaviAgent`. V2 uses these features:

### Local Indexing

- `LightweightComprehension` indexes workspace footage through lvmm-core.
- SQLite rows and sqlite-vec vectors are stored under `~/lvmm-data/local.db`.
- Current table/collection names are `summary`, `keyframe`, `video_transcript`, and `transcript`.

### MaviAgent (used by `ask_memories` tool)

- `MaviAgent.ask(question, video_id=...)` -- natural language Q&A against indexed videos
- VEA calls it once per indexed video when a multi-video project needs whole-workspace coverage.
- Returns a `MaviTrace` with answer text and reranked hit counts.

### Querier (used by `search_footage` tool)

- `Querier.search(question, video_ids=..., collections=...)` -- clip-level semantic search
- Returns `list[dict]` items with video id, timestamps, score, and text
- Results are normalized by ToolExecutor into the stable VEA clip shape

### Lightweight Comprehension (V2 indexing)

`LightweightComprehension` (`src/pipelines/v2/comprehension/lightweight_comprehension.py`):
1. Finds video files in the workspace `footage/` directory
2. Indexes each through lvmm-core (or reuses existing local rows/vectors)
3. Gets a per-video gist via MaviAgent
4. Saves `SessionData` with video entries and combined gist

This is much faster than V1 indexing (no scene-by-scene analysis). Detailed understanding happens on-demand during the agent conversation.

---

## LLM Provider Integration

V2 uses **two** LLM slots, wired in `src/services.py` at startup:

* **`main_llm`** — the text + tool-calling workhorse for the agent loop. Routes through `OpenRouterManager` (`lib/llm/OpenRouterManager.py`), so any frontier model works (Claude, GPT, Gemini, MiniMax, Qwen). Controlled by `MAIN_LLM_MODEL` in `config.json`.
* **`video_llm`** — only for tools that need native video input (`refine_clip_timestamps`, `verify_preview`). Controlled by `VIDEO_LLM_MODEL`. Bare names route via `GeminiGenaiManager` (Vertex AI, requires `GOOGLE_CLOUD_PROJECT`). Slash-prefixed IDs route via OpenRouter.

`gemini_manager` is a backwards-compat alias for `main_llm`. Both `main_llm` and `video_llm` expose the same `LLM_request()` surface (function calling, structured output, multimodal). Both can be swapped live: `services.set_main_llm(model_id)` / `services.set_video_llm(model_id)`, exposed via the dashboard header dropdown and the `POST /video-edit/v2/system/model` + `POST /video-edit/v2/system/video_model` endpoints.

### Agent Loop Usage

The agent session calls `self.gemini.genai_client.models.generate_content()` (i.e. `main_llm`) with:
- `contents`: conversation history (user/model/tool Content objects)
- `system_instruction`: rebuilt each turn with current scratchpad state and timeline view
- `tools`: `TOOL_DECLARATIONS` (10 function declarations)
- `safety_settings`: all categories set to `BLOCK_NONE` for creative content

### Structured Output (refine_clip_timestamps)

For timestamp refinement, Gemini is called via `GeminiGenaiManager.LLM_request()` with:
- Mixed inputs: local video file path + text prompt
- Pydantic schema `RefinedTimestamps` for structured JSON output
- Automatic retries with exponential backoff

### Structured Output (music selection)

For track selection, a simple text prompt is sent to Gemini asking it to return the index number of the best matching track.

---

## ElevenLabs TTS Integration

Used by the `generate_narration` tool:

- Uses the `elevenlabs` Python SDK
- Voice ID: `JBFqnCBsd6RMkjVDRZzb` (hardcoded)
- Model: `eleven_flash_v2_5`
- Output: `{workspace}/narration/narration.mp3`
- Script saved to `{workspace}/narration/narration_script.txt`
- Duration measured via `ffprobe`
- Requires `ELEVENLABS_API_KEY` environment variable

---

## Music Generation (Google Lyria 3)

Used by the `select_music` tool:

1. Calls Google Lyria 3 Pro (`google/lyria-3-pro-preview`) via the OpenRouter API
2. Sends the agent's text prompt with duration hint, requests `["audio", "text"]` modalities
3. Parses the audio bytes from the response and writes to `{workspace}/music/track.mp3`
4. Uses `OPENROUTER_API_KEY` — the same key used for the LLM agent loop. Cost: ~$0.08/song
5. Duration: up to ~3 minutes. The agent passes `duration_seconds` to match the timeline

---

## DaVinci Resolve Integration

### Status

DaVinci Resolve support is implemented but optional. It provides rendering of FCPXML timelines to video files (preview or final quality).

### Setup Requirements

- DaVinci Resolve Studio (not the free version -- external scripting requires Studio)
- Running in headless mode (`resolve -nogui`)
- Environment variables: `RESOLVE_SCRIPT_API`, `RESOLVE_SCRIPT_LIB`
- On Linux: virtual display via Xvfb required

### Health Check

`lib/utils/resolve_setup.py` provides:
- `check_resolve_status()` -- returns `{running, version, studio, error}`
- `print_setup_instructions()` -- platform-specific setup guide
- `ensure_pythonpath()` -- adds Resolve modules to sys.path

### Auto-Render

After FCPXML generation, the agent session kicks off two parallel renders:

- **FFmpeg** (always): `src/pipelines/v2/preview/ffmpeg_renderer.py` runs an FFmpeg-based render to `{workspace}/renders/ffmpeg.mp4`. Two quality modes — `"draft"` (480p shorter dimension, ultrafast preset, default) and `"full"` (timeline-native, slow preset). The mode is per-session and can be toggled from the dashboard render tab or via `set_ffmpeg_quality`.
- **Resolve** (optional): If DaVinci Resolve is available, `lib/utils/resolve_render.py` produces `{workspace}/renders/resolve.mp4` at the timeline's native resolution. Otherwise the resolve tab shows an idle message.

Both emit `render_start`, `render_progress`, and `render_complete` (or `render_error`) events. The dashboard shows both tabs side-by-side; UI edits (drag/gain/undo) call `schedule_ffmpeg_rerender()`, a debounced (~1.5s) auto-rerender so the preview always reflects the current edit.

### Logging

Every agent session writes a structured debug bundle to `{workspace}/logs/`. The goal is that a single zip of that folder gives someone else (you, a teammate, or the person who filed the bug report) everything they need to diagnose a broken run.

**How it's wired** — `src/pipelines/v2/logging_setup.py` defines a `WorkspaceLogScope` context manager backed by Python's `contextvars`. `AgentSession.handle_user_message`, `_render_ffmpeg`, and `_render_resolve` each enter the scope around their work, which activates a single root-logger JSONL handler for the duration. Any `logger.info(...)` call inside the scope (including inside asyncio tasks spawned from within it — contextvars are preserved across `asyncio.create_task`) routes to `logs/backend.jsonl` of the right project. Outside a scope, records are dropped rather than leaking to an unrelated project's bundle.

**What's captured**:

| File | Written by | Cleared on `clear/planning` |
|------|-----------|------------------------------|
| `manifest.json` | `logging_setup.write_manifest()` at `AgentSession.__init__` | yes (rewritten next session) |
| `backend.jsonl` | `_WorkspaceHandler` attached to root logger | yes |
| `llm.jsonl` | `append_llm_event()` at every main_llm call site in `agent_session.py` and video_llm call sites in `tools.py` | yes |
| `renders/ffmpeg-<ts>.log` | `_run_ffmpeg` via `_render_log_path` contextvar scoped by `render_ffmpeg_preview` | yes |
| `refine_clips/*` | `refine_clip_timestamps` tool (pre-existing, segment MP4s + `refine_debug.jsonl`) | yes |

**Redaction** — `redact()` in `logging_setup.py` masks dict keys matching `api_key`, `secret`, `authorization`, `password`, `access_token`, `auth_token`, `refresh_token`, or a standalone `token`, plus any `Bearer <...>` substring in free-form strings. The legitimate `tokens` field (LLM usage counts) is preserved.

### API Endpoints

- `GET /video-edit/v2/resolve/status` -- health check
- `POST /video-edit/v2/render` -- trigger render (preview or final quality)
- `GET /video-edit/v2/projects/{name}/renders/{filename}` -- serve rendered video

---

## Configuration

### config.json

```json
{
  "paths": {
    "videos_dir": "data/videos",
    "indexing_dir": "data/indexing",
    "outputs_dir": "data/outputs",
    "workspaces_dir": "data/workspaces"
  },
  "api_keys": {
    "MEMORIES_API_KEY": "... (optional; legacy V1 only)",
    "GOOGLE_CLOUD_PROJECT": "...",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "ELEVENLABS_API_KEY": "...",
  },
  "optional_features": {
    "enable_music": true,
    "enable_dynamic_cropping": true,
    "enable_subtitles": true,
    "enable_fcpxml_export": true
  },
  "video_processing": {
    "default_fps": 24,
    "summary_fps": 1,
    "summary_crf": 40
  }
}
```

All keys in `api_keys` are loaded into environment variables at startup by `src/config.py`.

### API Prefixes

- V1: `/video-edit/v1`
- V2: `/video-edit/v2`
- Dashboard: `/app` (serves `dashboard/dist/`)

### Google Cloud Authentication

Gemini access requires application-default credentials:

```bash
gcloud auth application-default login
```

---

## V2 API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v2/projects` | List all projects with summaries |
| POST | `/v2/index` | Lightweight indexing (upload + gist) |
| POST | `/v2/plan` | Start iterative planning loop (v1-style) |
| POST | `/v2/generate_fcpxml` | Generate FCPXML from storyboard |
| POST | `/v2/narration` | Generate narration audio |
| POST | `/v2/music` | Select and download music |
| POST | `/v2/crop` | Apply dynamic cropping to FCPXML |
| POST | `/v2/render` | Render via DaVinci Resolve |
| GET | `/v2/resolve/status` | Check Resolve availability |
| WS | `/v2/agent/{project}/chat` | Agentic editing chat session |
| WS | `/v2/session/{project}` | Planning loop live events |
| POST | `/v2/projects/{project}/clear/gists` | Clear gist data |
| POST | `/v2/projects/{project}/clear/planning` | Clear planning + chat |
| POST | `/v2/projects/{project}/clear/session` | Full local reset |
| POST | `/v2/projects/{project}/clear/memories` | Clear local lvmm-core index rows/vectors |

---

## Troubleshooting / Known Issues

### WebSocket disconnects during long agent operations
The agent loop can take minutes (especially with multiple `refine_clip_timestamps` calls). The WebSocket connection uses a 0.05s polling loop, so it should stay alive. If the client disconnects, the running agent task is cancelled. On reconnect, persisted state (scratchpads, chat_history, event_log, edit_decision) is sent in the `init` event.

### lvmm-core indexing is slow
Frame embedding and visual transcription are CPU/GPU-bound and can take minutes for long footage. Watch `[COMPREHENSION]` and `lvmm_core.*` logs for stage-level progress.

### FCPXML source file resolution
The compiler resolves `source_file` to `source_path` by matching filenames in the workspace `footage/` directory. If no match is found, the FCPXML will contain a `file:///media/{filename}` URI that requires manual media relinking in Final Cut Pro.

### Scratchpad overflow
Each pad is capped at 6000 characters. When exceeded, content is truncated from the start. The system prompt instructs the agent to use `replace` to consolidate rather than endlessly appending.

### Transitions disabled
Cross-dissolve transitions are commented out in the FCPXML compiler due to audio overlap issues. All edits use hard cuts.

### Agent duplication
Only one `AgentSession` exists per project. If two browser tabs connect to the same project, the second connection takes over the `emit` callback. Events from the agent loop will only reach the most recent connection.
