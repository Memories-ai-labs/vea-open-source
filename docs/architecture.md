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
+------------------+    REST     +-------------+-------------+
|  Memories.ai     |<---------->|   FastAPI Backend (8000)   |
|  (video index,   |            |                             |
|   chat, search)  |            |   AgentSession              |
+------------------+            |     +-- ScratchpadManager   |
                                |     +-- ToolExecutor        |
+------------------+            |     +-- WorkspaceManager    |
|  Gemini          |<---------->|                             |
|  (Vertex AI)     |            +----+---+---+---+---+-------+
|  (structured out,|                 |   |   |   |   |
|   function call) |                 |   |   |   |   +-- DaVinci Resolve
+------------------+                 |   |   |   +------ Soundstripe
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

The agent has 7 tools, declared as Gemini `FunctionDeclaration` objects in `tools.py`:

| Tool | Purpose | Calls |
|------|---------|-------|
| `ask_memories` | Ask natural-language questions about footage | Memories.ai Chat API |
| `search_footage` | Find clips by query, returns timestamps + scores | Memories.ai Search API (BY_CLIP) |
| `refine_clip_timestamps` | Precise in/out point selection using video + audio analysis | ffmpeg extract + downsample, then Gemini structured output |
| `update_scratchpad` | Write to one of 4 persistent scratchpads | Local filesystem |
| `generate_fcpxml` | Compile EditDecision JSON to FCPXML 1.10 | Deterministic compiler |
| `generate_narration` | Convert script to voiceover audio | ElevenLabs TTS API |
| `select_music` | Search music library, LLM picks best match, download | Soundstripe API + Gemini selection |
| `message_user` | Send visible message to user in chat | WebSocket event |

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
  +-- clips: List[ClipDecision]        # ordered, sequential on spine
  |     id, source_file, source_path
  |     source_start, source_end        # seconds
  |     label, description
  |     gain_db                         # audio level adjustment
  |     speed: SpeedChange              # rate (0.5 = slow-mo, 2.0 = fast)
  |     transform: TransformSettings    # crop/reframe
  |     transition_after: TransitionSpec # cross-dissolve, fade-in, fade-out
  |
  +-- narration: List[NarrationSegment]
  |     file, timeline_offset, start, duration, gain_db
  |
  +-- music: MusicTrack (optional)
  |     file, start, duration, gain_db (-12 default)
  |
  +-- titles: List[TextOverlay]
        text, timeline_offset, duration, font_size, lane
```

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
+-- session.json                # SessionData: video entries, status, planning state
+-- context.md                  # Accumulated context (gist + planning notes)
+-- storyboard.json             # Storyboard model (v1 planning loop output)
+-- clips.json                  # RetrievedClip list from planning loop
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
+-- iterations/                 # Planning loop snapshots
|   +-- iter_0_tool_plan.json
|   +-- iter_0_storyboard.json
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
+-- renders/                    # DaVinci Resolve output
|   +-- preview.mp4
|
+-- logs/                       # Run logs
    +-- run.log
```

### Session States

The `session.json` `status` field tracks the project lifecycle:

| Status | Meaning |
|--------|---------|
| `new` | Directory exists but no indexing has been done |
| `indexed` | Videos uploaded to Memories.ai, gist generated |
| `planning` | Iterative planning loop is running (v1 flow) |
| `fcpxml_ready` | FCPXML has been generated |
| `rendered` | DaVinci Resolve has produced a render |

---

## Dashboard Architecture

### Technology Stack

- React 18 + TypeScript
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

Features a "Manage" dropdown for: re-indexing, clearing gists, clearing planning/chat, deleting from Memories.ai.

**NLETimeline** (`NLETimeline.tsx`): A custom-built non-linear editing timeline visualization:
- Builds tracks from EditDecision: V1 (video spine), T1 (titles), A1 (narration), A2 (music)
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

## Memories.ai Integration

The `MemoriesAiManager` (`lib/llm/MemoriesAiManager.py`) wraps the Memories.ai API. V2 uses these features:

### Upload and Indexing

- `upload_video_file(file_path)` -- multipart upload with streaming
- `wait_for_ready(video_no)` -- polls until status is `PARSE` (30s interval, rate limit backoff)
- `find_video_by_name(name)` -- fuzzy match to reuse existing uploads

### Chat API (used by `ask_memories` tool)

- `chat(video_nos, prompt)` -- natural language Q&A against indexed videos
- Returns `ChatResponse` with text answer + `MemoriesReference` objects (timestamped citations)
- References include `video_no`, `video_name`, `timestamp` (HH:MM:SS), `description`

### Search API (used by `search_footage` tool)

- `search(query, search_type="BY_CLIP", video_nos=...)` -- clip-level semantic search
- Returns items with `videoNo`, `startTime`, `endTime`, `score`
- Results are capped at 15 clips by ToolExecutor

### Lightweight Comprehension (V2 indexing)

`LightweightComprehension` (`src/pipelines/v2/comprehension/lightweight_comprehension.py`):
1. Finds video files in the workspace `footage/` directory
2. Uploads each to Memories.ai (or reuses existing by name match)
3. Waits for all videos to reach `PARSE` status
4. Gets a per-video gist via the Chat API
5. Saves `SessionData` with video entries and combined gist

This is much faster than V1 indexing (no scene-by-scene analysis). Detailed understanding happens on-demand during the agent conversation.

---

## Gemini Integration

`GeminiGenaiManager` (`lib/llm/GeminiGenaiManager.py`) wraps Google's Vertex AI Gemini API.

### Agent Loop Usage

The agent session calls Gemini directly via `genai_client.models.generate_content()` with:
- `contents`: conversation history (user/model/tool Content objects)
- `system_instruction`: rebuilt each turn with current scratchpad state
- `tools`: `TOOL_DECLARATIONS` (7 function declarations)
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

## Soundstripe Music Integration

Used by the `select_music` tool:

1. Fetches tracks from Soundstripe API (3 pages of 50 tracks, with `audio_files` sideloaded)
2. Formats track metadata (title, mood, genre, energy, BPM, description)
3. Sends track list to Gemini, asks it to pick the best index number
4. Downloads the selected track's MP3 via the resolved `_mp3_url`
5. Saves to `{workspace}/music/track.mp3`
6. Requires `SOUNDSTRIPE_KEY` environment variable

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

After FCPXML generation, the agent session kicks off a background preview render:
- `_auto_render_preview()` creates an asyncio task
- Emits `render_start`, `render_progress`, `render_complete` events
- Output written to `{workspace}/renders/preview.mp4`
- Non-fatal if Resolve is unavailable

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
    "MEMORIES_API_KEY": "...",
    "GOOGLE_CLOUD_PROJECT": "...",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "ELEVENLABS_API_KEY": "...",
    "SOUNDSTRIPE_KEY": "..."
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
| POST | `/v2/projects/{project}/clear/memories` | Delete from Memories.ai |

---

## Troubleshooting / Known Issues

### WebSocket disconnects during long agent operations
The agent loop can take minutes (especially with multiple `refine_clip_timestamps` calls). The WebSocket connection uses a 0.05s polling loop, so it should stay alive. If the client disconnects, the running agent task is cancelled. On reconnect, persisted state (scratchpads, chat_history, event_log, edit_decision) is sent in the `init` event.

### Memories.ai rate limiting
The upload/status polling uses 30-second intervals with exponential backoff on rate limit errors. If you see repeated "Rate limited" messages, reduce concurrent uploads or wait.

### FCPXML source file resolution
The compiler resolves `source_file` to `source_path` by matching filenames in the workspace `footage/` directory. If no match is found, the FCPXML will contain a `file:///media/{filename}` URI that requires manual media relinking in Final Cut Pro.

### Scratchpad overflow
Each pad is capped at 6000 characters. When exceeded, content is truncated from the start. The system prompt instructs the agent to use `replace` to consolidate rather than endlessly appending.

### Transitions disabled
Cross-dissolve transitions are commented out in the FCPXML compiler due to audio overlap issues. All edits use hard cuts.

### Agent duplication
Only one `AgentSession` exists per project. If two browser tabs connect to the same project, the second connection takes over the `emit` callback. Events from the agent loop will only reach the most recent connection.
