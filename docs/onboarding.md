# VEA V2 Onboarding Guide

This guide walks through setting up and running VEA's V2 agentic editing system from scratch. For architectural details, see [architecture.md](architecture.md).

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Dashboard build |
| npm | (bundled with Node) | Dashboard dependencies |
| ffmpeg | any recent | Video processing (extract, downsample, probe, render) |
| uv | latest | Python package management |
| gcloud CLI | latest | **Optional** — only needed if you use Vertex AI instead of OpenRouter |

Install on macOS:

```bash
brew install python@3.12 node ffmpeg
brew install astral-sh/tap/uv
brew install --cask google-cloud-sdk   # optional
```

---

## Step-by-Step Setup

### 1. Clone and install

```bash
git clone https://github.com/Memories-ai-labs/vea-open-source.git
cd vea-open-source

# Python dependencies
uv sync

# Dashboard dependencies
cd dashboard && npm install && cd ..
```

### 2. Configure API keys

```bash
cp config.example.json config.json
```

Edit `config.json` and fill in the `api_keys` section:

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `MEMORIES_API_KEY` | Yes | https://memories.ai/app/service/key |
| `OPENROUTER_API_KEY` | One of these two | https://openrouter.ai |
| `GOOGLE_CLOUD_PROJECT` | One of these two | Your GCP project ID (Vertex AI enabled) |
| `GOOGLE_CLOUD_LOCATION` | No | Defaults to `us-central1` (Vertex only) |
| `ELEVENLABS_API_KEY` | No | https://elevenlabs.io -- needed for narration |

VEA needs one LLM provider for the agent loop. Pick **one**:

- **OpenRouter (simplest):** set `OPENROUTER_API_KEY`. Default model is `google/gemini-2.5-flash`; override with `OPENROUTER_MODEL`. No further auth needed.
- **Vertex AI Gemini:** set `GOOGLE_CLOUD_PROJECT` and run the gcloud step below. If both are configured, OpenRouter wins; force Vertex with `LLM_PROVIDER=vertex`.

### 3. Authenticate Google Cloud (Vertex AI users only)

```bash
gcloud auth application-default login
```

Skip this step if you're using OpenRouter.

### 4. Build the dashboard

```bash
cd dashboard && npm run build && cd ..
```

This creates `dashboard/dist/` which FastAPI serves at `/app`.

---

## Starting the System

### Using dev.sh (recommended)

The `dev.sh` script handles setup validation, dashboard builds, and starting the backend:

```bash
# First time: runs setup interactively (prompts for API keys)
./dev.sh setup

# Start the backend (builds dashboard if needed)
./dev.sh up

# Start with Vite dev server for frontend hot-reload
./dev.sh up --frontend-dev

# Start with ngrok (needed for v1 webhook-based indexing only)
./dev.sh up --with-ngrok

# Health check
./dev.sh doctor

# Stop background processes (ngrok, frontend dev server)
./dev.sh down
```

### Manual start

```bash
# Terminal 1: Backend
source .venv/bin/activate
python -m src.app
# Server starts on http://localhost:8000

# Terminal 2 (optional): Dashboard dev server with hot-reload
cd dashboard
npm run dev
# Dev server on http://localhost:5173
```

### Accessing the dashboard

- Production build: http://localhost:8000/app
- Dev server: http://localhost:5173 (if using `--frontend-dev` or `npm run dev`)
- API docs: http://localhost:8000/docs

---

## Creating a Test Project

### 1. Create a workspace directory

```bash
mkdir -p data/workspaces/my-test-project/footage
```

### 2. Add footage

Copy or symlink video files into the footage directory:

```bash
cp ~/Videos/keynote.mp4 data/workspaces/my-test-project/footage/
```

Supported formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.mpg`, `.mpeg`, `.m4v`, `.ts`

### 3. Open the dashboard

Navigate to http://localhost:8000/app (or http://localhost:5173 with dev server).

You should see your project in the project browser. Click on it to open the workspace.

### 4. Index footage

If the footage hasn't been indexed yet, the dashboard shows an **"Index footage"** banner with a button — click it. (You can also re-index at any time from the **Manage** dropdown.) Indexing uploads the video files to Memories.ai and generates a content gist; progress streams live to the banner.

Alternatively, use the API:

```bash
curl -X POST http://localhost:8000/video-edit/v2/index \
  -H "Content-Type: application/json" \
  -d '{"project_name": "my-test-project"}'
```

Indexing takes 1-5 minutes per video depending on length and upload speed. The footage strip in the dashboard will show a green checkmark next to each indexed file.

### 5. Start chatting

Once indexing completes, type a message in the chat input:

> "Create a 90-second highlight reel of this keynote, focusing on the product demo and audience reactions."

The agent will:
1. Query Memories.ai to understand the footage
2. Update the comprehension scratchpad
3. Propose an edit plan
4. Search for specific clips
5. Refine timestamps on each clip
6. Generate FCPXML

You can see tool calls and progress in the chat timeline. The NLE timeline in the middle row populates after FCPXML generation.

---

## Common Workflows

### Start a new editing session

1. Create workspace directory with footage
2. Open project in dashboard
3. Index footage via Manage menu
4. Send a creative brief in chat

### Iterate on an edit

After the agent generates an initial FCPXML:

> "The intro is too long. Cut it down to 10 seconds and add a faster pace."

The agent will update the planning scratchpad, re-search clips if needed, and regenerate the FCPXML.

### Add narration

> "Add a narration voiceover. Use a professional, energetic tone. Here's the script: ..."

The agent will call `generate_narration`, then include the narration segments in the next FCPXML generation.

### Add background music

> "Add some upbeat electronic background music, medium energy."

The agent will generate an original track via ElevenLabs Eleven Music and include it in the FCPXML.

### Export and use the FCPXML

The generated FCPXML is at `data/workspaces/{project}/fcpxml/edit_v1.fcpxml`. Import it into Final Cut Pro, DaVinci Resolve, or any FCPXML-compatible editor.

You may need to relink media -- the FCPXML references source files by their absolute path in the workspace `footage/` directory.

### Reset a project

Use the **Manage** dropdown in the dashboard:

- **Clear gists**: Removes gist text but keeps video_no mappings
- **Clear planning + chat**: Removes scratchpads, chat history, storyboard, clips, FCPXML -- keeps indexing
- **Delete from Memories.ai**: Removes the uploaded videos from Memories.ai cloud (requires re-upload to re-index)

---

## Debugging Tips

### Check backend logs

The backend prints structured log lines prefixed with tags:

- `[AGENT]` -- Agent session lifecycle
- `[AGENT WS]` -- WebSocket connection events
- `[MEMORIES]` -- Memories.ai API calls
- `[COMPREHENSION]` -- Indexing pipeline
- `[COMPILER]` -- FCPXML compilation
- `[MUSIC]` -- ElevenLabs music generation

### WebSocket connection issues

If the dashboard shows "Offline":

1. Verify the backend is running on port 8000
2. Check the browser console for WebSocket errors
3. The hook reconnects with exponential backoff (500ms to 5s) -- wait a few seconds
4. If using the Vite dev server, make sure the proxy is configured (check `dashboard/vite.config.ts`)

### Gemini errors

- "Gemini not configured" -- run `gcloud auth application-default login` and set `GOOGLE_CLOUD_PROJECT` in `config.json`
- Rate limit errors -- reduce concurrent requests, the agent loop already has built-in retry
- Empty responses -- check safety settings (all set to `BLOCK_NONE` but may still occasionally block)

### Memories.ai issues

- "Memories.ai not configured" -- set `MEMORIES_API_KEY` in `config.json`
- Upload timeouts -- large files get longer timeouts automatically (5min + 1min per 100MB)
- "Rate limited" during status polling -- backoff is automatic, wait
- Videos stuck in `UNPARSE` -- check your Memories.ai dashboard for quota issues

### FCPXML not generating

- "Invalid EditDecision" -- the agent sent malformed JSON; check the chat for error messages
- "Source file not found" -- footage filenames must match what the agent references; check `data/workspaces/{project}/footage/`
- Missing clips in Final Cut Pro -- relink media to the footage directory

### Scratchpads not updating

- The agent is instructed to update scratchpads immediately, but may occasionally forget
- Check the scratchpad tabs in the dashboard -- content updates appear in real-time
- If the comprehension pad is thin, ask the agent to learn more about the footage

---

## Key Files by Area

### Backend core
- `src/app.py` -- FastAPI server, all endpoints and WebSocket handlers
- `src/config.py` -- Configuration loading, path constants

### Agent system
- `src/pipelines/v2/agent/agent_session.py` -- Agent loop, history, persistence
- `src/pipelines/v2/agent/tools.py` -- Tool declarations and executor
- `src/pipelines/v2/agent/system_prompt.py` -- System prompt template
- `src/pipelines/v2/agent/scratchpad.py` -- Scratchpad manager

### Data models
- `src/pipelines/v2/schemas.py` -- SessionData, EditDecision, ClipDecision, etc.
- `src/schema.py` -- API request/response models (Pydantic)

### Video understanding
- `lib/llm/MemoriesAiManager.py` -- Memories.ai client (upload, chat, search)
- `src/pipelines/v2/comprehension/lightweight_comprehension.py` -- V2 indexing

### FCPXML
- `src/pipelines/v2/fcpxml/edit_compiler.py` -- Deterministic FCPXML 1.10 compiler
- `src/pipelines/common/fcpxml_exporter.py` -- Shared FCPXML utilities (fractions, formatting)

### Workspace
- `src/pipelines/v2/workspace.py` -- File I/O for workspaces

### LLM clients
- `lib/llm/GeminiGenaiManager.py` -- Gemini / Vertex AI client
- `lib/llm/MemoriesAiManager.py` -- Memories.ai client

### Dashboard
- `dashboard/src/App.tsx` -- Root component, routing
- `dashboard/src/hooks/useAgentChat.ts` -- WebSocket hook, all real-time state
- `dashboard/src/components/AgentChat.tsx` -- Main workspace UI
- `dashboard/src/components/NLETimeline.tsx` -- Timeline visualization
- `dashboard/src/components/ProjectBrowser.tsx` -- Project list
- `dashboard/src/api.ts` -- REST API client functions

### Rendering
- `lib/utils/resolve_setup.py` -- DaVinci Resolve health check
- `lib/utils/resolve_render.py` -- Resolve rendering (if available)

---

## Gotchas and Things to Watch Out For

1. **One agent session per project.** If you open the same project in two tabs, only the most recent tab receives events. The first tab will appear frozen.

2. **Indexing is idempotent.** Re-indexing reuses existing Memories.ai uploads if the video name matches. Use "Delete from Memories.ai" in Manage to force re-upload.

3. **Scratchpad size limit.** Each scratchpad is capped at 6000 characters. Long editing sessions can hit this. The agent should consolidate with `replace` rather than accumulating with `append`.

4. **Agent turns end on plain text.** If the agent responds with text but no tool calls, that is its final response for the turn. It will not execute tools after a text-only response. This is by design in the Gemini function calling protocol.

5. **FCPXML media paths are absolute.** The generated FCPXML uses absolute file paths from your local filesystem. If you move the project or share the FCPXML, you will need to relink media in your NLE.

6. **ffmpeg is required at runtime.** The `refine_clip_timestamps` tool calls ffmpeg to extract and downsample video segments. Without ffmpeg, this tool will fail.

7. **Narration and music are opt-in.** The system prompt explicitly forbids the agent from calling `generate_narration` or `select_music` unless the user requests it. If you want narration or music, ask for it.

8. **DaVinci Resolve is optional.** The auto-render after FCPXML generation is a non-blocking background task. If Resolve is not available, the render silently fails and the FCPXML is still usable in any compatible editor.

9. **The dev.sh script runs setup automatically.** If `.venv`, `config.json`, or `dashboard/dist` are missing, `./dev.sh up` will trigger `./dev.sh setup` first.

10. **Dashboard production build vs dev server.** The production build (`npm run build`) is served by FastAPI at `/app`. The dev server (`npm run dev`) runs on port 5173 with hot-reload but requires the backend to be running separately on port 8000 for API access.
