<div align="center">

  <h1>VEA: Video Editing Agent</h1>

  <h3>
    Autonomous Comprehension of
    Long-Form, Story-Driven Media
  </h3>

  <p>
    <a href="https://arxiv.org/abs/2509.16811"><strong>📄 Paper</strong></a> ·
    <a href="https://github.com/Memories-ai-labs/vea-open-source"><strong>💻 Code</strong></a>
  </p>

  <p>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    </a>
    <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Node-18+-green.svg" alt="Node">
    <img src="https://img.shields.io/badge/FFmpeg-Required-orange.svg" alt="FFmpeg">
    <a href="https://github.com/astral-sh/uv">
      <img src="https://img.shields.io/badge/Package_Manager-uv-purple.svg" alt="uv">
    </a>
  </p>

  <br>

  <p align="center">
    <strong>VEA</strong> is an AI-powered video editing service that turns raw footage into polished short-form content through a natural-language conversation with an editing agent.
  </p>
</div>

---

## What VEA does

You drop video files into a project folder, then chat with an agent that:

* 🧠 **Understands** your footage via [Memories.ai](https://memories.ai) (uploads, indexes, answers questions about visuals and dialogue).
* 🎬 **Plans and selects clips** based on your creative brief, refining cut points using LLM video analysis.
* 🗣️ **Narrates** with [ElevenLabs](https://elevenlabs.io) text-to-speech (optional, on request).
* 🎵 **Adds music** from [Soundstripe](https://www.soundstripe.com) with automatic loudness balancing (optional).
* 📦 **Exports** as both rendered MP4 (via FFmpeg or DaVinci Resolve) and Final Cut Pro XML (importable into FCP, Resolve, Premiere).
* 💬 **Iterates** on your feedback in real time — "make the intro shorter", "add more b-roll", "use a different song".

The whole workflow happens in a React dashboard with a live NLE-style timeline.

---

## 🚀 Quick start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Dashboard build |
| ffmpeg | recent | Video processing (extract, downsample, probe, render) |
| uv | latest | Python package management |

Optional:

* **Google Cloud SDK** — only if you use Vertex AI Gemini directly (instead of OpenRouter)
* **DaVinci Resolve Studio** — for high-quality final renders (the system also supports an FFmpeg-based draft render that needs no Resolve)

Install on macOS:

```bash
brew install python@3.12 node ffmpeg
brew install astral-sh/tap/uv
```

### 1. Clone and install

```bash
git clone https://github.com/Memories-ai-labs/vea-open-source.git
cd vea-open-source

# Python deps
uv sync

# Dashboard deps + production build
cd dashboard && npm install && npm run build && cd ..
```

### 2. Configure API keys

```bash
cp config.example.json config.json
```

Edit `config.json` and fill in the `api_keys` section:

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `MEMORIES_API_KEY` | **Yes** | https://memories.ai/app/service/key |
| `OPENROUTER_API_KEY` | **One of these two** | https://openrouter.ai |
| `GOOGLE_CLOUD_PROJECT` | **One of these two** | A GCP project with Vertex AI enabled |
| `ELEVENLABS_API_KEY` | Optional | https://elevenlabs.io — needed for narration |
| `SOUNDSTRIPE_KEY` | Optional | https://www.soundstripe.com — needed for music |

VEA needs an LLM provider for the agent loop. You can use **either**:

* **OpenRouter** (simplest): set `OPENROUTER_API_KEY` in your environment and you're done. Optionally set `OPENROUTER_MODEL` (default `google/gemini-2.5-flash`).
* **Vertex AI Gemini**: set `GOOGLE_CLOUD_PROJECT` and run `gcloud auth application-default login`.

If both are set, OpenRouter wins. Force Vertex by setting `LLM_PROVIDER=vertex`.

### 3. Start the backend

```bash
./dev.sh up
```

`dev.sh` runs setup if needed (creates `.venv`, installs deps, builds the dashboard) and then starts the FastAPI server on port 8000. Open the dashboard at:

> **http://localhost:8000/app**

For frontend hot-reload while iterating on UI:

```bash
./dev.sh up --frontend-dev   # also starts Vite dev server on 5173
```

Or run everything by hand:

```bash
source .venv/bin/activate
python -m src.app
# In another terminal, optional:
cd dashboard && npm run dev
```

---

## 🎬 Using the app

### 1. Create a project

```bash
mkdir -p data/workspaces/my-project/footage
cp ~/Videos/*.mp4 data/workspaces/my-project/footage/
```

Supported formats: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.mpg`, `.mpeg`, `.m4v`, `.ts`.

### 2. Open the project in the dashboard

Navigate to **http://localhost:8000/app**, click your project, and you'll land in the editing workspace.

### 3. Index footage

If the footage hasn't been indexed yet, the dashboard shows an **"Index footage"** banner with a button. Click it. The indexer uploads each file to Memories.ai and generates a content gist. Progress streams live to the UI.

> Indexing takes 1–5 minutes per video depending on length and upload speed. Once finished, every footage pill shows a green check.

### 4. Chat with the agent

Type a brief in the chat box:

> *"Create a 90-second highlight reel of this keynote, focusing on the product demo and audience reactions. Add a short narration intro and upbeat background music."*

The agent will:

1. Query Memories.ai to understand the footage
2. Update its `comprehension` and `creative_direction` scratchpads
3. Propose an edit plan (you'll see it in the chat)
4. Search for clips, refine in/out points with frame-accurate analysis
5. Generate narration and pick a music track (only because you asked for them)
6. Compile the edit to a JSON `edit_decision` and FCPXML
7. Auto-render a 480p draft (via FFmpeg) so you can preview it immediately

You watch all of this happen in real time:

* **Chat panel** — the agent's messages and tool calls
* **Scratchpad tabs** — its persistent memory (comprehension / creative direction / planning / fcpxml)
* **NLE timeline** — multi-track view (V1 video, T1 titles, A1 narration, A2 music) with hover details
* **Preview** — the rendered draft (and final, when DaVinci Resolve is available)

### 5. Iterate

Just keep talking:

> *"The intro feels too long, trim it. Use a different second clip — something more emotional."*

The agent updates the plan, regenerates the edit, and re-renders.

### 6. Export

When you're happy:

* **MP4** — `data/workspaces/my-project/renders/draft.mp4` (FFmpeg) or `final.mp4` (DaVinci Resolve, if installed)
* **FCPXML** — `data/workspaces/my-project/fcpxml/edit_v1.fcpxml`, importable into Final Cut Pro, DaVinci Resolve, or Premiere Pro

---

## 📂 Workspace layout

Each project is self-contained under `data/workspaces/{project_name}/`:

```
{project_name}/
├── footage/                 # Source video files (you put these here)
├── session.json             # Project state, video entries, indexing status
├── chat_history.json        # Persisted conversation
├── event_log.json           # Tool call / result history
├── scratchpads/             # Agent's persistent memory
│   ├── comprehension.md
│   ├── creative_direction.md
│   ├── planning.md
│   └── fcpxml.md
├── narration/               # ElevenLabs voiceover (when requested)
│   └── narration.mp3
├── music/                   # Soundstripe track (when requested)
│   └── track.mp3
├── fcpxml/
│   ├── edit_decision.json   # Structured edit (LLM output)
│   └── edit_v1.fcpxml       # Compiled FCPXML 1.10
├── renders/
│   ├── draft.mp4            # Quick FFmpeg draft
│   └── final.mp4            # DaVinci Resolve final (optional)
└── logs/run.log
```

---

## 🤖 Agent tools

The agent has 10 tools, all declared in `src/pipelines/v2/agent/tool_definitions.py`:

| Tool | Purpose |
|------|---------|
| `ask_memories` | Natural-language Q&A against indexed footage (Memories.ai chat) |
| `search_footage` | Semantic clip search returning timestamps + dialogue transcripts |
| `refine_clip_timestamps` | Frame-accurate in/out point selection via Gemini video analysis |
| `update_scratchpad` | Write to one of the 4 persistent scratchpads |
| `generate_fcpxml` | Compile edit decision JSON → FCPXML, validate clips, auto-render draft |
| `generate_narration` | ElevenLabs TTS with word-level timestamps (only if user asks) |
| `select_music` | Soundstripe search + LLM track selection (only if user asks) |
| `generate_subtitles` | STT-based subtitles for the current edit |
| `message_user` | Send a message to the dashboard mid-flow |
| `finish_turn` | Explicitly end the agent's turn with an optional summary |

---

## 🧪 API endpoints

The full FastAPI app is at **http://localhost:8000/docs**.

V2 (current, agent-driven) is at `/video-edit/v2`. The most useful endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v2/projects` | List all projects |
| `POST` | `/v2/index` | Trigger indexing for a project |
| `WS` | `/v2/agent/{project}/chat` | Agent chat WebSocket (used by dashboard) |
| `GET` | `/v2/projects/{project}/renders/{filename}` | Stream rendered MP4 |
| `POST` | `/v2/projects/{project}/clear/planning` | Clear chat + scratchpads + edit |
| `POST` | `/v2/projects/{project}/clear/memories` | Delete uploaded videos from Memories.ai |

A legacy V1 pipeline-style API still lives at `/video-edit/v1` (index → flexible_respond). It's the original system from the paper and is kept for reproducibility, but the dashboard and agent flow only use V2.

---

## 📚 More documentation

* [docs/onboarding.md](docs/onboarding.md) — step-by-step developer setup, common workflows, debugging tips
* [docs/architecture.md](docs/architecture.md) — technical architecture: agent loop, tool system, scratchpads, FCPXML compiler, dashboard
* [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) — orientation for AI coding assistants working in this repo
* [context/fcpxml_formatting_guide.md](context/fcpxml_formatting_guide.md) — FCPXML 1.10 reference

---

## 🐳 Docker

```bash
docker build -t vea .
docker run -p 8000:8000 \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/data:/app/data \
  vea
```

Then open http://localhost:8000/app.

---

## 🖊️ Citation

```bibtex
@article{ding2025prompt,
  title={Prompt-Driven Agentic Video Editing System: Autonomous Comprehension of Long-Form, Story-Driven Media},
  author={Ding, Zihan and Wang, Xinyi and Chen, Junlong and Kristensson, Per Ola and Shen, Junxiao},
  journal={arXiv preprint arXiv:2509.16811},
  year={2025}
}
```

---

<p align="center">
Copyright © 2026 Memories.ai Platforms, Inc.<br>
Released under the <a href="LICENSE">MIT License</a>.
</p>
