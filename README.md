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
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/FFmpeg-Required-orange.svg" alt="FFmpeg">
    <a href="https://github.com/astral-sh/uv">
      <img src="https://img.shields.io/badge/Package_Manager-uv-purple.svg" alt="uv">
    </a>
  </p>

  <br>

  <p align="center">
    <strong>VEA</strong> is an AI-powered video editing service that transforms movies, documentaries, and long-form videos into engaging short-form content.
    <br>
    It automates the entire pipeline: from video understanding to final rendering.
  </p>
</div>

---

## ✨ Features

VEA integrates state-of-the-art models to automate the editing workflow:

* 🧠 **Understand**: Deep video content analysis using **Memories.ai**.
* 📝 **Generate**: AI-scripted narration tailored to your specific prompt.
* 🎬 **Select**: Intelligent clip selection relevant to the narrative.
* 🗣️ **Narrate**: High-quality text-to-speech narration via **ElevenLabs**.
* 📱 **Crop**: Smart dynamic cropping (9:16) for vertical mobile viewing.
* 🎵 **Music**: Background music automatically synced to the beat.
* 📦 **Export**: Delivers both rendered `.mp4` video and **Final Cut Pro XML**.

---

## 🚀 Quick Start

### Prerequisites

* **Python 3.12+**
* **FFmpeg** (must be installed on system)
* **ngrok** (required for video indexing — the Caption API uses async webhooks)
* **[uv](https://github.com/astral-sh/uv)** package manager

### 1. Install System Dependencies

**FFmpeg:**

| OS | Command |
| --- | --- |
| **Ubuntu/Debian** | `sudo apt install ffmpeg` |
| **macOS** | `brew install ffmpeg` |
| **Windows** | Download from [ffmpeg.org](https://ffmpeg.org/download.html) |

**ngrok** (needed for indexing):

| OS | Command |
| --- | --- |
| **Ubuntu/Debian** | `sudo snap install ngrok` |
| **macOS** | `brew install ngrok` |
| **Windows** | Download from [ngrok.com/download](https://ngrok.com/download) |

Sign up at [ngrok.com](https://ngrok.com) and authenticate:
```bash
ngrok config add-authtoken <your-token>
```

### 2. Install Python Dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 3. Set Up ViNet (AI Cropping Model)

```bash
python -m lib.utils.vinet_setup
```

This downloads the pretrained saliency model weights used for dynamic 16:9 → 9:16 cropping.

### 4. Configuration

Copy the example config and populate your keys:

```bash
cp config.example.json config.json
```

**Required Keys in `config.json`:**

* `MEMORIES_API_KEY`: [Get Key](https://memories.ai/app/service/key)
* `GOOGLE_CLOUD_PROJECT`: [Google Cloud Console](https://console.cloud.google.com)
* `ELEVENLABS_API_KEY`: [ElevenLabs](https://elevenlabs.io/docs/api-reference/)
* `SOUNDSTRIPE_KEY`: [Soundstripe](https://www.soundstripe.com/) *(optional, for background music)*

### 5. Run Server

The recommended way to start the server — handles ngrok automatically:

```bash
gcloud auth application-default login  # Authenticate GCP (once)
./run.sh
```

Or manually, if you already have ngrok running:

```bash
# In one terminal: start ngrok
ngrok http 8000

# In another terminal: start the server with the callback URL
MEMORIES_CAPTION_CALLBACK_URL=https://<your-ngrok-url>/webhooks/memories/caption \
  source .venv/bin/activate && python -m src.app
```

> The API server starts at `http://localhost:8000`. You can also add `MEMORIES_CAPTION_CALLBACK_URL` directly to `config.json` under `api_keys` to avoid setting it as an env var each time.

---

## 🛠️ API Usage

### Footage Folder Structure

All source footage for a project must be placed under a single folder:

```
data/videos/
└── MyProject/
    ├── video1.mp4
    ├── video2.mp4
    └── clip.mp4        ← one or many files, all treated as one project
```

Pass the folder path as `blob_path` in all API calls.

### Step 1: Index Video

*Required once per project before generating content.*

```bash
curl -X POST http://localhost:8000/video-edit/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "blob_path": "data/videos/MyProject/"
  }'
```

> Indexing uploads footage to Memories.ai, generates scene descriptions, summaries, and people metadata. This can take several minutes for long videos. The resulting index is saved to `data/indexing/MyProject/media_indexing.json`.
>
> Use `"start_fresh": true` to force re-indexing from scratch.

### Step 2: Generate Response

*Create a recap, highlight reel, or story.*

```bash
curl -X POST http://localhost:8000/video-edit/v1/flexible_respond \
  -H "Content-Type: application/json" \
  -d '{
    "blob_path": "data/videos/MyProject/",
    "prompt": "Create a 2-minute recap of this movie",
    "video_response": true,
    "music": true,
    "narration": true,
    "aspect_ratio": 0.5625,
    "subtitles": true
  }'
```

### 📂 Output Structure

```
data/
├── videos/
│   └── MyProject/          ← place source footage here
│       └── video.mp4
├── indexing/
│   └── MyProject/
│       └── media_indexing.json   ← generated by /index
└── outputs/
    └── MyProject/
        └── {run_id}/
            └── video_response.mp4    ← final rendered video
```

---

## 📂 Project Structure

```text
vea-playground/
├── src/
│   ├── app.py                  # FastAPI server
│   ├── pipelines/              # Core logic (Indexing, Response, Shorts, etc.)
│   └── pipelines/common/       # Timeline, Dynamic Cropping (ViNet), FCPXML
├── lib/
│   ├── llm/                    # Memories.ai + Gemini clients
│   ├── oss/                    # Local storage abstraction
│   └── utils/                  # FFmpeg helpers, ViNet setup
├── data/
│   ├── videos/                 # Place source footage here (organized by project folder)
│   ├── indexing/               # Auto-generated scene/story indexes
│   └── outputs/                # Final rendered videos
├── vinet_v2/                   # ViNet saliency model weights (downloaded by vinet_setup.py)
├── config.json                 # Your API keys and settings (copy from config.example.json)
└── run.sh                      # Recommended start script (handles ngrok automatically)
```

---

## V2: Agentic Architecture

V2 replaces the fixed pipeline with an interactive agent that collaborates with you in real time. You describe what you want in natural language; the agent explores your footage, proposes edits, refines timestamps, and generates production-ready FCPXML -- all through a chat interface with a live NLE timeline.

### How it works

The system pairs a Gemini-based agent with a set of editing tools:

- **ask_memories** -- query indexed footage via Memories.ai
- **search_footage** -- find clips by semantic search
- **refine_clip_timestamps** -- precise in/out point selection using Gemini video analysis
- **generate_fcpxml** -- compile edit decisions to Final Cut Pro XML
- **generate_narration** -- text-to-speech voiceover via ElevenLabs
- **select_music** -- search and download background music from Soundstripe
- **update_scratchpad** -- persistent memory that survives the conversation window

Four scratchpads (comprehension, creative direction, planning, fcpxml) give the agent durable memory across long editing sessions.

### Quick start (V2)

```bash
# First-time setup (installs deps, prompts for API keys, builds dashboard)
./dev.sh setup

# Start backend + dashboard
./dev.sh up
# Open http://localhost:8000/app
```

Or manually:

```bash
uv sync
cd dashboard && npm install && npm run build && cd ..
gcloud auth application-default login
python -m src.app
# Open http://localhost:8000/app
```

### V2 workflow

1. **Create a workspace.** Make a directory under `data/workspaces/{project_name}/footage/` and drop in your video files.
2. **Index footage.** Open the project in the dashboard and click Manage > Index footage. This uploads to Memories.ai and generates a content gist.
3. **Chat with the agent.** Type a creative brief: "Create a 90-second highlight reel focusing on the product demo." The agent explores footage, proposes a plan, searches for clips, refines timestamps, and generates FCPXML.
4. **Iterate.** Give feedback ("the intro is too long", "add narration", "use more upbeat music") and the agent adjusts.
5. **Export.** The FCPXML at `data/workspaces/{project}/fcpxml/edit_v1.fcpxml` can be imported into Final Cut Pro, DaVinci Resolve, or any FCPXML-compatible editor.

### V2 workspace structure

```
data/workspaces/{project_name}/
+-- footage/              # Source video files
+-- session.json          # Index state, video entries
+-- scratchpads/          # Agent's persistent memory (4 markdown files)
+-- chat_history.json     # Persisted conversation
+-- fcpxml/
|   +-- edit_decision.json  # Structured edit decisions (JSON)
|   +-- edit_v1.fcpxml      # Compiled FCPXML 1.10
+-- narration/            # Generated voiceover audio
+-- music/                # Downloaded music track
+-- renders/              # DaVinci Resolve output (optional)
```

### Dashboard features

- **Project browser** -- list, create, and manage workspaces
- **Chat panel** -- conversational interface with tool call visibility
- **Scratchpad tabs** -- live view of agent's comprehension, creative direction, planning, and FCPXML state
- **NLE timeline** -- multi-track visualization (video, titles, narration, music) with zoom, pan, and hover tooltips
- **Video preview** -- rendered preview when DaVinci Resolve is available
- **Manage menu** -- re-index, clear planning, delete from Memories.ai

### Further reading

- [docs/architecture.md](docs/architecture.md) -- detailed technical architecture
- [docs/onboarding.md](docs/onboarding.md) -- step-by-step developer onboarding guide

---

## V1: Pipeline API (original)

The V1 API is still available at `/video-edit/v1`. It uses a fixed pipeline: index once, then generate responses to prompts. See the V1 API usage section below for details.

---

## 🐳 Docker Support

```bash
docker build -t vea .
docker run -p 8000:8000 -v $(pwd)/config.json:/app/config.json -v $(pwd)/data:/app/data vea

```

---

## 🖊️ Citation

If you find this project useful in your research, please consider citing our paper:

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
Copyright © 2026 Memories.ai Platforms, Inc.




Released under the <a href="LICENSE">MIT License</a>.
</p>
