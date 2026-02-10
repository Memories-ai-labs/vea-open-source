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

* **Python 3.11+**
* **FFmpeg** (Must be installed on system)
* **[uv](https://github.com/astral-sh/uv)** package manager

### 1. Installation

```bash
# Install uv
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

```

### 2. Install FFmpeg

| OS | Command |
| --- | --- |
| **Ubuntu/Debian** | `sudo apt install ffmpeg` |
| **macOS** | `brew install ffmpeg` |
| **Windows** | Download from [ffmpeg.org](https://ffmpeg.org/download.html) |

### 3. Configuration

Copy the example config and populate your keys:

```bash
cp config.example.json config.json

```

**Required Keys in `config.json`:**

* `MEMORIES_API_KEY`: [Get Key](https://memories.ai/app/service/key)
* `GOOGLE_CLOUD_PROJECT`: [Google Cloud Console](https://console.cloud.google.com)
* `ELEVENLABS_API_KEY`: [ELEVENLABS](https://elevenlabs.io/docs/api-reference/)

### 4. Run Server

```bash
gcloud auth application-default login  # Authenticate GCP
source .venv/bin/activate
python -m src.app

```

> The API server will start at `http://localhost:8000`.

---

## 🛠️ API Usage

### Step 1: Index Video

*Required before generating content.*

```bash
curl -X POST http://localhost:8000/video-edit/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "blob_path": "data/videos/MyProject/",
    "start_fresh": true
  }'

```

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

Generated files are saved in `data/outputs/{ProjectName}/`:

* `MyProject.mp4`: Final rendered video.
* `MyProject.fcpxml`: Final Cut Pro XML for manual tweaking.
* `clip_plan.json`, `narrations/`, `music/`: Intermediate assets.

---

## 📂 Project Structure

```text
vea-playground/
├── src/
│   ├── app.py                  # FastAPI server
│   ├── pipelines/              # Core logic (Indexing, Response)
│   └── common/                 # Timeline, Dynamic Cropping (ViNet), FCPXML
├── data/
│   ├── videos/                 # Place source videos here
│   └── outputs/                # Generated results
└── vinet_v2/                   # Saliency model for cropping

```

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
