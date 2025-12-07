# VEA - Video Editing Automation

VEA is an AI-powered video editing service that automatically creates short-form video content from long-form videos. Give it a movie or documentary, ask it to create a recap or highlight reel, and it will:

1. **Understand** the video content using AI (Memories.ai)
2. **Generate** a narration script tailored to your prompt
3. **Select** relevant clips from the source video
4. **Add** text-to-speech narration (ElevenLabs)
5. **Apply** intelligent dynamic cropping for vertical video (9:16)
6. **Include** background music synced to the beat
7. **Export** both a rendered video and Final Cut Pro XML for further editing

## Requirements

- Python 3.11+
- FFmpeg (must be installed on your system)
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### 1. Install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 3. Configure API keys

Copy the example config and fill in your API keys:

```bash
cp config.example.json config.json
```

Edit `config.json` with your API keys:

| Key | Required | Description | Get it from |
|-----|----------|-------------|-------------|
| `MEMORIES_API_KEY` | Yes | Video understanding AI | [memories.ai](https://memories.ai/app/service/key) |
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project for Gemini | [Google Cloud Console](https://console.cloud.google.com) |
| `ELEVENLABS_API_KEY` | Yes | Text-to-speech narration | [elevenlabs.io](https://elevenlabs.io) |
| `SOUNDSTRIPE_KEY` | No | Background music | [soundstripe.com](https://soundstripe.com) |

### 4. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

### 5. Organize your videos

Place your source videos in project folders:

```
data/videos/
├── MyProject/
│   └── source_video.mp4
└── AnotherProject/
    ├── video1.mp4
    └── video2.mp4
```

### 6. Run the server

```bash
source .venv/bin/activate
python -m src.app
```

The API server will start at `http://localhost:8000`.

## API Usage

### Index a video (required before generating responses)

```bash
curl -X POST http://localhost:8000/video-edit/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "blob_path": "data/videos/MyProject/",
    "start_fresh": true
  }'
```

### Generate a video response

```bash
curl -X POST http://localhost:8000/video-edit/v1/flexible_respond \
  -H "Content-Type: application/json" \
  -d '{
    "blob_path": "data/videos/MyProject/",
    "prompt": "Create a 2-minute recap of this movie",
    "video_response": true,
    "original_audio": false,
    "music": true,
    "narration": true,
    "aspect_ratio": 0.5625,
    "subtitles": true
  }'
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `blob_path` | string | Path to video folder (e.g., `data/videos/MyProject/`) |
| `prompt` | string | What kind of video to create |
| `video_response` | bool | Generate a video (true) or just text (false) |
| `original_audio` | bool | Include original video audio |
| `music` | bool | Add background music |
| `narration` | bool | Add AI-generated narration |
| `aspect_ratio` | float | Output aspect ratio (0.5625 = 9:16 vertical, 1.778 = 16:9 horizontal) |
| `subtitles` | bool | Generate subtitles |
| `snap_to_beat` | bool | Sync clip cuts to music beats |

## Output

Generated files are saved to `data/outputs/{ProjectName}/`:

```
data/outputs/MyProject/
├── footage/           # Normalized source videos
├── narrations/        # Generated audio narrations
├── music/             # Background music
├── clip_plan.json     # Clip selection metadata
├── MyProject.mp4      # Final rendered video
└── MyProject.fcpxml   # Final Cut Pro XML for editing
```

## Project Structure

```
vea-playground/
├── src/
│   ├── app.py                    # FastAPI server
│   ├── config.py                 # Configuration loading
│   └── pipelines/
│       ├── videoComprehension/   # Video indexing pipeline
│       ├── flexibleResponse/     # Response generation pipeline
│       └── common/               # Shared components
│           ├── timeline_constructor.py  # Video assembly
│           ├── dynamic_cropping.py      # AI cropping (ViNet)
│           └── fcpxml_exporter.py       # FCPXML export
├── lib/
│   ├── llm/                      # LLM integrations
│   └── utils/                    # Utility functions
├── data/
│   ├── videos/                   # Source videos (organized by project)
│   ├── indexing/                 # Video indexing metadata
│   └── outputs/                  # Generated outputs
└── vinet_v2/                     # ViNet saliency model
```

## Docker (Optional)

Build and run with Docker:

```bash
# Build
docker build -t vea .

# Run
docker run -p 8000:8000 -v $(pwd)/config.json:/app/config.json -v $(pwd)/data:/app/data vea
```

## Troubleshooting

### "Failed to read the first frame of video"
- Check that FFmpeg is installed: `ffmpeg -version`
- Ensure the video file is not corrupted
- Try deleting cached files in `data/outputs/{project}/footage/`

### SIGSEGV during dynamic cropping
- This can happen with GPU/CPU conflicts
- The code forces CPU mode for ViNet to avoid this
- If it persists, try clearing the cache and restarting

### "media_indexing.json not found"
- You need to index the video first using the `/index` endpoint
- Make sure `blob_path` matches between index and respond calls

## License

[Add your license here]
