# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VEA Playground is a video editing automation service that uses LLMs to understand video content and generate edited video responses. The system indexes long-form videos, comprehends their content, and responds to user prompts by extracting relevant clips, adding narration, music, and dynamic cropping to create polished short-form content.

**LLM Architecture (Hybrid):**
- **Memories.ai** - Video understanding (upload, indexing, chat with timestamped references)
- **Gemini (Vertex AI)** - Structured JSON output (schemas, formatting)

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

**Required system dependencies:**
- `ffmpeg` must be installed on the system for video processing

**Configuration files needed:**
- `config.json` - Copy from `config.example.json` and fill in your API keys:
  - `MEMORIES_API_KEY` - Memories.ai API key (required for video understanding)
  - `GOOGLE_CLOUD_PROJECT` - GCP project ID (required for Gemini structured output)
  - `ELEVENLABS_API_KEY` - ElevenLabs API key (required for narration TTS)
  - `SOUNDSTRIPE_KEY` - Soundstripe API key (optional, for background music)

**Google Cloud authentication:**
- Run `gcloud auth application-default login` for Gemini access via Vertex AI

### Running the Service

```bash
# Run locally (without Docker)
source .venv/bin/activate
python -m src.app

# Build Docker container
sudo docker build -t vea-recap .

# Run Docker container
sudo docker run -p 8000:8000 vea-recap
```

The service runs a FastAPI server on port 8000.

## Architecture Overview

### Pipeline-Based Processing

The system is organized around **5 main pipelines** that handle different video processing workflows:

1. **ComprehensionPipeline** (`src/pipelines/videoComprehension/`) - Indexes videos by generating comprehensive metadata:
   - Preprocesses videos into segments at different time scales (15min and 5min intervals)
   - Performs rough comprehension to understand overall narrative
   - Scene-by-scene analysis to capture detailed visual content
   - Refines the story into structured JSON and text summaries
   - Outputs: `story.txt`, `story.json`, `scenes.json`, `people.txt`

2. **FlexibleResponsePipeline** (`src/pipelines/flexibleResponse/`) - Generates responses to user prompts:
   - Three response types: text-only, text with evidence clips, or full video response
   - For video responses: generates narration scripts, selects relevant clips, adds music
   - Runs `edit_video_response.py` as subprocess to assemble final video
   - Uses evidence retrieval to find relevant timestamps in indexed content

3. **MovieToShortsPipeline** (`src/pipelines/movieToShort/`) - Converts long-form content to short-form clips

4. **ScreenplayPipeline** (`src/pipelines/screenplay/`) - Generates screenplay-style text from video content

5. **QualityAssessmentPipeline** (`src/pipelines/qualityAnalysis/`) - Evaluates generated video quality using quizzing

### Key Architectural Patterns

**Two-phase video processing:**
- Phase 1: Indexing (ComprehensionPipeline) - run once per video, stores metadata in GCS at `indexing/{video_name}/media_indexing.json`
- Phase 2: Response generation - uses cached indexing data to quickly respond to prompts

**Subprocess isolation for video editing:**
The `FlexibleResponsePipeline` calls `edit_video_response.py` as a subprocess (not imported) to free memory. The subprocess uses `TimelineConstructor` for video assembly.

**Caching strategy:**
- Videos are cached in `.cache/comprehension_media/` and `.cache/gcs_videos/`
- Indexing JSON files are cached in `.cache/media_indexing/`
- Temporary directories created with `tempfile.mkdtemp()` are cleaned via `clean_stale_tempdirs()`

**Cloud storage organization:**
- Source videos: `gs://{BUCKET_NAME}/movie_library/`
- Indexing data: `gs://{BUCKET_NAME}/indexing/{video_name}/media_indexing.json`
- Output videos: `gs://{BUCKET_NAME}/outputs/{video_name}/{run_id}/`

### Common Utilities

**`lib/utils/media.py`** - Core video processing functions:
- `get_video_info(video_path)` - Extract video metadata using ffprobe
- `get_video_duration(video_path)` - Get duration in seconds
- `downsample_video()` - Reduce resolution/fps for LLM processing
- `preprocess_long_video()` - Split video into time-based segments
- `extract_video_segment()` - Extract clip by timestamp (HH:MM:SS format)
- `download_and_cache_video()` - Download from GCS with local caching
- `parse_time_to_seconds()` - Convert HH:MM:SS[,mmm] to float seconds
- `seconds_to_hhmmss()` - Convert float seconds to HH:MM:SS,mmm

**`lib/llm/MemoriesAiManager.py`** - Memories.ai video understanding:
- `upload_video_url(video_url)` - Upload video by URL, returns `videoNo`
- `upload_video_file(file_path)` - Upload local video file
- `wait_for_ready(video_no)` - Poll until video indexing completes
- `chat(video_nos, prompt)` - Query indexed videos, returns text + timestamped references
- `search(query, search_type)` - Search indexed videos (BY_VIDEO, BY_AUDIO, BY_CLIP)
- All methods include `[MEMORIES]` prefixed logging for debugging
- Set `debug=True` for full raw API response logging

**`lib/llm/GeminiGenaiManager.py`** - Gemini LLM interaction (for structured output):
- `LLM_request(prompt_contents, schema=None)` - Call Gemini with structured output
- Accepts mixed inputs: local file paths (Path), GCS URIs (`gs://...`), or text strings
- Supports Pydantic schema for structured JSON responses
- Automatic retries with exponential backoff
- Uses Vertex AI (not direct Genai API)

**`lib/oss/gcp_oss.py`** - Google Cloud Storage operations via `GoogleCloudStorage` class

### FCPXML Export

The system can export edits as Final Cut Pro XML (FCPXML 1.10) for professional editing software:
- `src/pipelines/common/fcpxml_exporter.py` - Generates valid FCPXML with precise frame timing
- `context/fcpxml_formatting_guide.md` - Comprehensive guide for FCPXML format
- Supports multi-shot dynamic cropping, time mapping, audio roles, and narration lanes
- Uses rational number fractions (`1/24s`) for frame-accurate positioning

### Video Assembly and Compositing

**`TimelineConstructor`** (`src/pipelines/common/timeline_constructor.py`) - Assembles final videos:
- Handles multi-round editing with LLM feedback
- Dynamic cropping using ViNet saliency detection
- Audio mixing (original audio, narration, music)
- Subtitle generation and overlay
- Snap-to-beat synchronization with music
- Exports both MP4 and FCPXML

**Dynamic Cropping** (`src/pipelines/common/dynamic_cropping.py`):
- Uses ViNet model to detect salient regions
- Reframes 16:9 video to 9:16 vertical format
- Multi-shot support: splits clips based on saliency changes

## API Endpoints

All endpoints use prefix: `/video-edit/v1`

- `GET /movies` - List available videos in GCS
- `POST /index` - Index a video (runs ComprehensionPipeline)
- `POST /flexible_respond` - Generate response to user prompt
- `POST /movie_to_shorts` - Convert movie to shorts
- `POST /screenplay` - Generate screenplay from video
- `POST /quality_assessment` - Assess video quality

**Webhook endpoints** (for Memories.ai async callbacks):
- `POST /webhooks/memories/caption` - Callback for Video Caption API completion
- `POST /webhooks/memories/upload` - Callback for video upload/indexing completion

## Cloud Deployment

**GCP Kubernetes Setup:**
- Cluster: `video-edit-cluster`
- Deployment: `video-edit` workload
- Docker registry: `us-docker.pkg.dev/gen-lang-client-0057517563/vea-service/video-edit:latest`

**Deploy commands:**
```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Delete deployment
kubectl delete -f deployment.yaml

# Push image to GCP
docker tag <local-image> us-docker.pkg.dev/gen-lang-client-0057517563/vea-service/video-edit:latest
docker push us-docker.pkg.dev/gen-lang-client-0057517563/vea-service/video-edit:latest
```

## Important Conventions

**Pipeline task structure:**
- Each pipeline has a main class (e.g., `ComprehensionPipeline`) with async `run()` method
- Complex pipelines split work into `tasks/` subdirectory with focused task classes
- Tasks are composable async functions/classes that return structured data

**Media indexing JSON format:**
The `media_indexing.json` file contains:
- `media_files[]` - Array of file-level metadata with:
  - `name`, `cloud_storage_path` - File identification
  - `story.txt` - Linear narrative summary
  - `story.json` - Structured story with segments
  - `scenes.json` - Scene metadata with timestamps and visual content
  - `people.txt` - Character descriptions and relationships
- `manifest` - Description of each field's purpose

**Timestamp formats:**
- Internal processing uses float seconds
- User-facing/subtitle formats use `HH:MM:SS,mmm` (SRT standard)
- FCPXML uses rational fractions like `1/24s` or `1001/30000s`

**Video preprocessing for LLM:**
- Videos are downsampled to 480p height, low FPS (0.5-1fps), CRF 30-40
- This drastically reduces token count while preserving visual information
- Use `downsample_video()` before sending to Gemini

**Config management:**
- Static config in `src/config.py`: API_PREFIX, VIDEO_EXTS, BUCKET_NAME
- Environment-specific config via env vars: ENV (development/production)
- Secrets in `config/apiKeys.json` loaded into environment variables

## ViNet Saliency Model

The `vinet_v2/` directory contains a PyTorch-based visual saliency detection model used for intelligent cropping:
- **ViNet_A** - Audio-visual saliency (considers both video and audio)
- **ViNet_S** - Visual-only saliency
- Model setup via `lib/utils/vinet_setup.py`
- Used by dynamic cropping to identify important regions when reframing video

## Testing Tools

**Postman** is recommended for API testing (mentioned in README).

## Notes on Code Quality

- Pipelines run with commented-out try/except blocks in `src/app.py` for easier debugging
- Subprocess pattern prevents memory leaks during video processing
- GCS client initialized once at app startup with credentials from file
- All video processing should clean up temp files via `clean_stale_tempdirs()`
