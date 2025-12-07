# app.py

import asyncio
import logging
import traceback
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException, Request

from lib.oss.storage_factory import get_storage_client
from lib.llm.MemoriesAiManager import MemoriesAiManager, create_memories_manager
from src.schema import (
    MovieFile,
    IndexRequest,
    IndexResponse,
    FlexibleResponseRequest,
    FlexibleResponseResult,
    ShortsRequest,
    ShortsResponse,
    ScreenplayRequest,
    ScreenplayResponse,
    QualityAssessmentRequest,
    QualityAssessmentResponse,
)

from src.config import (
    API_PREFIX,
    BUCKET_NAME,
    VIDEOS_DIR,
    get_storage_mode,
    ensure_local_directories,
)

from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.movieToShort.movie_to_short_pipeline import MovieToShortsPipeline
from src.pipelines.screenplay.screenplay_pipeline import ScreenplayPipeline
from src.pipelines.qualityAnalysis.quality_assesment_pipeline import QualityAssessmentPipeline


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI()

# --- Initialize Storage client (local or cloud based on config) ---
storage_client = get_storage_client()
logger.info(f"Storage mode: {get_storage_mode()}")
ensure_local_directories()

# Alias for backward compatibility with existing code
gcp_oss = storage_client

# --- Initialize Memories.ai client (optional - only if API key is configured) ---
memories_manager: Optional[MemoriesAiManager] = None
_memories_pending_callbacks: Dict[str, asyncio.Future] = {}

try:
    if os.environ.get("MEMORIES_API_KEY"):
        memories_manager = create_memories_manager(debug=True)  # Set debug=False to reduce output
        logger.info("Memories.ai client initialized")
    else:
        logger.info("Memories.ai API key not configured - video understanding will use Gemini")
except Exception as e:
    logger.warning(f"Failed to initialize Memories.ai client: {e}")

# Caption API callback URL - loaded from config.json api_keys section
# Set to your public ngrok/server URL, e.g., "https://xxx.ngrok-free.app/webhooks/memories/caption"
_callback_url = os.environ.get("MEMORIES_CAPTION_CALLBACK_URL", "")
CAPTION_CALLBACK_URL = _callback_url if _callback_url else None  # Treat empty string as None
if CAPTION_CALLBACK_URL:
    logger.info(f"Video Caption API callback URL: {CAPTION_CALLBACK_URL}")
else:
    logger.info("MEMORIES_CAPTION_CALLBACK_URL not set - will use Chat API instead of Caption API")


@app.get("/")
async def root():
    return {"message": "FastAPI inference service is running."}

@app.get(f"{API_PREFIX}/movies", response_model=List[MovieFile])
async def list_available_movies() -> List[MovieFile]:
    """
    List available movies in storage (local or cloud).
    """
    try:
        logger.info("Fetching list of available movies...")
        blobs = storage_client.list_folder(BUCKET_NAME, f"{VIDEOS_DIR}/")
        movies = [MovieFile(name=blob[0], blob_path=blob[1]) for blob in blobs]
        logger.info(f"Found {len(movies)} movies.")
        return movies
    except Exception as e:
        logger.error(f"Error fetching movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")

@app.post(f"{API_PREFIX}/index")
async def index_longform(request: IndexRequest):
    try:
        logger.info(f"Received index request for blob: {request.blob_path} | Start fresh: {request.start_fresh}")

        # Check required dependencies
        if not memories_manager:
            raise HTTPException(status_code=500, detail="Memories.ai not configured. Set MEMORIES_API_KEY in config.json")
        if not CAPTION_CALLBACK_URL:
            raise HTTPException(status_code=500, detail="Caption callback URL not set. Run with ./run.sh to set up ngrok")

        # Auto-generate debug output dir based on video name
        from pathlib import Path
        video_stem = Path(request.blob_path).stem
        debug_output_dir = f"debug_output/{video_stem}"

        pipeline = ComprehensionPipeline(
            request.blob_path,
            start_fresh=request.start_fresh,
            debug_output_dir=debug_output_dir,
            memories_manager=memories_manager,
            caption_callback_url=CAPTION_CALLBACK_URL,
            register_caption_callback=register_memories_callback,
        )
        await pipeline.run()

        return IndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    
@app.post(f"{API_PREFIX}/flexible_respond", response_model=FlexibleResponseResult)
async def flexible_respond(request: FlexibleResponseRequest):
    try:
        logger.info(f"Flexible response for: {request.blob_path} with prompt: {request.prompt}")
        pipeline = FlexibleResponsePipeline(request.blob_path)
        response = await pipeline.run(request.prompt, request.video_response, request.original_audio, request.music, request.narration, request.aspect_ratio, request.subtitles, request.snap_to_beat, request.output_path)
        return response
    except Exception as e:
        logger.error(f"Flexible response error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Flexible response failed: {str(e)}")
    

@app.post(f"{API_PREFIX}/movie_to_shorts", response_model=ShortsResponse)
async def movie_to_shorts(request: ShortsRequest):
    """
    Generate all 1-minute shorts for a movie using the MovieToShortsPipeline.
    """
    try:
        pipeline = MovieToShortsPipeline(request.blob_path)
        shorts = await pipeline.run()
        return ShortsResponse(shorts=shorts)
    except Exception as e:
        logger.error(f"Error generating shorts for {request.blob_path}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate shorts: {str(e)}")
        
@app.post(f"{API_PREFIX}/screenplay", response_model=ScreenplayResponse)
async def generate_screenplay(request: ScreenplayRequest):
    try:
        logger.info(f"Starting screenplay generation for: {request.blob_path}")
        pipeline = ScreenplayPipeline(request.blob_path)
        gcs_output_path = await pipeline.run()

        return ScreenplayResponse(
            message=f"Screenplay generated for: {request.blob_path}",
            output_path=gcs_output_path
        )
    except Exception as e:
        logger.error(f"Screenplay generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate screenplay.")

@app.post(f"{API_PREFIX}/quality_assessment", response_model=QualityAssessmentResponse)
async def assess_quality(request: QualityAssessmentRequest):
    """
    Assess the quality of a generated video using LLM.
    """
    try:
        logger.info(f"Starting quality assessment for: {request.blob_path}")
        pipeline = QualityAssessmentPipeline(
            cloud_storage_video_path=request.blob_path,
            ground_truth_text=request.ground_truth,
            user_prompt=request.user_prompt,
        )
        result = await pipeline.run()

        return QualityAssessmentResponse(
            message=f"Quality assessment completed for: {request.blob_path}",
            result=result,
        )
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to assess video quality.")


# --- Memories.ai Webhook Endpoints ---

@app.post("/webhooks/memories/caption")
async def memories_caption_callback(request: Request):
    """
    Webhook endpoint for Memories.ai Video Caption API callbacks.

    When using the async Video Caption API, Memories.ai will POST results
    to this endpoint when processing completes.
    """
    import json
    from pathlib import Path

    try:
        data = await request.json()
        task_id = data.get("task_id")

        logger.info(f"[MEMORIES WEBHOOK] Received caption callback for task: {task_id}")

        # Always save callback data to file for debugging/analysis
        output_dir = Path("debug_output/caption_callbacks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[MEMORIES WEBHOOK] Saved callback to {output_file}")

        # Log response preview
        response_text = data.get("data", {}).get("response", "")
        if response_text:
            preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            logger.info(f"[MEMORIES WEBHOOK] Response preview: {preview}")

        if task_id and task_id in _memories_pending_callbacks:
            future = _memories_pending_callbacks.pop(task_id)
            if not future.done():
                future.set_result(data)
            logger.info(f"[MEMORIES WEBHOOK] Task {task_id} completed and callback resolved")
        else:
            logger.warning(f"[MEMORIES WEBHOOK] Unknown task_id or no pending callback: {task_id}")

        return {"status": "ok", "task_id": task_id}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


@app.post("/webhooks/memories/upload")
async def memories_upload_callback(request: Request):
    """
    Webhook endpoint for Memories.ai Upload API callbacks.

    Called when video upload and indexing completes.
    """
    try:
        data = await request.json()
        video_no = data.get("videoNo") or data.get("video_no")
        status = data.get("status")

        logger.info(f"[MEMORIES WEBHOOK] Upload callback - videoNo: {video_no}, status: {status}")

        # Store or process the upload completion
        if video_no and video_no in _memories_pending_callbacks:
            future = _memories_pending_callbacks.pop(video_no)
            if not future.done():
                future.set_result(data)

        return {"status": "ok", "video_no": video_no}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing upload callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


def register_memories_callback(task_id: str) -> asyncio.Future:
    """
    Register a pending callback for a Memories.ai async operation.

    Returns a Future that will be resolved when the webhook is called.

    Usage:
        future = register_memories_callback(task_id)
        result = await asyncio.wait_for(future, timeout=300)
    """
    loop = asyncio.get_running_loop()  # Use running loop, not default event loop
    future = loop.create_future()
    _memories_pending_callbacks[task_id] = future
    logger.info(f"[CALLBACK] Registered callback for task: {task_id} (pending: {len(_memories_pending_callbacks)})")
    return future


def cleanup_orphaned_callbacks(max_age_seconds: int = 3600):
    """
    Remove callbacks that have been pending for too long.

    Call this periodically to prevent memory leaks from failed requests.
    """
    # For now, just log the count - in production you'd track timestamps
    if _memories_pending_callbacks:
        logger.warning(f"[CALLBACK] {len(_memories_pending_callbacks)} pending callbacks: {list(_memories_pending_callbacks.keys())[:5]}...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
