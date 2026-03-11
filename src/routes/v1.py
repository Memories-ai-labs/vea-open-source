# src/routes/v1.py
"""V1 legacy endpoints and Memories.ai webhook handlers."""

import json
import logging
import traceback
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request

from src.schema import (
    MovieFile,
    IndexRequest,
    IndexResponse,
    FlexibleResponseRequest,
    FlexibleResponseResult,
)
from src import config as _config
from src import services

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(f"{_config.API_PREFIX}/movies", response_model=List[MovieFile])
async def list_available_movies() -> List[MovieFile]:
    """List available movies in storage (local or cloud)."""
    try:
        logger.info("Fetching list of available movies...")
        blobs = services.storage_client.list_folder(_config.BUCKET_NAME, f"{_config.VIDEOS_DIR}/")
        movies = [MovieFile(name=blob[0], blob_path=blob[1]) for blob in blobs]
        logger.info(f"Found {len(movies)} movies.")
        return movies
    except Exception as e:
        logger.error(f"Error fetching movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")


@router.post(f"{_config.API_PREFIX}/index")
async def index_longform(request: IndexRequest):
    try:
        logger.info(f"Received index request for blob: {request.blob_path} | Start fresh: {request.start_fresh}")

        if not services.memories_manager:
            raise HTTPException(status_code=500, detail="Memories.ai not configured. Set MEMORIES_API_KEY in config.json")
        if not services.CAPTION_CALLBACK_URL:
            raise HTTPException(status_code=500, detail="Caption callback URL not set. Run with ./run.sh to set up ngrok")

        video_stem = Path(request.blob_path).stem
        debug_output_dir = f"debug_output/{video_stem}"

        # fmt: off
        from src.pipelines.v1_legacy.videoComprehension.comprehensionPipeline import ComprehensionPipeline  # noqa: E501
        # fmt: on
        pipeline = ComprehensionPipeline(
            request.blob_path,
            start_fresh=request.start_fresh,
            debug_output_dir=debug_output_dir,
            memories_manager=services.memories_manager,
            caption_callback_url=services.CAPTION_CALLBACK_URL,
            register_caption_callback=services.register_memories_callback,
        )
        await pipeline.run()

        return IndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@router.post(f"{_config.API_PREFIX}/generate_edit", response_model=FlexibleResponseResult)
async def generate_edit(request: FlexibleResponseRequest):
    """V1: Generate an edited video response from a long-form source video."""
    try:
        logger.info(f"Generate edit for: {request.blob_path} | prompt: {request.prompt}")
        # fmt: off
        from src.pipelines.v1_legacy.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline  # noqa: E501
        # fmt: on
        pipeline = FlexibleResponsePipeline(request.blob_path)
        response = await pipeline.run(
            request.prompt, request.video_response, request.original_audio,
            request.music, request.narration, request.aspect_ratio,
            request.subtitles, request.snap_to_beat, request.output_path
        )
        return response
    except Exception as e:
        logger.error(f"Generate edit error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Edit generation failed: {str(e)}")


# --- Memories.ai Webhook Endpoints ---

@router.post("/webhooks/memories/caption")
async def memories_caption_callback(request: Request):
    """Webhook endpoint for Memories.ai Video Caption API callbacks."""
    try:
        data = await request.json()
        task_id = data.get("task_id")

        logger.info(f"[MEMORIES WEBHOOK] Received caption callback for task: {task_id}")

        output_dir = Path("debug_output/caption_callbacks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[MEMORIES WEBHOOK] Saved callback to {output_file}")

        response_text = (data.get("data") or {}).get("response", "") or (data.get("data") or {}).get("text", "")
        if response_text:
            preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            logger.info(f"[MEMORIES WEBHOOK] Response preview: {preview}")

        if task_id and task_id in services._memories_pending_callbacks:
            future = services._memories_pending_callbacks.pop(task_id)
            if not future.done():
                future.set_result(data)
            logger.info(f"[MEMORIES WEBHOOK] Task {task_id} completed and callback resolved")
        else:
            logger.warning(f"[MEMORIES WEBHOOK] Unknown task_id or no pending callback: {task_id}")

        return {"status": "ok", "task_id": task_id}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")


@router.post("/webhooks/memories/upload")
async def memories_upload_callback(request: Request):
    """Webhook endpoint for Memories.ai Upload API callbacks."""
    try:
        data = await request.json()
        video_no = data.get("videoNo") or data.get("video_no")
        status = data.get("status")

        logger.info(f"[MEMORIES WEBHOOK] Upload callback - videoNo: {video_no}, status: {status}")

        if video_no and video_no in services._memories_pending_callbacks:
            future = services._memories_pending_callbacks.pop(video_no)
            if not future.done():
                future.set_result(data)

        return {"status": "ok", "video_no": video_no}
    except Exception as e:
        logger.error(f"[MEMORIES WEBHOOK] Error processing upload callback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")
