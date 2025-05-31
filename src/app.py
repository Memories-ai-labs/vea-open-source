# app.py

import logging
from typing import List, Optional
import os
from fastapi import FastAPI, HTTPException

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.schema import (
    MovieFile,
    MovieIndexRequest,
    MovieIndexResponse,
    MovieRecapRequest,
    MovieRecapResponse,
    FlexibleResponseRequest,
    FlexibleResponseResult,
    IndexCheckRequest, 
    IndexCheckResponse,
    ShortFormIndexRequest, 
    ShortFormIndexResponse
)

from src.config import (
    API_PREFIX,
    CREDENTIAL_PATH,
    BUCKET_NAME,
    MOVIE_LIBRARY,
)
from src.pipelines.longFormComprehension.longFormComprehensionPipeline import LongFormComprehensionPipeline
from src.pipelines.movieRecapEditing.movieRecapEditingPipeline import MovieRecapEditingPipeline
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.shortFormComprehension.shortFormComprehensionPipeline import ShortFormComprehensionPipeline


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI()

# --- Initialize GCP OSS client ---
gcp_oss = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))


@app.get("/")
async def root():
    return {"message": "FastAPI inference service is running."}

@app.get(f"{API_PREFIX}/movies", response_model=List[MovieFile])
async def list_available_movies() -> List[MovieFile]:
    """
    List available movies stored in GCS.
    """
    try:
        logger.info("Fetching list of available movies from GCS...")
        blobs = gcp_oss.list_folder(BUCKET_NAME, f"{MOVIE_LIBRARY}/")
        movies = [MovieFile(name=blob[0], blob_path=blob[1]) for blob in blobs]
        logger.info(f"Found {len(movies)} movies.")
        return movies
    except Exception as e:
        logger.error(f"Error fetching movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")

@app.post(f"{API_PREFIX}/index_longform")
async def index_longform(request: MovieIndexRequest):
    try:
        logger.info(f"Received index request for blob: {request.blob_path} | Start fresh: {request.start_fresh}")
        pipeline = LongFormComprehensionPipeline(request.blob_path, start_fresh=request.start_fresh)
        await pipeline.run()

        return MovieIndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video.")


@app.post(f"{API_PREFIX}/edit_movie", response_model=MovieRecapResponse)
async def edit_movie(request: MovieRecapRequest):
    try:
        logger.info(f"Editing movie recap: {request.blob_path}")
        pipeline = MovieRecapEditingPipeline(request.blob_path)
        url = await pipeline.run(
            user_context=request.user_context,
            user_prompt=request.user_prompt,
            output_language=request.output_language or "English",
            user_music=request.user_music_path
        )
        return MovieRecapResponse(message="Recap generated.", url=url)
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(status_code=500, detail="Editing failed.")
    
@app.post(f"{API_PREFIX}/flexible_respond", response_model=FlexibleResponseResult)
async def flexible_respond(request: FlexibleResponseRequest):
    """
    Run the flexible response pipeline on a movie.
    """
    try:
        logger.info(f"Flexible response for: {request.blob_path} with prompt: {request.prompt}")
        pipeline = FlexibleResponsePipeline(request.blob_path)
        response = await pipeline.run(request.prompt, request.video_response)

        return response
    except Exception as e:
        logger.error(f"Flexible response error: {e}")
        raise HTTPException(status_code=500, detail="Flexible response failed.")
    

@app.post(f"{API_PREFIX}/check_index", response_model=IndexCheckResponse)
async def check_index_status(request: IndexCheckRequest):
    """
    Check if all required indexing files exist for a given movie.
    """
    try:
        media_name = os.path.basename(request.blob_path)
        media_base_name = os.path.splitext(media_name)[0]
        gcs_prefix = f"indexing/{media_base_name}/"  # Ensure trailing slash

        all_exist = gcp_oss.all_files_exist(
            bucket=BUCKET_NAME,
            base_path=gcs_prefix,
            filenames=request.required_files
        )

        return IndexCheckResponse(blob_path=request.blob_path, all_exist=all_exist)
    except Exception as e:
        logger.error(f"[ERROR] Index check failed for {request.blob_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check index status.")
    
@app.post(f"{API_PREFIX}/index_shortform", response_model=ShortFormIndexResponse)
async def index_shortform(request: ShortFormIndexRequest):
    """
    Handle short form video comprehension (folder of short videos).
    """
    try:
        logger.info(f"Indexing shortform videos from: {request.blob_path} | Start fresh: {request.start_fresh}")
        pipeline = ShortFormComprehensionPipeline(request.blob_path, start_fresh=request.start_fresh)
        await pipeline.run()
        return ShortFormIndexResponse(message=f"Successfully indexed shortform videos from: {request.blob_path}")
    except Exception as e:
        logger.error(f"Shortform indexing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to index shortform videos.")


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)

 