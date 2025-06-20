# app.py

import logging
from typing import List
import os
from fastapi import FastAPI, HTTPException

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.schema import (
    MovieFile,
    IndexRequest,
    IndexResponse,
    FlexibleResponseRequest,
    FlexibleResponseResult,
    ShortsRequest,
    ShortsResponse
)

from src.config import (
    API_PREFIX,
    CREDENTIAL_PATH,
    BUCKET_NAME,
    MOVIE_LIBRARY,
)

from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.movieToShort.movie_to_short_pipeline import MovieToShortsPipeline

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

@app.post(f"{API_PREFIX}/index")
async def index_longform(request: IndexRequest):
    try:
        logger.info(f"Received index request for blob: {request.blob_path} | Start fresh: {request.start_fresh}")
        pipeline = ComprehensionPipeline(request.blob_path, start_fresh=request.start_fresh)
        await pipeline.run()

        return IndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video.")
    
@app.post(f"{API_PREFIX}/flexible_respond", response_model=FlexibleResponseResult)
async def flexible_respond(request: FlexibleResponseRequest):
    # try:
        logger.info(f"Flexible response for: {request.blob_path} with prompt: {request.prompt}")
        pipeline = FlexibleResponsePipeline(request.blob_path)
        response = await pipeline.run(request.prompt, request.video_response, request.narration, request.music, request.narration, request.aspect_ratio, request.subtitles, request.snap_to_beat)

        return response
    # except Exception as e:
    #     logger.error(f"Flexible response error: {e}")
    #     raise HTTPException(status_code=500, detail="Flexible response failed.")
    

@app.post(f"{API_PREFIX}/movie_to_shorts", response_model=ShortsResponse)
async def movie_to_shorts(request: ShortsRequest):
        """
        Generate all 1-minute shorts for a movie using the MovieToShortsPipeline.
        """
    # try:
        pipeline = MovieToShortsPipeline(request.blob_path)
        shorts = await pipeline.run()
        return ShortsResponse(shorts=shorts)
    # except Except
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 