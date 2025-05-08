# app.py

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.schema import MovieFile, MovieIndexRequest, MovieIndexResponse
from src.config import (
    API_PREFIX,
    CREDENTIAL_PATH,
    BUCKET_NAME,
    MOVIE_LIBRARY,
)
from pipelines.longForm.longFormComprehensionPipeline import LongFormComprehensionPipeline
from pipelines.movieRecapEditing.movieRecapEditingPipeline import MovieRecapEditingPipeline


# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI()

# --- Initialize GCP OSS client ---
gcp_oss = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))


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


@app.post(f"{API_PREFIX}/index_movie")
async def index_movie(request: MovieIndexRequest):
    """
    Handle movie index request.
    """
    try:
        logger.info(f"Received index request for blob: {request.blob_path}")

        # Placeholder: Add video processing logic here
        download_url: Optional[str] = request.blob_path
        print(download_url)
        pipeline = LongFormComprehensionPipeline(request.blob_path)
        await pipeline.run()

        return MovieIndexResponse(
            message=f"Successfully indexed movie: {request.blob_path}."
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video.")



@app.post(f"{API_PREFIX}/edit_movie")
async def edit_movie(request: MovieIndexRequest):
    try:
        logger.info(f"Editing movie recap: {request.blob_path}")
        pipeline = MovieRecapEditingPipeline(request.blob_path)
        await pipeline.run()
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(status_code=500, detail="Editing failed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
