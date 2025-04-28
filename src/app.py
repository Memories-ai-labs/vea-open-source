# app.py

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.schema import MovieFile, EditRequest, EditResponse
from src.config import (
    API_PREFIX,
    CREDENTIAL_PATH,
    BUCKET_NAME,
    MOVIE_LIBRARY,
)


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


@app.post(f"{API_PREFIX}/edit", response_model=EditResponse)
async def edit_movie(request: EditRequest) -> EditResponse:
    """
    Handle movie edit request.
    """
    try:
        logger.info(f"Received edit request for blob: {request.blob_path}")

        # Placeholder: Add video processing logic here
        download_url: Optional[str] = None

        return EditResponse(
            message=f"Successfully processed movie: {request.blob_path}.",
            url=download_url or ""
        )
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
