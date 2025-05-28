import os
import asyncio
import logging
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import (
    CREDENTIAL_PATH
)
from src.pipelines.longFormComprehension.longFormComprehensionPipeline import LongFormComprehensionPipeline
from src.pipelines.movieRecapEditing.movieRecapEditingPipeline import MovieRecapEditingPipeline
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.shortFormComprehension.shortFormComprehensionPipeline import ShortFormComprehensionPipeline

# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize GCP OSS client ---
gcp_oss = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

def list_all(folder):
    """
    List available movies stored in GCS.
    """
    logger.info("Fetching list of available TV shows from GCS...")
    items = gcp_oss.list_folder("openinterx-vea", f"{folder}/")
    paths = []
    for item in items:
        paths.append(item[1])

    return paths

async def index_longform(path):
    print(f"Received index request for blob: {path}")
    pipeline = LongFormComprehensionPipeline(path, False)
    await pipeline.run()

async def index_all():
    folder = "tv_show"
    for path in list_all(folder):
        await index_longform(path)

if __name__ == "__main__":
    asyncio.run(index_all())
 