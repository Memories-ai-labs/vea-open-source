import os
import asyncio
import logging
from collections import defaultdict
import itertools

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

INDEX_LOG = "indexed_files.txt"

def load_indexed():
    if not os.path.exists(INDEX_LOG):
        return set()
    with open(INDEX_LOG, "r") as f:
        return set(line.strip() for line in f.readlines())
def mark_indexed(path):
    with open(INDEX_LOG, "a") as f:
        f.write(path + "\n")

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

async def recap_longform(path):
    print(f"Received index request for blob: {path}")
    pipeline = MovieRecapEditingPipeline(path)
    await pipeline.run()

async def index_all():
    folder = "tv_show"
    paths = list_all(folder)

    # Load previously indexed paths
    already_indexed = load_indexed()

    # Group by the second-level folder name
    folder_to_paths = defaultdict(list)
    for path in paths:
        parts = path.split('/')
        if len(parts) >= 3:
            show_folder = parts[1]
            folder_to_paths[show_folder].append(path)

    # Sort each folder's episodes
    for episodes in folder_to_paths.values():
        episodes.sort()

    # Round-robin over all shows
    for round_group in itertools.zip_longest(*folder_to_paths.values()):
        for path in round_group:
            if path and path not in already_indexed:
                try:
                    await index_longform(path)
                    mark_indexed(path)
                    await recap_longform(path)
                except Exception as e:
                    logger.error(f"Error indexing {path}: {e}")


if __name__ == "__main__":
    asyncio.run(index_all())
 