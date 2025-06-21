import os
import asyncio
import logging
from collections import defaultdict
import itertools
from tqdm import tqdm

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline

# # --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize GCP OSS client ---
gcp_oss = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))


def list_all(folder):
    """
    List available media files stored in GCS, excluding image files.
    """
    items = gcp_oss.list_folder(BUCKET_NAME, f"{folder}/")
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".svg"}
    exclude_names = {"Mandalorian", "Clancy's", "WandaVision"}
    paths = []
    for item in items:
        path = item[1]
        ext = os.path.splitext(path)[1].lower()
        if ext not in image_exts and not any(excluded in path for excluded in exclude_names):
            paths.append(path)
    return paths

async def file_exists(gcs_path):
    """Returns True if the file exists in GCS, else False."""
    return gcp_oss.path_exists(BUCKET_NAME, gcs_path)

async def index_and_recap_all():
    all_paths = list_all("tv_show")
    # Group by show name
    shows = defaultdict(list)
    for path in all_paths:
        parts = path.split('/')
        if len(parts) >= 3:
            show_name = parts[1]
            shows[show_name].append(path)

    # Sort each show's episodes
    for ep_list in shows.values():
        ep_list.sort()

    # Prepare a round-robin list of episodes across shows
    episode_lists = list(shows.values())
    round_robin_eps = list(filter(None, itertools.chain.from_iterable(itertools.zip_longest(*episode_lists))))
    print(f"[INFO] Total episodes to check: {len(round_robin_eps)}")

    # Progress bar for the round-robin recap process
    with tqdm(total=len(round_robin_eps), desc="Recapping episodes", ncols=90) as pbar:
        for ep_path in round_robin_eps:
            if ep_path is None:
                pbar.update(1)
                continue
            filename = os.path.basename(ep_path)
            file_no_ext = os.path.splitext(filename)[0]

            recap_gcs_path = f"outputs/{file_no_ext}/recap.mp4"
            if await file_exists(recap_gcs_path):
                # Already recapped, skip
                pbar.set_postfix({"episode": filename, "status": "recap exists"})
                pbar.update(1)
                continue

            # Check if index exists
            index_gcs_path = f"indexing/{file_no_ext}/media_indexing.json"
            if not await file_exists(index_gcs_path):
                # Run video comprehension pipeline
                pbar.set_postfix({"episode": filename, "status": "indexing..."})
                print(f"\n[INFO] Indexing {filename}...")
                pipeline = ComprehensionPipeline(ep_path)
                await pipeline.run()

            # Run flexible response pipeline (recap)
            pbar.set_postfix({"episode": filename, "status": "recapping..."})
            print(f"\n[INFO] Recapping {filename}...")
            fr_pipeline = FlexibleResponsePipeline(ep_path)
            await fr_pipeline.run(
                user_prompt="I am a youtube channel that creates movie/tv recap videos. Give me an engaging, clear, approximately 5-minute recap of this episode for viewers who don't have time to watch it. Focus on major plot points, main characters, and key events.",
                video_response=True,
                original_audio=True,
                music=True,
                narration_enabled=True,
                aspect_ratio=0,
                subtitles=True,
                snap_to_beat=False,
                output_path=f"outputs/{file_no_ext}/recap.mp4"
            )
            pbar.set_postfix({"episode": filename, "status": "done"})
            pbar.update(1)

if __name__ == "__main__":
    asyncio.run(index_and_recap_all())