import os
import asyncio
import logging
from collections import defaultdict
import itertools
from tqdm import tqdm

from lib.oss.storage_factory import get_storage_client
from src.config import BUCKET_NAME, INDEXING_DIR, OUTPUTS_DIR
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline

# --- Initialize logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize GCP OSS client ---
gcp_oss = get_storage_client()

FAILED_LOG_PATH = "failed_recaps.txt"

def list_all(folder):
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

def load_failed_episodes():
    failed = set()
    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                # Store both filename and full path for extra safety
                if len(parts) >= 2:
                    failed.add(parts[0].strip())   # filename
                    failed.add(parts[1].strip())   # gcs path
    return failed

async def file_exists(gcs_path):
    return gcp_oss.path_exists(BUCKET_NAME, gcs_path)

async def index_and_recap_all():
    all_paths = list_all("tv_show")
    shows = defaultdict(list)
    for path in all_paths:
        parts = path.split('/')
        if len(parts) >= 3:
            show_name = parts[1]
            shows[show_name].append(path)
    for ep_list in shows.values():
        ep_list.sort()
    episode_lists = list(shows.values())
    round_robin_eps = list(filter(None, itertools.chain.from_iterable(itertools.zip_longest(*episode_lists))))
    print(f"[INFO] Total episodes to check: {len(round_robin_eps)}")

    failed_episodes = []
    failed_set = load_failed_episodes()

    with tqdm(total=len(round_robin_eps), desc="Recapping episodes", ncols=90) as pbar:
        for ep_path in round_robin_eps:
            if ep_path is None:
                pbar.update(1)
                continue
            filename = os.path.basename(ep_path)
            file_no_ext = os.path.splitext(filename)[0]

            # Skip if failed previously
            if filename in failed_set or ep_path in failed_set:
                pbar.set_postfix({"episode": filename, "status": "skipped (failed before)"})
                pbar.update(1)
                continue

            recap_gcs_path = f"{OUTPUTS_DIR}/{file_no_ext}/recap.mp4"
            try:
                if await file_exists(recap_gcs_path):
                    pbar.set_postfix({"episode": filename, "status": "recap exists"})
                    pbar.update(1)
                    continue

                index_gcs_path = f"{INDEXING_DIR}/{file_no_ext}/media_indexing.json"
                if not await file_exists(index_gcs_path):
                    pbar.set_postfix({"episode": filename, "status": "indexing..."})
                    print(f"\n[INFO] Indexing {filename}...")
                    pipeline = ComprehensionPipeline(ep_path)
                    await pipeline.run()

                pbar.set_postfix({"episode": filename, "status": "recapping..."})
                print(f"\n[INFO] Recapping {filename}...")
                fr_pipeline = FlexibleResponsePipeline(ep_path)
                await fr_pipeline.run(
                    user_prompt="I am a youtube channel that creates movie/tv recap videos. Give me an engaging, clear, approximately 5-minute recap of this episode for viewers who don't have time to watch it. Focus on major plot points, main characters, and key events. include a short introduction and conclusion with each recap. ",
                    video_response=True,
                    original_audio=True,
                    music=True,
                    narration_enabled=True,
                    aspect_ratio=0,
                    subtitles=True,
                    snap_to_beat=False,
                    output_path=f"{OUTPUTS_DIR}/{file_no_ext}/recap.mp4"
                )
                pbar.set_postfix({"episode": filename, "status": "done"})
            except Exception as e:
                error_msg = f"{filename} | {ep_path} | {type(e).__name__}: {str(e)}"
                print(f"[ERROR] Failed to recap {filename}: {e}")
                failed_episodes.append(error_msg)
                # Immediately write to disk
                with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(error_msg + "\n")
                pbar.set_postfix({"episode": filename, "status": "FAILED"})
            finally:
                pbar.update(1)

    if failed_episodes:
        print(f"\n[INFO] {len(failed_episodes)} episodes failed. See: {FAILED_LOG_PATH}")
    else:
        print("\n[INFO] All episodes recapped successfully!")

if __name__ == "__main__":
    asyncio.run(index_and_recap_all())
