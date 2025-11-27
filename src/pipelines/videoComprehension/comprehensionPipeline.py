import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import json
import asyncio

from lib.oss.storage_factory import get_storage_client
from lib.utils.media import (
    correct_segment_number_based_on_time,
    preprocess_long_video,
    preprocess_short_video,
    get_video_info,
    get_video_duration,
    clean_stale_tempdirs,
    download_and_cache_video,
)
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import BUCKET_NAME, VIDEO_EXTS, INDEXING_DIR
from src.pipelines.videoComprehension.tasks.rough_comprehension import RoughComprehension
from src.pipelines.videoComprehension.tasks.scene_by_scene_comprehension import SceneBySceneComprehension
from src.pipelines.videoComprehension.tasks.refine_story import RefineStory    # <-- updated import

class ComprehensionPipeline:
    def __init__(self, blob_path, start_fresh=False, debug_output_dir=None):
        # Accept both file or folder
        is_gcs_file = not blob_path.endswith("/") and Path(blob_path).suffix.lower() in VIDEO_EXTS
        self.is_gcs_file = is_gcs_file

        # Determine naming
        if is_gcs_file:
            self.cloud_storage_media_path = blob_path
            self.media_folder_name = Path(blob_path).stem
        else:
            self.cloud_storage_media_path = blob_path.rstrip("/") + "/"
            self.media_folder_name = os.path.basename(self.cloud_storage_media_path.rstrip("/"))

        self.cloud_storage_indexing_dir = f"{INDEXING_DIR}/{self.media_folder_name}/"
        self.indexing_file_path = self.cloud_storage_indexing_dir + "media_indexing.json"
        self.cloud_storage_client = get_storage_client()
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash")

        self.start_fresh = start_fresh
        self.debug_output_dir = debug_output_dir

        if self.start_fresh:
            print(f"[INFO] start_fresh enabled. Deleting: {self.cloud_storage_indexing_dir}")
            self.cloud_storage_client.delete_folder(BUCKET_NAME, self.cloud_storage_indexing_dir)

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.workdir, "full_videos")
        os.makedirs(self.videos_dir, exist_ok=True)
        print(f"[INFO] Temp directory created: {self.workdir}")

    def _get_video_files(self):
        files = []
        for f in os.listdir(self.videos_dir):
            path = Path(os.path.join(self.videos_dir, f))
            if path.suffix.lower() in VIDEO_EXTS:
                files.append(path)
        return sorted(files, key=lambda x: x.name)

    async def run_for_file(self, local_media_path, cloud_storage_media_path):
        media_name = os.path.basename(local_media_path)
        long_segments_dir = tempfile.mkdtemp()
        short_segments_dir = tempfile.mkdtemp()

        print("[INFO] Preprocessing long segments...")
        long_segments = await preprocess_long_video(
            local_media_path, long_segments_dir, interval_seconds=15 * 60, fps=1, crf=30)

        print(f"[INFO] Generated {len(long_segments)} long segments.")

        print("[INFO] Preprocessing short segments...")
        short_segments = await preprocess_long_video(
            local_media_path, short_segments_dir, interval_seconds=5 * 60, fps=1, crf=30)
        short_segments = correct_segment_number_based_on_time(long_segments, short_segments)
        print(f"[INFO] Generated {len(short_segments)} short segments.")

        print("[INFO] Starting rough comprehension...")
        rc = RoughComprehension(self.llm)
        rough_summary_draft, people = await rc(long_segments)
        print("[INFO] Rough comprehension complete.")

        if self.debug_output_dir:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            draft_path = os.path.join(self.debug_output_dir, f"{Path(local_media_path).stem}_rough_summary.txt")
            with open(draft_path, "w", encoding="utf-8") as f:
                f.write(rough_summary_draft)
            print(f"[DEBUG] Saved rough summary draft to {draft_path}")

        print("[INFO] Starting scene-by-scene comprehension...")
        sc = SceneBySceneComprehension(self.llm)
        scenes = await sc(short_segments, rough_summary_draft, people)
        print(f"[INFO] Scene-by-scene comprehension generated {len(scenes)} scenes.")

        print("[INFO] Refining story...")
        rs = RefineStory(self.llm)
        story_json, story_text = await rs(rough_summary_draft, scenes)
        print("[INFO] Refined story complete.")

        return {
            "name": media_name,
            "cloud_storage_path": cloud_storage_media_path,
            "story.txt": story_text,
            "story.json": story_json,
            "people.txt": people,
            "scenes.json": scenes,
        }

    async def run(self):
        if self.cloud_storage_client.path_exists(BUCKET_NAME, self.indexing_file_path) and not self.start_fresh:
            print(f"[SKIP] Indexing already exists at {self.indexing_file_path}")
            return

        media_files = []
        if self.cloud_storage_media_path.endswith("/") or not Path(self.cloud_storage_media_path).suffix.lower() in VIDEO_EXTS:
            cache_root = Path(".cache/comprehension_media")
            target_dir = cache_root / self.media_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            if not any(target_dir.iterdir()) or self.start_fresh:
                print(f"[INFO] Downloading media folder {self.cloud_storage_media_path} → {target_dir}")
                self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, str(target_dir))
            else:
                print(f"[CACHE] Using cached media folder {target_dir}")
            local_files = [
                str(target_dir / f)
                for f in os.listdir(target_dir)
                if Path(f).suffix.lower() in VIDEO_EXTS
            ]
            for local_media_path in local_files:
                cloud_path = self.cloud_storage_media_path.rstrip("/") + "/" + os.path.basename(local_media_path)
                print(f"[INFO] Processing file: {local_media_path}")
                media_files.append(await self.run_for_file(local_media_path, cloud_path))
        else:
            cache_root = Path(".cache/comprehension_media")
            cache_root.mkdir(parents=True, exist_ok=True)
            local_media_path = download_and_cache_video(
                self.cloud_storage_client,
                BUCKET_NAME,
                self.cloud_storage_media_path,
                str(cache_root),
            )
            print(f"[INFO] Processing file: {local_media_path}")
            media_files.append(await self.run_for_file(local_media_path, self.cloud_storage_media_path))

        indexing_object = {
            "media_files": media_files,
            "manifest": {
                "story.txt": "A linear summary of the story or events, broken into segments.",
                "story.json": "A linear summary of the story or events, broken into segments in json form.",
                "people.txt": "Descriptions of all people, characters, or key individuals and their relationships.",
                "scenes.json": "Scene metadata including timestamps and visual content."
            }
        }

        output_path = os.path.join(self.workdir, "media_indexing.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(indexing_object, f, ensure_ascii=False, indent=2)

        if self.debug_output_dir:
            debug_indexing_path = os.path.join(self.debug_output_dir, "media_indexing.json")
            shutil.copy(output_path, debug_indexing_path)
            print(f"[DEBUG] Saved indexing to {debug_indexing_path}")

        print(f"[INFO] Uploading indexing file to GCS: {self.indexing_file_path}")
        self.cloud_storage_client.upload_files(BUCKET_NAME, output_path, self.indexing_file_path)
        print(f"[SUCCESS] Uploaded indexing: {self.indexing_file_path}")
