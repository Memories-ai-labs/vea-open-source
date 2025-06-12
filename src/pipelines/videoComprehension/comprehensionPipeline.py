import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import json
import asyncio

from lib.oss.auth import credentials_from_file
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.utils.media import (
    correct_segment_number_based_on_time,
    preprocess_long_video,
    preprocess_short_video,
    get_video_info,
    get_video_duration,
    clean_stale_tempdirs
)
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import CREDENTIAL_PATH, BUCKET_NAME, VIDEO_EXTS
from src.pipelines.videoComprehension.tasks.rough_comprehension import RoughComprehension
from src.pipelines.videoComprehension.tasks.scene_by_scene_comprehension import SceneBySceneComprehension
from src.pipelines.videoComprehension.tasks.refine_story import RefineStory    # <-- updated import

class ComprehensionPipeline:
    def __init__(self, blob_path, start_fresh=False):
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

        self.cloud_storage_indexing_dir = f"indexing/{self.media_folder_name}/"
        self.indexing_file_path = self.cloud_storage_indexing_dir + "media_indexing.json"
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.llm = GeminiGenaiManager()

        if start_fresh:
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
        # Preprocessing and comprehension logic (what you have in run())
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

        print("[INFO] Starting scene-by-scene comprehension...")
        sc = SceneBySceneComprehension(self.llm)
        scenes = await sc(short_segments, rough_summary_draft, people)
        print(f"[INFO] Scene-by-scene comprehension generated {len(scenes)} scenes.")

        print("[INFO] Refining story...")
        rs = RefineStory(self.llm)
        story_json, story_text = await rs(rough_summary_draft, scenes)
        print("[INFO] Refined story complete.")

        # Assemble file-specific indexing entry
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

        # Download single file or all files in folder
        media_files = []
        if self.cloud_storage_media_path.endswith("/") or not Path(self.cloud_storage_media_path).suffix.lower() in VIDEO_EXTS:
            # Folder: download all video files to temp
            videos_dir = os.path.join(tempfile.mkdtemp(), "videos")
            self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, videos_dir)
            local_files = [
                os.path.join(videos_dir, f)
                for f in os.listdir(videos_dir)
                if Path(f).suffix.lower() in VIDEO_EXTS
            ]
            for local_media_path in local_files:
                cloud_path = self.cloud_storage_media_path.rstrip("/") + "/" + os.path.basename(local_media_path)
                print(f"[INFO] Processing file: {local_media_path}")
                media_files.append(await self.run_for_file(local_media_path, cloud_path))
        else:
            # Single file: download to temp
            local_media_path = os.path.join(tempfile.mkdtemp(), os.path.basename(self.cloud_storage_media_path))
            self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, local_media_path)
            print(f"[INFO] Processing file: {local_media_path}")
            media_files.append(await self.run_for_file(local_media_path, self.cloud_storage_media_path))

        # Assemble manifest
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

        print(f"[INFO] Uploading indexing file to GCS: {self.indexing_file_path}")
        self.cloud_storage_client.upload_files(BUCKET_NAME, output_path, self.indexing_file_path)
        print(f"[SUCCESS] Uploaded indexing: {self.indexing_file_path}")
