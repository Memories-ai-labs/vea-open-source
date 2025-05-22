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
    preprocess_long_video,
    preprocess_short_video,
    get_video_info,
    get_video_duration,
    clean_stale_tempdirs
)
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.shortFormComprehension.tasks.general_comprehension import GeneralComprehension
from src.pipelines.shortFormComprehension.tasks.content_transcription import ContentTranscription


class ShortFormComprehensionPipeline:
    def __init__(self, cloud_storage_media_path: str, start_fresh: bool = False):
        self.cloud_storage_media_path = cloud_storage_media_path.rstrip("/") + "/"
        self.media_folder_name = os.path.basename(self.cloud_storage_media_path.rstrip("/"))
        self.cloud_storage_indexing_dir = f"indexing/{self.media_folder_name}/"
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.llm = GeminiGenaiManager()
        self.general_comprehension = GeneralComprehension(self.llm)
        self.content_transcription = ContentTranscription(self.llm)

        if start_fresh:
            print(f"[INFO] start_fresh enabled. Deleting: {self.cloud_storage_indexing_dir}")
            self.cloud_storage_client.delete_folder(BUCKET_NAME, self.cloud_storage_indexing_dir)

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.workdir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)
        print(f"[INFO] Temp directory created: {self.workdir}")

    def _get_video_files(self):
        files = []
        for f in os.listdir(self.videos_dir):
            path = Path(os.path.join(self.videos_dir, f))
            if path.suffix.lower() in [".mp4", ".mov", ".mkv"]:
                files.append(path)
        return sorted(files, key=lambda x: x.name)

    async def run(self):
        output_index_path = os.path.join(self.workdir, "media_indexing.json")
        gcs_output_path = self.cloud_storage_indexing_dir + "media_indexing.json"

        if self.cloud_storage_client.path_exists(BUCKET_NAME, gcs_output_path):
            print(f"[SKIP] media_indexing.json already exists at: {gcs_output_path}")
            return

        print(f"[INFO] Downloading videos from GCS path: {self.cloud_storage_media_path}")
        self.cloud_storage_client.download_files(
            BUCKET_NAME, self.cloud_storage_media_path, self.videos_dir
        )

        # Step 1: Detect video files and initialize entries
        media_index = {
            "media_files": [],
            "manifest": {
                "scene_descriptions.json": "Describes what happens every 10 seconds in each video segment.",
                "transcription.json": "Verbatim speech and visible text from each segment, with timestamps."
            }
        }

        media_entry_map = {}
        all_segments = []
        chunks_dir = os.path.join(self.videos_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        for video_path in self._get_video_files():
            video_name = video_path.name
            gcs_path = self.cloud_storage_media_path + video_name

            duration = get_video_duration(str(video_path))
            if duration > 600:
                print(f"[INFO] Splitting long video: {video_name} ({duration:.2f}s)")
                segments = await preprocess_long_video(
                    input_path=str(video_path),
                    output_dir=chunks_dir,
                    interval_seconds=600,
                    fps=1,
                    crf=30,
                    target_height=480
                )
            else:
                print(f"[INFO] Downsampling short video: {video_name} ({duration:.2f}s)")
                segments = [await preprocess_short_video(
                    input_path=str(video_path),
                    output_dir=chunks_dir,
                    crf=30,
                    target_height=480,
                    fps=1
                )]

            all_segments.extend(segments)

            media_entry = {
                "name": video_name,
                "cloud_storage_path": gcs_path,
                "scene_descriptions.json": [],
                "transcription.json": []
            }
            media_index["media_files"].append(media_entry)
            media_entry_map[str(video_path)] = media_entry

        print(f"[INFO] Total segments across all media: {len(all_segments)}")

        # Step 2: Run comprehension + transcription
        print(f"[INFO] Running general comprehension...")
        scene_dict = await self.general_comprehension(all_segments)

        print(f"[INFO] Running content transcription...")
        transcription_dict = await self.content_transcription(all_segments)

        # Step 3: Attach results
        for parent_path, scene_data in scene_dict.items():
            filename = Path(parent_path).name
            entry = next((e for e in media_index["media_files"] if e["name"] == filename), None)
            if entry:
                entry["scene_descriptions.json"].extend(scene_data)

        for parent_path, transcription_data in transcription_dict.items():
            filename = Path(parent_path).name
            entry = next((e for e in media_index["media_files"] if e["name"] == filename), None)
            if entry:
                entry["transcription.json"].extend(transcription_data)

        # Step 4: Save and upload
        with open(output_index_path, "w", encoding="utf-8") as f:
            json.dump(media_index, f, ensure_ascii=False, indent=2)

        self.cloud_storage_client.upload_files(BUCKET_NAME, output_index_path, gcs_output_path)
        print(f"[SUCCESS] Uploaded media indexing file: {gcs_output_path}")
