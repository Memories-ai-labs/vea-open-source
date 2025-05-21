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
        self.media_folder_name = os.path.basename(cloud_storage_media_path.rstrip("/"))
        self.cloud_storage_media_path = cloud_storage_media_path.rstrip("/") + "/"
        self.cloud_storage_indexing_dir = f"free_indexing/{self.media_folder_name}/"
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

    def _get_capture_time(self, video_path: Path) -> str | None:
        try:
            metadata = get_video_info(video_path)
            return metadata["format"].get("tags", {}).get("creation_time", None)
        except Exception:
            return None

    async def run(self):
        gcs_output_path = self.cloud_storage_indexing_dir + "general_comprehension.json"

        if self.cloud_storage_client.path_exists(BUCKET_NAME, gcs_output_path):
            print(f"[SKIP] general_comprehension.json already exists at: {gcs_output_path}")
            return

        print(f"[INFO] Downloading videos from GCS path: {self.cloud_storage_media_path}")
        self.cloud_storage_client.download_files(
            BUCKET_NAME, self.cloud_storage_media_path, self.videos_dir
        )

        files = []
        for f in os.listdir(self.videos_dir):
            path = Path(os.path.join(self.videos_dir, f))
            if path.suffix.lower() in [".mp4", ".mov", ".mkv"]:
                capture_time = self._get_capture_time(path)
                files.append((path, capture_time))

        files.sort(key=lambda x: (x[1] if x[1] else x[0].name))

        all_segments = []
        for file_path, capture_time in files:
            duration = get_video_duration(str(file_path))
            output_dir = os.path.join(self.videos_dir, "chunks")
            if duration > 600:
                print(f"[INFO] Splitting long video: {file_path.name} ({duration:.2f}s)")
                clips = await preprocess_long_video(
                    input_path=str(file_path),
                    output_dir=output_dir,
                    interval_seconds=600,
                    fps=1,
                    crf=30,
                    target_height=480
                )
            else:
                print(f"[INFO] Downsampling short video: {file_path.name} ({duration:.2f}s)")
                clips = [await preprocess_short_video(
                    input_path=str(file_path),
                    output_dir=output_dir,
                    crf=30,
                    target_height=480,
                    fps=1
                )]
            for i, clip in enumerate(clips):
                all_segments.append((clip, file_path.name, i, capture_time))

        print(f"[INFO] Total preprocessed segments: {len(all_segments)}")

        comprehension_entries = []
        for clip_path, original_file, seg_idx, captured_time in all_segments:
            try:
                print(f"[INFO] Running general comprehension for: {clip_path.name}")
                description = await self.general_comprehension(
                    video_path=clip_path,
                    metadata={
                        "file_name": original_file,
                        "segment_number": seg_idx,
                    }
                )
                print(f"[INFO] Running content transcription for: {clip_path.name}")
                transcription = await self.content_transcription(clip_path)

                comprehension_entries.append({
                    "file_name": original_file,
                    "segment_number": seg_idx,
                    "captured_time": captured_time,
                    "scene_descriptions": description,
                    "transcription": transcription,
                })

            except Exception as e:
                print(f"[ERROR] Failed on {clip_path.name}: {e}")

        master_json_path = os.path.join(self.workdir, "general_comprehension.json")
        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(comprehension_entries, f, ensure_ascii=False, indent=2)

        self.cloud_storage_client.upload_files(BUCKET_NAME, master_json_path, gcs_output_path)
        print(f"[SUCCESS] Uploaded full comprehension to: {gcs_output_path}")
