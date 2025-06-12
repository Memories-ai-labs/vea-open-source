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
    split_video_into_segments,
    extract_audio_from_video,
    extract_images_ffmpeg,
    preprocess_long_video,
    preprocess_short_video,
    get_video_info,
    get_video_duration,
    clean_stale_tempdirs
)
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import CREDENTIAL_PATH, BUCKET_NAME, VIDEO_EXTS
from src.pipelines.videoComprehension.tasks.transcription import Transcription
from src.pipelines.videoComprehension.tasks.rough_comprehension import RoughComprehension
from src.pipelines.videoComprehension.tasks.scene_by_scene_comprehension import SceneBySceneComprehension
from pipelines.videoComprehension.tasks.refine_story import RefinePlotSummary
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

    async def run(self):
        output_index_path = os.path.join(self.workdir, "media_indexing.json")
        gcs_output_path = self.cloud_storage_indexing_dir + "media_indexing.json"

        if self.cloud_storage_client.path_exists(BUCKET_NAME, gcs_output_path):
            print(f"[SKIP] media_indexing.json already exists at: {gcs_output_path}")
            return

        print(f"[INFO] Downloading videos from GCS path: {self.cloud_storage_media_path}")

        # Download logic handles single file or folder
        if self.is_gcs_file:
            # Single file: download just this file to videos_dir
            filename = os.path.basename(self.cloud_storage_media_path)
            local_path = os.path.join(self.videos_dir, filename)
            self.cloud_storage_client.download_files(
                BUCKET_NAME, self.cloud_storage_media_path, local_path
            )
        else:
            # Folder: download all files to videos_dir
            self.cloud_storage_client.download_files(
                BUCKET_NAME, self.cloud_storage_media_path, self.videos_dir
            )

        # set up index
        media_index = {
            "media_files": [],
        }

        for video_path in self._get_video_files():
            video_name = video_path.name
            video_folder = os.path.join(self.workdir, Path(video_name).stem)
            os.makedirs(video_folder, exist_ok=True)
            gcs_path = (
                self.cloud_storage_media_path
                if self.is_gcs_file
                else self.cloud_storage_media_path + video_name
            )
            media_entry = {
                "name": video_name,
                "cloud_storage_path": gcs_path,
                "transcript": []
            }
            
            # --- Split video and extract audio for each segment ---
            segments_dir = os.path.join(video_folder, "segments")
            segment_video_paths = split_video_into_segments(str(video_path), segments_dir, max_seconds=10*60)
            transcriber = Transcription(self.llm)
            for idx, seg_video_path in enumerate(segment_video_paths):
                audio_folder = os.path.join(video_folder, f"audio_seg_{idx}")
                os.makedirs(audio_folder, exist_ok=True)
                seg_audio_path = os.path.join(audio_folder, "audio.mp3")
                extract_audio_from_video(str(seg_video_path), seg_audio_path)
                # Gemini transcription returns structured directly
                transcript = await transcriber(Path(seg_audio_path))
                media_entry["transcript"].extend(transcript)

            media_index["media_files"].append(media_entry)

        with open("debug_indexing.json", "w") as f:
            json.dump(media_index, f, indent=2, ensure_ascii=False)

        # run transcription and media type classification on all files
