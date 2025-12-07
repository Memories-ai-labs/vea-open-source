import os
import shutil
import tempfile
import logging
from pathlib import Path
import json
from typing import Optional, Callable
import asyncio

from lib.oss.storage_factory import get_storage_client
from lib.utils.media import (
    get_video_duration,
    clean_stale_tempdirs,
    download_and_cache_video,
)
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.llm.MemoriesAiManager import MemoriesAiManager
from src.config import BUCKET_NAME, VIDEO_EXTS, INDEXING_DIR
from src.pipelines.videoComprehension.tasks.caption_comprehension import CaptionComprehension

logger = logging.getLogger(__name__)


class ComprehensionPipeline:
    """
    Video comprehension pipeline using Memories.ai hybrid approach.

    Uses Chat API for upload/indexing/summary/people, and Caption API for scene descriptions.
    """

    def __init__(
        self,
        blob_path: str,
        start_fresh: bool = False,
        debug_output_dir: Optional[str] = None,
        memories_manager: MemoriesAiManager = None,
        caption_callback_url: str = None,
        register_caption_callback: Callable[[str], asyncio.Future] = None,
    ):
        """
        Initialize the ComprehensionPipeline.

        Args:
            blob_path: Path to video file or folder in cloud storage
            start_fresh: If True, delete existing indexing and re-process
            debug_output_dir: Optional directory to save debug outputs
            memories_manager: MemoriesAiManager instance (required)
            caption_callback_url: Public webhook URL for Caption API callbacks (required)
            register_caption_callback: Function to register pending callback (required)
        """
        if not memories_manager:
            raise ValueError("memories_manager is required")
        if not caption_callback_url:
            raise ValueError("caption_callback_url is required")
        if not register_caption_callback:
            raise ValueError("register_caption_callback is required")

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

        # Store settings
        self.start_fresh = start_fresh
        self.debug_output_dir = debug_output_dir

        # Memories.ai integration
        self.memories_manager = memories_manager
        self.caption_callback_url = caption_callback_url
        self.register_caption_callback = register_caption_callback

        logger.info("[COMPREHENSION] Using Memories.ai hybrid approach (Chat + Caption API)")
        if self.start_fresh:
            logger.info("[COMPREHENSION] start_fresh enabled - will re-upload to Memories.ai")

        if self.start_fresh:
            print(f"[INFO] start_fresh enabled. Deleting: {self.cloud_storage_indexing_dir}")
            self.cloud_storage_client.delete_folder(BUCKET_NAME, self.cloud_storage_indexing_dir)

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.videos_dir = os.path.join(self.workdir, "full_videos")
        os.makedirs(self.videos_dir, exist_ok=True)
        print(f"[INFO] Temp directory created: {self.workdir}")

    async def _process_video(
        self,
        local_media_path: str,
        cloud_storage_media_path: str,
    ) -> dict:
        """
        Process a single video file using the hybrid Chat + Caption API approach.

        Args:
            local_media_path: Path to local video file
            cloud_storage_media_path: Path in cloud storage

        Returns:
            Dict with media file metadata and comprehension results
        """
        media_name = os.path.basename(local_media_path)
        stem = Path(local_media_path).stem
        logger.info(f"[COMPREHENSION] Processing: {media_name}")

        # Create Caption comprehension task (hybrid: Chat + Caption API)
        cc = CaptionComprehension(
            memories_manager=self.memories_manager,
            gemini_llm=self.llm,
            callback_url=self.caption_callback_url,
            register_callback=self.register_caption_callback,
            scene_interval_seconds=20,
            force_reupload=self.start_fresh,
        )

        # Run comprehension
        result = await cc(local_media_path)

        if self.debug_output_dir:
            os.makedirs(self.debug_output_dir, exist_ok=True)

            # Save summary
            summary_path = os.path.join(self.debug_output_dir, f"{stem}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(result["rough_summary"])
            print(f"[DEBUG] Saved summary to {summary_path}")

            # Save people descriptions
            people_path = os.path.join(self.debug_output_dir, f"{stem}_people.txt")
            with open(people_path, "w", encoding="utf-8") as f:
                f.write(result["people"])
            print(f"[DEBUG] Saved people to {people_path}")

            # Save scenes
            scenes_path = os.path.join(self.debug_output_dir, f"{stem}_scenes.json")
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(result["scenes"], f, ensure_ascii=False, indent=2)
            print(f"[DEBUG] Saved scenes to {scenes_path}")

        return {
            "name": media_name,
            "cloud_storage_path": cloud_storage_media_path,
            "story.txt": result["rough_summary"],
            "story.json": None,
            "people.txt": result["people"],
            "scenes.json": result["scenes"],
            "memories_video_no": result.get("video_no"),
        }

    async def run(self):
        """Run the comprehension pipeline."""
        if self.cloud_storage_client.path_exists(BUCKET_NAME, self.indexing_file_path) and not self.start_fresh:
            print(f"[SKIP] Indexing already exists at {self.indexing_file_path}")
            return

        media_files = []

        # Handle folder of videos
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
                media_files.append(await self._process_video(local_media_path, cloud_path))

        # Handle single video file
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
            media_files.append(await self._process_video(local_media_path, self.cloud_storage_media_path))

        # Build indexing object
        indexing_object = {
            "media_files": media_files,
            "manifest": {
                "story.txt": "A linear summary of the story or events.",
                "story.json": "Structured story data (optional, may be null).",
                "people.txt": "Descriptions of all people, characters, or key individuals.",
                "scenes.json": "Scene metadata including timestamps and visual content."
            }
        }

        # Save locally
        output_path = os.path.join(self.workdir, "media_indexing.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(indexing_object, f, ensure_ascii=False, indent=2)

        if self.debug_output_dir:
            debug_indexing_path = os.path.join(self.debug_output_dir, "media_indexing.json")
            shutil.copy(output_path, debug_indexing_path)
            print(f"[DEBUG] Saved indexing to {debug_indexing_path}")

        # Save to storage
        print(f"[INFO] Saving indexing file to: {self.indexing_file_path}")
        self.cloud_storage_client.upload_files(BUCKET_NAME, output_path, self.indexing_file_path)
        print(f"[SUCCESS] Saved indexing: {self.indexing_file_path}")
