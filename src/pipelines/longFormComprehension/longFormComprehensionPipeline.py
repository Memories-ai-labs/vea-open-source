import os
import tempfile
import shutil
from pathlib import Path
import asyncio
import glob
import json

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.utils.media import preprocess_long_video, clean_stale_tempdirs
from src.config import CREDENTIAL_PATH, BUCKET_NAME

from src.pipelines.longFormComprehension.tasks.rough_comprehension import RoughComprehension
from src.pipelines.longFormComprehension.tasks.scene_by_scene_comprehension import SceneBySceneComprehension
from src.pipelines.longFormComprehension.tasks.refine_plot_summary import RefinePlotSummary
from src.pipelines.longFormComprehension.tasks.artistic_analysis import ArtisticAnalysis

class LongFormComprehensionPipeline:
    def __init__(self, cloud_storage_media_path, start_fresh: bool = False):
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.cloud_storage_root = f"indexing/{self.media_base_name}/"
        self.indexing_file_path = self.cloud_storage_root + "media_indexing.json"

        self.start_fresh = start_fresh
        self.llm = GeminiGenaiManager("gemini-2.0-flash")
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.long_segments_dir = tempfile.mkdtemp()
        self.short_segments_dir = tempfile.mkdtemp()
        self.local_media_path = os.path.join(tempfile.mkdtemp(), self.media_name)

    async def run(self):
        if self.cloud_storage_client.path_exists(BUCKET_NAME, self.indexing_file_path) and not self.start_fresh:
            print(f"[SKIP] Indexing already exists at {self.indexing_file_path}")
            return

        print(f"[INFO] Downloading media file: {self.cloud_storage_media_path}")
        self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, self.local_media_path)

        print("[INFO] Preprocessing long segments...")
        long_segments = await preprocess_long_video(
            self.local_media_path, self.long_segments_dir, interval_seconds=15 * 60, fps=1, crf=30)
        long_segment_paths = [Path(d["path"]) for d in long_segments]
        print(f"[INFO] Generated {len(long_segments)} long segments.")

        print("[INFO] Preprocessing short segments...")
        short_segments = await preprocess_long_video(
            self.local_media_path, self.short_segments_dir, interval_seconds=5 * 60, fps=1, crf=30)
        print(f"[INFO] Generated {len(short_segments)} short segments.")

        print("[INFO] Starting rough comprehension...")
        rc = RoughComprehension(self.llm)
        rough_summary_draft, characters = await rc(long_segments)
        print("[INFO] Rough comprehension complete.")

        print("[INFO] Starting scene-by-scene comprehension...")
        sc = SceneBySceneComprehension(self.llm)
        scenes = await sc(short_segments, rough_summary_draft, characters)
        print(f"[INFO] Scene-by-scene comprehension generated {len(scenes)} scenes.")

        print("[INFO] Refining plot summary...")
        rp = RefinePlotSummary(self.llm)
        plot_json, plot_text = await rp(rough_summary_draft, scenes)
        print("[INFO] Refined plot summary complete.")

        print("[INFO] Performing artistic analysis...")
        aa = ArtisticAnalysis(self.llm)
        artistic_annotations = await aa(long_segments, plot_text)
        print(f"[INFO] Artistic analysis complete with {len(artistic_annotations)} entries.")

        print("[INFO] Assembling final indexing structure...")
        indexing_object = {
            "media_files": [
                {
                    "name": self.media_name,
                    "cloud_storage_path": self.cloud_storage_media_path,
                    "plot.txt": plot_text,
                    "plot.json": plot_json,
                    "characters.txt": characters,
                    "scenes.json": scenes,
                    "artistic.json": artistic_annotations,
                }
            ],
            "manifest": {
                "plot.txt": "A linear summary of the plot, broken into segments.",
                "plot.json": "A linear summary of the plot, broken into segments in json form.",
                "characters.txt": "Descriptions of all relevant characters and their relationships.",
                "scenes.json": "Scene metadata including timestamps and visual content.",
                "artistic.json": "Artistic breakdown of visual and audio elements of the movie.",
            }
        }

        output_path = os.path.join(self.workdir, "media_indexing.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(indexing_object, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Uploading indexing file to GCS: {self.indexing_file_path}")
        self.cloud_storage_client.upload_files(BUCKET_NAME, output_path, self.indexing_file_path)
        print(f"[SUCCESS] Uploaded indexing: {self.indexing_file_path}")
