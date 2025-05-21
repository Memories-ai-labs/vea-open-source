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
from google.cloud.exceptions import NotFound

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

        self.start_fresh = start_fresh
        self.llm = GeminiGenaiManager("gemini-2.0-flash")
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.long_segments_dir = tempfile.mkdtemp()
        self.short_segments_dir = tempfile.mkdtemp()
        self.local_media_path = os.path.join(tempfile.mkdtemp(), self.media_name)

    async def run(self):
        if self.start_fresh:
            print(f"[INFO] Start fresh: Deleting existing GCS folder at {self.cloud_storage_root}")
            self.cloud_storage_client.delete_folder(BUCKET_NAME, self.cloud_storage_root)

        # preprocess media
        long_segment_cspath = self.cloud_storage_root + "long_segments/"
        if self.cloud_storage_client.path_exists(BUCKET_NAME, long_segment_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, long_segment_cspath, self.long_segments_dir)
            long_segment_paths = [Path(f) for f in glob.glob(os.path.join(self.long_segments_dir, "*")) if os.path.isfile(f)]
        else:
            # Download media file from cloud_storage to local temp folder
            self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, self.local_media_path)

            long_segment_paths = await preprocess_long_video(self.local_media_path, self.long_segments_dir, interval_seconds=15*60, fps=1, crf=30)
            self.cloud_storage_client.upload_files(BUCKET_NAME, self.long_segments_dir, long_segment_cspath)

        short_segment_cspath = self.cloud_storage_root + "short_segments/"
        if self.cloud_storage_client.path_exists(BUCKET_NAME, short_segment_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, short_segment_cspath, self.short_segments_dir)
            short_segment_paths = [Path(f) for f in glob.glob(os.path.join(self.short_segments_dir, "*")) if os.path.isfile(f)]
        else:
            self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, self.local_media_path)

            short_segment_paths = await preprocess_long_video(self.local_media_path, self.short_segments_dir, interval_seconds=5*60, fps=1, crf=30)
            self.cloud_storage_client.upload_files(BUCKET_NAME, self.short_segments_dir, short_segment_cspath)

        # perform rough comprehension
        rough_summary_draft_cspath = self.cloud_storage_root + "rough_summary_draft.txt"
        rough_summary_draft_path = os.path.join(self.workdir, "rough_summary_draft.txt")
        characters_cspath = self.cloud_storage_root + "characters.txt"
        characters_path = os.path.join(self.workdir, "characters.txt")
        if self.cloud_storage_client.path_exists(BUCKET_NAME, rough_summary_draft_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, rough_summary_draft_cspath, rough_summary_draft_path)
            self.cloud_storage_client.download_files(BUCKET_NAME, characters_cspath, characters_path)
            with open(rough_summary_draft_path, "r", encoding="utf-8") as f:
                rough_summary_draft = f.read()
            with open(characters_path, "r", encoding="utf-8") as f:
                characters = f.read()
        else:
            rc = RoughComprehension(self.llm)
            rough_summary_draft, characters = await rc(long_segment_paths)
            with open(rough_summary_draft_path, "w", encoding="utf-8") as f:
                f.write(rough_summary_draft)
            with open(characters_path, "w", encoding="utf-8") as f:
                f.write(characters)
            self.cloud_storage_client.upload_files(BUCKET_NAME, rough_summary_draft_path, rough_summary_draft_cspath)
            self.cloud_storage_client.upload_files(BUCKET_NAME, characters_path, characters_cspath)

        # perform scene by scene comprehension
        scenes_cspath = self.cloud_storage_root + "scenes.json"
        scenes_path = os.path.join(self.workdir, "scenes.json")
        if self.cloud_storage_client.path_exists(BUCKET_NAME, scenes_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, scenes_cspath, scenes_path)
            with open(scenes_path, "r", encoding="utf-8") as f:
                scenes = json.load(f)
        else:
            sc = SceneBySceneComprehension(self.llm)
            scenes = await sc(short_segment_paths, long_segment_paths, rough_summary_draft, characters)
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(scenes, f, ensure_ascii=False, indent=4)
            self.cloud_storage_client.upload_files(BUCKET_NAME, scenes_path, scenes_cspath)
        
        # perform refined plot summary
        plot_json_cspath = self.cloud_storage_root + "plot.json"
        plot_text_cspath = self.cloud_storage_root + "plot.txt"
        plot_json_path = os.path.join(self.workdir, "plot.json")
        plot_text_path = os.path.join(self.workdir, "plot.txt")
        if self.cloud_storage_client.path_exists(BUCKET_NAME, plot_json_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, plot_json_cspath, plot_json_path)
            self.cloud_storage_client.download_files(BUCKET_NAME, plot_text_cspath, plot_text_path)
            with open(plot_json_path, "r", encoding="utf-8") as f:
                plot_json = json.load(f)
            with open(plot_text_path, "r", encoding="utf-8") as f:
                plot_text = f.read()
        else:
            rp = RefinePlotSummary(self.llm)
            plot_json, plot_text = await rp(rough_summary_draft, scenes)
            with open(plot_json_path, "w", encoding="utf-8") as f:
                json.dump(plot_json, f, ensure_ascii=False, indent=4)
            with open(plot_text_path, "w", encoding="utf-8") as f:
                f.write(plot_text)
            self.cloud_storage_client.upload_files(BUCKET_NAME, plot_json_path, plot_json_cspath)
            self.cloud_storage_client.upload_files(BUCKET_NAME, plot_text_path, plot_text_cspath)

        # --- Artistic analysis ---
        artistic_cspath = self.cloud_storage_root + "artistic.json"
        artistic_path = os.path.join(self.workdir, "artistic.json")
        if self.cloud_storage_client.path_exists(BUCKET_NAME, artistic_cspath):
            self.cloud_storage_client.download_files(BUCKET_NAME, artistic_cspath, artistic_path)
            with open(artistic_path, "r", encoding="utf-8") as f:
                artistic_annotations = json.load(f)
        else:
            
            aa = ArtisticAnalysis(self.llm)
            artistic_annotations = await aa(short_segment_paths, plot_text)
            with open(artistic_path, "w", encoding="utf-8") as f:
                json.dump(artistic_annotations, f, ensure_ascii=False, indent=4)
            self.cloud_storage_client.upload_files(BUCKET_NAME, artistic_path, artistic_cspath)