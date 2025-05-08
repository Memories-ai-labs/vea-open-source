import os
import json
import tempfile
from pathlib import Path
import asyncio
import shutil
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from google.cloud.exceptions import NotFound

from src.config import CREDENTIAL_PATH, BUCKET_NAME

from src.pipelines.movieRecapEditing.tasks.choose_clip_for_recap import ChooseClipForRecap
from src.pipelines.movieRecapEditing.tasks.generate_narration import GenerateNarrationForClips
from src.pipelines.movieRecapEditing.tasks.music_selection import MusicSelection
from src.pipelines.movieRecapEditing.tasks.edit_recap_video import EditMovieRecapVideo


class MovieRecapEditingPipeline:
    def __init__(self, cloud_storage_media_path):
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]

        self.cloud_storage_indexing_dir = f"indexing/{self.media_base_name}/"
        self.output_cloud_storage_dir = f"outputs/{self.media_base_name}/"

        self.llm = GeminiGenaiManager()
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

        self.clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.text_to_voice_dir = tempfile.mkdtemp()
        self.final_output_path = os.path.join(self.workdir, "recap.mp4")
        self.final_output_cspath = os.path.join(self.output_cloud_storage_dir, "recap.mp4")

    def clean_stale_tempdirs(self):
        print("Cleaning stale temp directories...")
        tmp_root = tempfile.gettempdir()  # Usually /tmp
        for name in os.listdir(tmp_root):
            path = os.path.join(tmp_root, name)
            if os.path.isdir(path) and name.startswith("tmp"):
                try:
                    shutil.rmtree(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    async def run(self):
        # Helper to download GCS file to local path
        def gcs_download(gcs_rel_path, local_filename):
            local_path = os.path.join(self.workdir, local_filename)
            self.cloud_storage_client.download_files(
                BUCKET_NAME,
                self.cloud_storage_indexing_dir + gcs_rel_path,
                local_path
            )
            return local_path

        # 1. Download summary and scene info
        plot_json_path = gcs_download("plot.json", "plot.json")
        scenes_path = gcs_download("scenes.json", "scenes.json")
        with open(plot_json_path, "r", encoding="utf-8") as f:
            plot_json = json.load(f)
        with open(scenes_path, "r", encoding="utf-8") as f:
            scenes = json.load(f)

        # 2. Choose Clips
        clip_selector = ChooseClipForRecap(self.llm)
        chosen_clips = await clip_selector(plot_json, scenes)

        # 3. Generate Narration
        narrator = GenerateNarrationForClips(self.text_to_voice_dir)
        await narrator(chosen_clips)

        # 4. Choose Music
        
        full_plot_text = "\n".join([entry["sentence_text"] for entry in plot_json])
        music_selector = MusicSelection(self.llm, self.workdir)
        chosen_music_path = await music_selector(full_plot_text)

        # 5. Create Video
        self.local_media_path = os.path.join(self.workdir, self.media_name)
        self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, self.local_media_path)
        editor = EditMovieRecapVideo(self.final_output_path)
        await editor(
            chosen_clips,
            self.local_media_path, 
            self.text_to_voice_dir,
            chosen_music_path
        )

        # 6. Upload Final Recap
        self.cloud_storage_client.upload_files(
            BUCKET_NAME,
            self.final_output_path,
            self.final_output_cspath
        )

        print(f"[INFO] Final video uploaded")
        return self.final_output_cspath