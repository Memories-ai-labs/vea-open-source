import os
import json
import tempfile
from pathlib import Path
import asyncio
import shutil
from pydub import AudioSegment

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.utils.media import clean_stale_tempdirs

from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.movieRecapEditing.tasks.user_customization import UserCustomization
from src.pipelines.movieRecapEditing.tasks.choose_clip_for_recap import ChooseClipForRecap
from src.pipelines.movieRecapEditing.tasks.translate_recap import TranslateRecap
from src.pipelines.common.generate_narration_audio import GenerateNarrationAudio
from src.pipelines.common.music_selection import MusicSelection
from src.pipelines.common.edit_video_response import EditVideoResponse


class MovieRecapEditingPipeline:
    def __init__(self, cloud_storage_media_path):
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.cloud_storage_indexing_dir = f"indexing/{self.media_base_name}/"

        self.llm = GeminiGenaiManager()
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

        clean_stale_tempdirs()
        self.workdir = tempfile.mkdtemp()
        self.text_to_voice_dir = tempfile.mkdtemp()

    async def run(self, user_context=None, user_prompt=None, output_language="English", user_music=None):
        # Helper to download GCS file to local path
        def gcs_download(gcs_rel_path, local_filename):
            local_path = os.path.join(self.workdir, local_filename)
            self.cloud_storage_client.download_files(
                BUCKET_NAME,
                self.cloud_storage_indexing_dir + gcs_rel_path,
                local_path
            )
            return local_path

        # 1. Download unified media_indexing.json
        indexing_json_path = gcs_download("media_indexing.json", "media_indexing.json")
        with open(indexing_json_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        # 2. Extract media data for this movie
        media_entry = next((entry for entry in index_data["media_files"] if entry["name"] == self.media_name), None)
        if not media_entry:
            raise ValueError(f"No media entry found for: {self.media_name}")

        manifest = index_data.get("manifest", {})
        plot_json = media_entry.get("plot.json", [])
        scenes = media_entry.get("scenes.json", [])
        characters = media_entry.get("characters.txt", "")
        artistic = media_entry.get("artistic.json", {})
        filename = media_entry.get("name")
        cloud_storage_media_path = media_entry.get("cloud_storage_path")

        # 3. Run User Customization
        customizer = UserCustomization(self.llm)
        customized_plot_json = await customizer(
            user_context=user_context,
            user_instruction=user_prompt,
            plot_json=plot_json,
            scenes=scenes,
            characters=characters,
            artistic_elements=artistic
        )

        # 4. Choose Clips
        clip_selector = ChooseClipForRecap(self.llm)
        chosen_clips = await clip_selector(customized_plot_json, scenes, filename, cloud_storage_media_path)

        # # 5. Translate Clips
        # translator = TranslateRecap(self.llm)
        # chosen_clips = await translator(chosen_clips, output_language)

        # 6. Generate Narration
        narrator = GenerateNarrationAudio(self.text_to_voice_dir)
        await narrator(chosen_clips)

        # 7. Choose Music
        if not user_music:
            full_plot_text = "\n".join([entry["sentence_text"] for entry in plot_json])
            music_selector = MusicSelection(self.llm, self.workdir, user_prompt)
            chosen_music_path = await music_selector(full_plot_text)
        else:
            # If user provided specific music, download it
            local_music_path = os.path.join(self.workdir, "user_music.mp3")
            self.cloud_storage_client.download_files(
                BUCKET_NAME,
                user_music,
                local_music_path
            )
            original_audio = AudioSegment.from_mp3(local_music_path)
            one_hour_ms = 60 * 60 * 1000
            loop_count = one_hour_ms // len(original_audio) + 1
            long_audio = (original_audio * loop_count)[:one_hour_ms]

            chosen_music_path = os.path.join(self.workdir, "user_music_1hour_loop.mp3")
            long_audio.export(chosen_music_path, format="mp3")

        # 8. Create Video
        # self.local_media_path = os.path.join(self.workdir, self.media_name)
        # self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, self.local_media_path)
        final_output_path = os.path.join(self.workdir, "recap.mp4")
        editor = EditVideoResponse(
            output_path=final_output_path,
            gcs_client=self.cloud_storage_client,
            bucket_name=BUCKET_NAME,
            workdir=self.workdir
            )
        
        await editor(
            clips=chosen_clips,
            narration_dir=self.text_to_voice_dir,
            background_music_path=chosen_music_path,
            narration_enabled=True
        )

        # Upload the result to GCS
        final_gcs_path = f"outputs/{self.media_base_name}/recap.mp4"
        print(f"[INFO] Uploading final video to: {final_gcs_path}")
        self.cloud_storage_client.upload_files(BUCKET_NAME, final_output_path, final_gcs_path)

        print(f"[INFO] Final video uploaded")
        return final_gcs_path
