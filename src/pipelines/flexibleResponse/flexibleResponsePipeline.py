import os
import json
import tempfile
import random
import string

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.utils.media import clean_stale_tempdirs
from src.config import CREDENTIAL_PATH, BUCKET_NAME

from src.pipelines.flexibleResponse.tasks.flexible_gemini_answer import FlexibleGeminiAnswer
from src.pipelines.flexibleResponse.tasks.classify_response_type import ClassifyResponseType
from src.pipelines.flexibleResponse.tasks.evidence_retrieval import EvidenceRetrieval
from src.pipelines.flexibleResponse.tasks.clip_extraction import ClipExtractor
from src.pipelines.flexibleResponse.tasks.generate_narration_script import GenerateNarrationScript
from src.pipelines.flexibleResponse.tasks.generate_video_clip_plan import GenerateVideoClipPlan
from src.pipelines.common.refine_clip_timestamps import RefineClipTimestamps
from pipelines.common.generate_narration_audio import GenerateNarrationAudio
from pipelines.common.music_selection import MusicSelection 
from pipelines.common.edit_video_response import EditVideoResponse

class FlexibleResponsePipeline:
    def __init__(self, cloud_storage_media_path):
        # Parse basic paths and setup
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path.rstrip("/"))
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.cloud_storage_indexing_dir = f"indexing/{self.media_base_name}/"
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash-preview-04-17")
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.workdir = tempfile.mkdtemp()
        self.media_indexing_json = None

        clean_stale_tempdirs()
        self._load_media_indexing_json()

    def _load_media_indexing_json(self):
        """Download and parse media_indexing.json for this media file."""
        print("[INFO] Downloading media_indexing.json...")
        index_local_path = os.path.join(self.workdir, "media_indexing.json")
        gcs_index_path = self.cloud_storage_indexing_dir + "media_indexing.json"
        self.cloud_storage_client.download_files(BUCKET_NAME, gcs_index_path, index_local_path)
        with open(index_local_path, "r", encoding="utf-8") as f:
            self.media_indexing_json = json.load(f)
        if not self.media_indexing_json:
            raise RuntimeError(f"media_indexing.json not found at {gcs_index_path}")
        print("[INFO] media_indexing.json loaded successfully.")

    def _flatten_indexing(self):
        """Extracts file-level metadata and manifest from the loaded media indexing json."""
        indexing_data = self.media_indexing_json["media_files"]
        file_descriptions = self.media_indexing_json["manifest"]
        return indexing_data, file_descriptions

    async def run(self, user_prompt: str, video_response: bool, original_audio: bool, music: bool, narration_enabled: bool, aspect_ratio: float, subtitles: bool, snap_to_beat: bool):
        """
        Orchestrates the flexible response pipeline for a given media.
        Returns the output path(s) and metadata.
        """
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        print(f"[INFO] Starting flexible response pipeline. Run ID: {run_id}")

        indexing_data, file_descriptions = self._flatten_indexing()
        print("[INFO] Flattened indexing data and manifest loaded.")

        # 1. Step 1: Generate the initial LLM response
        print("[INFO] Requesting initial Gemini response...")
        gemini_task = FlexibleGeminiAnswer(self.llm)
        initial_response = await gemini_task(
            user_prompt=user_prompt,
            indexing_data=indexing_data,
            file_descriptions=file_descriptions
        )
        print("[INFO] Initial Gemini response received.")
        # print(f"[DEBUG] Initial response: {initial_response}")

        if not video_response:
            # --- Text/Evidence Response Path ---
            print("[INFO] Classifying response type (text, text+evidence)...")
            classify = ClassifyResponseType(self.llm)
            response_type = await classify(user_prompt, initial_response)
            print(f"[INFO] Response type classified as: {response_type}")

            if response_type == "text_only":
                print("[INFO] Returning text-only response.")
                return {
                    "response": initial_response,
                    "response_type": "text_only",
                    "evidence_paths": [],
                    "run_id": run_id
                }

            elif response_type == "text_and_evidence":
                print("[INFO] Retrieving evidence clips from Gemini...")
                evidence_task = EvidenceRetrieval(self.llm)
                selected_clips = await evidence_task(
                    initial_response=initial_response,
                    indexing_data=indexing_data,
                    file_descriptions=file_descriptions
                )
                print(f"[INFO] {len(selected_clips)} evidence clips selected.")

                print("[INFO] Extracting and uploading evidence clips...")
                clipper = ClipExtractor(
                    workdir=self.workdir,
                    gcs_client=self.cloud_storage_client,
                    bucket_name=BUCKET_NAME
                )
                gcs_clip_paths = clipper.extract_and_upload_clips(selected_clips, run_id)
                print(f"[INFO] Uploaded {len(gcs_clip_paths)} evidence clips to GCS.")

                return {
                    "response": initial_response,
                    "response_type": "text_and_evidence",
                    "evidence_paths": gcs_clip_paths,
                    "run_id": run_id
                }
        else:
            # --- Video Response Path ---
            if narration_enabled:
                print("[INFO] Generating narration script for video response...")
                narration_task = GenerateNarrationScript(self.llm)
                refined_script = await narration_task(
                    initial_response=initial_response,
                    user_prompt=user_prompt,
                    indexing_data=indexing_data,
                    file_descriptions=file_descriptions
                )
                print("[INFO] Narration script generated.")
                # print(f"[DEBUG] Refined narration script: {refined_script}")
            else:
                print("[INFO] Skipping narration script generation as narration is disabled.")
                refined_script = initial_response

            # Generate the clip plan (either for narration or not)
            print("[INFO] Generating video clip plan...")
            clip_plan_task = GenerateVideoClipPlan(self.llm)
            selected_narrated_clips = await clip_plan_task(
                narration_script=refined_script,
                user_prompt=user_prompt,
                indexing_data=indexing_data,
                file_descriptions=file_descriptions,
                narration_enabled=narration_enabled
            )
            print(f"[INFO] Video clip plan generated. {len(selected_narrated_clips)} clips planned.")

            # refine trim timestamps for clips
            refiner = RefineClipTimestamps(self.llm, self.workdir, self.cloud_storage_client, BUCKET_NAME)
            selected_narrated_clips = await refiner(selected_narrated_clips)
            print("[INFO] Refined clip timestamps based on dialogue transcription.")

            # Generate narration audio files if enabled
            if narration_enabled:
                narration_audio_dir = os.path.join(self.workdir, "voice")
                os.makedirs(narration_audio_dir, exist_ok=True)
                print("[INFO] Generating narration audio files for each clip...")
                narration_generator = GenerateNarrationAudio(narration_audio_dir)
                await narration_generator(selected_narrated_clips)
                print("[INFO] Narration audio generation complete.")
            else:
                narration_audio_dir = None

            # Choose background music for the video
            if music:
                music_selector = MusicSelection(self.llm, self.workdir)
                print("[INFO] Selecting background music based on media indexing...")
                chosen_music_path = await music_selector(self.media_indexing_json, user_prompt)
                print(f"[INFO] Chosen background music: {chosen_music_path}")
            else:
                chosen_music_path = None

            # Assemble and render the final video
            final_output_path = os.path.join(self.workdir, "video_response.mp4")
            print("[INFO] Assembling final video using editor...")
            editor = EditVideoResponse(
                output_path=final_output_path,
                gcs_client=self.cloud_storage_client,
                bucket_name=BUCKET_NAME,
                workdir=self.workdir,
                llm=self.llm
            )
            
            await editor(
                clips=selected_narrated_clips,
                narration_dir=narration_audio_dir,
                background_music_path=chosen_music_path,
                original_audio=original_audio,
                narration_enabled=narration_enabled,
                aspect_ratio=aspect_ratio,
                subtitles = subtitles,
                snap_to_beat= snap_to_beat
            )
            print("[INFO] Video response assembly complete.")

            # Upload the result to GCS
            final_gcs_path = f"outputs/{self.media_base_name}/{run_id}/video_response.mp4"
            print(f"[INFO] Uploading final video to: {final_gcs_path}")
            self.cloud_storage_client.upload_files(BUCKET_NAME, final_output_path, final_gcs_path)
            print(f"[SUCCESS] Video response uploaded to GCS.")

            return {
                "response": refined_script,
                "response_type": "video",
                "evidence_paths": [final_gcs_path],
                "run_id": run_id
            }
