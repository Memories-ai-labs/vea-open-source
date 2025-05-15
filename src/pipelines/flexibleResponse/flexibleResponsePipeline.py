import os
import json
import tempfile
import random
import string
import shutil

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH, BUCKET_NAME

from src.pipelines.flexibleResponse.tasks.flexible_gemini_answer import FlexibleGeminiAnswer
from src.pipelines.flexibleResponse.tasks.classify_response_type import ClassifyResponseType
from src.pipelines.flexibleResponse.tasks.evidence_retrieval import EvidenceRetrieval
from src.pipelines.flexibleResponse.tasks.clip_extraction import ClipExtractor
from src.pipelines.flexibleResponse.tasks.generate_narration_script import GenerateNarrationScript
from src.pipelines.flexibleResponse.tasks.generate_video_clip_plan import GenerateVideoClipPlan
from src.pipelines.flexibleResponse.tasks.generate_narration_audio import GenerateNarrationForVideoResponse
from src.pipelines.flexibleResponse.tasks.edit_video_response import EditFlexibleVideoResponse

class FlexibleResponsePipeline:
    def __init__(self, cloud_storage_media_path):
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]

        self.clean_stale_tempdirs()
        self.cloud_storage_indexing_dir = f"indexing/{self.media_base_name}/"
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash-preview-04-17")
        self.cloud_storage_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.workdir = tempfile.mkdtemp()

        self.files_of_interest = {
            "plot.txt": "A linear summary of the plot, broken into segments.",
            "characters.txt": "Descriptions of all relevant characters and their relationships.",
            "scenes.json": "Scene metadata including timestamps and visual content.",
            "artistic.json": "Artistic breakdown of visual and audio elements of the movie.",
        }
        
        self.local_indexing_data = {}
        self.filtered_file_descriptions = {}
        self._load_indexing_files()

    def _load_indexing_files(self):
        for fname, desc in self.files_of_interest.items():
            try:
                local_path = os.path.join(self.workdir, fname)
                self.cloud_storage_client.download_files(
                    BUCKET_NAME,
                    self.cloud_storage_indexing_dir + fname,
                    local_path
                )
                with open(local_path, "r", encoding="utf-8") as f:
                    content = json.load(f) if fname.endswith(".json") else f.read()
                self.local_indexing_data[fname] = content
                self.filtered_file_descriptions[fname] = desc
            except Exception as e:
                print(f"[WARNING] Skipping {fname}: {e}")

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

    async def run(self, user_prompt: str, video_response: bool):
        # Generate 8-char alphanumeric ID
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

        gemini_task = FlexibleGeminiAnswer(self.llm)
        initial_response = await gemini_task(
            user_prompt=user_prompt,
            indexing_data=self.local_indexing_data,
            file_descriptions=self.filtered_file_descriptions
        )
    
        if not video_response:
            classify = ClassifyResponseType(self.llm)
            response_type = await classify(user_prompt, initial_response)

            if response_type == "text_only":
                return {
                    "response": initial_response,
                    "response_type": "text_only",
                    "evidence_paths": []
                }
            elif response_type == "text_and_evidence":
                # Step 1: Retrieve structured clip metadata
                evidence_task = EvidenceRetrieval(self.llm)
                selected_clips = await evidence_task(
                    initial_response=initial_response,
                    indexing_data=self.local_indexing_data,
                    file_descriptions=self.filtered_file_descriptions
                )
                print(selected_clips)

                # Step 2: Download movie and extract clips
                print("[INFO] downloading original media")
                movie_local_path = os.path.join(self.workdir, self.media_name)
                self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, movie_local_path)

                clipper = ClipExtractor(movie_local_path)
                local_clips = clipper.extract_clips(selected_clips)

                # Step 3: Upload clips and return their GCS paths
                gcs_clip_paths = []
                for path in local_clips:
                    gcs_rel_path = f"evidence/{self.media_base_name}/{run_id}/{os.path.basename(path)}"
                    self.cloud_storage_client.upload_files(BUCKET_NAME, path, gcs_rel_path)
                    gcs_clip_paths.append(gcs_rel_path)
                print(gcs_clip_paths)

                return {
                    "response": initial_response,
                    "response_type": "text_and_evidence",
                    "evidence_paths": gcs_clip_paths,
                    "run_id": run_id
                }
        else:
            narration_task = GenerateNarrationScript(self.llm)
            refined_script = await narration_task(
                initial_response=initial_response,
                user_prompt=user_prompt,
                indexing_data=self.local_indexing_data,
                file_descriptions=self.filtered_file_descriptions
            )
            # print(refined_script)

            # Generate clip plan
            clip_plan_task = GenerateVideoClipPlan(self.llm)
            selected_narrated_clips = await clip_plan_task(
                narration_script=refined_script,
                user_prompt=user_prompt,
                indexing_data=self.local_indexing_data,
                file_descriptions=self.filtered_file_descriptions
            )
            # print(selected_narrated_clips)

            # Step 3: Generate narration audio
            narration_audio_dir = os.path.join(self.workdir, "voice")
            os.makedirs(narration_audio_dir, exist_ok=True)
            narration_generator = GenerateNarrationForVideoResponse(narration_audio_dir)
            await narration_generator(selected_narrated_clips)

            # Step 4: Download original movie
            movie_local_path = os.path.join(self.workdir, self.media_name)
            self.cloud_storage_client.download_files(BUCKET_NAME, self.cloud_storage_media_path, movie_local_path)

            # Step 5: Generate final video
            final_output_path = os.path.join(self.workdir, "video_response.mp4")
            editor = EditFlexibleVideoResponse(final_output_path)
            await editor(
                clips=selected_narrated_clips,
                movie_path=movie_local_path,
                narration_dir=narration_audio_dir,
            )

            # Step 6: Upload video
            final_gcs_path = f"outputs/{self.media_base_name}/{run_id}/video_response.mp4"
            self.cloud_storage_client.upload_files(BUCKET_NAME, final_output_path, final_gcs_path)

            return {
                "response": refined_script,
                "response_type": "video",
                "evidence_paths": [final_gcs_path],
                "run_id": run_id
            }


