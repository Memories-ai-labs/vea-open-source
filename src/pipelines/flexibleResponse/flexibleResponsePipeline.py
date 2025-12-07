import os
import json
import logging
import tempfile
import random
import string
import subprocess
import uuid
import shutil

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.storage_factory import get_storage_client
from lib.utils.media import clean_stale_tempdirs, download_and_cache_video, get_video_duration, parse_time_to_seconds
from lib.utils.metrics_collector import metrics_collector
from src.config import BUCKET_NAME, INDEXING_DIR, OUTPUTS_DIR

from src.pipelines.flexibleResponse.tasks.flexible_gemini_answer import FlexibleGeminiAnswer
from src.pipelines.flexibleResponse.tasks.classify_response_type import ClassifyResponseType
from src.pipelines.flexibleResponse.tasks.evidence_retrieval import EvidenceRetrieval
from src.pipelines.flexibleResponse.tasks.clip_extraction import ClipExtractor
from src.pipelines.flexibleResponse.tasks.generate_narration_script import GenerateNarrationScript
from src.pipelines.flexibleResponse.tasks.generate_video_clip_plan import GenerateVideoClipPlan
from src.pipelines.common.refine_clip_timestamps import RefineClipTimestamps
from src.pipelines.common.generate_narration_audio import GenerateNarrationAudio
from src.pipelines.common.music_selection import MusicSelection

logger = logging.getLogger(__name__)


class FlexibleResponsePipeline:
    def __init__(
        self,
        cloud_storage_media_path: str,
    ):
        """
        Initialize the FlexibleResponsePipeline.

        Args:
            cloud_storage_media_path: Path to the video file in cloud storage
        """
        # Parse basic paths and setup
        self.cloud_storage_media_path = cloud_storage_media_path

        # Extract project name from path:
        # - If path is a folder (ends with /): use folder name
        # - If path is a file: use parent folder name
        path = cloud_storage_media_path.rstrip("/")
        if cloud_storage_media_path.endswith("/") or not os.path.splitext(path)[1]:
            # It's a folder path
            self.project_name = os.path.basename(path)
        else:
            # It's a file path - use parent folder name
            self.project_name = os.path.basename(os.path.dirname(path))

        # Fallback to file stem if no folder structure
        if not self.project_name:
            self.project_name = os.path.splitext(os.path.basename(path))[0]

        self.media_name = os.path.basename(path)
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.cloud_storage_indexing_dir = f"{INDEXING_DIR}/{self.project_name}/"
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash")
        self.cloud_storage_client = get_storage_client()
        self.workdir = tempfile.mkdtemp()
        self.media_indexing_json = None

        clean_stale_tempdirs()
        self._load_media_indexing_json()

    def _load_media_indexing_json(self):
        """Download and parse media_indexing.json for this media file."""
        print("[INFO] Downloading media_indexing.json...")
        gcs_index_path = self.cloud_storage_indexing_dir + "media_indexing.json"
        cache_dir = os.path.join(".cache", "media_indexing")
        os.makedirs(cache_dir, exist_ok=True)
        index_local_path = download_and_cache_video(
            self.cloud_storage_client,
            BUCKET_NAME,
            gcs_index_path,
            cache_dir,
        )
        if os.path.dirname(index_local_path) != self.workdir:
            working_copy = os.path.join(self.workdir, "media_indexing.json")
            os.makedirs(self.workdir, exist_ok=True)
            shutil.copy2(index_local_path, working_copy)
            index_local_path = working_copy
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

    def _filter_out_of_bounds_clips(self, clips: list) -> list:
        """
        Filter out clips with timestamps that exceed the source video duration.
        Also filters clips with invalid timestamps (start >= end, negative values, etc.)
        """
        if not clips:
            return clips

        # Cache video durations to avoid repeated lookups
        duration_cache = {}
        valid_clips = []

        for clip in clips:
            try:
                # Get source video path
                source_path = clip.get("cloud_storage_path", "")
                if not source_path:
                    print(f"[WARN] Clip {clip.get('id', '?')} has no cloud_storage_path, skipping")
                    continue

                # Get or cache video duration
                if source_path not in duration_cache:
                    # Download video to get duration
                    cache_dir = os.path.join(".cache", "duration_check")
                    os.makedirs(cache_dir, exist_ok=True)
                    local_path = download_and_cache_video(
                        self.cloud_storage_client,
                        BUCKET_NAME,
                        source_path,
                        cache_dir
                    )
                    duration_cache[source_path] = get_video_duration(local_path)

                video_duration = duration_cache[source_path]

                # Parse clip timestamps
                start_str = clip.get("start", "00:00:00,000")
                end_str = clip.get("end", "00:00:00,000")
                start_sec = parse_time_to_seconds(start_str)
                end_sec = parse_time_to_seconds(end_str)

                # Validate timestamps
                if start_sec < 0:
                    print(f"[WARN] Clip {clip.get('id', '?')} has negative start time ({start_str}), skipping")
                    continue

                if end_sec <= start_sec:
                    print(f"[WARN] Clip {clip.get('id', '?')} has end <= start ({end_str} <= {start_str}), skipping")
                    continue

                if start_sec >= video_duration:
                    print(f"[WARN] Clip {clip.get('id', '?')} start ({start_str} = {start_sec:.1f}s) exceeds video duration ({video_duration:.1f}s), skipping")
                    continue

                if end_sec > video_duration:
                    # Clamp end to video duration instead of skipping
                    from lib.utils.media import seconds_to_hhmmss
                    old_end = end_str
                    clip["end"] = seconds_to_hhmmss(video_duration)
                    print(f"[WARN] Clip {clip.get('id', '?')} end ({old_end}) exceeds video duration, clamped to {clip['end']}")

                valid_clips.append(clip)

            except Exception as e:
                print(f"[WARN] Error validating clip {clip.get('id', '?')}: {e}, skipping")
                continue

        filtered_count = len(clips) - len(valid_clips)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} invalid clips, {len(valid_clips)} remaining")

        return valid_clips

    async def run(self, user_prompt: str, video_response: bool, original_audio: bool, music: bool,
                  narration_enabled: bool, aspect_ratio: float, subtitles: bool, snap_to_beat: bool,
                  output_path=None):
        """
        Orchestrates the flexible response pipeline for a given media.
        Returns the output path(s) and metadata.
        """
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        print(f"[INFO] Starting flexible response pipeline. Run ID: {run_id}")

        indexing_data, file_descriptions = self._flatten_indexing()
        print("[INFO] Flattened indexing data and manifest loaded.")

        # Step 1: Generate the initial LLM response using Gemini with pre-indexed data
        print("[INFO] Requesting initial Gemini response...")
        gemini_task = FlexibleGeminiAnswer(self.llm)
        with metrics_collector.track_step("flexible_gemini_answer"):
            initial_response = await gemini_task(
                user_prompt=user_prompt,
                indexing_data=indexing_data,
                file_descriptions=file_descriptions
            )
        print("[INFO] Initial Gemini response received.")

        if not video_response:
            # --- Text/Evidence Response Path ---
            print("[INFO] Classifying response type (text, text+evidence)...")
            classify = ClassifyResponseType(self.llm)
            with metrics_collector.track_step("classify_response_type"):
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
                try:
                    print("[INFO] Retrieving evidence clips from Gemini...")
                    evidence_task = EvidenceRetrieval(self.llm)
                    with metrics_collector.track_step("evidence_retrieval"):
                        selected_clips = await evidence_task(
                            initial_response=initial_response,
                            indexing_data=indexing_data,
                            file_descriptions=file_descriptions
                        )
                    print(f"[INFO] {len(selected_clips)} evidence clips selected.")

                    # Filter out clips with invalid/out-of-bounds timestamps
                    selected_clips = self._filter_out_of_bounds_clips(selected_clips)
                    if not selected_clips:
                        raise RuntimeError("No valid evidence clips remaining after filtering")

                    print("[INFO] Extracting and uploading evidence clips...")
                    clipper = ClipExtractor(
                        workdir=self.workdir,
                        gcs_client=self.cloud_storage_client,
                        bucket_name=BUCKET_NAME
                    )
                    with metrics_collector.track_step("clip_extraction"):
                        gcs_clip_paths = clipper.extract_and_upload_clips(selected_clips, run_id)
                    print(f"[INFO] Uploaded {len(gcs_clip_paths)} evidence clips to GCS.")

                    return {
                        "response": initial_response,
                        "response_type": "text_and_evidence",
                        "evidence_paths": gcs_clip_paths,
                        "run_id": run_id
                    }
                except:
                    return {
                        "response": initial_response,
                        "response_type": "text_only",
                        "evidence_paths": [],
                        "run_id": run_id
                    }
        else:
            # --- Video Response Path ---
            if narration_enabled:
                print("[INFO] Generating narration script for video response...")
                narration_task = GenerateNarrationScript(self.llm)
                with metrics_collector.track_step("generate_narration_script"):
                    refined_script = await narration_task(
                        initial_response=initial_response,
                        user_prompt=user_prompt,
                        indexing_data=indexing_data,
                        file_descriptions=file_descriptions
                    )
                print("[INFO] Narration script generated.")
            else:
                print("[INFO] Skipping narration script generation as narration is disabled.")
                refined_script = initial_response

            # Generate the clip plan (either for narration or not)
            print("[INFO] Generating video clip plan...")
            clip_plan_task = GenerateVideoClipPlan(self.llm)
            with metrics_collector.track_step("generate_clip_plan"):
                selected_narrated_clips = await clip_plan_task(
                    narration_script=refined_script,
                    user_prompt=user_prompt,
                    indexing_data=indexing_data,
                    file_descriptions=file_descriptions,
                    narration_enabled=narration_enabled
                )
            print(f"[INFO] Video clip plan generated. {len(selected_narrated_clips)} clips planned.")
            print(selected_narrated_clips)

            # Filter out clips with invalid/out-of-bounds timestamps
            selected_narrated_clips = self._filter_out_of_bounds_clips(selected_narrated_clips)
            if not selected_narrated_clips:
                raise RuntimeError("No valid clips remaining after filtering out-of-bounds timestamps")

            # refine trim timestamps for clips
            if original_audio:
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
            editor_input = {
                "project_name": self.project_name,
                "clips": selected_narrated_clips,
                "narration_dir": narration_audio_dir,
                "background_music_path": chosen_music_path,
                "original_audio": original_audio,
                "narration_enabled": narration_enabled,
                "aspect_ratio": aspect_ratio,
                "subtitles": subtitles,
                "snap_to_beat": snap_to_beat,
                "bucket_name": BUCKET_NAME,
            }

            editor_input_path = os.path.join(self.workdir, f"edit_input_{uuid.uuid4().hex}.json")
            with open(editor_input_path, "w", encoding="utf-8") as f:
                json.dump(editor_input, f, indent=2)

            # Generate temp metrics file path for subprocess
            subprocess_metrics_file = os.path.join(self.workdir, f"subprocess_metrics_{uuid.uuid4().hex}.json")

            # --- Call CLI subprocess with metrics output path
            print("[INFO] Running EditVideoResponse as subprocess to free memory...")
            subprocess.run([
                "python", "-m", "src.pipelines.common.edit_video_response",
                editor_input_path,
                subprocess_metrics_file
            ], check=True)
            print("[INFO] EditVideoResponse subprocess complete.")

            # Read result file to get actual video path
            result_file = editor_input_path.replace(".json", "_result.json")
            with open(result_file, "r", encoding="utf-8") as f:
                edit_result = json.load(f)
            local_video_path = edit_result.get("video_path")
            print(f"[INFO] Video created at: {local_video_path}")

            # Merge subprocess metrics into parent
            metrics_collector.merge_from_file(subprocess_metrics_file)

            # Clean up temp files
            try:
                os.remove(subprocess_metrics_file)
                os.remove(result_file)
            except:
                pass

            # Upload the result to GCS
            final_gcs_path = f"{OUTPUTS_DIR}/{self.media_base_name}/{run_id}/video_response.mp4"
            if output_path:
                final_gcs_path = output_path
            print(f"[INFO] Uploading final video to: {final_gcs_path}")
            self.cloud_storage_client.upload_files(BUCKET_NAME, local_video_path, final_gcs_path)
            print(f"[SUCCESS] Video response uploaded to GCS.")

            return {
                "response": refined_script,
                "response_type": "video",
                "evidence_paths": [final_gcs_path],
                "run_id": run_id
            }
