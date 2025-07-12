import math
import os
import json
import asyncio
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from lib.utils.media import seconds_to_hhmmss, get_video_duration
from src.pipelines.movieToShort.schema import ShortsPlan

class MovieToShortsPipeline:
    """
    Selects the most iconic, visually striking, or emotionally resonant scenes in a movie to generate addictive and memorable 1-minute shorts.
    """
    def __init__(self, blob_path: str, short_duration: int = 60, aspect_ratio: float = 9/16):
        self.blob_path = blob_path
        self.short_duration = short_duration  # Output short length in seconds
        self.aspect_ratio = aspect_ratio
        self.flexible_pipeline = FlexibleResponsePipeline(blob_path)
        self.gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

    async def run(self):
        print("[SHORTS] Getting movie duration...")
        local_path = await self._download_movie()
        total_duration = get_video_duration(local_path)
        print(f"[SHORTS] Total duration: {total_duration:.2f} sec")

        movie_name = os.path.splitext(os.path.basename(self.blob_path))[0]

        # ---- STEP 1: Generate a plan for the shorts ----
        plan_prompt = (
            "You are a professional social media video strategist. Your task is to extract iconic or highly entertaining 1-minute shorts from the movie. "
            "These can be memorable or funny dialogue scenes, emotionally powerful moments, or visually striking action scenes. "
            "Each short should be digestible on its own, requiring minimal context to understand. Favor scenes where the viewer can quickly grasp the stakes or meaning.\n"
            "For example, choose: (1) funny or iconic dialogue that mostly takes place in a tight time span, with optional payoff scenes from later in the movie; (2) action scenes with just enough setup or reaction dialogue to give them meaning. "
            "You can jump forward in time to include resolution or payoff moments if it improves the impact.\n"
            "For each short, provide:\n"
            "- short_index: number\n"
            "- description: what the short contains in detail and why it's addictive, and give rough timestamps where the content can be found in the footage\n"
            "- start: approximate start time (HH:MM:SS) of the main segment\n"
            "- end: approximate end time (HH:MM:SS) of the main segment\n"
            "- supporting_clips: list of additional timestamps (start/end pairs) to include payoff or resolution clips\n"
            "for movies, aim for around 20 shorts, and for tv episodes, aim for around 8 shorts.\n"
            # "Be very selective what which shorts to include, since we want to create the most engaging and addictive 1-minute shorts possible. "
            f"Movie duration: {int(total_duration//60)} min {int(total_duration%60)} sec."
        )

        print("[SHORTS] Generating plan for shorts (text-only response)...")
        plan_result = await self.flexible_pipeline.run(
            user_prompt=plan_prompt,
            video_response=False,
            original_audio=True,
            music=False,
            narration_enabled=False,
            aspect_ratio=self.aspect_ratio,
            subtitles=False,
            snap_to_beat=False,
        )
        shorts_plan_text = plan_result.get("response", "")

        # print("\n[SHORTS PLAN]\n", shorts_plan_text)
        if not shorts_plan_text.strip():
            raise RuntimeError("Failed to generate a shorts plan!")

        # ---- STEP 1B: Clean up and structure plan for parsing ----
        format_prompt = (
            "Format the following shorts plan into a JSON list. For each short, extract:\n"
            "- short_index: integer\n"
            "- description: the full sentence(s) describing the short\n"
            "- start: main segment start timestamp (HH:MM:SS)\n"
            "- end: main segment end timestamp (HH:MM:SS)\n"
            "- supporting_clips: optional list of start/end timestamp pairs in HH:MM:SS format\n"
            "Return only a JSON list, no explanation.\n"
            "Shorts plan:\n"
            "-------------------\n"
            f"{shorts_plan_text}\n"
            "-------------------"
        )

        shorts_plan_json = await asyncio.to_thread(
            self.flexible_pipeline.llm.LLM_request,
            [format_prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[ShortsPlan]
            }
        )

        print("\n[SHORTS PLAN STRUCTURED JSON]\n", json.dumps(shorts_plan_json, indent=2))
        lines = shorts_plan_json
        n_shorts = len(lines)
        print(f"[SHORTS] Targeting {n_shorts} shorts (based on plan).")

        # ---- STEP 2: Generate each short using the plan ----
        shorts = []
        for i, plan_entry in enumerate(lines):
            print(f"\n[SHORTS] Generating short #{i+1}")
            gcs_output_path = f"outputs/movie2short/{movie_name}/{i}.mp4"
            plan_desc = plan_entry.get("description", "")
            plan_start = plan_entry.get("start")
            plan_end = plan_entry.get("end")
            extras = plan_entry.get("supporting_clips", [])

            per_short_prompt = (
                "You are a professional video editor creating addictive 1-minute shorts for social media.\n"
                f"This is short #{i+1} in the series.\n"
                "Description of short: " + plan_desc + "\n"
            )
            if plan_start or plan_end:
                per_short_prompt += f"Main clip: {plan_start} to {plan_end}\n"
            if extras:
                per_short_prompt += f"Include supporting moment(s) at: {extras}\n"

            per_short_prompt += (
                "Choose the most powerful 1-minute edit using clips that are close in time, but may include payoff or resolution clips later in the movie.\n"
                "Avoid wide jumps unless there's a clear setup/payoff. Emphasize memorable dialogue or visually intense action.\n"
                "Ensure the short is emotionally or narratively satisfying and easy to follow without prior context.\n"
                "The final video should be between 40–90 seconds, engaging from the first second, with subtitles included."
            )

            try:
                result = await self.flexible_pipeline.run(
                    user_prompt=per_short_prompt,
                    video_response=True,
                    original_audio=True,
                    music=False,
                    narration_enabled=False,
                    aspect_ratio=self.aspect_ratio,
                    # subtitles=True,
                    subtitles=False,

                    snap_to_beat=False,
                    output_path=gcs_output_path
                )
                shorts.append({
                    "short_index": i,
                    "gcs_output_path": gcs_output_path,
                    "plan_entry": plan_entry,
                    "response": result,
                })
            except:
                pass

        print(f"[SHORTS] Generation complete! {len(shorts)} shorts created.")
        return shorts

    async def _download_movie(self):
        import tempfile
        fname = os.path.basename(self.blob_path)
        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, fname)
        if not os.path.exists(local_path):
            print(f"[SHORTS] Downloading {self.blob_path} to {local_path} ...")
            self.gcs_client.download_files(BUCKET_NAME, self.blob_path, local_path)
        else:
            print(f"[SHORTS] Using cached {local_path}")
        return local_path
