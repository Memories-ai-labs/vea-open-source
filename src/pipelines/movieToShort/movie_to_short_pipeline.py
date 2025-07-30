import math
import os
import json
import asyncio
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from lib.utils.media import seconds_to_hhmmss, get_video_duration
from src.pipelines.movieToShort.schema import ShortsPlans
from typing import List

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
            "You are a professional video strategist. Your goal is to create a series of short clips that, when viewed in order, effectively retell the main story of the movie.\n"
            "Each short should correspond to an important or emotionally impactful part of the plot. Avoid including boring or unimportant scenes. Focus on story progression, emotional development, key reveals, and satisfying resolutions.\n"
            "\n"
            "The shorts should:\n"
            "- Be arranged chronologically according to the movie's timeline.\n"
            "- Skip filler scenes or sections that don’t add much to the story.\n"
            "- Highlight turning points, major conflicts, character growth, climactic moments, and payoffs.\n"
            "\n"
            "For each short, provide:\n"
            "- short_index: number\n"
            "- description: what the short is about and why it is an important part of the story\n"
            "- start: approximate HH:MM:SS timestamp for the main segment of this short (i.e., where this story beat begins)\n"
            "- end: approximate HH:MM:SS timestamp for the main segment’s end\n"
            "- supporting_clips: optional list of HH:MM:SS start/end pairs for any other key moments that should be included to strengthen the short (e.g., payoffs, reactions, flashbacks)\n"
            "\n"
            "You are not required to use the entire movie. It’s okay to skip unimportant or dull scenes.\n"
            "Make sure the total number of shorts allows the whole story to be followed in ~20 parts for a full movie, or ~8 for an TV episode.\n"
            f"The movie duration is approximately {int(total_duration//60)} minutes {int(total_duration%60)} seconds.\n"
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
            "Format the following chronological shorts plan into a JSON list. Each short should include:\n"
            "- short_index: integer (starts from 0)\n"
            "- description: what this short contains and why it is important to the story, in detail\n"
            "- start: HH:MM:SS timestamp for the main segment’s beginning\n"
            "- end: HH:MM:SS timestamp for the main segment’s end\n"
            "- supporting_clips: optional list of additional segments to include, each with a start and end in HH:MM:SS format\n"
            "\n"
            "Return only a JSON list of objects. Do not include any explanation.\n"
            "Shorts plan:\n"
            "-------------------\n"
            f"{shorts_plan_text}\n"
            "-------------------"
        )


        shorts_plan_json = await asyncio.to_thread(
            self.flexible_pipeline.llm.LLM_request,
            [format_prompt],
            ShortsPlans
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
                "Choose the most powerful 1-minute edit using clips that reflect this part of the movie’s plot. "
                "Keep the edit engaging and impactful while preserving the story flow. "
                "You must choose clips in chronological order, and try your best to avoid choosing clips outside the time range specified in the plan.\n"
                "Ensure the short is understandable on its own but contributes to the overall chronological storytelling across all shorts. "
                "Final video should be between 40–90 seconds, with subtitles included."
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
