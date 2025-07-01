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
    Generates a set of 1-minute shorts for a movie, using a global plan and strict index prompts.
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
            "I am an online creator making 1-minute addictive shorts edits of the movie. Your task is to divide the movie into sections of digestible story points, "
            "each suitable for a 1-minute short (typically 5-15 min of story, depending on the movie pace). "
            "Plan out the shorts so the viewing experience is cohesive and entertaining, the story is easy to follow, and the series of shorts saves time compared to watching the movie. "
            "For each short, describe what should be covered and provide a rough in/out timestamp (HH:MM:SS) if possible. "
            "Return a numbered list, one short per line, like: '1: [description and suggested timestamps]'. "
            "IMPORTANT: This is a text-only request—do not suggest specific clips or evidence, just plan the content for each short. "
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

        print("\n[SHORTS PLAN]\n", shorts_plan_text)
        if not shorts_plan_text.strip():
            raise RuntimeError("Failed to generate a shorts plan!")

        # ---- STEP 1B: Clean up and structure plan for parsing ----
        format_prompt = (
            "Format the following shorts plan into a JSON list. For each short, extract:\n"
            "- short_index: integer (the number at the start of the line)\n"
            "- description: the full sentence(s) describing what to cover in this short\n"
            "- start: the first timestamp (HH:MM:SS) mentioned for this short, or null if none\n"
            "- end: the second timestamp (HH:MM:SS) mentioned for this short, or null if none\n"
            "If only one timestamp is present, treat it as 'start' and leave 'end' as null. If no timestamps, set both to null. "
            "Respond ONLY with a valid JSON list, no explanation or comments.\n"
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

            per_short_prompt = (
                "you are a professional video editor. Your task is to edit the movie into compact and digestible shorts that are fun and addictive, but also tells the story of the movie. "
                f"You are to create short #{i+1} for a movie-to-shorts series. "
                "-------------------\n"
                f"Short description for this short: {plan_desc}\n"
            )
            if plan_start or plan_end:
                per_short_prompt += f"Suggested movie timestamps for this short: start={plan_start}, end={plan_end}\n"
            per_short_prompt += (
                f"You should create a 1-minute short (60 seconds) that is highly engaging and entertaining, choosing the most poignant clips to best tell the story.  "
                "Prioritize scenes with dialogue, especially dialogue that is engaging, clear, and moves the story forward. "
                "If no timestamps are provided, choose the most fitting segment. Deliver a satisfying, highly engaging short with subtitles. "
                "Do not recap previous shorts, focus only on this segment."
                "IMPORTANT: the short should be approximately 1 minute long, no longer than 1.5 minutes and no shorter than 40 seconds. "
                "dont choose timestamps that makes clips so short that its jarring to watch. maintain a fluid and natural viewing experience\n"
            )

            try:
                result = await self.flexible_pipeline.run(
                    user_prompt=per_short_prompt,
                    video_response=True,
                    original_audio=True,
                    music=False,
                    narration_enabled=False,
                    aspect_ratio=self.aspect_ratio,
                    subtitles=True,
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
        """
        Downloads the movie file from GCS to a temp location if not already present, returns local path.
        """
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

# Usage:
# pipeline = MovieToShortsPipeline("your/gcs/path/to/movie.mp4")
# asyncio.run(pipeline.run())
