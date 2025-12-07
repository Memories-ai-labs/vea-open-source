import os
import json
import asyncio
from lib.oss.storage_factory import get_storage_client
from src.config import BUCKET_NAME
from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from lib.utils.media import get_video_duration, download_and_cache_video
from lib.utils.metrics_collector import metrics_collector
from src.pipelines.movieToShort.schema import ShortsPlans
from src.pipelines.common.schema import convert_timestamp_to_string

class MovieToShortsPipeline:
    """
    Selects the most iconic, visually striking, or emotionally resonant scenes in a movie to generate addictive and memorable 1-minute shorts.
    """
    def __init__(self, blob_path: str, short_duration: int = 60, aspect_ratio: float = 9/16):
        self.blob_path = blob_path
        self.short_duration = short_duration  # Output short length in seconds
        self.aspect_ratio = aspect_ratio
        self.flexible_pipeline = FlexibleResponsePipeline(blob_path)
        self.gcs_client = get_storage_client()

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
            "- Skip filler scenes or sections that don't add much to the story.\n"
            "- Highlight turning points, major conflicts, character growth, climactic moments, and payoffs.\n"
            "\n"
            "For each short, provide:\n"
            "- short_index: number\n"
            "- description: what the short is about and why it is an important part of the story\n"
            "- start: timestamp for the main segment's beginning as an object with separate numeric fields:\n"
            "  {\"hours\": <0-23>, \"minutes\": <0-59>, \"seconds\": <0-59>, \"milliseconds\": <0-999>}\n"
            "  Examples:\n"
            "  - 2 minutes 5 seconds = {\"hours\": 0, \"minutes\": 2, \"seconds\": 5, \"milliseconds\": 0}\n"
            "  - 1 hour 15 minutes = {\"hours\": 1, \"minutes\": 15, \"seconds\": 0, \"milliseconds\": 0}\n"
            "  - 45 seconds = {\"hours\": 0, \"minutes\": 0, \"seconds\": 45, \"milliseconds\": 0}\n"
            "- end: timestamp for the main segment's end (same format as start)\n"
            "- supporting_clips: optional list of additional segments, each with start and end in the same format\n"
            "\n"
            "You are not required to use the entire movie. It's okay to skip unimportant or dull scenes.\n"
            "Make sure the total number of shorts allows the whole story to be followed in ~20 parts for a full movie, or ~8 for an TV episode.\n"
            f"The movie duration is approximately {int(total_duration//60)} minutes {int(total_duration%60)} seconds.\n"
        )


        print("[SHORTS] Generating plan for shorts (text-only response)...")
        with metrics_collector.track_step("planning"):
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

        if not shorts_plan_text.strip():
            raise RuntimeError("Failed to generate a shorts plan!")

        # ---- STEP 1B: Clean up and structure plan for parsing ----
        format_prompt = (
            "Format the following chronological shorts plan into a JSON list. Each short should include:\n"
            "- short_index: integer (starts from 0)\n"
            "- description: what this short contains and why it is important to the story, in detail\n"
            "- start: timestamp as an object with separate numeric fields:\n"
            "  {\"hours\": <0-23>, \"minutes\": <0-59>, \"seconds\": <0-59>, \"milliseconds\": <0-999>}\n"
            "  Examples:\n"
            "  - 2 minutes 5 seconds = {\"hours\": 0, \"minutes\": 2, \"seconds\": 5, \"milliseconds\": 0}\n"
            "  - 1 hour 15 minutes = {\"hours\": 1, \"minutes\": 15, \"seconds\": 0, \"milliseconds\": 0}\n"
            "  - 45 seconds = {\"hours\": 0, \"minutes\": 0, \"seconds\": 45, \"milliseconds\": 0}\n"
            "- end: timestamp for the segment's end (same format as start)\n"
            "- supporting_clips: optional list of additional segments, each with start and end in the same format\n"
            "\n"
            "Return only a JSON list of objects. Do not include any explanation.\n"
            "Shorts plan:\n"
            "-------------------\n"
            f"{shorts_plan_text}\n"
            "-------------------"
        )


        with metrics_collector.track_step("format_plan"):
            shorts_plan_json = await asyncio.to_thread(
                self.flexible_pipeline.llm.LLM_request,
                [format_prompt],
                ShortsPlans,
                context="format_plan"
            )

        # Convert structured timestamps to strings for downstream processing
        for entry in shorts_plan_json:
            if "start" in entry:
                entry["start"] = convert_timestamp_to_string(entry["start"])
            if "end" in entry:
                entry["end"] = convert_timestamp_to_string(entry["end"])
            if "supporting_clips" in entry and entry["supporting_clips"]:
                for clip in entry["supporting_clips"]:
                    if "start" in clip:
                        clip["start"] = convert_timestamp_to_string(clip["start"])
                    if "end" in clip:
                        clip["end"] = convert_timestamp_to_string(clip["end"])

        # Save planning metrics before resetting for individual shorts
        os.makedirs(".cache/metrics", exist_ok=True)
        planning_metrics_file = ".cache/metrics/planning_metrics.json"
        metrics_collector.write_report(planning_metrics_file)
        metrics_collector.print_report()
        print(f"[METRICS] Planning metrics written to {planning_metrics_file}\n")

        print("\n[SHORTS PLAN STRUCTURED JSON]\n", json.dumps(shorts_plan_json, indent=2))
        lines = shorts_plan_json
        n_shorts = len(lines)
        print(f"[SHORTS] Targeting {n_shorts} shorts (based on plan).")

        # ---- STEP 2: Generate each short using the plan ----
        shorts = []
        for i, plan_entry in enumerate(lines):
            # Reset metrics for this short
            metrics_collector.reset()
            metrics_collector.short_index = i

            print(f"\n[SHORTS] Generating short #{i+1}")
            gcs_output_path = f"output/newmovie2short/{movie_name}/{i}.mp4"
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
                "Choose the most powerful 1-minute edit using clips that reflect this part of the movie's plot. "
                "Keep the edit engaging and impactful while preserving the story flow. "
                "You must choose clips in chronological order, and try your best to avoid choosing clips outside the time range specified in the plan.\n"
                "Ensure the short is understandable on its own but contributes to the overall chronological storytelling across all shorts. "
                "Final video should be between 40–90 seconds, with subtitles included."
            )

            try:
                with metrics_collector.track_step(f"short_{i}_generation"):
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

                # Write metrics to file and print report
                os.makedirs(".cache/metrics", exist_ok=True)
                metrics_file = f".cache/metrics/short_{i}_metrics.json"
                metrics_collector.write_report(metrics_file)
                metrics_collector.print_report()
                print(f"[METRICS] Written to {metrics_file}\n")

            except Exception as e:
                print(f"[ERROR] Failed to generate short #{i}: {e}")
                pass

        print(f"[SHORTS] Generation complete! {len(shorts)} shorts created.")
        return shorts

    async def _download_movie(self):
        cache_dir = os.path.join(".cache", "movie_to_short")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = download_and_cache_video(
            self.gcs_client,
            BUCKET_NAME,
            self.blob_path,
            cache_dir,
        )
        print(f"[SHORTS] Using source from {local_path}")
        return local_path
