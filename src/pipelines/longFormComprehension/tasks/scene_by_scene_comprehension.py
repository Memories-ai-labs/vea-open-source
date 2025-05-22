import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path

from lib.utils.media import seconds_to_hhmmss, parse_time_to_seconds
from src.pipelines.longFormComprehension.schema import Scene

class SceneBySceneComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, short_segments: list[dict], summary_draft: str, characters: str):
        scenes = []
        scene_id = 1

        for seg in sorted(short_segments, key=lambda x: x["start"]):
            file_path = Path(seg["path"])
            start_seconds = seg["start"]
            segment_number = seg["segment_number"]

            print(f"[INFO] Transcribing scene-by-scene for segment: {file_path.name}")

            prompt = (
                f"{summary_draft}\n\n"
                f"{characters}\n\n"
                "Provided is a segment of a long-form narrative media, such as a movie, TV show, or documentary, along with two references: "
                "- a JSON with the full plot summary and segment number, "
                "- a character list with names, roles, and relationships. "
                "Every 20 seconds, describe the scene in detail, such as the characters involved and their actions. "
                "Use the plot and characters to help deduce who is in each scene and what is happening. "
                "Ignore scenes that are just studio logo animation or end credits. "
                "Be sure to use the whole segment. Format each scene as JSON with: "
                "- start_timestamp (HH:MM:SS)\n"
                "- end_timestamp (HH:MM:SS)\n"
                "- description\n"
                "You should output in English except for character names, which should be in the original language."
            )

            scene_data = await asyncio.to_thread(
                self.llm.LLM_request,
                [file_path, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[Scene],
                }
            )

            for scene in scene_data:
                try:
                    raw_start = parse_time_to_seconds(scene["start_timestamp"])
                    raw_end = parse_time_to_seconds(scene["end_timestamp"])
                    scene["start_timestamp"] = seconds_to_hhmmss(start_seconds + raw_start)
                    scene["end_timestamp"] = seconds_to_hhmmss(start_seconds + raw_end)
                    scene["id"] = scene_id
                    scene["segment_num"] = segment_number
                    scene_id += 1
                except Exception as e:
                    print(f"[WARN] Timestamp fix failed for scene: {scene} | {e}")

            scenes.extend(scene_data)
            print(f"[INFO] Segment {file_path.name} transcribed successfully.")
            time.sleep(1)

        print("[INFO] Scenes transcribed successfully.")
        return scenes
