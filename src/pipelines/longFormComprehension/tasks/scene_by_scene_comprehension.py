import os
import json
import time
import asyncio
from datetime import timedelta

from src.pipelines.longFormComprehension.schema import Scene

class SceneBySceneComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, short_segments, long_segments, summary_draft, characters):

        long_segment_bounds = []
        for i, file in enumerate(sorted(long_segments, key=lambda f: f.name)):
            parts = file.stem.rsplit("_", 2)
            start_sec = int(parts[1])
            end_sec = int(parts[2])
            long_segment_bounds.append((i + 1, start_sec, end_sec))
            
        short_segments.sort(key=lambda f: f.name)
        scenes = []
        scene_id = 1

        for file_path in short_segments:
            path_parts = file_path.stem.rsplit("_", 2)
            start_seconds = int(path_parts[1])

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

            segment_num = None
            for idx, seg_start, seg_end in long_segment_bounds:
                if seg_start <= start_seconds < seg_end:
                    segment_num = idx
                    break

            for scene in scene_data:
                try:
                    scene_start_sec = self._parse_time_to_seconds(scene["start_timestamp"])
                    scene_end_sec = self._parse_time_to_seconds(scene["end_timestamp"])
                    scene["start_timestamp"] = self._seconds_to_hhmmss(start_seconds + scene_start_sec)
                    scene["end_timestamp"] = self._seconds_to_hhmmss(start_seconds + scene_end_sec)
                    scene["id"] = scene_id
                    scene["segment_num"] = segment_num
                    scene_id += 1
                except Exception as e:
                    print(f"[WARN] Timestamp fix failed for scene: {scene} | {e}")

            scenes.extend(scene_data)
            print(f"[INFO] Segment {file_path.name} transcribed successfully.")
            time.sleep(1)

        print("[INFO] Scenes transcribed successfully.")
        return scenes

    def _parse_time_to_seconds(self, t: str) -> int:
        parts = list(map(int, t.split(":")))
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            raise ValueError(f"Invalid timestamp format: {t}")
        return h * 3600 + m * 60 + s

    def _seconds_to_hhmmss(self, seconds: int) -> str:
        return str(timedelta(seconds=int(seconds)))
