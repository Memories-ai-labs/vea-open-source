import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path

from lib.utils.media import seconds_to_hhmmss, parse_time_to_seconds
from src.pipelines.videoComprehension.schema import Scene

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
                "Provided is a segment of a long-form video. All videos, regardless of their content, convey a story or sequence of events—sometimes profound, sometimes simple or surface-level. "
                "You are also given two references: "
                "- a JSON with the full story summary and segment number, "
                "- a list of people/characters with names, roles, and relationships. "
                "\n\n"
                "Depending on the type of media, you should adjust the scene descriptions accordingly:\n"
                "- If the video is story-driven (e.g. movie, TV show, documentary), focus on story, characters, and actions, but you may briefly mention artistic elements such as camera angles, lighting, or music when relevant.\n"
                "- If the media is content-driven (e.g. lecture, presentation, interview), include what content is being discussed or displayed, along with relevant context and key points.\n"
                "- If the media is visually driven (e.g. travel footage, montage), focus on the visual content, notable scenes, locations, or changes in setting, and you may also note things like style, music, or atmosphere.\n"
                "\n"
                "Every 20 seconds, describe the scene in detail, such as the individuals involved and their actions. "
                "Use the story and people/characters to help deduce who is in each scene and what is happening. "
                "Ignore scenes that are just logo animation, credits, or unrelated filler. "
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
