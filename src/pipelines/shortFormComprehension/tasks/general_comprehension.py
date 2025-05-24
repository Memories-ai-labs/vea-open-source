import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path
from typing import List

from src.pipelines.shortFormComprehension.schema import Scene
from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss

class GeneralComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, segments: List[dict]) -> dict:
        """
        Performs general comprehension on grouped video segments.
        Segments with the same parent path are merged into a single scene list.
        Returns: dict[parent_path -> List[Scene]]
        """
        grouped = {}
        for seg in segments:
            parent = str(seg.get("parent_path"))
            grouped.setdefault(parent, []).append(seg)

        results = {}
        for parent_path, seg_group in grouped.items():
            merged_scenes = []
            print(f"[INFO] Running general comprehension for: {parent_path} ({len(seg_group)} segments)")
            for seg in sorted(seg_group, key=lambda s: s["start"]):
                print(f"[INFO] Running general comprehension for: {seg["path"]}")
                video_path = Path(seg["path"])
                start_offset = seg["start"]
                segment_id = seg["segment_number"]

                prompt = (
                    "You are given a video segment. Describe what happens every 10 seconds in the form of structured JSON.\n\n"
                    "Each description should include:\n"
                    "- start_timestamp (HH:MM:SS)\n"
                    "- end_timestamp (HH:MM:SS)\n"
                    "- scene_description: a concise and informative summary of what occurs in this window.\n\n"
                    "Include details such as the location, actions, people involved, emotions, objects, and atmosphere, but avoid repetition.\n"
                    "Be especially detailed when describing people's actions, such as sports plays or performance moves at a concert.\n"
                    "When in doubt, the focus of detail should align with the video content: e.g., actions for sports/concerts, setting for travel, objects for museum videos, etc.\n"
                    "Please use the entire video, even if it cuts off abruptly.\n"
                    "Avoid generic language or non-visual content like credits or watermarks.\n"
                )

                chunk_descriptions: List[Scene] = await asyncio.to_thread(
                    self.llm.LLM_request,
                    [video_path, prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[Scene],
                    }
                )

                for item in chunk_descriptions:
                    try:
                        raw_start = parse_time_to_seconds(item["start_timestamp"])
                        raw_end = parse_time_to_seconds(item["end_timestamp"])
                        item["start_timestamp"] = seconds_to_hhmmss(raw_start + start_offset)
                        item["end_timestamp"] = seconds_to_hhmmss(raw_end + start_offset)
                        item["segment_id"] = segment_id
                    except Exception as e:
                        print(f"[WARN] Timestamp fix failed: {item} | {e}")

                merged_scenes.extend(chunk_descriptions)
                time.sleep(1)

            results[parent_path] = merged_scenes

        return results
