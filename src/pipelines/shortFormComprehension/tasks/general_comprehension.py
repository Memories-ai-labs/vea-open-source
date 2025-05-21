import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path
from typing import List

from src.pipelines.shortFormComprehension.schema import Scene

class GeneralComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, video_path: Path, metadata: dict = None):
        """
        Performs general comprehension on a single video segment.
        For each ~10s chunk, it describes the setting, people involved, and their actions.
        """
        print(f"[INFO] Generating general descriptions for: {video_path.name}")

        prompt = (
            "You are given a video segment. Describe what happens every 10 seconds in the form of structured JSON.\n\n"
            "Each description should include:\n"
            "- start_timestamp (HH:MM:SS)\n"
            "- end_timestamp (HH:MM:SS)\n"
            "- scene_description: a concise and informative summary of what occurs in this window.\n\n"
            "Include details such as the location, actions, people involved, emotions, objects, and atmosphere, but avoid repetition.\n"
            "Be especially detailed when describing people's action, such as sports plays or performance moves at a concert. When in doubt, " 
            "the aspect of the video you should be detailed on depends on the general content of the video. for example, focus on actions for " \
            "sports game or concert videos, focus on landscape and setting for travel videos, and focus on objects for museum videos... and so on."
            "please use the entire video, even if it cuts off abruptly."
            "Avoid generic language or non-visual content like credits or watermarks.\n\n"
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
                item["segment_id"] = metadata.get("segment_number") if metadata else None
            except Exception as e:
                print(f"[WARN] Failed to add metadata to chunk: {item} | {e}")

        return chunk_descriptions
