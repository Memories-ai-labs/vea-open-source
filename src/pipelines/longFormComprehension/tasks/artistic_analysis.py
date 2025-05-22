import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path

from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss
from src.pipelines.longFormComprehension.schema import ArtisticSegment


class ArtisticAnalysis:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, short_segments: list[dict], plot_text: str):
        segments = []
        segment_id = 1

        for seg in sorted(short_segments, key=lambda x: x["start"]):
            file_path = Path(seg["path"])
            start_seconds = seg["start"]

            print(f"[INFO] Analyzing artistic elements for: {file_path.name}")

            prompt = (
                f"{plot_text}\n\n"
                "The above is a refined plot summary for a long-form narrative video such as a movie, TV show, or documentary.\n"
                "You are now given a specific video segment to analyze every 20 seconds. Focus on the artistic elements, such as:\n"
                "- Camera work (e.g., angles, movement, framing)\n"
                "- Lighting (e.g., natural vs artificial, intensity, shadows)\n"
                "- Color palette and tone (e.g., warm, muted, saturated)\n"
                "- Set design and visual composition\n"
                "- Sound design (e.g., sound effects, ambient noise)\n"
                "- Music and how it interacts with visuals\n"
                "- Any noticeable special effects (CGI, transitions)\n\n"
                "Format each interval as a JSON object with:\n"
                "- start_timestamp (HH:MM:SS)\n"
                "- end_timestamp (HH:MM:SS)\n"
                "- visual_elements\n"
                "- audio_elements\n\n"
                "Keep your answer focused on artistic elements, and don't include descriptions of the plot or characters.\n"
            )

            results = await asyncio.to_thread(
                self.llm.LLM_request,
                [file_path, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[ArtisticSegment]
                }
            )

            for seg_data in results:
                try:
                    seg_start = parse_time_to_seconds(seg_data["start_timestamp"])
                    seg_end = parse_time_to_seconds(seg_data["end_timestamp"])
                    seg_data["start_timestamp"] = seconds_to_hhmmss(start_seconds + seg_start)
                    seg_data["end_timestamp"] = seconds_to_hhmmss(start_seconds + seg_end)
                    seg_data["id"] = segment_id
                    segment_id += 1
                except Exception as e:
                    print(f"[WARN] Timestamp fix failed: {seg_data} | {e}")

            segments.extend(results)
            print(f"[INFO] Segment {file_path.name} analysis complete.")
            time.sleep(1)

        print("[INFO] Artistic analysis complete.")
        return segments
