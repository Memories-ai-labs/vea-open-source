import os
import json
import time
import asyncio
from datetime import timedelta
from pathlib import Path
from src.pipelines.longForm.schema import ArtisticSegment

class ArtisticAnalysis:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, short_segments: list[Path], plot_text: str):
        segments = []
        segment_id = 1

        for video_file in sorted(short_segments, key=lambda f: f.name):
            path_parts = video_file.stem.rsplit("_", 2)
            start_seconds = int(path_parts[1])

            print(f"[INFO] Analyzing artistic elements for: {video_file.name}")

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
                "keep your answer brief, and dont include descriptions of the plot or characters, focus on the artistry.\n"
            )

            results = await asyncio.to_thread(
                self.llm.LLM_request,
                [video_file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[ArtisticSegment]
                }
            )

            for seg in results:
                try:
                    seg["start_timestamp"] = self._shift_timestamp(seg["start_timestamp"], start_seconds)
                    seg["end_timestamp"] = self._shift_timestamp(seg["end_timestamp"], start_seconds)
                    seg["id"] = segment_id
                    segment_id += 1
                except Exception as e:
                    print(f"[WARN] Timestamp fix failed: {seg} | {e}")

            segments.extend(results)
            time.sleep(1)

        print("[INFO] Artistic analysis complete.")
        return segments

    def _shift_timestamp(self, timestamp: str, shift_seconds: int) -> str:
        parts = list(map(int, timestamp.split(":")))
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        total = h * 3600 + m * 60 + s + shift_seconds
        return str(timedelta(seconds=total))

