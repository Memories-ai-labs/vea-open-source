import asyncio
from typing import List
from pathlib import Path
from src.pipelines.shortFormComprehension.schema import TranscribedLine
from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss

class ContentTranscription:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, segments: List[dict]) -> dict:
        """
        Ask Gemini to transcribe a list of video segments using structured output.
        Segments with the same parent file are merged together into a single transcript list.
        Returns: dict[parent_path -> List[TranscribedLine]]
        """
        grouped = {}
        for seg in segments:
            parent = str(seg.get("parent_path"))
            grouped.setdefault(parent, []).append(seg)

        results = {}
        for parent_path, seg_group in grouped.items():
            merged_lines = []
            print(f"[INFO] Running transcription for: {parent_path} ({len(seg_group)} segments)")
            for seg in sorted(seg_group, key=lambda s: s["start"]):
                print(f"[INFO] Running transcription for: {seg["path"]}")
                video_path = Path(seg["path"])
                start_offset = seg["start"]

                prompt = (
                    "You are given a video segment. Transcribe all content in structured JSON.\n\n"
                    "Include:\n"
                    "- Spoken words (verbatim)\n"
                    "- Readable visible text (from signs, posters, presentation slides, blackboards)\n"
                    "- A timestamp in HH:MM:SS for when the content appears\n\n"
                    "If there is no spoken content or visible text, return nothing. Do not include placeholder or filler lines.\n"
                )

                response = await asyncio.to_thread(
                    self.llm.LLM_request,
                    [video_path, prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[TranscribedLine],
                    }
                )

                for line in response:
                    try:
                        raw_ts = parse_time_to_seconds(line["timestamp"])
                        line["timestamp"] = seconds_to_hhmmss(raw_ts + start_offset)
                    except Exception as e:
                        print(f"[WARN] Failed to normalize timestamp: {line} | {e}")

                merged_lines.extend(response)

            results[parent_path] = merged_lines

        return results
