import os
import json
import asyncio
import tempfile
import time
import shutil

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.storage_factory import get_storage_client
from lib.utils.media import (
    download_and_cache_video,
    parse_time_to_seconds,
    extract_video_segment
)
from src.config import BUCKET_NAME, INDEXING_DIR
from src.pipelines.common.generate_subtitles import GenerateSubtitles
from src.pipelines.screenplay.schema import SegmentTimestamp, SegmentTimestamps, SectionScreenplay


class ScreenplayPipeline:
    def __init__(self, cloud_storage_media_path):
        if cloud_storage_media_path.endswith("/"):
            raise ValueError("Provided blob path must be a file, not a folder.")
        
        self.cloud_storage_media_path = cloud_storage_media_path
        self.media_name = os.path.basename(cloud_storage_media_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.cloud_storage_indexing_dir = f"{INDEXING_DIR}/{self.media_base_name}/"
        self.llm = GeminiGenaiManager(model="gemini-2.5-pro")
        self.cloud_storage_client = get_storage_client()
        self.workdir = tempfile.mkdtemp()
        self.subtitle_generator = GenerateSubtitles(output_dir=os.path.join(self.workdir, "subs"))

        self.indexing = self._load_indexing()

    def _load_indexing(self):
        gcs_path = self.cloud_storage_indexing_dir + "media_indexing.json"
        cache_dir = os.path.join(".cache", "screenplay_indexing")
        os.makedirs(cache_dir, exist_ok=True)
        cached_path = download_and_cache_video(
            self.cloud_storage_client,
            BUCKET_NAME,
            gcs_path,
            cache_dir,
        )
        local_path = os.path.join(self.workdir, "media_indexing.json")
        shutil.copy2(cached_path, local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("media_files"):
            raise ValueError("Missing 'media_files' in indexing file.")
        return data["media_files"][0]  # Assume one media file per indexing

    def _dict_to_segment(self, seg_dict):
        return SegmentTimestamp(start=seg_dict["start"], end=seg_dict["end"])

    async def _segment_video(self, plot_text, story_sentences, scenes):
        plot_summary = "\n".join([f"(Segment {s['segment_num']}): {s['sentence_text']}" for s in story_sentences])
        prompt = (
            "Split this movie into logical segments at natural plot breaks. "
            "Each segment must be 10 minutes or less.\n\n"
            "For each segment, provide start and end as timestamp objects with separate numeric fields:\n"
            "{\"hours\": <0-23>, \"minutes\": <0-59>, \"seconds\": <0-59>, \"milliseconds\": <0-999>}\n"
            "Examples:\n"
            "- 2 minutes 5 seconds = {\"hours\": 0, \"minutes\": 2, \"seconds\": 5, \"milliseconds\": 0}\n"
            "- 1 hour 15 minutes = {\"hours\": 1, \"minutes\": 15, \"seconds\": 0, \"milliseconds\": 0}\n"
            "- 45 seconds = {\"hours\": 0, \"minutes\": 0, \"seconds\": 45, \"milliseconds\": 0}\n\n"
            f"Plot:\n{plot_text}\n\n"
            f"Detailed Plot Breakdown:\n{plot_summary}\n\n"
            f"Scene List:\n{json.dumps(scenes, indent=2, ensure_ascii=False)}"
        )
        segment_dicts = self.llm.LLM_request([prompt], SegmentTimestamps)
        return [self._dict_to_segment(d) for d in segment_dicts]

    async def _generate_screenplay_for_segment(self, segment, movie_path: str):
        # Convert Timestamp objects to strings for downstream functions
        start_str = segment.start.to_string()
        end_str = segment.end.to_string()
        start_seconds = segment.start.to_seconds()
        end_seconds = segment.end.to_seconds()

        segment_video_path = extract_video_segment(
            movie_path, self.workdir, start_str, end_str,
            output_name=f"segment_{start_str.replace(':','').replace(',','')}.mp4"
        )

        transcription = None
        for attempt in range(3):
            try:
                transcription = await asyncio.to_thread(
                    self.subtitle_generator,
                    audio_path=segment_video_path,
                    global_start_time=start_seconds
                )
                break
            except Exception as e:
                print(f"[WARN] Transcription attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2)

        text = " ".join([w["text"] for w in transcription.get("words", []) if w["type"] == "word"])

        scenes = [
            s for s in self.indexing.get("scenes.json", [])
            if start_seconds <= parse_time_to_seconds(s["start_timestamp"]) <= end_seconds
        ]
        scene_descriptions = "\n".join([f"{s['start_timestamp']} - {s['end_timestamp']}: {s['scene_description']}" for s in scenes])
        plot = self.indexing.get("story.txt", "")
        characters = self.indexing.get("people.txt", "")

        prompt = (
            "Write a screenplay for the following movie segment using traditional screenplay format.\n"
            "- Make sure dialogue is complete and not omitted.\n"
            "- Use only plain text. Do NOT use Markdown, HTML, or rich formatting.\n"
            "- Format all dialogue in simple lines prefixed by the character name in uppercase.\n"
            "- Do not center-align or apply special spacing for names or actions.\n"
            "- Maintain the correct order of scenes and events.\n\n"
            f"Transcript:\n{text}\n\n"
            f"Plot Summary:\n{plot}\n\n"
            f"Scene Breakdown:\n{scene_descriptions}\n\n"
            f"Character Descriptions:\n{characters}"
        )

        screenplay = self.llm.LLM_request([prompt])
        return SectionScreenplay(segment=segment, screenplay=screenplay)

    async def _merge_screenplay_sections(self, sections):
        merged = "\n\n---\n\n".join([s.screenplay for s in sections])
        prompt = (
            "You are a professional screenwriter. Merge the following segmented screenplay sections into a single, clean, complete movie script.\n"
            "- Preserve all content and do NOT omit any lines of dialogue.\n"
            "- Ensure plain text format. Do NOT use any HTML, Markdown, or rich formatting.\n"
            "- Format dialogue with character names in uppercase, followed by their lines.\n"
            "- Avoid layout styles that require a screenplay viewer. This output should be readable in a plain text editor.\n"
            "- Maintain scene order and ensure smooth transitions between segments.\n\n"
            f"{merged}"
        )

        final = self.llm.LLM_request([prompt])
        return final

    async def run(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir, exist_ok=True)

        print("[INFO] Downloading full movie for processing...")
        movie_path = download_and_cache_video(
            self.cloud_storage_client,
            BUCKET_NAME,
            self.cloud_storage_media_path,
            self.workdir
        )

        print("[INFO] Segmenting movie into screenplay chunks...")
        plot_text = self.indexing.get("story.txt", "")
        story_sentences = self.indexing.get("story.json", [])
        scenes = self.indexing.get("scenes.json", [])
        segments = await self._segment_video(plot_text, story_sentences, scenes)

        print(f"[INFO] {len(segments)} segments planned.")
        semaphore = asyncio.Semaphore(15)

        async def process_segment_with_limit(seg):
            async with semaphore:
                print(f"[INFO] Generating screenplay for {seg.start.to_string()} - {seg.end.to_string()}")
                return await self._generate_screenplay_for_segment(seg, movie_path)

        screenplays = await asyncio.gather(
            *[process_segment_with_limit(seg) for seg in segments]
        )

        final_screenplay = await self._merge_screenplay_sections(screenplays)

        local_output_path = os.path.join(self.workdir, f"{self.media_base_name}_screenplay.txt")
        with open(local_output_path, "w", encoding="utf-8") as f:
            f.write(final_screenplay)
        print(f"[SUCCESS] Final screenplay saved to {local_output_path}")

        output_path = f"screenplay/{self.media_base_name}/{self.media_base_name}_screenplay.txt"
        self.cloud_storage_client.upload_files(BUCKET_NAME, local_output_path, output_path)
        print(f"[SUCCESS] Saved to: {output_path}")
        return output_path
