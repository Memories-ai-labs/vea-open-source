import os
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.pipelines.common.schema import RefinedClipTimestamps
from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss
from src.pipelines.common.generate_subtitles import GenerateSubtitles

class RefineClipTimestamps:
    def __init__(self, llm, workdir, gcs_client, bucket_name):
        self.llm = llm
        self.workdir = workdir
        self.gcs_client = gcs_client
        self.bucket_name = bucket_name
        self.downloaded_files = {}
        self.semaphore = asyncio.Semaphore(6)  # Max 6 concurrent threads
        self.executor = ThreadPoolExecutor(max_workers=6)

        # Init ElevenLabs subtitle generator
        self.subtitle_generator = GenerateSubtitles(
            output_dir=os.path.join(self.workdir, "subs"),
            model_id="scribe_v1",
        )

    def _download_if_needed(self, cloud_path):
        filename = os.path.basename(cloud_path)
        if filename in self.downloaded_files:
            return self.downloaded_files[filename]
        local_path = os.path.join(self.workdir, filename)
        print(f"[INFO] Downloading {cloud_path} → {local_path}")
        self.gcs_client.download_files(self.bucket_name, cloud_path, local_path)
        self.downloaded_files[filename] = local_path
        return local_path

    def _export_expanded_clip(self, clip):
        start_sec = max(0, parse_time_to_seconds(clip["start"]) - 10)
        end_sec = parse_time_to_seconds(clip["end"]) + 10
        duration = end_sec - start_sec

        movie_path = self._download_if_needed(clip["cloud_storage_path"])
        out_path = os.path.join(self.workdir, f"expanded_clip_{clip['id']}.mp4")

        print(f"[INFO] Exporting clip {clip['id']} with padding: {seconds_to_hhmmss(start_sec)}–{seconds_to_hhmmss(end_sec)}")
        cmd = [
            "ffmpeg", "-ss", str(start_sec), "-i", movie_path, "-t", str(duration),
            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path, start_sec

    async def _process_single_clip(self, clip):
        async with self.semaphore:
            print(f"[TASK] Refining timestamps for clip ID: {clip['id']} | {clip['start']}–{clip['end']}")

            expanded_path, expanded_start_sec = await asyncio.to_thread(self._export_expanded_clip, clip)

            transcription_result = await asyncio.to_thread(
                self.subtitle_generator,
                audio_path=expanded_path,
                global_start_time=expanded_start_sec
            )
            words = transcription_result.get("words", [])
            if not words:
                print(f"[WARN] No transcription words for clip ID: {clip['id']}")
                return clip

            pretty_transcript = self.subtitle_generator.pretty_print_words(words)

            prompt = (
                "You are refining the start and end times of a single video clip so it aligns naturally with spoken sentences.\n\n"
                "Transcript format explanation:\n"
                "- Each line represents a word or a pause (spacing).\n"
                "- Lines with `type: word` are spoken words with exact start and end timestamps.\n"
                "- Lines with `type: spacing` represent silent gaps between words, also with start/end times.\n\n"
                "Refinement Rules:\n"
                "- avoid starting the clip in the middle of a sentence or ending the clip in the middle of a sentence. cutting off speech is jarring\n"
                "- leave some buffer time before the first word and after the last word if space allows\n"
                "- try to keep full sentences if possible, avoid super short clips, and make sure the refined clip isnt too much shorter or longer than the original\n"
                "- if the clip does not contain any spoken words, you can leave the timestamps unchanged\n"
                "- try to respect the natural pace and flow of the spoken content\n\n"
                
                "Output JSON format:\n"
                "{\n"
                "  \"id\": integer,\n"
                "  \"refined_start\": \"HH:MM:SS\",\n"
                "  \"refined_end\": \"HH:MM:SS\"\n"
                "}\n\n"
                f"Clip ID: {clip['id']}\n"
                f"File: {clip['file_name']}\n"
                f"Original range: {clip['start']} - {clip['end']}\n"
                f"Narration: {clip.get('narration', '')}\n\n"
                "Transcript word timings:\n"
                f"{pretty_transcript}"
            )

            refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": RefinedClipTimestamps
                }
            )

            print(f"[UPDATE] Clip {clip['id']}: {clip['start']}–{clip['end']} → {refined['refined_start']}–{refined['refined_end']}")
            clip["start"] = refined["refined_start"]
            clip["end"] = refined["refined_end"]
            return clip

    async def __call__(self, clips):
        tasks = [self._process_single_clip(clip) for clip in clips]
        refined_clips = await asyncio.gather(*tasks)

        # --- Ensure no overlaps between adjacent clips ---
        refined_clips.sort(key=lambda c: parse_time_to_seconds(c["start"]))
        for i in range(1, len(refined_clips)):
            prev_end = parse_time_to_seconds(refined_clips[i - 1]["end"])
            current_start = parse_time_to_seconds(refined_clips[i]["start"])
            if current_start < prev_end:
                # Push start forward by 0.1s after previous end
                new_start_sec = prev_end + 0.1
                refined_clips[i]["start"] = seconds_to_hhmmss(new_start_sec)
                print(f"[FIX] Adjusted clip {refined_clips[i]['id']} start to avoid overlap: {refined_clips[i]['start']}")

        return refined_clips
