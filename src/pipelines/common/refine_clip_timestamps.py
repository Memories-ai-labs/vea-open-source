import os
import asyncio
import subprocess
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

        # Initialize ElevenLabs subtitle generator
        self.subtitle_generator = GenerateSubtitles(
            output_dir=os.path.join(self.workdir, "subs"),
            model_id="scribe_v1",
        )

    def _download_if_needed(self, cloud_path):
        filename = os.path.basename(cloud_path)
        if filename in self.downloaded_files:
            print(f"[DEBUG] Reusing cached: {filename}")
            return self.downloaded_files[filename]
        local_path = os.path.join(self.workdir, filename)
        print(f"[INFO] Downloading {cloud_path} → {local_path}")
        self.gcs_client.download_files(self.bucket_name, cloud_path, local_path)
        self.downloaded_files[filename] = local_path
        return local_path

    def _export_expanded_clip(self, clip, index):
        start_sec = max(0, parse_time_to_seconds(clip["start"]) - 10)
        end_sec = parse_time_to_seconds(clip["end"]) + 10
        duration = end_sec - start_sec

        movie_path = self._download_if_needed(clip["cloud_storage_path"])
        out_path = os.path.join(self.workdir, f"expanded_clip_{index}.mp4")

        print(f"[INFO] Exporting clip {clip['id']} with padding: {seconds_to_hhmmss(start_sec)}–{seconds_to_hhmmss(end_sec)}")
        cmd = [
            "ffmpeg", "-ss", str(start_sec), "-i", movie_path, "-t", str(duration),
            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path, start_sec

    async def __call__(self, chosen_clips):
        print("[TASK] Refining timestamps using ElevenLabs + Gemini...")

        all_clips_for_prompt = []
        for i, clip in enumerate(chosen_clips):
            print(f"[INFO] Processing clip ID: {clip['id']} | {clip['start']}–{clip['end']}")
            expanded_path, expanded_start_sec = self._export_expanded_clip(clip, i)

            # Run ElevenLabs transcription and apply global offset
            transcription_result = self.subtitle_generator(
                audio_path=expanded_path,
                global_start_time=expanded_start_sec
            )
            words = transcription_result.get("words", [])

            # Get pretty-printed text for Gemini
            pretty_transcript = self.subtitle_generator.pretty_print_words(words)

            all_clips_for_prompt.append({
                "id": clip["id"],
                "file_name": clip["file_name"],
                "original_start": clip["start"],
                "original_end": clip["end"],
                "narration": clip.get("narration", ""),
                "pretty_transcript": pretty_transcript
            })

        # --- Batch Gemini refinement ---
        BATCH_SIZE = 1
        refined_by_id = {}

        for batch_idx in range(0, len(all_clips_for_prompt), BATCH_SIZE):
            batch = all_clips_for_prompt[batch_idx:batch_idx + BATCH_SIZE]

            prompt_lines = []
            for clip_info in batch:
                prompt_lines.append(
                    f"Clip ID {clip_info['id']} (file: {clip_info['file_name']})\n"
                    f"Original range: {clip_info['original_start']} - {clip_info['original_end']}\n"
                    f"Narration: {clip_info['narration']}\n"
                    f"Transcript word timings:\n" +
                    clip_info["pretty_transcript"]
                )

            prompt = (
                f"""
                \n\n
                You are refining the start and end times of selected video clips to ensure they align naturally with spoken sentences.
                ### How to Read the Transcript Timings Below:
                - The transcript for each clip is formatted as a **word-level pretty print list**.
                - Each line represents either a word or a pause (spacing) from the speech-to-text output.
                #### Example line format:
                [Word Text]        start: HH:MM:SS,mmm, end: HH:MM:SS,mmm, type: word    , speaker: None  
                [Spacing]          start: HH:MM:SS,mmm, end: HH:MM:SS,mmm, type: spacing , speaker: None  
                - Lines with `type: word` represent spoken words, with exact start and end timestamps.
                - Lines with `type: spacing` represent silent gaps between words, also with start and end times.
                ---
                ### Your Refinement Rules:
                1. **Start Time Rule:**  
                For each clip, set the `refined_start` **at the timestamp from a word's `start` time that occurs before the beginning of a sentence**, leaving **~0.5 seconds of buffer** before the sentence starts if space allows within the original clip range.  
                If no buffer is available, then start exactly at the first word's `start` timestamp.
                ---
                2. **End Time Rule:**  
                Set the `refined_end` **at the timestamp from a word's `end` time that falls after the end of the final sentence in the clip**, leaving **~0.5 seconds of buffer** after the sentence if space allows.  
                If no room for buffer, end exactly after the final word of the sentence.
                ---
                3. **Identifying Sentences:**  
                - Use the `spacing` lines as clues for sentence breaks:  
                - **Short gaps (under ~500 ms)** mean words are part of the same sentence.  
                - **Longer gaps (more than ~500 ms)** likely indicate the end of a sentence.
                - **Do not start or end a clip in the middle of a spoken sentence.**  
                - **It's acceptable to include multiple full sentences in one clip.**
                ---
                4. **Narration Reference:**  
                Use the `narration` field as a hint for what content the clip is meant to cover.
                ---
                "### Clip Details:\n\n"
                {"".join(prompt_lines)}
                ### Output format:  
                For each clip, output a JSON object like:  
                "id": "clip_id",  
                "refined_start": "HH:MM:SS",  
                "refined_end": "HH:MM:SS"  
                \n\n\n
                """
            )

            print(f"[LLM] Sending batch {batch_idx // BATCH_SIZE + 1} to Gemini...")
            refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[RefinedClipTimestamps]
                }
            )
            print(f"[LLM] Gemini batch {batch_idx // BATCH_SIZE + 1} complete.")

            # Parse results
            if refined and isinstance(refined[0], dict):
                refined = [RefinedClipTimestamps(**c) for c in refined]

            for ref in refined:
                refined_by_id[ref.id] = ref

        # --- Apply refinements back to clips ---
        for clip in chosen_clips:
            ref = refined_by_id.get(clip["id"])
            if ref:
                print(f"[UPDATE] Clip {clip['id']}: {clip['start']}–{clip['end']} → {ref.refined_start}–{ref.refined_end}")
                clip["start"] = ref.refined_start
                clip["end"] = ref.refined_end
            else:
                print(f"[WARN] No refined timestamps returned for clip ID: {clip['id']}")

        return chosen_clips
