import os
import json
import asyncio
import subprocess
import whisperx
from whisperx import align

from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss
from src.pipelines.common.schema import RefinedClipTimestamps

class RefineClipTimestamps:
    def __init__(self, llm, workdir, gcs_client, bucket_name, device="cpu"):
        self.llm = llm
        self.workdir = workdir
        self.gcs_client = gcs_client
        self.bucket_name = bucket_name
        self.device = device
        print(f"[INFO] Loading WhisperX transcription model on {device}...")
        self.whisperx_model = whisperx.load_model("base", self.device, compute_type="float32")
        self.align_model = None
        self.align_metadata = None
        self.downloaded_files = {}

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

    def _transcribe_and_align_clip(self, video_path):
        print(f"[INFO] Running WhisperX transcription on: {video_path}")
        audio = whisperx.load_audio(video_path)
        result = self.whisperx_model.transcribe(audio)
        language = result["language"]

        if self.align_model is None or self.align_metadata is None or self.align_metadata["language"] != language:
            print(f"[INFO] Loading WhisperX align model for language: {language}")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code=language, device=self.device)

        print(f"[INFO] Running forced alignment on: {video_path}")
        aligned_segments = align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            device=self.device
        )

        return aligned_segments

    def _format_segments_with_global_times(self, segments, expanded_clip_start_sec):
        lines = []
        for seg in segments:
            global_start = expanded_clip_start_sec + seg['start']
            global_end = expanded_clip_start_sec + seg['end']
            line = f"{seconds_to_hhmmss(global_start)} - {seconds_to_hhmmss(global_end)}: {seg['text'].strip()}"
            lines.append(line)
        return lines

    async def __call__(self, chosen_clips):
        print("[TASK] Refining timestamps using WhisperX (with alignment) + Gemini...")

        all_clips_for_prompt = []
        for i, clip in enumerate(chosen_clips):
            print(f"[INFO] Processing clip ID: {clip['id']} | {clip['start']}–{clip['end']}")
            expanded_path, expanded_start_sec = self._export_expanded_clip(clip, i)
            whisperx_result = self._transcribe_and_align_clip(expanded_path)
            segments_text_lines = self._format_segments_with_global_times(
                whisperx_result["segments"], max(0, parse_time_to_seconds(clip["start"]) - 10)
            )
            all_clips_for_prompt.append({
                "id": clip["id"],
                "file_name": clip["file_name"],
                "original_start": clip["start"],
                "original_end": clip["end"],
                "narration": clip.get("narration", ""),
                "segments": segments_text_lines,
            })

        # Refine in batches of 8
        BATCH_SIZE = 8
        refined_by_id = {}
        for batch_idx in range(0, len(all_clips_for_prompt), BATCH_SIZE):
            batch = all_clips_for_prompt[batch_idx:batch_idx + BATCH_SIZE]

            prompt_lines = []
            for clip_info in batch:
                prompt_lines.append(
                    f"Clip ID {clip_info['id']} (file: {clip_info['file_name']})\n"
                    f"Original range: {clip_info['original_start']} - {clip_info['original_end']}\n"
                    f"Narration: {clip_info['narration']}\n"
                    f"Transcript segments:\n" +
                    "\n".join(clip_info['segments'])
                )
            prompt = (
                "You are refining the start and end times of selected video clips so that they do not cut off spoken dialogue or important sentences, and so the content is well-aligned with the provided narration. "
                "For each clip, use the transcript segments with their global timecodes to make sure you do not start or end a clip in the middle of someone's sentence or important action. "
                "Make sure your cut points are natural breaks in people's speech, this is the most important! "
                "Try to match the narration, but always avoid cutting off spoken audio.\n\n"
                "For each clip, return a JSON object with these fields:\n"
                "- `id`: integer\n"
                "- `refined_start`: adjusted global start time (HH:MM:SS)\n"
                "- `refined_end`: adjusted global end time (HH:MM:SS)\n"
                "The new refined range must be within the padded range you see above (never before or after the transcript's earliest/latest segment).\n\n"
                "Here are the clips and their transcript segments:\n\n" +
                "\n\n".join(prompt_lines)
            )

            print(f"[LLM] Sending batch {batch_idx // BATCH_SIZE + 1} to Gemini for refinement...")
            refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[RefinedClipTimestamps]
                }
            )
            print(f"[LLM] Gemini batch {batch_idx // BATCH_SIZE + 1} refinement complete.")

            # Accept both dicts and models
            if refined and isinstance(refined[0], dict):
                refined = [RefinedClipTimestamps(**c) for c in refined]
            for ref in refined:
                refined_by_id[ref.id] = ref

        # Apply refinements
        for clip in chosen_clips:
            ref = refined_by_id.get(clip["id"])
            if ref:
                print(f"[UPDATE] Clip {clip['id']}: {clip['start']}–{clip['end']} → {ref.refined_start}–{ref.refined_end}")
                clip["start"] = ref.refined_start
                clip["end"] = ref.refined_end
            else:
                print(f"[WARN] No refined timestamps returned for clip ID: {clip['id']}")

        return chosen_clips
