import os
import subprocess
import asyncio
import cv2
import json

from concurrent.futures import ThreadPoolExecutor
from src.pipelines.common.schema import RefinedClipTimestamps
from lib.utils.media import parse_time_to_seconds, seconds_to_hhmmss, download_and_cache_video
from src.pipelines.common.generate_subtitles import GenerateSubtitles

def overlay_debug_info(
    video_path,
    words,
    original_start,
    original_end,
    refined_start,
    refined_end,
    output_path,
    font_scale=1,
    font_thickness=2
):

    temp_output_path = output_path.replace(".mp4", "_silent.mp4")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    word_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Draw debug circles
        if original_start <= current_time_sec <= original_end:
            cv2.circle(frame, (width - 60, 60), 20, (255, 0, 0), -1)  # Blue
        if refined_start <= current_time_sec <= refined_end:
            cv2.circle(frame, (width - 60, 120), 20, (0, 255, 0), -1)  # Green

        # Show current spoken word
        while word_index < len(words) and words[word_index]['end'] < current_time_sec:
            word_index += 1
        if word_index < len(words):
            word = words[word_index]
            if word['start'] <= current_time_sec <= word['end']:
                text = word['text']
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int((width - text_width) / 2)
                text_y = int(height - 50)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)

        out.write(frame)

    cap.release()
    out.release()

    # Use ffmpeg to merge audio back
    audio_path = video_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_output_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_output_path)

    print(f"[DEBUG] Saved visual debug video to: {output_path}")

class RefineClipTimestamps:
    def __init__(self, llm, workdir, gcs_client, bucket_name):
        self.llm = llm
        self.workdir = workdir
        self.gcs_client = gcs_client
        self.bucket_name = bucket_name
        self.downloaded_files = {}
        self.semaphore = asyncio.Semaphore(6)  # Max 6 concurrent threads
        self.executor = ThreadPoolExecutor(max_workers=6)

        self.debug = True  # Enable debug mode for visual output


        # Init ElevenLabs subtitle generator
        self.subtitle_generator = GenerateSubtitles(
            output_dir=os.path.join(self.workdir, "subs"),
            model_id="scribe_v1",
        )

    def _download_if_needed(self, cloud_path):
        filename = os.path.basename(cloud_path)
        if filename in self.downloaded_files:
            return self.downloaded_files[filename]
        cache_dir = os.path.join(".cache", "refine_clips")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = download_and_cache_video(
            self.gcs_client,
            self.bucket_name,
            cloud_path,
            cache_dir,
        )
        print(f"[INFO] Using source {local_path}")
        self.downloaded_files[filename] = local_path
        return local_path

    def _export_expanded_clip(self, clip):
        start_ts = clip["start"]
        end_ts = clip["end"]

        start_sec = max(0, parse_time_to_seconds(start_ts) - 10)
        end_sec = parse_time_to_seconds(end_ts) + 10
        duration = end_sec - start_sec

        assert end_sec > start_sec, (
            f"[ERROR] Invalid export range for clip {clip['id']}: "
            f"start={start_ts} ({start_sec:.3f}s), "
            f"end={end_ts} ({end_sec:.3f}s)"
        )

        movie_path = self.downloaded_files[os.path.basename(clip["cloud_storage_path"])]
        out_path = os.path.join(self.workdir, f"expanded_clip_{clip['id']}.mp4")

        print(f"[INFO] Exporting clip {clip['id']} with padding: {seconds_to_hhmmss(start_sec)}–{seconds_to_hhmmss(end_sec)}")

        cmd = [
            "ffmpeg",
            "-i", movie_path,
            "-ss", str(start_sec),
            "-t", str(duration),
            "-vf", "scale=iw:ih",   # no change but triggers decode
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            out_path,
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

            # Save raw transcript for debugging
            if self.debug:
                transcript_path = os.path.join(self.workdir, f"transcript_clip_{clip['id']}.txt")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(pretty_transcript)
                print(f"[DEBUG] Saved raw transcript to: {transcript_path}")

            # -- Step 1: Gemini reasoning --
            reasoning_prompt = (
                "You are analyzing a video clip transcript to determine the best natural start and end time boundaries. "
                "These should preserve full spoken sentences and avoid cutting off dialogue. "
                "Consider the narration and scene description to keep the part of the clip that supports its purpose.\n\n"
                "Refinement Goals:\n"
                "- Preserve full sentences and do not cut off speech in the middle of a sentence. so you should mostly only cut at a period.\n"
                "- Based on the provided narration and scene description, adjust the boundaries to keep the most relevant content.\n"
                "- If there's no speech, timestamps can be unchanged.\n"
                "- Keep the length relatively close to the original, but making sure you dont cut off or start in the middle of a sentence is more important.\n\n"
                f"Clip ID: {clip['id']}\n"
                f"File: {clip['file_name']}\n"
                f"Original range: {clip['start']} - {clip['end']}\n"
                f"Narration: {clip.get('narration', 'N/A')}\n"
                f"Scene description: {clip.get('scene_description', 'N/A')}\n\n"
                "Transcript word timings:\n"
                f"{pretty_transcript}\n\n"
                "Please describe in text how you would determine the ideal boundaries. and clearly state the new start and end times in HH:MM:SS,mmm format.\n"
            )

            # Run Gemini reasoning prompt (text-only response)
            reasoning_text = await asyncio.to_thread(
                self.llm.LLM_request,
                [reasoning_prompt]
            )

            # Build schema-only prompt with reasoning transition
            schema_prompt = (
                f"{reasoning_text}\n\n"
                "Based on your explanation above, you have identified a natural and coherent region in the clip "
                "that aligns with full sentences and respects the purpose of the scene (e.g., narration or dialogue).\n\n"
                "Now, please convert your reasoning into a structured JSON output using the following format:\n"
                "{\n"
                "  \"id\": integer,\n"
                "  \"refined_start\": \"HH:MM:SS,mmm\",\n"
                "  \"refined_end\": \"HH:MM:SS,mmm\"\n"
                "}\n\n"
                "Only output the JSON object, and ensure the timestamps reflect your reasoning above."
            )

            # Get structured JSON output
            refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [schema_prompt],
                RefinedClipTimestamps
            )

            print(f"[UPDATE] Clip {clip['id']}: {clip['start']}–{clip['end']} → {refined['refined_start']}–{refined['refined_end']}")
            original_start_sec = parse_time_to_seconds(clip["start"])
            original_end_sec = parse_time_to_seconds(clip["end"])
            clip["start"] = refined["refined_start"]
            clip["end"] = refined["refined_end"]

            if self.debug:
                debug_file_path = os.path.join(self.workdir, f"debug_clip_{clip['id']}.txt")
                with open(debug_file_path, "w", encoding="utf-8") as f:
                    f.write("========== PROMPT ==========\n")
                    f.write(reasoning_prompt.strip() + "\n\n")
                    f.write("======= REASONING TEXT =======\n")
                    f.write(reasoning_text.strip() + "\n\n")
                    f.write("===== STRUCTURED OUTPUT =====\n")
                    f.write(json.dumps(refined, indent=2) + "\n")
                print(f"[DEBUG] Saved full prompt + reasoning + output to: {debug_file_path}")

                debug_output = os.path.join(self.workdir, f"debug_clip_{clip['id']}.mp4")
                unoffset_words = self.subtitle_generator.offset_words(words, -expanded_start_sec)
                overlay_debug_info(
                    video_path=expanded_path,
                    words=unoffset_words,
                    original_start=original_start_sec - expanded_start_sec,
                    original_end=original_end_sec - expanded_start_sec,
                    refined_start=parse_time_to_seconds(refined["refined_start"]) - expanded_start_sec,
                    refined_end=parse_time_to_seconds(refined["refined_end"]) - expanded_start_sec,
                    output_path=debug_output
                )

            return clip


    async def __call__(self, clips):
        # Download each movie file once before threading
        all_cloud_paths = {clip["cloud_storage_path"] for clip in clips}
        for path in all_cloud_paths:
            self._download_if_needed(path)

        # Run clip refinements in parallel (max 6 threads)
        tasks = [self._process_single_clip(clip) for clip in clips]
        refined_clips = await asyncio.gather(*tasks)

        # --- Ensure no overlaps between adjacent clips ---
        refined_clips.sort(key=lambda c: parse_time_to_seconds(c["start"]))
        for i in range(1, len(refined_clips)):
            prev_end = parse_time_to_seconds(refined_clips[i - 1]["end"])
            current_start = parse_time_to_seconds(refined_clips[i]["start"])
            current_end = parse_time_to_seconds(refined_clips[i]["end"])

            if current_start < prev_end:
                # Shift current clip's start just after prev clip's end, but ensure it stays before end
                new_start_sec = min(prev_end + 0.1, current_end - 0.1)
                if new_start_sec < current_end:
                    refined_clips[i]["start"] = seconds_to_hhmmss(new_start_sec)
                    print(f"[FIX] Adjusted clip {refined_clips[i]['id']} start to avoid overlap: {refined_clips[i]['start']}")
                else:
                    print(f"[WARN] Could not fix overlap for clip {refined_clips[i]['id']}: zero or negative duration")


        return refined_clips
