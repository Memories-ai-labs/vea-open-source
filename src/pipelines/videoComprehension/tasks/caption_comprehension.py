"""
Hybrid comprehension task using Memories.ai Chat + Caption APIs.

- Chat API (RAG-based): Used for video indexing, rough summary, and people descriptions.
  This leverages Memories.ai's indexed video data for efficient context retrieval.

- Caption API (direct LLM): Used ONLY for scene-by-scene descriptions.
  This provides better timestamped scene analysis than Chat API's RAG search.

For long videos (>20 min), splits into segments and processes each segment
separately to avoid output token limits, then merges the results.
"""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from lib.llm.MemoriesAiManager import MemoriesAiManager
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.utils.media import (
    get_video_duration,
    seconds_to_hhmmss,
    seconds_to_mmss,
    downsample_video,
    preprocess_long_video,
)

logger = logging.getLogger(__name__)

# Segment duration for Caption API (10 minutes)
# Shorter segments are more reliable and avoid output token limits
SEGMENT_DURATION_SECONDS = 10 * 60  # 10 minutes


class CaptionComprehension:
    """
    Hybrid comprehension using Chat API + Caption API.

    - Chat API: Upload, index, summary, and people descriptions (RAG-based, efficient)
    - Caption API: Scene-by-scene descriptions only (direct LLM analysis, better timestamps)
    """

    def __init__(
        self,
        memories_manager: MemoriesAiManager,
        gemini_llm: GeminiGenaiManager,
        callback_url: str,
        register_callback: Callable[[str], asyncio.Future],
        scene_interval_seconds: int = 20,
        force_reupload: bool = False,
    ):
        """
        Args:
            memories_manager: Memories.ai API client.
            gemini_llm: Gemini LLM for structured output parsing.
            callback_url: Public webhook URL for Caption API callbacks.
            register_callback: Function to register a pending callback (returns Future).
            scene_interval_seconds: Target interval for scene descriptions.
            force_reupload: If True, re-upload video even if already indexed.
        """
        self.memories = memories_manager
        self.gemini = gemini_llm
        self.callback_url = callback_url
        self.register_callback = register_callback
        self.scene_interval = scene_interval_seconds
        self.force_reupload = force_reupload

    async def __call__(
        self,
        local_video_path: str,
        video_url: Optional[str] = None,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Process a video file using Chat API (summary/people) + Caption API (scenes).

        Args:
            local_video_path: Path to local video file.
            video_url: Optional public URL (if available, used for Chat API upload).
            timeout: Maximum seconds to wait for Caption API callbacks.

        Returns:
            Dict with rough_summary, people, scenes, and video_no.
        """
        print(f"[CAPTION COMPREHENSION] Starting hybrid comprehension for: {local_video_path}")

        # Get video duration
        duration = get_video_duration(local_video_path)
        if not duration:
            duration = 300.0
            print("[CAPTION COMPREHENSION] Could not determine duration, defaulting to 5 minutes")
        print(f"[CAPTION COMPREHENSION] Video duration: {duration:.1f} seconds ({duration/60:.1f} min)")

        # =========================================================================
        # STEP 1: Upload and index via Chat API (same as MemoriesComprehension)
        # =========================================================================
        print("[CAPTION COMPREHENSION] Step 1/4: Uploading to Memories.ai (Chat API)...")
        video_no = await self._upload_and_index(local_video_path, video_url)
        print(f"[CAPTION COMPREHENSION] Video indexed, video_no: {video_no}")

        # =========================================================================
        # STEP 2: Get rough summary via Chat API (RAG-based, efficient)
        # =========================================================================
        print("[CAPTION COMPREHENSION] Step 2/4: Getting rough summary (Chat API)...")
        rough_summary = await self._get_rough_summary_chat(video_no)
        print(f"[CAPTION COMPREHENSION] Summary: {rough_summary[:200]}...")

        # =========================================================================
        # STEP 3: Get people descriptions via Chat API (RAG-based, efficient)
        # =========================================================================
        print("[CAPTION COMPREHENSION] Step 3/4: Getting people descriptions (Chat API)...")
        people = await self._get_people_description_chat(video_no, rough_summary)
        print(f"[CAPTION COMPREHENSION] People: {people[:200]}...")

        # =========================================================================
        # STEP 4: Get scene-by-scene via Caption API (direct LLM, better timestamps)
        # This uses summary + people as context for better character identification
        # =========================================================================
        print("[CAPTION COMPREHENSION] Step 4/4: Getting scene descriptions (Caption API)...")
        scenes = await self._get_scenes_caption(
            local_video_path, video_url, duration, rough_summary, people, timeout
        )
        print(f"[CAPTION COMPREHENSION] Got {len(scenes)} scenes")

        return {
            "video_no": video_no,
            "rough_summary": rough_summary,
            "people": people,
            "scenes": scenes,
        }

    # =========================================================================
    # Chat API methods (for upload, summary, people)
    # =========================================================================

    async def _upload_and_index(
        self,
        local_video_path: str,
        video_url: Optional[str],
    ) -> str:
        """Upload video to Memories.ai and wait for indexing (Chat API)."""
        original_path = Path(local_video_path)
        base_name = original_path.stem.replace("_memories_480p", "").replace("_memories_720p", "")

        # Check file size to determine if we need to downsample
        file_size_mb = os.path.getsize(local_video_path) / (1024 * 1024)
        if file_size_mb > 200 or "_memories_480p" in original_path.stem:
            memories_video_name = f"{base_name}_memories_480p"
        else:
            memories_video_name = base_name

        # Check if already exists (unless force_reupload)
        if not self.force_reupload:
            print(f"[CAPTION COMPREHENSION] Checking if '{memories_video_name}' exists...")
            existing = await self.memories.find_video_by_name(memories_video_name)

            if existing and existing.status == "PARSE":
                print(f"[CAPTION COMPREHENSION] Video already indexed: {existing.video_no}")
                return existing.video_no
            elif existing and existing.status in ("UNPARSE", "PENDING"):
                print(f"[CAPTION COMPREHENSION] Video uploading, waiting: {existing.video_no}")
                await self.memories.wait_for_ready(existing.video_no, timeout=600)
                return existing.video_no

        # Need to upload
        print("[CAPTION COMPREHENSION] Uploading video...")

        if video_url:
            video_no = await self.memories.upload_video_url(video_url)
        else:
            # Check if we need to downsample
            upload_path = local_video_path

            if file_size_mb > 200:
                cache_dir = original_path.parent / ".memories_cache"
                cache_dir.mkdir(exist_ok=True)
                downsampled_path = str(cache_dir / f"{original_path.stem}_memories_480p.mp4")

                if os.path.exists(downsampled_path):
                    print(f"[CAPTION COMPREHENSION] Using cached downsampled: {downsampled_path}")
                    upload_path = downsampled_path
                else:
                    print(f"[CAPTION COMPREHENSION] Downsampling {file_size_mb:.0f}MB video...")
                    await asyncio.to_thread(
                        downsample_video,
                        local_video_path,
                        downsampled_path,
                        32, 480, 12  # crf, height, fps
                    )
                    upload_path = downsampled_path
                    new_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
                    print(f"[CAPTION COMPREHENSION] Downsampled to {new_size_mb:.0f}MB")

            video_no = await self.memories.upload_video_file(upload_path)

        # Wait for indexing
        print("[CAPTION COMPREHENSION] Waiting for video processing...")
        await self.memories.wait_for_ready(video_no, timeout=600)

        return video_no

    async def _get_rough_summary_chat(self, video_no: str) -> str:
        """Get summary via Chat API (RAG-based)."""
        prompt = (
            "This is a long-form video. All videos, regardless of genre, style, or purpose, "
            "contain a story—sometimes profound, sometimes simple or surface-level. "
            "Your task is to create a comprehensive summary to help someone understand the story, "
            "sequence of events, or key message without watching it.\n\n"
            "Please provide:\n"
            "1. A detailed summary of the main narrative or message\n"
            "2. Key events in the order they appear (note: events may not be chronological - "
            "long-form videos may include flashbacks, time jumps, or non-linear presentation)\n"
            "3. Important themes or topics discussed\n"
            "4. Any significant visual elements, locations, or scenes\n"
            "5. A list of relationships or interactions observed between people\n\n"
            "Paraphrase any dialogue you observe. If parts of the video contain only credits, "
            "logo animations, or unrelated filler footage, omit those from your description.\n\n"
            "Be thorough and detailed."
        )

        response = await self.memories.chat(video_nos=[video_no], prompt=prompt)
        return response.text

    async def _get_people_description_chat(self, video_no: str, rough_summary: str) -> str:
        """Get people descriptions via Chat API (RAG-based)."""
        prompt = (
            f"Based on this video summary:\n{rough_summary}\n\n"
            "Using the video and the summary above, create a clean and complete list of ALL individuals "
            "observed in the video.\n\n"
            "For each person, include:\n"
            "- Name (if mentioned, shown on screen, or can be inferred)\n"
            "- Physical appearance and distinguishing features (clothing, hair, build, etc.)\n"
            "- Role, job, or position (if clear from context)\n"
            "- Key actions they perform or dialogue they speak (paraphrased)\n"
            "- Relationships or interactions with other people in the video\n\n"
            "If someone's name is unknown, refer to them by their role or distinguishing features.\n\n"
            "Group related individuals together if appropriate. Return plain text only."
        )

        response = await self.memories.chat(video_nos=[video_no], prompt=prompt)
        return response.text

    # =========================================================================
    # Caption API methods (for scene-by-scene only)
    # =========================================================================

    async def _get_scenes_caption(
        self,
        local_video_path: str,
        video_url: Optional[str],
        duration: float,
        rough_summary: str,
        people: str,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        """Get scene-by-scene descriptions using Caption API (direct LLM analysis).

        For long videos (>20 min), splits into segments and processes each.
        """
        # Prepare downsampled video for Caption API upload
        video_dir = Path(local_video_path).parent
        cache_dir = video_dir / ".memories_cache"
        cache_dir.mkdir(exist_ok=True)
        stem = Path(local_video_path).stem
        downsampled_path = cache_dir / f"{stem}_caption_480p.mp4"

        if not downsampled_path.exists():
            print(f"[CAPTION COMPREHENSION] Downsampling video for Caption API...")
            downsample_video(
                str(local_video_path),
                str(downsampled_path),
                target_height=480,
                fps=12,
                crf=28,
            )
            size_mb = downsampled_path.stat().st_size / (1024 * 1024)
            print(f"[CAPTION COMPREHENSION] Downsampled video: {size_mb:.1f}MB")
        else:
            size_mb = downsampled_path.stat().st_size / (1024 * 1024)
            print(f"[CAPTION COMPREHENSION] Using cached downsampled: {size_mb:.1f}MB")

        # For short videos (<= segment duration), process directly
        if duration <= SEGMENT_DURATION_SECONDS:
            print(f"[CAPTION COMPREHENSION] Short video ({duration/60:.1f} min), processing directly")
            return await self._get_scenes_for_segment(
                str(downsampled_path), video_url, duration, rough_summary, people, timeout,
                segment_start_seconds=0, segment_number=1
            )

        # For long videos, split into segments
        print(f"[CAPTION COMPREHENSION] Long video ({duration/60:.1f} min), splitting into {SEGMENT_DURATION_SECONDS//60} min segments...")

        segments_dir = tempfile.mkdtemp(prefix="caption_segments_")
        print(f"[CAPTION COMPREHENSION] Segment temp dir: {segments_dir}")

        # Split video into segments
        segments = await preprocess_long_video(
            local_video_path,
            segments_dir,
            interval_seconds=SEGMENT_DURATION_SECONDS,
            crf=28,
            target_height=480,
            fps=12,
        )

        print(f"[CAPTION COMPREHENSION] Created {len(segments)} segments")

        # Process all segments in PARALLEL for speed
        print(f"[CAPTION COMPREHENSION] Processing {len(segments)} segments in parallel...")

        async def process_segment(seg: dict) -> tuple[int, List[Dict[str, Any]]]:
            """Process a single segment and return (segment_num, scenes)."""
            segment_path = str(seg["path"])
            segment_start = seg["start"]
            segment_end = seg["end"]
            segment_num = seg["segment_number"]
            segment_duration = segment_end - segment_start

            print(f"[CAPTION COMPREHENSION] Starting segment {segment_num}/{len(segments)} "
                  f"({seconds_to_mmss(segment_start)} - {seconds_to_mmss(segment_end)})")

            try:
                segment_scenes = await self._get_scenes_for_segment(
                    segment_path,
                    video_url=None,
                    duration=segment_duration,
                    rough_summary=rough_summary,
                    people=people,
                    timeout=timeout,
                    segment_start_seconds=segment_start,
                    segment_number=segment_num,
                )
                print(f"[CAPTION COMPREHENSION] Segment {segment_num} completed: {len(segment_scenes)} scenes")
                return (segment_num, segment_scenes)

            except Exception as e:
                print(f"[CAPTION COMPREHENSION] Segment {segment_num} failed: {e}")
                return (segment_num, [])

        # Run all segments in parallel
        tasks = [process_segment(seg) for seg in segments]
        results = await asyncio.gather(*tasks)

        # Merge results in order and assign sequential IDs
        all_scenes = []
        scene_id = 1

        for segment_num, segment_scenes in sorted(results, key=lambda x: x[0]):
            for scene in segment_scenes:
                scene["id"] = scene_id
                scene_id += 1
            all_scenes.extend(segment_scenes)

        # Cleanup
        try:
            import shutil
            shutil.rmtree(segments_dir)
        except Exception as e:
            print(f"[CAPTION COMPREHENSION] Failed to cleanup: {e}")

        print(f"[CAPTION COMPREHENSION] Total scenes: {len(all_scenes)}")
        return all_scenes

    async def _caption_and_wait(
        self,
        local_video_path: str,
        video_url: Optional[str],
        user_prompt: str,
        system_prompt: str,
        timeout: float,
    ) -> str:
        """Send Caption API request and wait for callback result."""
        import time

        # Send the caption request
        if video_url:
            task_id = await self.memories.caption_video_url(
                video_url=video_url,
                user_prompt=user_prompt,
                callback_url=self.callback_url,
                system_prompt=system_prompt,
                thinking=False,
            )
        else:
            task_id = await self.memories.caption_video_file(
                file_path=local_video_path,
                user_prompt=user_prompt,
                callback_url=self.callback_url,
                system_prompt=system_prompt,
                thinking=False,
            )

        # Register callback and wait
        future = self.register_callback(task_id)
        print(f"[CAPTION COMPREHENSION] Waiting for callback (task: {task_id})...")

        # Wait with periodic progress updates
        start_time = time.time()
        progress_interval = 30.0

        while True:
            try:
                wait_time = min(progress_interval, timeout - (time.time() - start_time))
                if wait_time <= 0:
                    raise asyncio.TimeoutError()

                result = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=wait_time
                )
                break

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"[CAPTION COMPREHENSION] Timeout after {int(elapsed)}s "
                        f"(task: {task_id}). Check webhook connectivity."
                    )
                print(f"[CAPTION COMPREHENSION] Still waiting... ({int(elapsed)}s elapsed)")

        # Check result
        if result.get("status") != 0:
            raise RuntimeError(f"[CAPTION COMPREHENSION] Caption failed: {result}")

        text = result.get("data", {}).get("text", "")
        elapsed = int(time.time() - start_time)
        print(f"[CAPTION COMPREHENSION] Got result in {elapsed}s - {len(text)} chars")

        return text

    async def _get_scenes_for_segment(
        self,
        segment_path: str,
        video_url: Optional[str],
        duration: float,
        rough_summary: str,
        people: str,
        timeout: float,
        segment_start_seconds: float = 0,
        segment_number: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get scene descriptions for a single segment using Caption API."""
        system_prompt = (
            "You are a professional video analyst. Provide detailed scene-by-scene "
            "descriptions with accurate timestamps. Format exactly as requested."
        )

        max_ts = seconds_to_mmss(duration)

        # Calculate expected number of scenes for this segment
        expected_scenes = max(1, int(duration / self.scene_interval))

        user_prompt = (
            f"Video summary for context:\n{rough_summary[:1000]}...\n\n"
            f"People in the video:\n{people[:500]}...\n\n"
            f"This is segment {segment_number}. Duration: {int(duration)} seconds (~{int(duration/60)} min).\n\n"
            "Provide scene-by-scene descriptions covering the ENTIRE segment from start to finish.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            f"1. Each scene should be approximately {self.scene_interval} seconds long\n"
            f"2. You MUST produce approximately {expected_scenes} scenes to cover the full {int(duration)} seconds\n"
            "3. Scenes must be CONSECUTIVE with no gaps - each scene's start time equals the previous scene's end time\n"
            "4. Cover from 0:00 all the way to the end of the segment\n\n"
            "For each scene include:\n"
            "- Timestamp range in [MM:SS - MM:SS] format\n"
            "- Who appears (use names from people list when possible)\n"
            "- What actions take place\n"
            "- Any dialogue or text shown (paraphrased)\n\n"
            "IMPORTANT: Timestamps are RELATIVE to this segment (starting at 0:00). "
            f"Valid range: 0:00 to {max_ts}.\n\n"
            "Format example:\n"
            f"[0:00 - 0:{self.scene_interval:02d}] Description of first scene...\n"
            f"[0:{self.scene_interval:02d} - 0:{self.scene_interval*2:02d}] Description of second scene...\n"
            f"...continue until you reach {max_ts}"
        )

        text = await self._caption_and_wait(
            segment_path, video_url, user_prompt, system_prompt, timeout
        )

        # Parse text into scenes
        scenes = self._parse_scene_response(text, duration)

        if not scenes:
            print(f"[CAPTION COMPREHENSION] No scenes parsed from segment {segment_number}")
            return []

        # Convert relative timestamps to absolute
        for scene in scenes:
            try:
                start_parts = scene['start_timestamp_raw'].split(':')
                end_parts = scene['end_timestamp_raw'].split(':')

                rel_start = int(start_parts[0]) * 60 + int(start_parts[1])
                rel_end = int(end_parts[0]) * 60 + int(end_parts[1])

                abs_start = segment_start_seconds + rel_start
                abs_end = segment_start_seconds + rel_end

                scene['start_timestamp'] = seconds_to_hhmmss(abs_start)
                scene['end_timestamp'] = seconds_to_hhmmss(abs_end)
                scene['segment_num'] = segment_number

                del scene['start_timestamp_raw']
                del scene['end_timestamp_raw']

            except (ValueError, IndexError, KeyError) as e:
                print(f"[CAPTION COMPREHENSION] Timestamp conversion failed: {e}")
                scene['start_timestamp'] = seconds_to_hhmmss(segment_start_seconds)
                scene['end_timestamp'] = seconds_to_hhmmss(segment_start_seconds + self.scene_interval)
                scene['segment_num'] = segment_number

        return scenes

    def _parse_scene_response(self, response: str, duration: float) -> List[Dict[str, str]]:
        """Parse text response into scene dicts."""
        import re
        scenes = []

        # Match patterns like [0:00 - 0:20] or [00:00-00:20]
        pattern = r'\[?(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})\]?\s*(.+?)(?=\[?\d{1,2}:\d{2}|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for start_ts, end_ts, description in matches:
            scenes.append({
                "start_timestamp_raw": start_ts.strip(),
                "end_timestamp_raw": end_ts.strip(),
                "scene_description": description.strip()
            })

        # Fallback: line-by-line parsing
        if not scenes:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                ts_match = re.match(r'(\d{1,2}:\d{2})\s*[-–]?\s*(\d{1,2}:\d{2})?\s*[:\-]?\s*(.+)', line)
                if ts_match:
                    start_ts = ts_match.group(1)
                    end_ts = ts_match.group(2) or self._add_seconds(start_ts, self.scene_interval)
                    description = ts_match.group(3)
                    scenes.append({
                        "start_timestamp_raw": start_ts,
                        "end_timestamp_raw": end_ts,
                        "scene_description": description
                    })

        print(f"[CAPTION COMPREHENSION] Parsed {len(scenes)} scenes")
        return scenes

    def _add_seconds(self, timestamp: str, seconds: int) -> str:
        """Add seconds to MM:SS timestamp."""
        parts = timestamp.split(':')
        if len(parts) == 2:
            mins, secs = int(parts[0]), int(parts[1])
            total_secs = mins * 60 + secs + seconds
            return f"{total_secs // 60}:{total_secs % 60:02d}"
        return timestamp
