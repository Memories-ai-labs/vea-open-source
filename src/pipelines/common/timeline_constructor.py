"""
Timeline Constructor Module

Constructs video timelines from AI-generated clip plans. Handles:
- Downloading and normalizing source videos
- Attaching narration audio with priority modes
- Audio level balancing (LUFS-based)
- Dynamic cropping for aspect ratio conversion
- Background music mixing
- FCPXML export for professional editing software
"""

import os
import tempfile
import asyncio
import math
import subprocess
import json
import shutil
from copy import deepcopy
from pathlib import Path

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    AudioClip,
    CompositeAudioClip,
    concatenate_videoclips,
    vfx,
    afx,
)
import librosa
import numpy as np

from lib.utils.media import parse_time_to_seconds, download_and_cache_video
from lib.utils.metrics_collector import metrics_collector
from src.pipelines.common.dynamic_cropping import DynamicCropping
from src.pipelines.common.fcpxml_exporter import export_fcpxml
from src.pipelines.common.generate_subtitles import GenerateSubtitles
from src.pipelines.common import metadata_helpers
from src.pipelines.common import audio_processing


class TimelineConstructor:
    """
    Constructs video timelines from AI-generated clip plans.

    Priority Modes:
    - narration: TTS narration replaces clip audio, trim to narration length
    - clip_audio: Play twice - first with narration, then with boosted original audio
    - clip_video: Use entire segment uncut with original audio

    Output Structure:
        data/outputs/{project_name}/
        ├── {project_name}.mp4          # Final rendered video
        ├── {project_name}.fcpxml       # FCPXML for Final Cut Pro
        ├── clip_plan.json              # The clip plan used
        ├── footage/                    # Normalized source footage
        ├── narrations/                 # TTS narration audio files
        └── music/                      # Background music (if used)
    """

    ALLOWED_FRAME_RATES = (24, 25, 30, 50, 60, 120)
    DATA_OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs"

    def __init__(
        self,
        music_volume_multiplier: float = 0.5,
        gcs_client=None,
        bucket_name: str = None,
        llm=None,
    ):
        self.music_volume_multiplier = music_volume_multiplier
        self.gcs_client = gcs_client
        self.bucket_name = bucket_name
        self.llm = llm
        self._downloaded_files = {}
        self._output_dir = None  # Set when clips are processed

    # =========================================================================
    # Output Directory Management
    # =========================================================================

    def _setup_output_dir(self, project_name: str) -> Path:
        """Create output directory structure for the project."""
        self._output_dir = self.DATA_OUTPUTS_DIR / project_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self._output_dir / "footage").mkdir(exist_ok=True)
        (self._output_dir / "narrations").mkdir(exist_ok=True)
        (self._output_dir / "music").mkdir(exist_ok=True)

        print(f"[INFO] Output directory: {self._output_dir}")
        return self._output_dir

    def _load_audio_as_array(self, audio_path: str) -> AudioClip:
        """
        Load an audio file as a numpy-backed AudioClip to avoid FFmpeg reader state issues.

        The problem: AudioFileClip maintains an internal FFmpeg subprocess. When clips are
        concatenated or processed multiple times, this reader can get into a bad state where
        it tries to read past the end of the file, causing 'stdout is None' errors.

        The solution: Read the entire audio into memory as a numpy array, then create a
        new AudioClip from that array. This is stable and doesn't rely on FFmpeg state.
        """
        # Load audio with AudioFileClip to get metadata
        with AudioFileClip(audio_path) as temp_clip:
            fps = temp_clip.fps
            duration = temp_clip.duration
            nchannels = temp_clip.nchannels
            # Read entire audio into memory
            audio_array = temp_clip.to_soundarray(fps=fps)

        # Create a new AudioClip from the numpy array
        def make_frame(t):
            # Handle both scalar and array time inputs
            t = np.atleast_1d(t)
            # Convert time to sample indices
            indices = (t * fps).astype(int)
            # Clip to valid range
            indices = np.clip(indices, 0, len(audio_array) - 1)
            result = audio_array[indices]
            # Ensure consistent 2D shape (samples, channels)
            if result.ndim == 1 and nchannels > 1:
                result = result.reshape(-1, nchannels)
            return result

        stable_clip = AudioClip(make_frame, duration=duration, fps=fps)
        stable_clip.nchannels = nchannels
        return stable_clip

    # =========================================================================
    # Media Download and Normalization
    # =========================================================================

    def _download_media_file(self, file_name: str, cloud_storage_path: str) -> str:
        """Download and normalize a video file from cloud storage."""
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]["path"]

        # Ensure output dir is set up (fallback to file stem if called before run())
        if self._output_dir is None:
            self._setup_output_dir(Path(file_name).stem)

        # Download to cache, then normalize to output folder
        cache_dir = Path(tempfile.gettempdir()) / "vea_download_cache"
        cache_dir.mkdir(exist_ok=True)

        local_path = download_and_cache_video(
            self.gcs_client,
            self.bucket_name,
            cloud_storage_path,
            str(cache_dir),
        )
        normalized_info = self._normalize_frame_rate(file_name, local_path)
        self._downloaded_files[file_name] = normalized_info
        return normalized_info["path"]

    def _normalize_frame_rate(self, file_name: str, source_path: str) -> dict:
        """Normalize video to a standard frame rate and save to output footage folder."""
        original_fps = self._probe_frame_rate(source_path)
        target_fps = self._pick_target_fps(original_fps)

        # Save normalized footage to output directory
        footage_dir = self._output_dir / "footage"
        footage_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = footage_dir / f"{Path(file_name).stem}_fps{int(target_fps)}.mp4"

        # Check if valid normalized file already exists (not empty/corrupted)
        if normalized_path.exists() and normalized_path.stat().st_size > 1000:
            # Verify it's actually readable
            probe_fps = self._probe_frame_rate(str(normalized_path))
            if probe_fps > 0:
                return {
                    "path": str(normalized_path),
                    "fps": target_fps,
                    "original_fps": original_fps or target_fps,
                    "source_path": source_path,
                }
            else:
                print(f"[WARN] Existing normalized file is corrupted, re-normalizing: {normalized_path}")
                normalized_path.unlink()

        cmd = [
            "ffmpeg", "-y", "-i", source_path,
            "-vf", f"fps={target_fps}",
            "-vsync", "cfr",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart",
            str(normalized_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {
                "path": str(normalized_path),
                "fps": target_fps,
                "original_fps": original_fps or target_fps,
                "source_path": source_path,
            }
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN] Failed to normalize FPS for {file_name}: {e}")
            return {
                "path": source_path,
                "fps": original_fps or target_fps,
                "original_fps": original_fps or target_fps,
                "source_path": source_path,
            }

    def _probe_frame_rate(self, path: str) -> float:
        """Probe video frame rate using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams", [])
            if not streams:
                return 0.0
            stream = streams[0]
            avg = self._parse_rate(stream.get("avg_frame_rate"))
            return avg if avg > 0 else self._parse_rate(stream.get("r_frame_rate"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return 0.0

    @staticmethod
    def _parse_rate(rate_str: str) -> float:
        """Parse frame rate string like '30000/1001' or '30'."""
        if not rate_str or rate_str in {"0/0", "0", "N/A"}:
            return 0.0
        if "/" in rate_str:
            try:
                num, den = rate_str.split("/", 1)
                return float(num) / float(den) if float(den) != 0 else 0.0
            except ValueError:
                return 0.0
        try:
            return float(rate_str)
        except ValueError:
            return 0.0

    def _pick_target_fps(self, measured_fps: float) -> float:
        """Pick the closest standard frame rate."""
        if measured_fps <= 0:
            return 30.0
        return float(min(self.ALLOWED_FRAME_RATES, key=lambda x: abs(x - measured_fps)))

    # =========================================================================
    # Clip Processing
    # =========================================================================

    def _process_clip(
        self,
        clip: dict,
        narration_dir: str,
        narration_enabled: bool = True,
        original_audio: bool = True,
    ) -> list:
        """
        Process a single clip according to its priority mode.

        Returns a list of VideoFileClip objects (usually 1, but 2 for clip_audio priority).
        """
        clip_id = clip["id"]
        file_name = clip["file_name"]
        download_info = self._downloaded_files[file_name]
        video_path = download_info["path"]
        start_sec = parse_time_to_seconds(clip["start"])
        end_sec = parse_time_to_seconds(clip["end"])
        priority = clip.get("priority", "narration")

        # Build metadata for FCPXML export
        metadata = metadata_helpers._build_segment_metadata(
            clip, start_sec, end_sec, original_audio=original_audio
        )
        if download_info.get("fps"):
            metadata.setdefault("timeline", {})["frame_rate"] = download_info["fps"]

        # Load video
        video_clip = VideoFileClip(video_path)

        # Ensure video has audio (create silence if needed)
        if video_clip.audio is None:
            silent = AudioClip(lambda _: [0, 0], duration=video_clip.duration, fps=44100)
            video_clip = video_clip.with_audio(silent)

        video_clip = metadata_helpers._attach_metadata(video_clip, metadata)

        # If narration disabled, just return trimmed clip
        if not narration_enabled:
            video_clip = self._safe_subclip(video_clip, start_sec, end_sec)
            if not original_audio:
                video_clip = video_clip.with_effects([afx.MultiplyVolume(0)])
            metadata_helpers._register_clip_edit(clip, metadata)
            return [video_clip]

        # Load narration as numpy-backed clip to avoid FFmpeg reader state issues
        try:
            audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
            narration = self._load_audio_as_array(audio_path)
            narration_duration = float(narration.duration or 0.0)
        except Exception as e:
            print(f"[WARN] Failed to load narration for clip {clip_id}: {e}")
            video_clip = self._safe_subclip(video_clip, start_sec, end_sec)
            metadata_helpers._register_clip_edit(clip, metadata)
            return [video_clip]

        # Trim video to clip range
        trim_end = min(end_sec, float(video_clip.duration or end_sec))
        base_clip = self._safe_subclip(video_clip, start_sec, trim_end)
        video_duration = float(base_clip.duration or 0.0)

        # Slow down video if shorter than narration
        if video_duration < narration_duration and narration_duration > 0:
            speed_factor = video_duration / narration_duration
            metadata_helpers._record_retime_adjustment(
                metadata,
                speed_factor=speed_factor,
                reason="match_narration_length",
                original_duration=video_duration,
                target_duration=narration_duration,
            )
            base_clip = base_clip.with_effects([vfx.MultiplySpeed(speed_factor)])

        # Calculate volume adjustment for clip_audio priority
        volume_multiplier = self._calculate_volume_match(base_clip.audio, narration)

        # Mute original audio if requested
        if not original_audio:
            metadata_helpers._record_audio_mix(metadata, mix_type="muted", details={"gain_db": -100.0})
            base_clip = base_clip.with_effects([afx.MultiplyVolume(0)])

        # Build result based on priority mode
        results = []

        if priority == "clip_video":
            # Use whole clip with narration overlay
            narration_meta = deepcopy(metadata)
            metadata_helpers._record_audio_mix(
                narration_meta, mix_type="narration_overlay",
                details={"narration_path": audio_path, "priority": priority}
            )
            # Attach narration, then trim combined clip (correct pattern)
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = self._safe_subclip(clip_with_narration, 0, narration_duration)
            clip_with_narration = metadata_helpers._attach_metadata(clip_with_narration, narration_meta)
            results.append(clip_with_narration)
            metadata_helpers._register_clip_edit(clip, narration_meta)

        elif priority == "clip_audio":
            # Play twice: first with narration, then with boosted original audio
            narration_meta = deepcopy(metadata)
            metadata_helpers._record_audio_mix(
                narration_meta, mix_type="narration",
                details={"narration_path": audio_path, "priority": priority}
            )
            # First segment: narration audio
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = self._safe_subclip(clip_with_narration, 0, narration_duration)
            clip_with_narration = metadata_helpers._attach_metadata(clip_with_narration, narration_meta)
            results.append(clip_with_narration)
            metadata_helpers._register_clip_edit(clip, narration_meta)

            # Second segment: boosted original audio
            if original_audio and base_clip.audio is not None:
                original_meta = deepcopy(metadata)
                original_meta["segment_type"] = "clip_audio_followup"
                gain_db = 20.0 * math.log10(volume_multiplier) if volume_multiplier > 0 else -100.0
                metadata_helpers._record_audio_mix(
                    original_meta, mix_type="original_clip_boost",
                    details={"gain_multiplier": volume_multiplier, "gain_db": gain_db}
                )
                boosted = base_clip.with_effects([afx.MultiplyVolume(volume_multiplier)])
                boosted = metadata_helpers._attach_metadata(boosted, original_meta)
                results.append(boosted)
                metadata_helpers._register_clip_edit(clip, original_meta)

        else:  # priority == "narration" (default)
            narration_meta = deepcopy(metadata)
            metadata_helpers._record_audio_mix(
                narration_meta, mix_type="narration",
                details={"narration_path": audio_path, "priority": priority}
            )
            # Attach narration, then trim combined clip (correct pattern)
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = self._safe_subclip(clip_with_narration, 0, narration_duration)
            clip_with_narration = metadata_helpers._attach_metadata(clip_with_narration, narration_meta)
            results.append(clip_with_narration)
            metadata_helpers._register_clip_edit(clip, narration_meta)

        return results

    def _safe_subclip(self, clip, start: float, end: float, epsilon: float = 0.01):
        """Safely subclip with bounds checking."""
        safe_end = max(start, end - epsilon)
        if clip.duration is not None:
            safe_end = min(safe_end, clip.duration)
        return clip.subclipped(start, safe_end)

    def _calculate_volume_match(self, clip_audio, narration_audio) -> float:
        """Calculate volume multiplier to match clip audio to narration level."""
        try:
            clip_lufs = audio_processing.get_loudness(clip_audio)
            narration_lufs = audio_processing.get_loudness(narration_audio)
            if narration_lufs != 0 and clip_lufs != 0:
                return float(10 ** ((narration_lufs - clip_lufs) / 20))
        except Exception:
            pass
        return 1.0

    # =========================================================================
    # Music and Beat Snapping
    # =========================================================================

    def snap_clips_to_music_beats(
        self,
        clips: list,
        music_path: str,
        sample_rate: int = 22050,
        snap_window: float = 1.0,
    ) -> list:
        """Adjust clip speeds so they end on music beats."""
        y, _ = librosa.load(music_path, sr=sample_rate)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

        current_time = 0.0
        adjusted = []

        for clip in clips:
            duration = clip.duration
            intended_end = current_time + duration

            # Find beats within snap window
            candidates = beat_times[np.abs(beat_times - intended_end) <= snap_window]

            if len(candidates) > 0:
                snap_target = candidates[np.argmin(np.abs(candidates - intended_end))]
                new_duration = snap_target - current_time

                if new_duration > 0 and abs(new_duration - duration) > 0.05:
                    speed_factor = duration / new_duration
                    clip = clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                    clip = self._safe_subclip(clip, 0, new_duration)
                    duration = new_duration

            adjusted.append(clip)
            current_time += duration

        return adjusted

    # =========================================================================
    # Final Assembly
    # =========================================================================

    def _assemble_final_video(
        self,
        clips: list,
        music_path: str = None,
        music_volume: float = None,
    ):
        """Concatenate clips and mix in background music."""
        final_video = concatenate_videoclips(clips)

        if not music_path:
            return final_video

        try:
            # Load music as numpy-backed clip to avoid FFmpeg reader state issues
            music = self._load_audio_as_array(music_path)

            # Calculate music volume if not provided
            if music_volume is None:
                try:
                    music_lufs = audio_processing.get_loudness(music)
                    video_lufs = audio_processing.get_loudness(final_video.audio)
                    if video_lufs != 0 and music_lufs != 0:
                        music_volume = (10 ** ((video_lufs - music_lufs) / 20)) * self.music_volume_multiplier
                    else:
                        music_volume = self.music_volume_multiplier
                except Exception:
                    music_volume = self.music_volume_multiplier

            # Apply volume and trim to video length
            music = music.with_effects([afx.MultiplyVolume(music_volume)])
            music = music.subclipped(0, final_video.duration)

            # Mix audio
            mixed_audio = CompositeAudioClip([final_video.audio, music])
            final_video = final_video.with_audio(mixed_audio)

        except Exception as e:
            print(f"[WARN] Failed to mix background music: {e}")

        return final_video

    # =========================================================================
    # Subtitles
    # =========================================================================

    async def _generate_subtitles(self, video_path: str, aspect_ratio: float):
        """Generate and burn subtitles into the video."""
        print("[INFO] Generating subtitles...")

        try:
            subs_dir = self._output_dir / "subtitles"
            subs_dir.mkdir(exist_ok=True)

            generator = GenerateSubtitles(output_dir=str(subs_dir))
            transcription = await asyncio.to_thread(
                generator, audio_path=video_path, global_start_time=0.0
            )

            words = transcription.get("words", [])
            if not words:
                print("[WARN] No words detected, skipping subtitles")
                return

            srt_entries = GenerateSubtitles.words_to_srt_entries(words, max_words=8)
            srt_path = subs_dir / "subtitles.srt"
            GenerateSubtitles.write_srt(srt_entries, str(srt_path))

            # Style based on aspect ratio
            font_size = 24 if aspect_ratio < 1.0 else 18
            margin_v = 60 if aspect_ratio < 1.0 else 40

            temp_output = video_path.replace(".mp4", "_subs.mp4")
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"subtitles={srt_path}:force_style='FontSize={font_size},FontName=Arial,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Shadow=1,MarginV={margin_v}'",
                "-c:a", "copy", temp_output,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                shutil.move(temp_output, video_path)
                print(f"[INFO] Subtitles burned into video")
            else:
                print(f"[WARN] Failed to burn subtitles: {result.stderr[:200]}")

        except Exception as e:
            print(f"[WARN] Subtitle generation failed: {e}")

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(
        self,
        clips: list,
        narration_dir: str,
        background_music_path: str = None,
        original_audio: bool = True,
        narration_enabled: bool = True,
        aspect_ratio: float = 16 / 9,
        subtitles: bool = True,
        snap_to_beat: bool = False,
        project_name: str = None,
    ) -> Path:
        """
        Build video timeline from clip plan.

        All outputs are saved to: data/outputs/{project_name}/

        Args:
            clips: List of clip dicts with id, file_name, start, end, narration, priority
            narration_dir: Directory containing narration audio files
            background_music_path: Optional path to background music
            original_audio: Whether to include original clip audio
            narration_enabled: Whether to use TTS narration
            aspect_ratio: Target aspect ratio (e.g., 16/9 or 9/16)
            subtitles: Whether to generate subtitles
            snap_to_beat: Whether to snap clip ends to music beats
            project_name: Name for the output folder (defaults to first clip's file name)

        Returns:
            Path to output directory containing all files
        """
        if not clips:
            raise ValueError("No clips provided")

        # Set up output directory - use provided project name or fall back to first file name
        if not project_name:
            project_name = Path(clips[0]["file_name"]).stem
        self._setup_output_dir(project_name)

        # Save clip plan to output directory
        clip_plan_path = self._output_dir / "clip_plan.json"
        with open(clip_plan_path, "w") as f:
            json.dump(clips, f, indent=2)
        print(f"[INFO] Clip plan saved to {clip_plan_path}")

        # Copy narrations to output directory
        output_narration_dir = self._output_dir / "narrations"
        if narration_dir and os.path.exists(narration_dir):
            for clip in clips:
                clip_id = clip["id"]
                src_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                if os.path.exists(src_path):
                    dst_path = output_narration_dir / f"{clip_id}.mp3"
                    shutil.copy2(src_path, dst_path)
            narration_dir = str(output_narration_dir)
            print(f"[INFO] Narrations copied to {output_narration_dir}")

        # Copy background music to output directory
        output_music_path = None
        if background_music_path and os.path.exists(background_music_path):
            music_filename = os.path.basename(background_music_path)
            output_music_path = self._output_dir / "music" / music_filename
            shutil.copy2(background_music_path, output_music_path)
            background_music_path = str(output_music_path)
            print(f"[INFO] Music copied to {output_music_path}")

        # Download all source videos (normalizes to footage/ folder)
        for clip in clips:
            self._download_media_file(clip["file_name"], clip["cloud_storage_path"])

        # Process each clip
        processed_clips = []
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            processed_clips.extend(
                self._process_clip(clip, narration_dir, narration_enabled, original_audio)
            )

        if not processed_clips:
            raise ValueError("No clips were processed successfully")

        # Calculate export dimensions
        if aspect_ratio == 0:
            aspect_ratio = 16 / 9
        if aspect_ratio >= 1.0:
            export_width, export_height = 1920, int(round(1920 / aspect_ratio))
        else:
            export_width, export_height = int(round(1920 * aspect_ratio)), 1920

        # Dynamic cropping (use output dir as workdir)
        dc = DynamicCropping(self.llm, str(self._output_dir))
        cropped_clips = await dc(export_width, export_height, processed_clips)

        # Snap to beats if requested
        if background_music_path and snap_to_beat:
            cropped_clips = self.snap_clips_to_music_beats(cropped_clips, background_music_path)

        # Collect metadata for FCPXML
        timeline_metadata = [
            deepcopy(getattr(c, "_vea_metadata", None))
            for c in cropped_clips
            if getattr(c, "_vea_metadata", None) is not None
        ]

        video_asset_map = {name: info["path"] for name, info in self._downloaded_files.items()}
        narration_asset_map = {}
        if narration_dir and os.path.exists(narration_dir):
            for meta in timeline_metadata:
                path = meta.get("audio", {}).get("details", {}).get("narration_path")
                if path and os.path.exists(path):
                    narration_asset_map[path] = path

        total_duration = sum(c.duration for c in cropped_clips)

        # Calculate music volume
        music_volume, music_gain_db = None, None
        if background_music_path and total_duration > 0:
            music_volume, music_gain_db = audio_processing.compute_music_adjustment(
                cropped_clips, background_music_path, total_duration, self.music_volume_multiplier
            )

        # Export FCPXML to output directory
        fcpxml_path = self._output_dir / f"{project_name}.fcpxml"
        export_fcpxml(
            timeline_metadata=timeline_metadata,
            video_asset_map=video_asset_map,
            narration_asset_map=narration_asset_map,
            music_asset_path=background_music_path,
            music_asset_name=os.path.basename(background_music_path) if background_music_path else None,
            music_duration=total_duration,
            music_gain_db=music_gain_db,
            output_path=str(fcpxml_path),
            project_name=project_name,
        )
        print(f"[INFO] FCPXML saved to {fcpxml_path}")

        # Assemble final video
        video_output_path = self._output_dir / f"{project_name}.mp4"
        with metrics_collector.track_step("video_assembly"):
            final_video = self._assemble_final_video(cropped_clips, background_music_path, music_volume)

            try:
                await asyncio.to_thread(
                    final_video.write_videofile,
                    str(video_output_path),
                    preset="ultrafast",
                    fps=24,
                    audio_codec="aac",
                )
                print(f"[INFO] Final video created: {video_output_path}")
            finally:
                try:
                    final_video.close()
                except Exception:
                    pass

        # Generate subtitles
        if subtitles:
            await self._generate_subtitles(str(video_output_path), aspect_ratio)

        # Cleanup
        for clip in processed_clips + cropped_clips:
            try:
                clip.close()
            except Exception:
                pass

        print(f"[INFO] All outputs saved to: {self._output_dir}")
        return self._output_dir, video_output_path

    # =========================================================================
    # Test Harnesses
    # =========================================================================

    @staticmethod
    def test_clip_processing(
        video_path: str,
        narration_audio_path: str = None,
        output_dir: str = "/home/alex/code/vea2/vea-playground/test_outputs/final_outputs",
        priority: str = "narration",
        start_sec: float = 0.0,
        end_sec: float = 10.0,
    ) -> dict:
        """
        Test single clip processing with narration attachment.

        This tests the core audio attachment pattern that caused the
        'Accessing time t=X seconds' error.

        Usage:
            from src.pipelines.common.timeline_constructor import TimelineConstructor
            result = TimelineConstructor.test_clip_processing(
                "/path/to/video.mp4",
                "/path/to/narration.mp3",
                priority="narration",
            )

        Args:
            video_path: Path to source video
            narration_audio_path: Path to narration MP3 (optional)
            output_dir: Where to save test outputs
            priority: One of "narration", "clip_audio", "clip_video"
            start_sec: Clip start time
            end_sec: Clip end time

        Returns:
            dict with paths and status
        """
        import json
        import numpy as np
        from pathlib import Path
        from moviepy import VideoFileClip, AudioFileClip, AudioClip
        from moviepy import vfx

        def load_audio_as_array(audio_path: str) -> AudioClip:
            """Load audio file as numpy-backed AudioClip (avoids FFmpeg reader issues)."""
            with AudioFileClip(audio_path) as temp_clip:
                fps = temp_clip.fps
                duration = temp_clip.duration
                nchannels = temp_clip.nchannels
                audio_array = temp_clip.to_soundarray(fps=fps)

            def make_frame(t):
                t = np.atleast_1d(t)
                indices = (t * fps).astype(int)
                indices = np.clip(indices, 0, len(audio_array) - 1)
                result = audio_array[indices]
                if result.ndim == 1 and nchannels > 1:
                    result = result.reshape(-1, nchannels)
                return result

            clip = AudioClip(make_frame, duration=duration, fps=fps)
            clip.nchannels = nchannels
            return clip

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = video_path.stem
        results = {"status": "started", "errors": []}

        print(f"[TEST] Loading video: {video_path}")
        video_clip = VideoFileClip(str(video_path))
        print(f"[TEST] Video duration: {video_clip.duration:.2f}s")

        # Ensure video has audio
        if video_clip.audio is None:
            print("[TEST] No audio in video, creating silence")
            silent = AudioClip(lambda _: [0, 0], duration=video_clip.duration, fps=44100)
            video_clip = video_clip.with_audio(silent)

        # Trim to range
        safe_end = min(end_sec, video_clip.duration - 0.01)
        base_clip = video_clip.subclipped(start_sec, safe_end)
        video_duration = base_clip.duration
        print(f"[TEST] Trimmed clip duration: {video_duration:.2f}s")

        # Load narration if provided (as numpy-backed clip)
        narration = None
        narration_duration = 0.0
        if narration_audio_path:
            narration_path = Path(narration_audio_path)
            if narration_path.exists():
                print(f"[TEST] Loading narration: {narration_path}")
                narration = load_audio_as_array(str(narration_path))
                narration_duration = narration.duration
                print(f"[TEST] Narration duration: {narration_duration:.2f}s")
            else:
                results["errors"].append(f"Narration not found: {narration_path}")

        # Process based on priority
        final_clips = []

        if narration is None:
            print("[TEST] No narration, outputting trimmed clip only")
            final_clips.append(base_clip)

        elif priority == "narration":
            print("[TEST] Priority=narration: Attaching TTS, trimming to narration length")

            # Slow down video if shorter than narration
            if video_duration < narration_duration:
                speed_factor = video_duration / narration_duration
                print(f"[TEST] Slowing video by factor {speed_factor:.3f}")
                base_clip = base_clip.with_effects([vfx.MultiplySpeed(speed_factor)])

            # KEY PATTERN: Attach first, then trim
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = clip_with_narration.subclipped(0, narration_duration)
            final_clips.append(clip_with_narration)

        elif priority == "clip_audio":
            print("[TEST] Priority=clip_audio: Two segments - narration then original")

            # First: with narration
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = clip_with_narration.subclipped(0, narration_duration)
            final_clips.append(clip_with_narration)

            # Second: with original audio
            final_clips.append(base_clip)

        elif priority == "clip_video":
            print("[TEST] Priority=clip_video: Full video with narration overlay")
            clip_with_narration = base_clip.with_audio(narration)
            if clip_with_narration.duration > narration_duration > 0:
                clip_with_narration = clip_with_narration.subclipped(0, narration_duration)
            final_clips.append(clip_with_narration)

        # Write outputs
        from moviepy import concatenate_videoclips

        if len(final_clips) > 1:
            final_video = concatenate_videoclips(final_clips)
        else:
            final_video = final_clips[0]

        output_path = output_dir / f"test_clip_{priority}_{stem}.mp4"
        print(f"[TEST] Writing output to: {output_path}")

        try:
            final_video.write_videofile(
                str(output_path),
                preset="ultrafast",
                fps=24,
                audio_codec="aac",
            )
            results["status"] = "success"
            results["output_video"] = str(output_path)
            results["final_duration"] = final_video.duration
            print(f"[TEST] SUCCESS: Wrote {output_path.name}")
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"[TEST] FAILED: {e}")
        finally:
            final_video.close()
            video_clip.close()
            if narration:
                narration.close()

        # Save results metadata
        meta_path = output_dir / f"test_clip_{priority}_{stem}.json"
        with open(meta_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[TEST] Results: {results['status']}")
        return results

