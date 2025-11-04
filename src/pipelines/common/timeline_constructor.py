import os
import tempfile
import asyncio
import gc
import traceback
import time
import shutil
import math
import subprocess
from copy import deepcopy
from pathlib import Path

from moviepy import *
import librosa
import numpy as np
import json

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.utils.media import parse_time_to_seconds, download_and_cache_video
from lib.utils.metrics_collector import metrics_collector
from src.pipelines.common.dynamic_cropping import DynamicCropping
from src.pipelines.common.fcpxml_exporter import export_fcpxml
from src.pipelines.common import metadata_helpers
from src.pipelines.common import audio_processing

class TimelineConstructor:
    ALLOWED_FRAME_RATES = (24, 30, 60, 120)

    def __init__(
        self, 
        output_path="video_response.mp4", 
        music_volume_multiplier=0.5, 
        gcs_client=None, 
        gcs_media_base_path=None, 
        bucket_name=None,
        workdir=None,
        llm=None
    ):
        self.output_path = output_path
        self.music_volume_multiplier = music_volume_multiplier
        self.gcs_client = gcs_client
        self.gcs_media_base_path = gcs_media_base_path
        self.bucket_name = bucket_name
        self.workdir = workdir or tempfile.mkdtemp()
        self.llm = llm
        self._downloaded_files = {}

    @staticmethod
    def _parse_ffprobe_rate(rate_str: str | None) -> float:
        if not rate_str or rate_str in {"0/0", "0", "N/A"}:
            return 0.0
        if "/" in rate_str:
            num_str, den_str = rate_str.split("/", 1)
            try:
                num = float(num_str)
                den = float(den_str)
                if den == 0:
                    return 0.0
                return num / den
            except ValueError:
                return 0.0
        try:
            return float(rate_str)
        except ValueError:
            return 0.0

    def _probe_frame_rate(self, path: str) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0.0

        try:
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams") or []
            if not streams:
                return 0.0
            stream = streams[0]
            avg_rate = self._parse_ffprobe_rate(stream.get("avg_frame_rate"))
            if avg_rate > 0:
                return avg_rate
            return self._parse_ffprobe_rate(stream.get("r_frame_rate"))
        except json.JSONDecodeError:
            return 0.0

    def _pick_target_fps(self, measured_fps: float) -> float:
        if measured_fps <= 0:
            return float(self.ALLOWED_FRAME_RATES[1])  # default to 30 fps
        return float(
            min(
                self.ALLOWED_FRAME_RATES,
                key=lambda allowed: abs(allowed - measured_fps),
            )
        )

    def _normalize_media_frame_rate(self, file_name: str, source_path: str) -> dict:
        original_fps = self._probe_frame_rate(source_path)
        target_fps = self._pick_target_fps(original_fps)

        normalized_dir = Path(self.workdir) / "normalized_media"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = normalized_dir / f"{Path(file_name).stem}_fps{int(target_fps)}.mp4"

        if normalized_path.exists():
            return {
                "path": str(normalized_path),
                "fps": target_fps,
                "original_fps": original_fps or target_fps,
                "source_path": source_path,
            }

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
            "-vf",
            f"fps={target_fps}",
            "-vsync",
            "cfr",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-af",
            "aresample=async=1:first_pts=0",
            "-movflags",
            "+faststart",
            str(normalized_path),
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {
                "path": str(normalized_path),
                "fps": target_fps,
                "original_fps": original_fps or target_fps,
                "source_path": source_path,
            }
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[WARN] Failed to normalize FPS for {file_name}: {exc}. Using source video.")
            return {
                "path": source_path,
                "fps": original_fps or target_fps,
                "original_fps": original_fps or target_fps,
                "source_path": source_path,
            }

    def _download_media_file(self, file_name, cloud_storage_path):
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]["path"]
        local_path = download_and_cache_video(
            self.gcs_client,
            self.bucket_name,
            cloud_storage_path,
            self.workdir,
        )
        normalized_info = self._normalize_media_frame_rate(file_name, local_path)
        self._downloaded_files[file_name] = normalized_info
        return normalized_info["path"]

    def safe_subclip(self, clip, start_time, end_time, epsilon=0.01):
        safe_end_time = max(start_time, end_time - epsilon)
        if clip.duration is not None:
            safe_end_time = min(safe_end_time, clip.duration)
        return clip.subclipped(start_time, safe_end_time)

    def _process_clip(
        self, clip, narration_dir, narration_enabled=True, original_audio=True
    ):
        clip_id = clip["id"]
        file_name = clip["file_name"]
        download_info = self._downloaded_files[file_name]
        video_path = download_info["path"]
        metadata_frame_rate = float(download_info.get("fps", 0.0)) or None
        original_frame_rate = float(download_info.get("original_fps", 0.0)) or None
        start_sec = parse_time_to_seconds(clip["start"])
        end_sec = parse_time_to_seconds(clip["end"])
        priority = clip.get("priority", False)

        metadata = metadata_helpers._build_segment_metadata(
            clip,
            start_sec,
            end_sec,
            original_audio=original_audio,
        )

        if metadata_frame_rate:
            metadata.setdefault("timeline", {})["frame_rate"] = metadata_frame_rate
            metadata.setdefault("source", {})["fps"] = metadata_frame_rate
        if original_frame_rate and original_frame_rate != metadata_frame_rate:
            metadata.setdefault("source", {})["original_fps"] = original_frame_rate

        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            silent_audio = AudioClip(
                lambda t: [0, 0],
                duration=video_clip.duration,
                fps=44100,
            )
            video_clip = video_clip.with_audio(silent_audio)
        video_clip = metadata_helpers._attach_metadata(video_clip, metadata)

        if narration_enabled:
            try:
                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                narration = AudioFileClip(audio_path)
                audio_duration = float(narration.duration or 0.0)

                priority_value = priority or "narration"

                trim_end = min(end_sec, float(video_clip.duration or end_sec))
                metadata["timings"]["applied_start"] = float(start_sec)
                metadata["timings"]["applied_end"] = float(trim_end)

                base_clip = metadata_helpers._apply_clip_transform(
                    video_clip,
                    metadata,
                    lambda c: self.safe_subclip(c, start_sec, trim_end),
                )

                video_duration = float(base_clip.duration or 0.0)
                if video_duration < audio_duration and audio_duration > 0:
                    speed_factor = video_duration / audio_duration if audio_duration > 0 else 1
                    metadata_helpers._record_retime_adjustment(
                        metadata,
                        speed_factor=speed_factor,
                        reason="match_narration_length",
                        original_duration=video_duration,
                        target_duration=audio_duration,
                    )
                    base_clip = metadata_helpers._apply_clip_transform(
                        base_clip,
                        metadata,
                        lambda c: c.with_effects([vfx.MultiplySpeed(speed_factor)]),
                    )
                    video_duration = float(base_clip.duration or 0.0)

                # Measure relative loudness using LUFS
                volume_multiplier = 1.0
                gain_db = 0.0
                try:
                    clip_volume = audio_processing.get_loudness(base_clip.audio)
                    narration_volume = audio_processing.get_loudness(narration)
                    if narration_volume != 0 and clip_volume != 0:
                        volume_multiplier = float(10 ** ((narration_volume - clip_volume) / 20))
                        gain_db = 20.0 * math.log10(volume_multiplier) if volume_multiplier > 0 else -100.0
                except Exception:
                    volume_multiplier = 1.0
                    gain_db = 0.0

                if not original_audio:
                    metadata_helpers._record_audio_mix(
                        metadata,
                        mix_type="muted",
                        details={"gain_db": -100.0},
                    )
                    base_clip = metadata_helpers._apply_clip_transform(
                        base_clip,
                        metadata,
                        lambda c: c.with_effects([afx.MultiplyVolume(0)]),
                    )

                # Build narration-first segment
                narration_metadata = deepcopy(metadata)
                mix_type = "narration_overlay" if priority_value == "clip_video" else "narration"
                metadata_helpers._record_audio_mix(
                    narration_metadata,
                    mix_type=mix_type,
                    details={
                        "narration_path": audio_path,
                        "narration_duration": audio_duration,
                        "narration_gain_db": 0.0,
                        "priority": priority_value,
                    },
                )

                clip_with_narration = metadata_helpers._apply_clip_transform(
                    base_clip,
                    narration_metadata,
                    lambda c: c.with_audio(narration),
                )

                if float(clip_with_narration.duration or 0.0) > audio_duration > 0:
                    clip_with_narration = metadata_helpers._apply_clip_transform(
                        clip_with_narration,
                        narration_metadata,
                        lambda c: self.safe_subclip(c, 0, audio_duration),
                    )

                results = [clip_with_narration]
                narration_metadata.pop("_lock_source_end", None)
                metadata_helpers._register_clip_edit(clip, narration_metadata)

                if priority_value == "clip_audio" and original_audio and base_clip.audio is not None:
                    original_metadata = deepcopy(metadata)
                    original_metadata["segment_type"] = "clip_audio_followup"
                    metadata_helpers._record_audio_mix(
                        original_metadata,
                        mix_type="original_clip_boost",
                        details={
                            "gain_multiplier": volume_multiplier,
                            "gain_db": gain_db,
                            "priority": "clip_audio",
                        },
                    )
                    boosted_clip = metadata_helpers._apply_clip_transform(
                        base_clip,
                        original_metadata,
                        lambda c: c.with_effects([afx.MultiplyVolume(volume_multiplier)]),
                    )
                    original_metadata.pop("_lock_source_end", None)
                    results.append(boosted_clip)
                    metadata_helpers._register_clip_edit(clip, original_metadata)

                return results

            except Exception as exc:
                metadata_helpers._append_note(metadata, f"Narration failed: {exc}")
                narration_enabled = False

        if not narration_enabled:
            video_clip = self.safe_subclip(video_clip, start_sec, end_sec)
            if not original_audio:
                video_clip = video_clip.with_effects([afx.MultiplyVolume(0)])
            metadata_helpers._register_clip_edit(clip, metadata)
            return [video_clip]
        
    def snap_clips_to_music_beats(self, processed_clips, background_music_path, sr=22050, strong_snap_sec=1.0):
        y, _ = librosa.load(background_music_path, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        strong_beats = beat_times
        
        current_time = 0.0
        adjusted_clips = []
        for clip in processed_clips:
            orig_duration = clip.duration
            intended_end = current_time + orig_duration
            strong_cands = strong_beats[np.abs(strong_beats - intended_end) <= strong_snap_sec]
            snap_target = None
            if len(strong_cands) > 0:
                snap_target = strong_cands[np.argmin(np.abs(strong_cands - intended_end))]
            if snap_target is not None:
                new_duration = snap_target - current_time
                if new_duration > 0 and abs(new_duration - orig_duration) > 0.05:
                    speed_factor = orig_duration / new_duration
                    clip = clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                    clip = self.safe_subclip(clip, 0, new_duration)
                    orig_duration = new_duration
            adjusted_clips.append(clip)
            current_time += orig_duration
        return adjusted_clips

    def _assemble_final_video(
        self,
        processed_clips,
        background_music_path=None,
        music_volume_override: float | None = None,
    ):
        final_video = concatenate_videoclips(processed_clips)
        video_audio = final_video.audio
        if background_music_path:
            try:
                background_music = AudioFileClip(background_music_path)
                if music_volume_override is not None:
                    volume_multiplier = music_volume_override
                else:
                    # Measure loudness and calculate appropriate music level
                    try:
                        music_volume = audio_processing.get_loudness(background_music)
                        video_volume = audio_processing.get_loudness(video_audio)
                        if video_volume == 0 or music_volume == 0:
                            volume_multiplier = self.music_volume_multiplier
                        else:
                            volume_multiplier = (
                                10 ** ((video_volume - music_volume) / 20)
                            ) * self.music_volume_multiplier
                    except Exception as e:
                        print(f"[WARN] Failed to measure loudness for music balancing: {e}")
                        volume_multiplier = self.music_volume_multiplier

                background_music = background_music.with_effects(
                    [afx.MultiplyVolume(volume_multiplier)]
                ).subclipped(0, final_video.duration)
                final_audio = CompositeAudioClip([video_audio, background_music])
                final_video = final_video.with_audio(final_audio)
            except Exception as e:
                print(f"[WARN] Failed to mix background music: {e}")
        return final_video

    def _ensure_audio_readers(self, audio_clip):
        """Re-initialize ffmpeg readers on audio clips that were previously closed."""
        if audio_clip is None:
            return

        # CompositeAudioClip exposes sub-clips via `clips`
        child_clips = getattr(audio_clip, "clips", None)
        if child_clips:
            for child in child_clips:
                self._ensure_audio_readers(child)

        reader = getattr(audio_clip, "reader", None)
        if reader is not None and getattr(reader, "proc", None) is None:
            initialize = getattr(reader, "initialize", None)
            if callable(initialize):
                try:
                    initialize()
                except Exception:
                    pass

    async def run(
        self,
        clips,
        narration_dir,
        background_music_path=None,
        original_audio=True,
        narration_enabled=True,
        aspect_ratio=16/9,
        subtitles=True,
        snap_to_beat=False,
        multi_round_mode=True
    ):
        for clip in clips:
            self._download_media_file(clip["file_name"], clip["cloud_storage_path"])

        original_clips = []
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            original_clips.extend(self._process_clip(clip, narration_dir, narration_enabled, original_audio))

        if not original_clips:
            raise ValueError("No clips were processed successfully.")

        if aspect_ratio == 0:
            aspect_ratio = 16/9

        export_width = 1920 if aspect_ratio >= 1.0 else int(round(1920 * aspect_ratio))
        export_height = int(round(1920 / aspect_ratio)) if aspect_ratio >= 1.0 else 1920

        dc = DynamicCropping(self.llm, self.workdir)
        cropped_clips = await dc(export_width, export_height, original_clips)

        if background_music_path and snap_to_beat:
            cropped_clips = self.snap_clips_to_music_beats(cropped_clips, background_music_path)

        timeline_metadata = [deepcopy(getattr(c, "_vea_metadata", None)) for c in cropped_clips if getattr(c, "_vea_metadata", None) is not None]

        video_asset_map = {
            name: info["path"]
            for name, info in self._downloaded_files.items()
        }

        narration_asset_map = {}
        if narration_dir and os.path.exists(narration_dir):
            for meta in timeline_metadata:
                details = meta.get("audio", {}).get("details", {})
                if details and details.get("narration_path"):
                    narration_path = details["narration_path"]
                    if os.path.exists(narration_path):
                        narration_asset_map[narration_path] = narration_path

        total_timeline_duration = sum(c.duration for c in cropped_clips)

        # Compute music adjustment for both video assembly and FCPXML export
        music_volume_override = None
        music_gain_db = None
        if background_music_path and total_timeline_duration > 0:
            music_volume_override, music_gain_db = audio_processing.compute_music_adjustment(
                cropped_clips,
                background_music_path,
                total_timeline_duration,
                self.music_volume_multiplier
            )

        project_name = os.path.splitext(os.path.basename(self.output_path))[0] or "VEA Edit"
        fcpxml_path = os.path.join(self.workdir, f"{project_name}.fcpxml")

        export_fcpxml(
            timeline_metadata=timeline_metadata,
            video_asset_map=video_asset_map,
            narration_asset_map=narration_asset_map,
            music_asset_path=background_music_path,
            music_asset_name=os.path.basename(background_music_path) if background_music_path else None,
            music_duration=total_timeline_duration,
            music_gain_db=music_gain_db,
            output_path=fcpxml_path,
            project_name=project_name,
        )
        print(f"[INFO] FCPXML saved to {fcpxml_path}")

        with metrics_collector.track_step("video_assembly"):
            final_video = self._assemble_final_video(cropped_clips, background_music_path, music_volume_override)
            self._ensure_audio_readers(final_video.audio)

            try:
                await asyncio.to_thread(
                    final_video.write_videofile,
                    self.output_path,
                    preset="ultrafast",
                    fps=24,
                )
                print(f"[INFO] Final video created: {self.output_path}")
            finally:
                try:
                    final_video.close()
                except Exception:
                    pass

        for clip in original_clips + cropped_clips:
            try:
                clip.close()
            except Exception:
                pass
        dc.cleanup()
        gc.collect()

        return fcpxml_path if multi_round_mode else self.output_path
