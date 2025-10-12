import os
import tempfile
import asyncio
import gc
import traceback
import time
import shutil
import math
from copy import deepcopy
from moviepy import *
import librosa
import numpy as np
import subprocess
import sys
import json
import whisperx
import pyloudnorm as pyln

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from lib.utils.media import parse_time_to_seconds
from src.pipelines.common.dynamic_cropping import DynamicCropping
from src.pipelines.common.generate_subtitles import GenerateSubtitles
from src.pipelines.common.fcpxml_exporter import export_fcpxml

MULTIROUND_MODE = True

class EditVideoResponse:
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
        self.timeline_metadata = []

    # ---------- Metadata Helpers ----------
    def _build_segment_metadata(
        self,
        clip_info,
        start_sec: float,
        end_sec: float,
        *,
        segment_type: str = "primary",
        original_audio: bool = True,
    ) -> dict:
        metadata = {
            "clip_id": clip_info.get("id") if isinstance(clip_info, dict) else None,
            "segment_type": segment_type,
            "source": {
                "file_name": clip_info.get("file_name") if isinstance(clip_info, dict) else None,
                "cloud_storage_path": clip_info.get("cloud_storage_path") if isinstance(clip_info, dict) else None,
            },
            "timings": {
                "source_start": float(start_sec),
                "source_end": float(end_sec),
                "applied_start": float(start_sec),
                "applied_end": float(end_sec),
            },
            "timeline": {
                "duration": max(float(end_sec) - float(start_sec), 0.0),
            },
            "retime": {
                "speed": 1.0,
                "adjustments": [],
            },
            "audio": {
                "original_audio_enabled": bool(original_audio),
                "mix": "original" if original_audio else "muted",
                "details": {},
            },
            "crop": None,
            "notes": [],
        }
        return metadata

    def _attach_metadata(self, clip_obj, metadata):
        duration = float(clip_obj.duration or 0.0)
        metadata.setdefault("timeline", {})["duration"] = duration
        timings = metadata.setdefault("timings", {})
        applied_start = timings.get("applied_start", timings.get("source_start", 0.0))
        applied_end = applied_start + duration
        timings["applied_end"] = applied_end
        if not metadata.get("_lock_source_end"):
            timings["source_end"] = applied_end
        setattr(clip_obj, "_vea_metadata", metadata)
        return clip_obj

    def _apply_clip_transform(self, clip_obj, metadata, transform_fn):
        transformed = transform_fn(clip_obj)
        return self._attach_metadata(transformed, metadata)

    def _record_retime_adjustment(
        self,
        metadata,
        *,
        speed_factor: float,
        reason: str,
        original_duration: float,
        target_duration: float,
    ) -> None:
        retime = metadata.setdefault("retime", {})
        retime.setdefault("adjustments", []).append(
            {
                "reason": reason,
                "speed_factor": float(speed_factor),
                "original_duration": float(original_duration),
                "target_duration": float(target_duration),
            }
        )
        retime["speed"] = float(retime.get("speed", 1.0) * speed_factor)
        timings = metadata.setdefault("timings", {})
        timings["source_end"] = timings.get("source_start", 0.0) + float(original_duration)
        metadata["_lock_source_end"] = True

    def _record_audio_mix(self, metadata, *, mix_type: str, details: dict | None = None) -> None:
        audio = metadata.setdefault("audio", {})
        audio["mix"] = mix_type
        if details:
            audio.setdefault("details", {}).update(details)
        if mix_type == "muted":
            audio.setdefault("details", {})["gain_db"] = -100.0

    def _average_clip_loudness(self, clips) -> float | None:
        total_energy = 0.0
        total_duration = 0.0
        for clip in clips:
            audio = getattr(clip, "audio", None)
            duration = float(getattr(clip, "duration", 0.0) or 0.0)
            if audio is None or duration <= 0:
                continue
            try:
                segment = audio.subclipped(0, duration)
                loudness = self.get_loudness(segment)
            except Exception as exc:
                print(f"[WARN] Failed to measure loudness for clip: {exc}")
                continue
            finally:
                try:
                    segment.close()
                except Exception:
                    pass
            total_energy += (10 ** (loudness / 10.0)) * duration
            total_duration += duration
        if total_duration <= 0 or total_energy <= 0:
            return None
        average_power = total_energy / total_duration
        # Convert back to LUFS-like dB scale, guard against log(0)
        return 10.0 * math.log10(max(average_power, 1e-12))

    def _compute_music_adjustment(self, clips, background_music_path: str, total_duration: float) -> tuple[float | None, float | None]:
        music_volume_multiplier = None
        music_gain_db = None
        if not background_music_path or total_duration <= 0:
            return music_volume_multiplier, music_gain_db

        reference_loudness = self._average_clip_loudness(clips)

        music_clip = None
        music_segment = None
        try:
            music_clip = AudioFileClip(background_music_path)
            music_segment = music_clip.subclipped(0, min(total_duration, float(music_clip.duration or total_duration)))
            music_loudness = self.get_loudness(music_segment)
        except Exception as exc:
            print(f"[WARN] Failed to measure background music loudness: {exc}")
            reference_loudness = reference_loudness  # no-op, keep for readability
            music_loudness = None
        finally:
            if music_segment is not None:
                try:
                    music_segment.close()
                except Exception:
                    pass
            if music_clip is not None:
                try:
                    music_clip.close()
                except Exception:
                    pass

        if reference_loudness is None or music_loudness is None:
            volume_multiplier = self.music_volume_multiplier
        else:
            volume_multiplier = (
                10 ** ((reference_loudness - music_loudness) / 20.0)
            ) * self.music_volume_multiplier

        if volume_multiplier is not None and volume_multiplier > 0:
            music_volume_multiplier = volume_multiplier
            music_gain_db = 20.0 * math.log10(volume_multiplier)
        else:
            music_volume_multiplier = None
            music_gain_db = -100.0

        return music_volume_multiplier, music_gain_db

    def _append_note(self, metadata, note: str) -> None:
        metadata.setdefault("notes", []).append(note)

    def _register_clip_edit(self, clip_dict, metadata) -> None:
        if isinstance(clip_dict, dict):
            clip_dict.setdefault("edits", []).append(deepcopy(metadata))

    def _prepare_multiround_assets(self, narration_dir, background_music_path=None):
        assets_root = os.path.join(self.workdir, "multiround_assets")
        os.makedirs(assets_root, exist_ok=True)

        video_asset_map = {}
        for meta in self.timeline_metadata:
            source = meta.get("source", {})
            file_name = source.get("file_name")
            if not file_name or file_name in video_asset_map:
                continue
            local_path = self._downloaded_files.get(file_name)
            if not local_path or not os.path.exists(local_path):
                continue
            dest_path = os.path.join(assets_root, file_name)
            if not os.path.exists(dest_path):
                shutil.copy2(local_path, dest_path)
            video_asset_map[file_name] = dest_path

        narration_asset_map = {}
        for meta in self.timeline_metadata:
            details = meta.get("audio", {}).get("details", {}) or {}
            narration_path = details.get("narration_path")
            if not narration_path or narration_path in narration_asset_map:
                continue
            if not os.path.exists(narration_path):
                continue
            dest_name = os.path.basename(narration_path)
            dest_path = os.path.join(assets_root, dest_name)
            if not os.path.exists(dest_path):
                shutil.copy2(narration_path, dest_path)
            narration_asset_map[narration_path] = dest_path

        music_asset_path = None
        music_asset_name = None
        if background_music_path and os.path.exists(background_music_path):
            music_asset_name = os.path.basename(background_music_path)
            music_dest = os.path.join(assets_root, music_asset_name)
            if not os.path.exists(music_dest):
                shutil.copy2(background_music_path, music_dest)
            music_asset_path = music_dest
            for meta in self.timeline_metadata:
                if meta.get("audio", {}).get("mix") == "music":
                    meta.setdefault("audio", {}).setdefault("details", {})["music_path"] = music_dest

        return assets_root, video_asset_map, narration_asset_map, music_asset_path, music_asset_name

    # ---------- File Download Helpers ----------
    def _download_media_file(self, file_name, cloud_storage_path):
        """Download the media file from GCS if not already present."""
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]
        local_path = os.path.join(self.workdir, file_name)
        self.gcs_client.download_files(self.bucket_name, cloud_storage_path, local_path)
        self._downloaded_files[file_name] = local_path
        return local_path


    def get_loudness(self, audio_clip, sample_rate=44100):
        """
        Calculate integrated LUFS loudness of an audio clip using pyloudnorm.
        
        Parameters:
        - audio_clip: a moviepy AudioClip (e.g., AudioFileClip)
        - sample_rate: sample rate in Hz (default is 44100)
        
        Returns:
        - Integrated LUFS loudness (float)
        """
        samples = audio_clip.to_soundarray(fps=sample_rate)
        # Convert stereo to mono by averaging channels if needed
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        meter = pyln.Meter(sample_rate)  # ITU-R BS.1770 loudness meter
        loudness = meter.integrated_loudness(samples)
        return loudness
    
    def safe_subclip(self, clip, start_time, end_time, epsilon=0.01):
        safe_end_time = max(start_time, end_time - epsilon)
        if clip.duration is not None:
            safe_end_time = min(safe_end_time, clip.duration)
        return clip.subclipped(start_time, safe_end_time)

    # ---------- Video Clip Assembly ----------
    def _process_clip(
        self,
        clip,
        narration_dir,
        narration_enabled=True,
        original_audio=True
    ):
        """
        Given a NarratedClip dict, returns a processed moviepy VideoClip.
        Logic: trims to narration if enabled, matches audio/narration/priority rules.
        """
        clip_id = clip["id"]
        file_name = clip["file_name"]
        video_path = self._downloaded_files[file_name]
        start_sec = parse_time_to_seconds(clip["start"])
        end_sec = parse_time_to_seconds(clip["end"])
        priority = clip.get("priority", False)

        metadata = self._build_segment_metadata(
            clip,
            start_sec,
            end_sec,
            original_audio=original_audio,
        )

        print(
            f"[DEBUG] _process_clip start id={clip_id} file={file_name} "
            f"start={start_sec:.2f}s end={end_sec:.2f}s priority={priority}"
        )

        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            silent_audio = AudioClip(
                lambda t: [0, 0],
                duration=video_clip.duration,
                fps=44100,
            )
            video_clip = video_clip.with_audio(silent_audio)
        video_clip = self._attach_metadata(video_clip, metadata)

        if narration_enabled:
            try:
                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                print(f"[DEBUG] Loading narration for clip {clip_id} from {audio_path}")
                narration = AudioFileClip(audio_path)
                audio_duration = float(narration.duration or 0.0)
                trim_end = min(end_sec, video_clip.duration)
                metadata["timings"]["applied_start"] = float(start_sec)
                metadata["timings"]["applied_end"] = float(trim_end)
                video_clip = self._apply_clip_transform(
                    video_clip,
                    metadata,
                    lambda c: self.safe_subclip(c, start_sec, trim_end),
                )
                video_duration = video_clip.duration
                if video_duration < audio_duration:
                    speed_factor = video_duration / audio_duration if audio_duration > 0 else 1
                    self._record_retime_adjustment(
                        metadata,
                        speed_factor=speed_factor,
                        reason="match_narration_length",
                        original_duration=video_duration,
                        target_duration=audio_duration,
                    )
                    video_clip = self._apply_clip_transform(
                        video_clip,
                        metadata,
                        lambda c: c.with_effects([vfx.MultiplySpeed(speed_factor)]),
                    )
                    video_duration = video_clip.duration
                clip_volume = self.get_loudness(video_clip.audio)
                narration_volume = self.get_loudness(narration)
                volume_multiplier = (
                    10 ** ((narration_volume - clip_volume) / 20)
                    if narration_volume != 0 and clip_volume != 0
                    else 1.0
                )

                if not original_audio:
                    self._record_audio_mix(
                        metadata,
                        mix_type="muted",
                        details={"gain_db": -100.0},
                    )
                    video_clip = self._apply_clip_transform(
                        video_clip,
                        metadata,
                        lambda c: c.with_effects([afx.MultiplyVolume(0)]),
                    )

                # Priority handling
                if not priority:
                    priority = "narration"
                if priority == "clip_audio":
                    # Play twice: once with narration, then with original audio boosted
                    copy_metadata = deepcopy(metadata)
                    copy_metadata["segment_type"] = "clip_audio_followup"
                    gain_multiplier = float(volume_multiplier)
                    gain_db = 20.0 * np.log10(gain_multiplier) if gain_multiplier > 0 else -100.0
                    self._record_audio_mix(
                        copy_metadata,
                        mix_type="original_clip_boost",
                        details={
                            "gain_multiplier": gain_multiplier,
                            "gain_db": gain_db,
                        },
                    )
                    video_clip_copy = self._apply_clip_transform(
                        video_clip,
                        copy_metadata,
                        lambda c: c.with_effects([afx.MultiplyVolume(volume_multiplier)]),
                    )
                    self._record_audio_mix(
                        metadata,
                        mix_type="narration",
                        details={
                            "narration_path": audio_path,
                            "narration_gain_db": 0.0,
                            "narration_duration": audio_duration,
                        },
                    )
                    video_clip = self._apply_clip_transform(
                        video_clip,
                        metadata,
                        lambda c: c.with_audio(narration),
                    )
                    if video_duration > audio_duration:
                        video_clip = self._apply_clip_transform(
                            video_clip,
                            metadata,
                            lambda c: self.safe_subclip(c, 0, audio_duration),
                        )
                    metadata.pop("_lock_source_end", None)
                    self._register_clip_edit(clip, metadata)
                    self._register_clip_edit(clip, copy_metadata)
                    return [video_clip, video_clip_copy]
                elif priority == "clip_video":
                    self._record_audio_mix(
                        metadata,
                        mix_type="narration_overlay",
                        details={
                            "narration_path": audio_path,
                            "narration_gain_db": 0.0,
                            "narration_duration": audio_duration,
                        },
                    )
                    video_clip = self._apply_clip_transform(
                        video_clip,
                        metadata,
                        lambda c: c.with_audio(narration),
                    )
                    metadata.pop("_lock_source_end", None)
                    self._register_clip_edit(clip, metadata)
                    return [video_clip]
                else:  # narration default
                    self._record_audio_mix(
                        metadata,
                        mix_type="narration",
                        details={
                            "narration_path": audio_path,
                            "narration_gain_db": 0.0,
                            "narration_duration": audio_duration,
                        },
                    )
                    video_clip = self._apply_clip_transform(
                        video_clip,
                        metadata,
                        lambda c: c.with_audio(narration),
                    )
                    if video_duration > audio_duration:
                        video_clip = self._apply_clip_transform(
                            video_clip,
                            metadata,
                            lambda c: self.safe_subclip(c, 0, audio_duration),
                        )
                    metadata.pop("_lock_source_end", None)
                    self._register_clip_edit(clip, metadata)
                    return [video_clip]
            except Exception as exc:
                print(
                    f"[WARN] Narration processing failed for clip {clip_id}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                # try the clip again but dont use narration
                self._append_note(metadata, f"Narration failed: {exc}")
                narration_enabled = False

        if not narration_enabled:
            # Reload original full-length clip
            video_clip = VideoFileClip(video_path)
            if not original_audio:
                self._record_audio_mix(
                    metadata,
                    mix_type="muted",
                    details={"gain_db": -100.0},
                )
                video_clip = video_clip.with_effects([afx.MultiplyVolume(0)])
            video_clip = self._attach_metadata(video_clip, metadata)
            metadata["timings"]["applied_start"] = float(start_sec)
            metadata["timings"]["applied_end"] = float(end_sec)
            video_clip = self._apply_clip_transform(
                video_clip,
                metadata,
                lambda c: self.safe_subclip(c, start_sec, end_sec),
            )
            self._record_audio_mix(
                metadata,
                mix_type="original" if original_audio else "muted",
                details={"gain_db": 0.0 if original_audio else -100.0},
            )
            metadata.pop("_lock_source_end", None)
            self._register_clip_edit(clip, metadata)
            return [video_clip]

            
    def snap_clips_to_music_beats(self, processed_clips, background_music_path, sr=22050, strong_snap_sec=1.0):
        """Adjusts each clip so the end lands on the nearest strong/weak beat, by changing its speed."""
        # 1. Analyze music for beat locations (in seconds)
        y, _ = librosa.load(background_music_path, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # Use all beats for both strong and weak, or apply more logic for downbeats if wanted
        strong_beats = beat_times  # You can filter for downbeats if you want, e.g., every N beats
        
        # 2. Adjust each clip to end on a nearby beat
        current_time = 0.0
        adjusted_clips = []
        for clip in processed_clips:
            orig_duration = clip.duration
            intended_end = current_time + orig_duration

            # Find beats within snapping window
            strong_cands = strong_beats[np.abs(strong_beats - intended_end) <= strong_snap_sec]
            snap_target = None

            if len(strong_cands) > 0:
                snap_target = strong_cands[np.argmin(np.abs(strong_cands - intended_end))]
            
            if snap_target is not None:
                new_duration = snap_target - current_time
                if new_duration > 0 and abs(new_duration - orig_duration) > 0.05:  # Only adjust if needed
                    speed_factor = orig_duration / new_duration
                    print(f"[SYNC] Retiming clip from {orig_duration:.2f}s to {new_duration:.2f}s, factor={speed_factor:.3f}")
                    # This will speed up or slow down the clip so it ends on the beat
                    clip = clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                    # Make absolutely sure duration matches
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
        """Concatenates all processed video clips and handles music mixing."""
        final_video = concatenate_videoclips(processed_clips)
        video_audio = final_video.audio
        if background_music_path:
            try:
                background_music = AudioFileClip(background_music_path)
                music_volume = self.get_loudness(background_music)
                video_volume = self.get_loudness(video_audio)
                if music_volume_override is not None:
                    volume_multiplier = music_volume_override
                elif video_volume == 0 or music_volume == 0:
                    volume_multiplier = self.music_volume_multiplier
                else:
                    volume_multiplier = (
                        10 ** ((video_volume - music_volume) / 20)
                    ) * self.music_volume_multiplier
                background_music = background_music.with_effects(
                    [afx.MultiplyVolume(volume_multiplier)]
                ).subclipped(0, final_video.duration)
                final_audio = CompositeAudioClip([video_audio, background_music])
                final_video = final_video.with_audio(final_audio)
            except Exception as e:
                print(f"[WARN] Failed to mix background music: {e}")
        else:
            print("[INFO] No background music provided. Exporting narration only.")
        
        return final_video

    
    def segment_subtitle_by_aspect(self, text, start, end, aspect_ratio):
        """
        Splits subtitle text into smaller segments depending on aspect ratio.
        - Horizontal (aspect >= 1.0): 2 lines × 10 words = 20 words max
        - Vertical   (aspect <  1.0): 2 lines × 5  words = 10 words max
        Each split segment shares the time interval evenly.
        """
        if aspect_ratio >= 1.0:
            max_words = 24
            max_words_per_line = 12
        else:
            max_words = 12
            max_words_per_line = 6

        words = text.strip().split()
        n_words = len(words)
        if n_words <= max_words:
            # No split needed
            return [(start, end, text, max_words_per_line)]

        # Split words into chunks
        segments = []
        n_segments = int(np.ceil(n_words / max_words))
        chunk_size = int(np.ceil(n_words / n_segments))
        total_duration = end - start

        for i in range(n_segments):
            chunk_words = words[i*chunk_size : (i+1)*chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunk_start = start + i * total_duration / n_segments
            chunk_end = start + (i+1) * total_duration / n_segments
            segments.append((chunk_start, chunk_end, chunk_text, max_words_per_line))
        return segments

    def wrap_lines(self, text, max_per_line):
        """
        Wraps a string into two lines with a given max words per line.
        """
        words = text.split()
        if len(words) <= max_per_line:
            return text
        # Allow only 2 lines
        line1 = ' '.join(words[:max_per_line])
        line2 = ' '.join(words[max_per_line:])
        return line1 + "\n" + line2

    
    def generate_subtitles_with_whisper(self, audio_path, srt_output_path, aspect_ratio=16/9):
        """
        Transcribes the given audio file using WhisperX and saves the subtitles in SRT format,
        adaptively splitting long subtitles by aspect ratio and wrapping to two lines.
        """
        model = whisperx.load_model("base", "cpu", compute_type="float32")
        result = model.transcribe(audio_path)
        segments = result["segments"]  # Each segment has start, end, text

        all_entries = []
        for segment in segments:
            # Split or keep as needed
            segs = self.segment_subtitle_by_aspect(
                segment["text"],
                segment["start"],
                segment["end"],
                aspect_ratio
            )
            all_entries.extend(segs)

        with open(srt_output_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text, max_per_line) in enumerate(all_entries, 1):
                start_ts = self._format_timestamp(start)
                end_ts = self._format_timestamp(end)
                wrapped_text = self.wrap_lines(text, max_per_line)
                f.write(f"{idx}\n{start_ts} --> {end_ts}\n{wrapped_text}\n\n")



    def _format_timestamp(self, seconds):
        """
        Formats the timestamp in SRT format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    def burn_subtitles(self, input_path, srt_path, output_path):
        """Burns subtitles from an SRT file into the video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)
        
    async def __call__(
        self,
        clips,
        narration_dir,
        background_music_path=None,
        original_audio=True,
        narration_enabled=True,
        aspect_ratio=16/9,
        subtitles=True,
        snap_to_beat=False
    ):
        print(f"[INFO] Processing {len(clips)} clips...")

        # Download all needed files first
        needed = {clip["file_name"]: clip["cloud_storage_path"] for clip in clips}
        for fname, cspath in needed.items():
            print(f"[DEBUG] Downloading source clip {fname} from {cspath}")
            original_path = self._download_media_file(fname, cspath)
            self._downloaded_files[fname] = original_path

        # Process and assemble video clips
        original_clips = []
        aspect_ratios = []
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            try:
                new_clips = self._process_clip(clip, narration_dir, narration_enabled, original_audio)
                print(f"[DEBUG] Clip {clip['id']} produced {len(new_clips)} segments")
                original_clips.extend(new_clips)
                for c in new_clips:
                    w, h = c.size
                    aspect_ratios.append(round(float(w) / float(h), 3))
            except Exception as e:
                print(f"[ERROR] Failed to process clip {clip['id']}: {e}\n{traceback.format_exc()}")
                continue

        if not original_clips:
            raise ValueError("No clips were processed successfully.")

        # --- Aspect Ratio Handling ---
        if aspect_ratio == 0:
            from collections import Counter
            aspect_counter = Counter(aspect_ratios)
            most_common_aspect, _ = aspect_counter.most_common(1)[0]
            aspect_ratio = most_common_aspect
            print(f"[INFO] Auto-detected aspect ratio: {aspect_ratio}")

        if aspect_ratio >= 1.0:
            export_width = 1920
            export_height = int(round(1920 / aspect_ratio))
        else:
            export_height = 1920
            export_width = int(round(1920 * aspect_ratio))

        print(f"[INFO] Target output resolution: {export_width}x{export_height}")

        # --- Cropping ---
        dc = DynamicCropping(self.llm, self.workdir)
        print(f"[DEBUG] Starting dynamic cropping for {len(original_clips)} clips")
        try:
            cropped_clips = await dc(export_width, export_height, original_clips)
        except Exception as exc:
            print(f"[ERROR] Dynamic cropping failed: {exc}\n{traceback.format_exc()}")
            dc.cleanup()
            raise
        print(f"[DEBUG] Dynamic cropping complete. Received {len(cropped_clips)} clips")

        if background_music_path and snap_to_beat:
            cropped_clips = self.snap_clips_to_music_beats(cropped_clips, background_music_path)

        self.timeline_metadata = [
            deepcopy(meta)
            for meta in (getattr(c, "_vea_metadata", None) for c in cropped_clips)
            if meta is not None
        ]

        total_timeline_duration = sum(
            float(meta.get("timeline", {}).get("duration", 0.0))
            for meta in self.timeline_metadata
        )

        music_volume_override = None
        music_gain_db = None
        if background_music_path and total_timeline_duration > 0:
            music_volume_override, music_gain_db = self._compute_music_adjustment(
                cropped_clips,
                background_music_path,
                total_timeline_duration,
            )
            music_meta = self._build_segment_metadata(
                {
                    "file_name": os.path.basename(background_music_path),
                    "cloud_storage_path": background_music_path,
                },
                0.0,
                float(total_timeline_duration),
                segment_type="music",
                original_audio=True,
            )
            music_meta["audio"]["mix"] = "music"
            music_meta["audio"]["details"] = {
                "music_path": background_music_path,
                "gain_multiplier": music_volume_override if music_volume_override is not None else self.music_volume_multiplier,
                "gain_db": music_gain_db if music_gain_db is not None else (
                    20.0 * math.log10(self.music_volume_multiplier) if self.music_volume_multiplier > 0 else -100.0
                ),
            }
            self.timeline_metadata.append(music_meta)

        try:
            debug_dump = {
                "clips": clips,
                "timeline_metadata": self.timeline_metadata,
            }
            tmp_dir = tempfile.gettempdir()
            dump_path = os.path.join(
                tmp_dir,
                f"movie_edit_metadata_{os.getpid()}_{int(time.time()*1000)}.json",
            )
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(debug_dump, f, indent=2, ensure_ascii=False, default=str)
            print(f"[DEBUG] Saved enriched clip metadata to {dump_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write metadata debug dump: {exc}")

        export_result_path = None

        final_video = None

        if MULTIROUND_MODE:
            if not self.timeline_metadata:
                raise RuntimeError("Timeline metadata is empty; cannot export FCPXML in multiround mode.")

            assets_root, video_asset_map, narration_asset_map, music_asset_path, music_asset_name = self._prepare_multiround_assets(
                narration_dir,
                background_music_path,
            )
            export_dir = os.path.join(self.workdir, "multiround_export")
            os.makedirs(export_dir, exist_ok=True)

            project_name = os.path.splitext(os.path.basename(self.output_path))[0] or "VEA Edit"
            fcpxml_path = os.path.join(export_dir, f"{project_name}.fcpxml")

            export_fcpxml(
                timeline_metadata=self.timeline_metadata,
                video_asset_map=video_asset_map,
                narration_asset_map=narration_asset_map,
                music_asset_path=music_asset_path,
                music_asset_name=music_asset_name,
                music_duration=total_timeline_duration,
                music_gain_db=music_gain_db,
                output_path=fcpxml_path,
                frame_rate=24,
                project_name=project_name,
                event_name="VEA Multiround Export",
            )

            print(f"[MULTIROUND] Assets copied to {assets_root}")
            if music_asset_path:
                print(f"[MULTIROUND] Background music copied to {music_asset_path}")
            print(f"[MULTIROUND] FCPXML saved to {fcpxml_path}")
            export_result_path = fcpxml_path
        else:
            # Concatenate and export
            final_video = self._assemble_final_video(
                cropped_clips,
                background_music_path,
                music_volume_override,
            )

            if subtitles:
                caption_dir = tempfile.mkdtemp()
                tmp_srt_path = os.path.join(caption_dir, "captions.srt")
                tmp_video_path = os.path.join(caption_dir, "no_caption.mp4")
                print(f"[DEBUG] Writing intermediate video (no captions) to {tmp_video_path}")
                await asyncio.to_thread(
                    final_video.write_videofile,
                    tmp_video_path,
                    preset='ultrafast',
                    fps=24,
                    threads=8,
                )
                audio_export_path = os.path.join(caption_dir, "final_audio.wav")
                await asyncio.to_thread(
                    final_video.audio.write_audiofile,
                    audio_export_path
                )

                subtitle_generator = GenerateSubtitles(output_dir=caption_dir)
                transcription_result = subtitle_generator(audio_export_path)

                words = transcription_result.get("words", [])
                if words:
                    srt_entries = subtitle_generator.words_to_srt_entries(words, max_words=12)
                    subtitle_generator.write_srt(srt_entries, tmp_srt_path)

                    try:
                        print(f"[DEBUG] Burning subtitles into video using {tmp_srt_path}")
                        self.burn_subtitles(tmp_video_path, tmp_srt_path, self.output_path)
                    except Exception as e:
                        print(f"[WARN] Failed to burn subtitles: {e}\n{traceback.format_exc()}")
                        self.output_path = tmp_video_path
            else:
                print(f"[DEBUG] Writing final video directly to {self.output_path}")
                await asyncio.to_thread(
                    final_video.write_videofile,
                    self.output_path,
                    preset='ultrafast',
                    fps=24,
                    threads=8,
                )
                export_result_path = self.output_path

        # Clean up
        print("[INFO] Cleaning up memory...")
        for clip in original_clips:
            try:
                clip.close()
            except Exception as e:
                print(f"[WARN] Failed to close original clip: {e}")
        for clip in cropped_clips:
            try:
                clip.close()
            except Exception as e:
                print(f"[WARN] Failed to close cropped clip: {e}")
        try:
            dc.cleanup()
        except Exception as e:
            print(f"[WARN] Dynamic cropping cleanup failed: {e}")
        if final_video is not None:
            try:
                final_video.close()
            except Exception as e:
                print(f"[WARN] Failed to close final video: {e}")

        gc.collect()

        if export_result_path is None:
            export_result_path = self.output_path

        print(f"[INFO] Final output: {export_result_path}")
        return export_result_path

        # Should never reach here but keep for safety
        return export_result_path
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Expected path to edit config JSON")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"[DEBUG] EditVideoResponse invoked with config {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(
        "[DEBUG] Parsed config summary:\n"
        f"  clips={len(config.get('clips', []))}\n"
        f"  narration_dir={config.get('narration_dir')}\n"
        f"  background_music_path={config.get('background_music_path')}\n"
        f"  aspect_ratio={config.get('aspect_ratio')}\n"
        f"  subtitles={config.get('subtitles')} snap_to_beat={config.get('snap_to_beat')}\n"
        f"  output_path={config.get('output_path')}"
    )

    # Reconstruct arguments
    clips = config["clips"]
    narration_dir = config["narration_dir"]
    background_music_path = config["background_music_path"]
    original_audio = config["original_audio"]
    narration_enabled = config["narration_enabled"]
    aspect_ratio = config["aspect_ratio"]
    subtitles = config["subtitles"]
    snap_to_beat = config["snap_to_beat"]
    workdir = config["workdir"]
    output_path = config["output_path"]
    bucket_name = config["bucket_name"]

    # Reinstantiate GCS and LLM
    gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
    llm = GeminiGenaiManager(model="gemini-1.5-flash")

    editor = EditVideoResponse(
        output_path=output_path,
        gcs_client=gcs_client,
        bucket_name=bucket_name,
        workdir=workdir,
        llm=llm
    )

    asyncio.run(editor(
        clips=clips,
        narration_dir=narration_dir,
        background_music_path=background_music_path,
        original_audio=original_audio,
        narration_enabled=narration_enabled,
        aspect_ratio=aspect_ratio,
        subtitles=subtitles,
        snap_to_beat=snap_to_beat
    ))