import os
import tempfile
import asyncio
from moviepy import *
from pydub.utils import mediainfo
import librosa
import numpy as np
import subprocess
from lib.utils.media import parse_time_to_seconds
from src.pipelines.common.dynamic_cropping import DynamicCropping
import whisper
import whisperx

class EditVideoResponse:
    def __init__(
        self, 
        output_path="video_response.mp4", 
        music_volume_multiplier=0.3, 
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

    # ---------- File Download Helpers ----------
    def _download_media_file(self, file_name, cloud_storage_path):
        """Download the media file from GCS if not already present."""
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]
        local_path = os.path.join(self.workdir, file_name)
        self.gcs_client.download_files(self.bucket_name, cloud_storage_path, local_path)
        self._downloaded_files[file_name] = local_path
        return local_path

    # ---------- Video Clip Assembly ----------
    def _process_clip(
        self, clip, narration_dir, narration_enabled=True, original_audio=True
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

        video_clip = VideoFileClip(video_path)

        if narration_enabled:
            try:
                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                narration = AudioFileClip(audio_path)
                audio_duration = narration.duration
                conservative_trim_end = max(end_sec, min(start_sec + audio_duration, video_clip.duration))
                video_clip = video_clip.subclipped(start_sec, conservative_trim_end)
                video_duration = video_clip.duration
                if video_duration < audio_duration:
                    speed_factor = video_duration / audio_duration if audio_duration > 0 else 1
                    video_clip = video_clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                clip_volume = video_clip.audio.max_volume().mean()
                narration_volume = narration.max_volume().mean()
                volume_multiplier = (narration_volume / clip_volume) if narration_volume != 0 and clip_volume != 0 else 1.0

                if not original_audio:
                    video_clip.with_effects([afx.MultiplyVolume(0)])

                # Priority handling
                if not priority:
                    priority = "narration"
                if priority == "clip_audio":
                    # Play twice: once with narration, then with original audio boosted
                    video_clip_copy = video_clip.with_effects([afx.MultiplyVolume(volume_multiplier)])
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = video_clip.subclipped(0, audio_duration)
                    return [video_clip, video_clip_copy]
                elif priority == "clip_video":
                    video_clip = video_clip.with_audio(narration)
                    return [video_clip]
                else:  # narration default
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = video_clip.subclipped(0, audio_duration)
                    return [video_clip]
            except:
                # try the clip again but dont use narration
                narration_enabled = False
                pass

        if not narration_enabled:
            # No narration: Use original audio, as is
            video_clip = video_clip.subclipped(start_sec, end_sec)
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
                    clip = clip.subclipped(0, new_duration)
                    orig_duration = new_duration

            adjusted_clips.append(clip)
            current_time += orig_duration

        return adjusted_clips

    def _assemble_final_video(
        self, processed_clips, background_music_path=None
    ):
        """Concatenates all processed video clips and handles music mixing."""
        final_video = concatenate_videoclips(processed_clips)
        video_audio = final_video.audio

        # If background music, mix it in at the correct level
        if background_music_path:
            try:
                background_music = AudioFileClip(background_music_path)
                music_volume = background_music.max_volume().mean()
                video_volume = video_audio.max_volume().mean()
                # Adjust music relative to overall video/narration volume
                if video_volume == 0 or music_volume == 0:
                    volume_multiplier = self.music_volume_multiplier
                else:
                    volume_multiplier = video_volume / music_volume * self.music_volume_multiplier
                background_music = background_music.with_effects(
                    [afx.MultiplyVolume(volume_multiplier)]
                ).subclipped(0, final_video.duration)
                final_audio = CompositeAudioClip([video_audio, background_music])
                final_video = final_video.with_audio(final_audio)
            except Exception as e:
                pass
        else:
            print("[INFO] No background music provided. Exporting narration only.")
        return final_video
    
    def generate_subtitles_with_whisper(self, audio_path, srt_output_path):
        """
        Transcribes the given audio file using WhisperX and saves the subtitles in SRT format.
        """
        self.whisperx_model = whisperx.load_model("base", "cpu", compute_type="float32")
        result = self.whisperx_model.transcribe(audio_path)
        segments = result["segments"]  # Each segment has start, end, text

        with open(srt_output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


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
        original_audio = True,
        narration_enabled=True,
        aspect_ratio=16/9,
        subtitles = True,
        snap_to_beat = False
    ):
        print(f"[INFO] Processing {len(clips)} clips...")

        # Download all needed files first
        needed = {clip["file_name"]: clip["cloud_storage_path"] for clip in clips}
        for fname, cspath in needed.items():
            original_path = self._download_media_file(fname, cspath)
            self._downloaded_files[fname] = original_path

        # Process and assemble video clips
        processed_clips = []
        aspect_ratios = []
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            try:
                new_clips = self._process_clip(clip, narration_dir, narration_enabled, original_audio)
                processed_clips.extend(new_clips)
                # Track aspect ratio for each processed clip
                for c in new_clips:
                    w, h = c.size
                    aspect_ratios.append(round(float(w) / float(h), 3))
            except Exception as e:
                print(f"[ERROR] Failed to process clip {clip['id']}: {e}")
                continue

        if not processed_clips:
            raise ValueError("No clips were processed successfully.")

        # --- Aspect Ratio Handling ---
        if aspect_ratio == 0:
            # Find most common aspect ratio among clips (rounded to 2 decimal places)
            from collections import Counter
            aspect_counter = Counter(aspect_ratios)
            most_common_aspect, _ = aspect_counter.most_common(1)[0]
            aspect_ratio = most_common_aspect
            print(f"[INFO] Auto-detected aspect ratio: {aspect_ratio}")

        # Calculate export width/height close to 1080p for at least one dimension
        if aspect_ratio >= 1.0:
            # Landscape: width = 1920, height = round(1920 / aspect)
            export_width = 1920
            export_height = int(round(1920 / aspect_ratio))
        else:
            # Portrait: height = 1920, width = round(1920 * aspect)
            export_height = 1920
            export_width = int(round(1920 * aspect_ratio))

        print(f"[INFO] Target output resolution: {export_width}x{export_height}")

        # --- Cropping ---
        dc = DynamicCropping(self.llm, self.workdir, batch_size=8)
        processed_clips = await dc(export_width, export_height, processed_clips)

        if background_music_path and snap_to_beat:
            processed_clips = self.snap_clips_to_music_beats(processed_clips, background_music_path)

        # Concatenate, mix music, and export to temp mp4
        final_video = self._assemble_final_video(
            processed_clips, background_music_path
        )

        if subtitles:
            caption_dir = tempfile.mkdtemp()
            tmp_srt_path = os.path.join(caption_dir, "captions.srt")
            tmp_video_path = os.path.join(caption_dir, "no_caption.mp4")
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
            self.generate_subtitles_with_whisper(audio_export_path, tmp_srt_path)
            try:
                self.burn_subtitles(tmp_video_path, tmp_srt_path, self.output_path)
            except:
                self.output_path = tmp_video_path
        else:
            await asyncio.to_thread(
                final_video.write_videofile,
                self.output_path,
                preset='ultrafast',
                fps=24,
                threads=8,
            )

        for clip in processed_clips:
            clip.close()
        final_video.close()

        print(f"[INFO] Final video created: {self.output_path}")
        return self.output_path
