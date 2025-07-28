import os
import tempfile
import asyncio
import gc
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
        if video_clip.audio is None:
            silent_audio = AudioClip(
                lambda t: [0, 0],
                duration=video_clip.duration,
                fps=44100,
            )
            video_clip = video_clip.with_audio(silent_audio)

        if narration_enabled:
            try:
                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                narration = AudioFileClip(audio_path)
                audio_duration = narration.duration
                conservative_trim_end = max(end_sec, min(start_sec + audio_duration, video_clip.duration))
                video_clip = self.safe_subclip(video_clip, start_sec, conservative_trim_end)
                video_duration = video_clip.duration
                if video_duration < audio_duration:
                    speed_factor = video_duration / audio_duration if audio_duration > 0 else 1
                    video_clip = video_clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                clip_volume = self.get_loudness(video_clip.audio)
                narration_volume = self.get_loudness(narration)
                volume_multiplier = (
                    10 ** ((narration_volume - clip_volume) / 20)
                    if narration_volume != 0 and clip_volume != 0
                    else 1.0
                )

                if not original_audio:
                    video_clip = video_clip.with_effects([afx.MultiplyVolume(0)])

                # Priority handling
                if not priority:
                    priority = "narration"
                if priority == "clip_audio":
                    # Play twice: once with narration, then with original audio boosted
                    video_clip_copy = video_clip.with_effects([afx.MultiplyVolume(volume_multiplier)])
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = self.safe_subclip(video_clip, 0, audio_duration)
                    return [video_clip, video_clip_copy]
                elif priority == "clip_video":
                    video_clip = video_clip.with_audio(narration)
                    return [video_clip]
                else:  # narration default
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = self.safe_subclip(video_clip, 0, audio_duration)
                    return [video_clip]
            except:
                # try the clip again but dont use narration
                narration_enabled = False
                pass

        if not narration_enabled:
            # Reload original full-length clip
            video_clip = VideoFileClip(video_path)
            if not original_audio:
                video_clip = video_clip.with_effects([afx.MultiplyVolume(0)])
            video_clip = self.safe_subclip(video_clip, start_sec, end_sec)
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
        self, processed_clips, background_music_path=None
    ):
        """Concatenates all processed video clips and handles music mixing."""
        final_video = concatenate_videoclips(processed_clips)
        video_audio = final_video.audio
        if background_music_path:
            try:
                background_music = AudioFileClip(background_music_path)
                music_volume = self.get_loudness(background_music)
                video_volume = self.get_loudness(video_audio)
                if video_volume == 0 or music_volume == 0:
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
            original_path = self._download_media_file(fname, cspath)
            self._downloaded_files[fname] = original_path

        # Process and assemble video clips
        original_clips = []
        aspect_ratios = []
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            try:
                new_clips = self._process_clip(clip, narration_dir, narration_enabled, original_audio)
                original_clips.extend(new_clips)
                for c in new_clips:
                    w, h = c.size
                    aspect_ratios.append(round(float(w) / float(h), 3))
            except Exception as e:
                print(f"[ERROR] Failed to process clip {clip['id']}: {e}")
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
        cropped_clips = await dc(export_width, export_height, original_clips)

        if background_music_path and snap_to_beat:
            cropped_clips = self.snap_clips_to_music_beats(cropped_clips, background_music_path)

        # Concatenate and export
        final_video = self._assemble_final_video(cropped_clips, background_music_path)

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

            subtitle_generator = GenerateSubtitles(output_dir=caption_dir)
            transcription_result = subtitle_generator(audio_export_path)

            words = transcription_result.get("words", [])
            if words:
                srt_entries = subtitle_generator.words_to_srt_entries(words, max_words=12)
                subtitle_generator.write_srt(srt_entries, tmp_srt_path)

                try:
                    self.burn_subtitles(tmp_video_path, tmp_srt_path, self.output_path)
                except Exception as e:
                    print(f"[WARN] Failed to burn subtitles: {e}")
                    self.output_path = tmp_video_path
        else:
            await asyncio.to_thread(
                final_video.write_videofile,
                self.output_path,
                preset='ultrafast',
                fps=24,
                threads=8,
            )

        # Clean up memory: close clips
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
            final_video.close()
        except Exception as e:
            print(f"[WARN] Failed to close final video: {e}")

        # Explicit garbage collection
        gc.collect()

        print(f"[INFO] Final video created: {self.output_path}")
        return self.output_path
    

if __name__ == "__main__":
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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
    llm = GeminiGenaiManager(model="gemini-2.5-flash")

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