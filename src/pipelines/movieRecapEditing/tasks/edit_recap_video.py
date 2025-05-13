import os
import json
import asyncio
from moviepy import *
from pathlib import Path
from pydub.utils import mediainfo
import subprocess
import tempfile


class EditMovieRecapVideo:
    def __init__(self, output_path="final_recap.mp4", music_volume_multiplier=0.5):
        self.output_path = output_path
        self.music_volume_multiplier = music_volume_multiplier

    def get_audio_duration(self, audio_path):
        info = mediainfo(audio_path)
        return float(info['duration'])

    def hhmmss_to_seconds(self, hhmmss):
        h, m, s = map(float, hhmmss.split(":"))
        return h * 3600 + m * 60 + s

    def trim_video(self, video_path, start_hms, end_hms, duration):
        start_sec = self.hhmmss_to_seconds(start_hms)
        end_sec = self.hhmmss_to_seconds(end_hms)
        window_duration = end_sec - start_sec

        video = VideoFileClip(video_path)
        max_duration = video.duration
        start_sec = min(start_sec, max_duration - window_duration)
        end_sec = min(end_sec, max_duration)
        video = video.subclipped(start_sec, end_sec)

        if duration <= window_duration:
            center = window_duration / 2
            sub_start = max(0, center - duration / 2)
            sub_end = sub_start + duration
            print(f"[INFO] Trimming centered subclip: {sub_start:.2f}s to {sub_end:.2f}s within {window_duration:.2f}s window")
            return video.subclipped(sub_start, sub_end)
        else:
            if window_duration == 0:
                return None
            speed_factor = window_duration / duration
            print(f"[WARN] Window too short. Slowing down by factor {1/speed_factor:.2f} to match {duration:.2f}s")
            return video.with_effects([vfx.MultiplySpeed(speed_factor)])

    def write_srt(self, clips, srt_path, narration_dir):
        def format_srt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        current_time = 0.0  # running timestamp in the final video

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, clip in enumerate(clips, start=1):
                clip_id = clip["id"]
                sentence = clip.get("corresponding_summary_sentence", "").replace("\n", " ").strip()

                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                duration = self.get_audio_duration(audio_path)

                start_sec = current_time
                end_sec = current_time + duration

                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}\n")
                f.write(f"{sentence}\n\n")

                current_time = end_sec  # move forward for the next caption


    def burn_subtitles(self, input_path, srt_path, output_path):
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite if exists
            "-i", input_path,
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)

    async def __call__(
        self,
        clips,
        movie_path,
        narration_dir,
        background_music_path,
    ):
        processed_clips = []

        for clip in sorted(clips, key=lambda c: int(c["id"])):
            clip_id = clip["id"]
            audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")

            audio_duration = self.get_audio_duration(audio_path)

            video = await asyncio.to_thread(
                self.trim_video,
                movie_path,
                clip["start_timestamp"],
                clip["end_timestamp"],
                audio_duration
            )

            if video:
                narration = AudioFileClip(audio_path)
                video = video.with_audio(narration)
                processed_clips.append(video)

        if not processed_clips:
            raise ValueError("No clips were processed successfully.")

        final_video = concatenate_videoclips(processed_clips)

        narration_audio = final_video.audio
        background_music = AudioFileClip(background_music_path).with_effects(
            [afx.MultiplyVolume(self.music_volume_multiplier)]
        ).subclipped(0, final_video.duration)
        final_audio = CompositeAudioClip([narration_audio, background_music])
        final_video = final_video.with_audio(final_audio)
        
        # render the final video and add captions
        caption_dir = tempfile.mkdtemp()
        tmp_srt_path = os.path.join(caption_dir, "captions.srt")
        tmp_video_path = os.path.join(caption_dir, "no_caption.mp4")
        await asyncio.to_thread(
            final_video.write_videofile,
            tmp_video_path,
            codec='libx264',
            audio_codec='aac'
        )

        # add captions
        self.write_srt(clips, tmp_srt_path, narration_dir)
        self.burn_subtitles(tmp_video_path, tmp_srt_path, self.output_path)


        print(f"[INFO] Final recap video created: {self.output_path}")
        return self.output_path
