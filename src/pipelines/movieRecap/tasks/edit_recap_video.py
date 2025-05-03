import os
import json
import asyncio
from moviepy import *
from pathlib import Path
from pydub.utils import mediainfo

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

        await asyncio.to_thread(
            final_video.write_videofile,
            self.output_path,
            codec='libx264',
            audio_codec='aac'
        )

        print(f"[INFO] Final recap video created: {self.output_path}")
