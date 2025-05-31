import os
import json
import asyncio
from moviepy import *
from pathlib import Path
from pydub.utils import mediainfo
import subprocess
import tempfile
from pydub import AudioSegment
from moviepy.tools import close_all_clips


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

    def write_srt(self, clips, srt_path, narration_dir, max_words_per_line=10):
        def format_srt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        def split_into_halves(text):
            words = text.split()
            half = len(words) // 2
            return ' '.join(words[:half]), ' '.join(words[half:])

        current_time = 0.0  # running timestamp in the final video

        with open(srt_path, "w", encoding="utf-8") as f:
            srt_index = 1

            for clip in clips:
                clip_id = clip["id"]
                sentence = clip.get("corresponding_summary_sentence", "").replace("\n", " ").strip()

                audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")
                duration = self.get_audio_duration(audio_path)

                if len(sentence.split()) > max_words_per_line * 2:
                    # Split long subtitles into two
                    part1, part2 = split_into_halves(sentence)

                    mid_time = current_time + duration / 2

                    # Write first part
                    f.write(f"{srt_index}\n")
                    f.write(f"{format_srt_time(current_time)} --> {format_srt_time(mid_time)}\n")
                    f.write(f"{part1}\n\n")
                    srt_index += 1

                    # Write second part
                    f.write(f"{srt_index}\n")
                    f.write(f"{format_srt_time(mid_time)} --> {format_srt_time(current_time + duration)}\n")
                    f.write(f"{part2}\n\n")
                    srt_index += 1
                else:
                    # Regular 1-part subtitle
                    f.write(f"{srt_index}\n")
                    f.write(f"{format_srt_time(current_time)} --> {format_srt_time(current_time + duration)}\n")
                    f.write(f"{sentence}\n\n")
                    srt_index += 1

                current_time += duration


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

        final_video = concatenate_videoclips(processed_clips, method="compose")

        narration_audio = final_video.audio

        # --- Calculate average volume of narration
        tmp_narration_path = os.path.join(tempfile.mkdtemp(), "combined_narration.mp3")
        final_video.audio.write_audiofile(tmp_narration_path, fps=44100, codec="libmp3lame")
        narration_volume = AudioSegment.from_file(tmp_narration_path).dBFS

        # --- Load background music and measure its volume
        music_segment = AudioSegment.from_file(background_music_path)
        music_volume = music_segment.dBFS

        # --- Adjust music volume to be ~6 dB lower than narration
        target_music_volume = narration_volume - 6.0
        delta_db = target_music_volume - music_volume
        volume_multiplier = 10 ** (delta_db / 20)

        background_music = AudioFileClip(background_music_path).with_effects(
            [afx.MultiplyVolume(volume_multiplier)]
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

        # cleanup
        for clip in processed_clips:
            clip.close()
        narration_audio.close()
        background_music.close()
        final_video.close()
        close_all_clips()


        # burn subtitles into the final video
        self.burn_subtitles(tmp_video_path, tmp_srt_path, self.output_path)


        print(f"[INFO] Final recap video created: {self.output_path}")
        return self.output_path
