import os
import tempfile
import subprocess
import asyncio
from moviepy import *
from pydub.utils import mediainfo
from pydub import AudioSegment

from lib.utils.media import parse_time_to_seconds


class EditFlexibleVideoResponse:
    def __init__(
        self, 
        output_path="video_response.mp4", 
        music_volume_multiplier=0.5, 
        gcs_client=None, 
        gcs_media_base_path=None, 
        bucket_name=None,
        workdir=None
    ):
        self.output_path = output_path
        self.music_volume_multiplier = music_volume_multiplier
        self.gcs_client = gcs_client
        self.gcs_media_base_path = gcs_media_base_path
        self.bucket_name = bucket_name
        self.workdir = workdir or tempfile.mkdtemp()
        self._downloaded_files = {}

    def get_audio_duration(self, audio_path):
        info = mediainfo(audio_path)
        return float(info['duration'])

    def _download_media_file(self, file_name, cloud_storage_path):
        # Only download each media file once (by file_name)
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]
        local_path = os.path.join(self.workdir, file_name)
        self.gcs_client.download_files(self.bucket_name, cloud_storage_path, local_path)
        self._downloaded_files[file_name] = local_path
        return local_path

    async def __call__(
            self,
            clips,
            narration_dir,
        ):
            processed_clips = []
            print(f"[INFO] Processing {len(clips)} clips...")

            # Download each media file needed (once)
            needed = {}
            for clip in clips:
                fname = clip["file_name"]
                cspath = clip["cloud_storage_path"]
                if fname not in needed:
                    needed[fname] = cspath

            for fname, cspath in needed.items():
                original_path = self._download_media_file(fname, cspath)
                self._downloaded_files[fname] = original_path

            for clip in sorted(clips, key=lambda c: int(c["id"])):
                clip_id = clip["id"]
                file_name = clip["file_name"]
                audio_path = os.path.join(narration_dir, f"{clip_id:04d}.mp3")
                video_path = self._downloaded_files[file_name]

                start_sec = parse_time_to_seconds(clip["start"])
                end_sec = parse_time_to_seconds(clip["end"])
                video_clip = VideoFileClip(video_path).subclipped(start_sec, end_sec).resized((1920, 1080))
                video_duration = video_clip.duration
                priority = clip.get("priority", False)
                video_clip_copy = video_clip

                audio_duration = self.get_audio_duration(audio_path)
                if video_duration < audio_duration:
                    speed_factor = video_duration / audio_duration if audio_duration > 0 else 1
                    video_clip = video_clip.with_effects([vfx.MultiplySpeed(speed_factor)])
                narration = AudioFileClip(audio_path)

                if priority == "clip_audio":
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = video_clip.subclipped(0, audio_duration)
                    processed_clips.append(video_clip)
                    processed_clips.append(video_clip_copy.with_effects([afx.MultiplyVolume(1.5)]))
                elif priority == "clip_video":
                    video_clip = video_clip.with_audio(narration)
                    processed_clips.append(video_clip)
                else:
                    video_clip = video_clip.with_audio(narration)
                    if video_duration > audio_duration:
                        video_clip = video_clip.subclipped(0, audio_duration)
                    processed_clips.append(video_clip)


            if not processed_clips:
                raise ValueError("No clips were processed successfully.")

            final_video = concatenate_videoclips(processed_clips)

            # render the final video
            await asyncio.to_thread(
                final_video.write_videofile,
                self.output_path,
                codec='libx264',
                audio_codec='aac'
            )

            print(f"[INFO] Final flexible video created: {self.output_path}")
            return self.output_path
