import os
import tempfile
import subprocess
import asyncio
from moviepy import *
from pydub.utils import mediainfo
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

    def trim_video(self, video_path, start_hms, end_hms, duration):
        start_sec = parse_time_to_seconds(start_hms)
        end_sec = parse_time_to_seconds(end_hms)
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

    def _download_media_file(self, file_name, cloud_storage_path):
        # Only download each media file once (by file_name)
        if file_name in self._downloaded_files:
            return self._downloaded_files[file_name]
        local_path = os.path.join(self.workdir, file_name)
        self.gcs_client.download_files(self.bucket_name, cloud_storage_path, local_path)
        self._downloaded_files[file_name] = local_path
        return local_path
    
    @staticmethod
    def preprocess_clip_ffmpeg(input_path, output_path, target_size=(1920, 1080), target_fps=24):
        width, height = target_size
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", input_path,
            "-vf", f"scale={width}:{height},fps={target_fps}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-an",  # No audio
            "-preset", "ultrafast",  # fastest for CPU
            output_path
        ]
        print(f"[FFMPEG] Preprocessing: {input_path} -> {output_path}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

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
                preprocessed_path = os.path.join(self.workdir, f"{os.path.splitext(fname)[0]}_pre.mp4")

                # Only preprocess if not already done
                if not os.path.exists(preprocessed_path):
                    print(f"[INFO] Preprocessing {fname} to {preprocessed_path} (1080p 24fps)")
                    await asyncio.to_thread(
                        self.preprocess_clip_ffmpeg,
                        original_path, preprocessed_path, (1920, 1080), 24
                    )
                else:
                    print(f"[INFO] Preprocessed file already exists: {preprocessed_path}")
                # Update the map to use preprocessed file
                self._downloaded_files[fname] = preprocessed_path

            for clip in sorted(clips, key=lambda c: int(c["id"])):
                clip_id = clip["id"]
                file_name = clip["file_name"]
                audio_path = os.path.join(narration_dir, f"{clip_id:04d}.mp3")
                audio_duration = self.get_audio_duration(audio_path)

                video_path = self._downloaded_files[file_name]

                video = await asyncio.to_thread(
                    self.trim_video,
                    video_path,
                    clip["start"],
                    clip["end"],
                    audio_duration
                )

                if video:
                    narration = AudioFileClip(audio_path)
                    video = video.with_audio(narration)
                    processed_clips.append(video)

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
