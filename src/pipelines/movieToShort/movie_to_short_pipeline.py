import math
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH
from src.config import BUCKET_NAME

from src.pipelines.flexibleResponse.flexibleResponsePipeline import FlexibleResponsePipeline
from lib.utils.media import seconds_to_hhmmss, get_video_duration  # you may already have get_video_duration

class MovieToShortsPipeline:
    """
    Generates all 1-minute shorts from a movie, each covering the best moments from sequential 5-minute windows.
    """
    def __init__(self, blob_path: str, short_duration: int = 60, window_size: int = 900, aspect_ratio: float = 1):
        self.blob_path = blob_path
        self.short_duration = short_duration  # Output short length in seconds (1 min)
        self.window_size = window_size        # Movie window per short (5 min)
        self.aspect_ratio = aspect_ratio
        self.flexible_pipeline = FlexibleResponsePipeline(blob_path)
        self.gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

    async def run(self):
        # Download the movie to get total duration
        print("[SHORTS] Getting movie duration...")
        local_path = await self._download_movie()
        total_duration = get_video_duration(local_path)
        print(f"[SHORTS] Total duration: {total_duration:.2f} sec")

        shorts = []
        n_shorts = math.ceil(total_duration / self.window_size)
        for i in range(n_shorts):
            window_start = i * self.window_size
            window_end = min(window_start + self.window_size, total_duration)
            prompt = (
                f"I am creating a short video from a movie. Select the most important and interesting moments from {seconds_to_hhmmss(window_start)} "
                f"to {seconds_to_hhmmss(window_end)} of the original movie. pick clips and dialogue that effectively tell the story, and do not cut off speech. "
                f"The goal is to create a highly engaging, approximately {self.short_duration}-second short using original video and audio for people to become interested in the movie and binge watch the shorts."
                f"Do not overlap in your clip selections, and avoid duplication. Edit together the best dialogue or visuals from this window into a {self.short_duration}-second short. each short should be somewhat "
                "self contained as a story that delivers satisfaction. Avoid recap or summary—just deliver the most exciting original content for short-form consumption."
            )
            print(f"[SHORTS] Generating short #{i+1} for window {seconds_to_hhmmss(window_start)} - {seconds_to_hhmmss(window_end)}")
            result = await self.flexible_pipeline.run(
                user_prompt=prompt,
                video_response=True,
                original_audio=True,
                music=False,
                narration_enabled=False,
                aspect_ratio=self.aspect_ratio,
                subtitles=True
            )
            shorts.append({
                "short_index": i,
                "window_start": seconds_to_hhmmss(window_start),
                "window_end": seconds_to_hhmmss(window_end),
                "response": result
            })
        return shorts

    async def _download_movie(self):
        """
        Downloads the movie file from GCS to a temp location if not already present, returns local path.
        """
        import tempfile
        import os
        fname = os.path.basename(self.blob_path)
        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, fname)
        if not os.path.exists(local_path):
            print(f"[SHORTS] Downloading {self.blob_path} to {local_path} ...")
            self.gcs_client.download_files(BUCKET_NAME, self.blob_path, local_path)
        else:
            print(f"[SHORTS] Using cached {local_path}")
        return local_path
