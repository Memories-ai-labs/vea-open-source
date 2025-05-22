import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

class ClipExtractor:
    def __init__(self, workdir, gcs_client, bucket_name):
        self.workdir = workdir
        self.gcs_client = gcs_client
        self.bucket_name = bucket_name
        self.downloaded_files = {}

    def _time_to_seconds(self, time_str):
        h, m, s = map(float, time_str.split(":"))
        return h * 3600 + m * 60 + s

    def _download_if_needed(self, cloud_storage_path):
        """
        Download the video file from the given GCS path (cloud_storage_path) only once.
        Uses the basename of the path for the local file.
        """
        filename = os.path.basename(cloud_storage_path)
        if filename in self.downloaded_files:
            return self.downloaded_files[filename]
        local_path = os.path.join(self.workdir, filename)
        self.gcs_client.download_files(self.bucket_name, cloud_storage_path, local_path)
        self.downloaded_files[filename] = local_path
        return local_path

    def _extract_single_clip(self, movie_path, clip, index):
        start = clip["start"]
        end = clip["end"]
        duration = self._time_to_seconds(end) - self._time_to_seconds(start)
        output_path = os.path.join(self.workdir, f"clip_{index}.mp4")
        cmd = [
            "ffmpeg", "-ss", start, "-i", movie_path, "-t", str(duration),
            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def extract_and_upload_clips(self, clips, run_id, upload_prefix="evidence"):
        """
        For each clip in clips, download the movie if needed using cloud_storage_path, extract, upload, and return a list of GCS paths.
        Each clip dict must contain 'cloud_storage_path', 'file_name', 'start', 'end'.
        """
        gcs_clip_paths = []
        for idx, clip in enumerate(clips):
            cloud_storage_path = clip["cloud_storage_path"]  # This is the path in GCS
            file_name = os.path.basename(cloud_storage_path)
            movie_local_path = self._download_if_needed(cloud_storage_path)
            local_clip_path = self._extract_single_clip(movie_local_path, clip, idx)
            # Make unique GCS path for each
            gcs_rel_path = f"{upload_prefix}/{os.path.splitext(file_name)[0]}/{run_id}/clip_{idx}.mp4"
            self.gcs_client.upload_files(self.bucket_name, local_clip_path, gcs_rel_path)
            gcs_clip_paths.append(gcs_rel_path)
        return gcs_clip_paths
