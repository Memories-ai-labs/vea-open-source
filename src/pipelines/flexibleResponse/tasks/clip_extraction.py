import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

class ClipExtractor:
    def __init__(self, movie_path):
        self.movie_path = movie_path
        self.clip_dir = tempfile.mkdtemp()

    def _time_to_seconds(self, time_str):
        h, m, s = map(float, time_str.split(":"))
        return h * 3600 + m * 60 + s

    def _extract_single_clip(self, clip, index):
        start = clip["start"]
        end = clip["end"]
        duration = self._time_to_seconds(end) - self._time_to_seconds(start)

        output_path = os.path.join(self.clip_dir, f"clip_{index}.mp4")
        cmd = [
            "ffmpeg", "-ss", start, "-i", self.movie_path, "-t", str(duration),
            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", output_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def extract_clips(self, clips: list[dict], max_workers=4):
        print(f"[INFO] Extracting {len(clips)} clips in parallel with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._extract_single_clip, clip, i) for i, clip in enumerate(clips)]
            return [f.result() for f in futures]
