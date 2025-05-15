
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import subprocess
import json
from typing import Optional
import time
import os, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

def get_video_info(video_path: str) -> dict:
    """
    Extract video metadata (width, height, fps, duration, etc.) using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=False)

    if result.returncode != 0:
        # 把错误流也 decode 成 utf-8，看清楚
        raise RuntimeError(f"ffprobe error: {result.stderr.decode('utf-8')}")

    metadata_json = result.stdout.decode('utf-8')  # ✅ 手动decode成utf-8
    metadata = json.loads(metadata_json)
    return metadata

def get_video_duration(video_path: str) -> dict:
    video_info = get_video_info(video_path)
    return float(video_info["format"]["duration"])

def build_ffmpeg_command(input_path, output_path, start, duration, filter_str, crf):
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(input_path), "-t", str(duration)]
    if filter_str:
        cmd += ["-vf", filter_str]
    cmd += ["-c:v", "libx264", "-crf", str(crf or 28), "-preset", "ultrafast", "-c:a", "copy", "-threads", "1", str(output_path)]
    return cmd

async def convert_video(
    input_path: str,
    output_dir: str,
    interval_seconds: int,
    fps: Optional[int] = None,
    crf: Optional[int] = 28,
    target_height: Optional[int] = 480,
    max_workers: int = 4,
) -> list[Path]:
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_duration = get_video_duration(input_path)
    filter_parts = []
    if target_height:
        filter_parts.append(f"scale=-2:{target_height}")
    if fps:
        filter_parts.append(f"fps={fps}")
    filter_str = ",".join(filter_parts) if filter_parts else None

    tasks = []
    paths = []

    current_start = 0
    while current_start < total_duration:
        start = int(current_start)
        end = int(min(current_start + interval_seconds, total_duration))
        duration = end - start
        output_path = output_dir / f"clip_{start:05d}_{end:05d}.mp4"
        cmd = build_ffmpeg_command(input_path, output_path, start, duration, filter_str, crf)
        tasks.append((cmd, output_path))
        current_start += interval_seconds

    def run_cmd(cmd):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cmd, cmd) for cmd, _ in tasks]
        for f in futures: f.result()
        paths = [p for _, p in tasks]

    end_time = time.time()
    print(f"[INFO] Completed {len(paths)} segments in {end_time - start_time:.2f} sec")
    return paths


# if __name__ == "__main__":
#     from pathlib import Path
#     import asyncio
#     video_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test/爱在日落黄昏时.mkv"
#     output_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test"
#     info = get_video_info(video_path)
#     # print(info)
#     asyncio.run(convert_video(video_path, Path(output_path), interval_seconds=1200, fps=1, crf=28))
