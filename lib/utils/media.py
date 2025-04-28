
import logging
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


import subprocess
import json
from typing import Optional
import time


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


async def convert_video(
    input_path: str,
    output_dir: str,
    interval_seconds: int,
    fps: Optional[int] = None,
    crf: Optional[int] = None,
    target_height: Optional[int] = 480,
) -> list[str]:
    """
    Split the input video into segments of length `interval_seconds`.
    
    Args:
        input_path (str): Path to the input video.
        output_dir (str): Directory to save the output segments.
        interval_seconds (int): Length of each segment (in seconds).
        fps (Optional[int]): Frames per second.
        crf (Optional[int]): Constant rate factor (quality control).
        target_height (Optional[int]): Resize video height (maintains aspect ratio).

    Returns:
        List[str]: List of output segment file paths, in order.
    """
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    total_duration = get_video_duration(input_path)
    logger.info(f"Total video duration: {total_duration:.2f} seconds")

    segment_idx = 0
    current_start = 0.0
    output_paths = []

    while current_start < total_duration:
        output_filename = os.path.join(output_dir, f"segment_{segment_idx:04d}.mp4")

        # Calculate actual duration for this segment
        duration = min(interval_seconds, total_duration - current_start)

        cmd = ["ffmpeg", "-y", "-ss", str(current_start), "-i", input_path, "-t", str(duration)]

        filters = []
        if target_height:
            filters.append(f"scale=-2:{target_height}")
        if fps:
            filters.append(f"fps={fps}")
        if filters:
            cmd += ["-vf", ",".join(filters)]

        cmd += ["-c:v", "libx264"]
        if crf:
            cmd += ["-crf", str(crf)]
        cmd += ["-preset", "medium"]
        cmd += ["-c:a", "copy"]
        cmd += [output_filename]

        logger.info(f"Generating segment {segment_idx}: start={current_start:.2f}s duration={duration:.2f}s")
        result = subprocess.run(cmd, capture_output=True, text=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error during segment {segment_idx}: {result.stderr}")

        output_paths.append(output_filename)  # record output path here

        segment_idx += 1
        current_start += interval_seconds

    end = time.time()
    logger.info(f"Completed splitting {input_path} into {segment_idx} segments in {end - start:.2f} seconds")

    return output_paths


async def trim_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float
) -> None:
    """
    Trim a video between start_time and end_time (in seconds).
    """
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")


if __name__ == "__main__":
    from pathlib import Path
    import asyncio
    video_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test/爱在日落黄昏时.mkv"
    output_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test"
    info = get_video_info(video_path)
    # print(info)
    asyncio.run(convert_video(video_path, Path(output_path), interval_seconds=1200, fps=1, crf=28))
    asyncio.run(trim_video(video_path, Path(output_path) / "trim.mp4", start_time=5.0, end_time=15.0))