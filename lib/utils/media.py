import logging
import os
from pathlib import Path
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import timedelta
import tempfile
import shutil

logging.basicConfig(level=logging.WARNING)  # Only show warnings, errors, and critical logs
logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-loglevel", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr.decode('utf-8')}")
    return json.loads(result.stdout.decode('utf-8'))

def get_video_duration(video_path: str) -> float:
    return float(get_video_info(video_path)["format"]["duration"])

def downsample_video(input_path: str, output_path: str, crf: int = 30, target_height: int = 480, fps: float = 0.5):
    """
    Create a downsampled version of the video with lower FPS and resolution, preserving audio.
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-vf", f"fps={fps},scale=-2:{target_height}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "ultrafast",
        "-c:a", "copy",
        output_path
    ]
    logger.info(f"[Downsampling] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def build_chunk_command(input_path, output_path, start, duration):
    """
    Build FFmpeg command to slice preprocessed video into a chunk without re-encoding.
    """
    return [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        "-threads", "1",
        output_path
    ]

async def preprocess_long_video(
    input_path: str,
    output_dir: str,
    interval_seconds: int,
    crf: Optional[int] = 30,
    target_height: Optional[int] = 480,
    fps: Optional[float] = 0.5,
) -> list[dict]:
    """
    Splits a downsampled video into segments and returns a list of dicts
    with clip path, start/end timestamps, parent file, and segment number.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    input_stem = input_path.stem

    downsampled_path = output_dir / f"{input_stem}_downsampled.mp4"
    if not downsampled_path.exists():
        logger.info(f"[INFO] Downsampling video: {input_path}")
        downsample_video(str(input_path), str(downsampled_path), crf=crf, target_height=target_height, fps=fps)
    else:
        logger.info(f"[INFO] Skipping downsampling (already exists)")

    total_duration = get_video_duration(str(downsampled_path))
    logger.info(f"[INFO] Total duration of downsampled video: {total_duration:.2f} seconds")

    tasks = []
    segments = []
    current_start = 0
    segment_number = 1

    while current_start < total_duration:
        start = int(current_start)
        end = int(min(current_start + interval_seconds, total_duration))
        duration = end - start

        output_path = output_dir / f"{input_stem}_{start:05d}_{end:05d}.mp4"
        cmd = build_chunk_command(str(downsampled_path), str(output_path), start, duration)
        tasks.append((cmd, output_path, start, end, segment_number))
        current_start += interval_seconds
        segment_number += 1

    max_workers = min(os.cpu_count(), 8)
    logger.info(f"[INFO] Using {max_workers} worker threads")

    def run_cmd(cmd):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cmd, cmd) for cmd, *_ in tasks]
        for i, f in enumerate(futures):
            f.result()
            _, path, start, end, seg_num = tasks[i]
            segments.append({
                "path": path,
                "start": start,
                "end": end,
                "segment_number": seg_num,
                "parent_path": str(input_path)
            })

    logger.info(f"[INFO] Completed {len(segments)} segments in {time.time() - start_time:.2f} sec")
    return segments

def correct_segment_number_based_on_time(reference_segments, target_segments):
    """
    For each segment in target_segments, find the reference_segment in reference_segments
    such that its 'start' <= target_segment['start'] < 'end', and set the segment_number accordingly.
    Modifies target_segments in-place.
    """
    for target in target_segments:
        target_start = target["start"]
        matched = False
        for ref in reference_segments:
            if ref["start"] <= target_start < ref["end"]:
                target["segment_number"] = ref["segment_number"]
                matched = True
                break
        if not matched:
            print(f"[WARN] No reference segment found for target segment starting at {target_start}")
    return target_segments  # not required, but useful for chaining


async def preprocess_short_video(
    input_path: str,
    output_dir: str,
    crf: int = 30,
    target_height: int = 480,
    fps: float = 0.5
) -> dict:
    """
    Downsample a video to a lower resolution and frame rate without splitting.
    Returns a dict with clip path, start/end timestamps, segment number, and parent file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    input_stem = input_path.stem
    downsampled_path = output_dir / f"{input_stem}_downsampled.mp4"

    if downsampled_path.exists():
        logger.info(f"[INFO] Skipping downsampling (already exists): {downsampled_path}")
    else:
        logger.info(f"[INFO] Downsampling video: {input_path}")
        downsample_video(str(input_path), str(downsampled_path), crf=crf, target_height=target_height, fps=fps)

    duration = get_video_duration(str(downsampled_path))
    return {
        "path": downsampled_path,
        "start": 0,
        "end": int(duration),
        "segment_number": 1,
        "parent_path": str(input_path)
    }

def parse_time_to_seconds(t: str) -> int:
    """
    Converts a timestamp string (HH:MM:SS or MM:SS) to total seconds.
    """
    parts = list(map(int, t.split(":")))
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        raise ValueError(f"Invalid timestamp format: {t}")
    return h * 3600 + m * 60 + s

def seconds_to_hhmmss(seconds: int) -> str:
    """
    Converts seconds to HH:MM:SS format.
    """
    return str(timedelta(seconds=int(seconds)))

def clean_stale_tempdirs():
        print("Cleaning stale temp directories...")
        tmp_root = tempfile.gettempdir()  # Usually /tmp
        for name in os.listdir(tmp_root):
            path = os.path.join(tmp_root, name)
            if os.path.isdir(path) and name.startswith("tmp"):
                try:
                    shutil.rmtree(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Skipping {path}: {e}")