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
         "-c:a", "aac",
        "-b:a", "128k",  #compress audio
        # "-strict -2",
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

def split_video_into_segments(video_path, output_dir, max_seconds=1200):
    """
    Splits a video into segments of max_seconds (default 20min) using ffmpeg.
    Returns list of segment video paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "seg_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c", "copy", "-map", "0",
        "-f", "segment",
        "-segment_time", str(max_seconds),
        output_pattern
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return sorted(Path(output_dir).glob("seg_*.mp4"))

def extract_audio_from_video(
    video_path: str,
    audio_output_path: str,
    audio_format: str = "mp3",
    bitrate: str = "64k",
    sample_rate: int = 16000,
    mono: bool = True,
):
    """
    Extracts audio from a video file, compressing to save space.
    """
    # Output extension check
    if not audio_output_path.lower().endswith(f".{audio_format}"):
        raise ValueError(f"audio_output_path must end with .{audio_format}")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame" if audio_format == "mp3" else audio_format,
        "-b:a", bitrate,
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]  # force mono

    cmd.append(str(audio_output_path))

    print(f"[INFO] Extracting compressed audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}")
    return audio_output_path


def extract_images_ffmpeg(
    video_path: str, 
    output_dir: str, 
    fps: float = 0.25, 
    target_height: int = 480, 
    jpeg_quality: int = 10
):
    """
    Extracts images from a video at the specified FPS using ffmpeg.
    Saves frames at smaller resolution (preserves aspect) and lower JPEG quality.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%05d.jpg")
    # -2 makes ffmpeg preserve aspect ratio
    vf_str = f"fps={fps},scale=-2:{target_height}"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf_str,
        "-q:v", str(jpeg_quality),
        output_pattern
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[INFO] Extraction complete.")


def download_and_cache_video(gcs_client, bucket_name, cloud_path, local_dir):
    """
    Downloads a video file from GCS if not already cached locally.
    Returns the local path.
    """
    filename = os.path.basename(cloud_path)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"[CACHE] Video already downloaded: {local_path}")
        return local_path

    print(f"[DOWNLOAD] Downloading video from GCS: {cloud_path} → {local_path}")
    gcs_client.download_files(bucket_name, cloud_path, local_path)
    return local_path

def extract_video_segment(full_video_path, output_dir, start_hhmmss, end_hhmmss, output_name="segment.mp4"):
    """
    Extracts a segment from a video using ffmpeg and saves it in output_dir.
    Returns the path to the segment.
    """
    start_sec = parse_time_to_seconds(start_hhmmss)
    end_sec = parse_time_to_seconds(end_hhmmss)
    duration = max(0.1, end_sec - start_sec)

    output_path = os.path.join(output_dir, output_name)
    cmd = [
        "ffmpeg", "-ss", str(start_sec), "-i", full_video_path, "-t", str(duration),
        "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path