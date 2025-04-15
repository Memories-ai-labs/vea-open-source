from moviepy.editor import VideoFileClip
import os
from typing import List, Union
import logging
import av
from pathlib import Path
import ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import yt_dlp
import time
from .file_utils import get_output_path, ensure_extension
import subprocess

logger = logging.getLogger(__name__)

def get_video_duration_pyav(video_path):
    container = av.open(video_path)
    duration = container.duration / av.time_base
    return duration

def split_video(file_path, interval_time, output_path=None):
    """
    Split a video file into segments of specified interval time using direct ffmpeg calls.
    
    Args:
        file_path: Path to the input video file
        interval_time: Time interval in seconds for each segment
        output_path: Optional path for output directory
        
    Returns:
        List of paths to the generated video segments
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
        
    if interval_time <= 0:
        raise ValueError("Interval time must be positive")
        
    # Ensure input file has correct extension
    file_path = ensure_extension(file_path)
    
    # If output_path is not provided, use the same directory as input
    if output_path is None:
        output_path = os.path.dirname(file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Get video duration using ffprobe
        probe_cmd = [
            "ffprobe", 
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Calculate number of segments
        num_segments = int(duration / interval_time) + (1 if duration % interval_time > 0 else 0)
        
        logger.info(f"Splitting video into {num_segments} segments of {interval_time} seconds each")
        
        segment_paths = []
        for i in range(num_segments):
            start_time = i * interval_time
            end_time = min((i + 1) * interval_time, duration)
            
            # Generate output path for this segment
            segment_path = get_output_path(
                os.path.join(output_path, f"{int(start_time)}_{int(end_time)}"),
                default_ext=os.path.splitext(file_path)[1]
            )
            logger.info(f"Processing segment {i+1}/{num_segments}: {start_time}s to {end_time}s")
            
            try:
                # Use ffmpeg to cut the segment
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-i", file_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-r", "1",  # Set fps to 1
                    "-avoid_negative_ts", "1",
                    segment_path
                ]
                
                # Run ffmpeg with no output to stdout/stderr
                subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                segment_paths.append(segment_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error processing segment {i+1}: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {str(e)}")
                raise
            
        return segment_paths
        
    except Exception as e:
        logger.error(f"Error splitting video: {str(e)}")
        # Clean up any partial segments
        for path in segment_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise

def get_video_duration(file_path: str) -> float:
    """
    Get the duration of a video file in seconds.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Duration in seconds
    """
    try:
        video = VideoFileClip(file_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        raise

def trim_video_clip(video_path, start_time, end_time, output_path, maintain_quality=True):
    """
    Extract a clip from a video file and save it to the specified output path.
    
    Args:
        video_path: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Optional path for output file
        maintain_quality: Whether to maintain original video quality
        
    Returns:
        Path to the trimmed video clip
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Ensure input file has correct extension
    video_path = ensure_extension(video_path)
    
    try:
        video = VideoFileClip(video_path)
        clip = video.subclip(start_time, end_time)
        
        # Set output settings based on quality preference
        output_settings = {
            'codec': 'libx264',
            'audio_codec': 'aac',
            'temp_audiofile': os.path.join(os.path.dirname(output_path), 'temp-audio.m4a'),
            'remove_temp': True,
            'verbose': False
        }
        
        clip.write_videofile(output_path, **output_settings)
        clip.close()
        video.close()
        return output_path
        
    except Exception as e:
        logger.error(f"Error trimming video: {str(e)}")
        raise

def get_video_info(file_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary containing video information
    """
    try:
        video = VideoFileClip(file_path)
        info = {
            'duration': video.duration,
            'size': video.size,
            'fps': video.fps,
            'filename': os.path.basename(file_path),
            'resolution': f"{video.size[0]}x{video.size[1]}"
        }
        video.close()
        return info
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise
