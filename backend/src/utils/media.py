from moviepy.editor import VideoFileClip
import os
from typing import List, Union
import logging
import av
from pathlib import Path
import ffmpeg
import yt_dlp
import time

logger = logging.getLogger(__name__)

def get_video_duration_pyav(video_path):
    container = av.open(video_path)
    duration = container.duration / av.time_base
    return duration

def split_video(
    file_path: str,
    interval_time: int,
    output_path: str
) -> List[str]:
    """
    Split a video file into segments based on interval time.
    
    Args:
        file_path: Path to the input video file
        interval_time: Time interval in seconds for each segment
        output_path: Directory path for output files
    
    Returns:
        List of paths to the generated video segments
    
    Raises:
        ValueError: If file_path doesn't exist or interval_time is invalid
        Exception: For video processing errors
    """
    try:
        # Input validation
        if not os.path.exists(file_path):
            raise ValueError(f"Video file not found: {file_path}")
        
        if interval_time <= 0:
            raise ValueError("Interval time must be positive")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video
        logger.info(f"Loading video: {file_path}")
        video = VideoFileClip(file_path)
        
        # Calculate segments
        duration = video.duration
        num_segments = int(duration // interval_time)
        if duration % interval_time > 0:
            num_segments += 1
            
        output_files = []
        
        # Process each segment
        for i in range(num_segments):
            start_time = i * interval_time
            end_time = min((i + 1) * interval_time, duration)
            
            # Create segment filename
            output_filename = f"{int(start_time)}_{int(end_time)}.mp4"
            output_file = output_dir / output_filename
            
            logger.info(f"Processing segment {i+1}/{num_segments}: {start_time}s to {end_time}s")
            
            try:
                # Extract and save segment
                segment = video.subclip(start_time, end_time)
                segment.write_videofile(
                    str(output_file),
                    codec='libx264',
                    audio_codec='aac',
                    remove_temp=True,
                    verbose=False
                )
                segment.close()
                
                output_files.append(str(output_file))
                logger.info(f"Successfully saved segment: {output_filename}")
                
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {str(e)}")
                # Try without audio if audio processing fails
                try:
                    logger.info(f"Retrying segment {i+1} without audio...")
                    segment = video.subclip(start_time, end_time)
                    segment.without_audio().write_videofile(
                        str(output_file),
                        codec='libx264',
                        verbose=False
                    )
                    segment.close()
                    output_files.append(str(output_file))
                    logger.info(f"Successfully saved segment without audio: {output_filename}")
                except Exception as e2:
                    logger.error(f"Failed to process segment even without audio: {str(e2)}")
                    raise
                
        # Cleanup
        video.close()
        
        logger.info(f"Successfully split video into {len(output_files)} segments")
        return output_files
        
    except Exception as e:
        logger.error(f"Error splitting video: {str(e)}")
        raise
    finally:
        # Ensure video is closed even if an error occurs
        if 'video' in locals():
            video.close()

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

def trim_video_clip(
    video_path: str, 
    start: Union[int, float], 
    end: Union[int, float], 
    output_video_path: str,
    maintain_quality: bool = True
) -> str:
    """
    Extracts a clip from the video between `start` and `end` timestamps and saves it to `output_video_path`.
    
    Args:
        video_path: Path to the input video file
        start: Start time in seconds
        end: End time in seconds
        output_video_path: Path where the trimmed video will be saved
        maintain_quality: If True, maintains original video quality (default: True)
    
    Returns:
        Path to the trimmed video file
        
    Raises:
        FileNotFoundError: If input video doesn't exist
        Exception: For other processing errors
    """
    try:
        # Input validation
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        assert start > 0, ("Start time cannot be negative")
            
        assert start < end, ("Start time must be less than end time")
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Load the video
        logger.info(f"Loading video: {video_path}")
        video = VideoFileClip(video_path)
        
        # Validate timestamps against video duration
        if end > video.duration:
            logger.warning(f"End time {end} exceeds video duration {video.duration}. Using video duration instead.")
            end = video.duration
            
        if start > video.duration:
            raise ValueError(f"Start time {start} exceeds video duration {video.duration}")
            
        # Extract the clip
        logger.info(f"Extracting clip from {start} to {end} seconds")
        clip = video.subclip(start, end)
        
        # Prepare output settings
        output_settings = {
            'temp_audiofile': os.path.join(os.path.dirname(output_video_path), 'temp-audio.m4a'),
            'remove_temp': True,
            'verbose': False
        }
        
        if maintain_quality:
            output_settings.update({
                'codec': 'libx264',
                'audio_codec': 'aac',
                'preset': 'medium',
                'bitrate': ffmpeg.probe(video_path)['streams'][0]['bit_rate']
            })
        
        # Write the clip
        logger.info(f"Writing trimmed video to: {output_video_path}")
        clip.write_videofile(output_video_path, **output_settings)
        
        # Clean up
        clip.close()
        video.close()
        
        logger.info("Video trimming completed successfully")
        return output_video_path
        
    except Exception as e:
        logger.error(f"Error trimming video: {str(e)}")
        # Clean up any partial output
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except:
                pass
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
