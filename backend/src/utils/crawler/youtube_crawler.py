from moviepy.editor import VideoFileClip
import os
import logging
import yt_dlp
import time


logger = logging.getLogger(__name__)


def nearest_resolution(width, height):
    # Standard YouTube resolutions
    resolutions = {
        '144p': (256, 144),
        '240p': (426, 240),
        '360p': (640, 360),
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '2160p': (3840, 2160),
        '4320p': (7680, 4320)
    }
    # Calculate the aspect ratio of the input resolution
    input_aspect_ratio = width / height
    # Find the nearest resolution
    nearest_res = None
    min_diff = float('inf')

    for label, (res_width, res_height) in resolutions.items():
        # Calculate the difference in aspect ratio
        res_aspect_ratio = res_width / res_height
        aspect_diff = abs(input_aspect_ratio - res_aspect_ratio)
        # Calculate the difference in pixel count (total pixels)
        pixel_diff = abs(width * height - res_width * res_height)
        # Choose the closest based on the aspect ratio difference
        total_diff = aspect_diff + pixel_diff / 1000  # weight pixel difference
        if total_diff < min_diff:
            min_diff = total_diff
            nearest_res = label
    return nearest_res


async def download_youtube_video(url: str, output_path: str) -> str:
    """
    Download a video from YouTube.
    
    Args:
        url: YouTube video URL
        output_path: Path where to save the video
        
    Returns:
        Path to the downloaded video file
        
    Raises:
        Exception: If download fails
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # First, get available formats
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Fetching video information from: {url}")
            info = ydl.extract_info(url, download=False)
            
            # Get available formats
            formats = info.get('formats', [])
            if not formats:
                raise Exception("No formats available for this video")
                
            # Log available formats
            logger.info(f"Found {len(formats)} available formats")
            
            # Define preferred resolutions in order
            preferred_resolutions = ["720p", "480p", "1080p"]
            
            # Select best format based on resolution
            selected_format = None
            
            # Create a dictionary of MP4 formats with calculated resolution
            mp4_formats = {}
            for f in formats:
                if f.get('ext') == 'mp4' and f.get('vcodec') != 'none':
                    width = f.get('width')
                    height = f.get('height')
                    if width and height:
                        # Calculate resolution (use height as the standard resolution indicator)
                        resolution = nearest_resolution(width, height)
                        mp4_formats[resolution] = f
                        logger.info(f"Found MP4 format: {width}x{height} ({resolution}p)")
            
            if not mp4_formats:
                raise Exception("No MP4 formats available")
            
            # Log available resolutions
            available_resolutions = sorted(mp4_formats.keys(), reverse=True)
            logger.info(f"Available resolutions: {available_resolutions}")
            
            # Try to find the best available resolution from preferred list
            for preferred_resolution in preferred_resolutions:
                if preferred_resolution in mp4_formats:
                    selected_format = mp4_formats[preferred_resolution]
                    logger.info(f"Selected format with resolution {preferred_resolution}p")
                    break
            
            if not selected_format:
                # If no preferred resolution found, use the highest available resolution
                highest_resolution = max(mp4_formats.keys())
                selected_format = mp4_formats[highest_resolution]
                logger.info(f"No preferred resolution found, using highest available: {highest_resolution}p")
            
            format_id = selected_format.get('format_id', 'best')
            width = selected_format.get('width', 'unknown')
            height = selected_format.get('height', 'unknown')
            logger.info(f"Selected format: {format_id} (Resolution: {width}x{height})")
        
        # Now download with selected format and optimized settings
        last_progress_time = 0
        last_progress_percent = 0
        
        def progress_hook(d):
            nonlocal last_progress_time, last_progress_percent
            current_time = time.time()
            
            # Extract percentage from the status dict, handling ANSI color codes
            percent_str = d.get('_percent_str', '0')
            # Remove ANSI color codes and percentage sign
            percent_str = ''.join(c for c in percent_str if c.isdigit() or c == '.')
            try:
                current_percent = float(percent_str)
            except ValueError:
                current_percent = 0
            
            # Only print if:
            # 1. It's been at least 5 seconds since last print, or
            # 2. Progress has increased by at least 5%
            if (current_time - last_progress_time >= 10 or 
                current_percent - last_progress_percent >= 10):
                logger.info(f"Download progress: {d['status']} - {percent_str}%")
                last_progress_time = current_time
                last_progress_percent = current_percent
        
        ydl_opts = {
            'format': format_id,
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'merge_output_format': 'mp4',
            'verbose': True,
            # Optimize download speed
            'concurrent_fragments': 5,  # Download multiple fragments at once
            'buffersize': 32768,  # Increase buffer size
            'http_chunk_size': 10485760,  # 10MB chunks
            'retries': 10,  # More retries for better reliability
            'fragment_retries': 10,
            'file_access_retries': 10,
            'extractor_retries': 10,
            'progress_hooks': [progress_hook],
        }
        
        logger.info(f"Starting download with format {format_id}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except yt_dlp.utils.DownloadError as e:
                logger.error(f"Download error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                raise
            
        if not os.path.exists(output_path):
            raise Exception(f"Video download failed - file not created at {output_path}")
            
        # Verify the file is valid
        try:
            video = VideoFileClip(output_path)
            duration = video.duration
            video.close()
            logger.info(f"Successfully downloaded video (duration: {duration:.2f}s) to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Downloaded file appears to be invalid: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise Exception("Downloaded video file is invalid")
        
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}")
        # Clean up any partial download
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise