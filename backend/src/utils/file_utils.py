import os
import logging

logger = logging.getLogger(__name__)

# Common video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']

def get_file_extension(file_path):
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The file extension including the dot (e.g., '.mp4', '.mkv')
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def ensure_extension(file_path, default_ext='.mp4'):
    """
    Ensure a file path has the specified extension.
    If the file already has an extension, it will be preserved.
    
    Args:
        file_path: Path to the file
        default_ext: Default extension to use if none is found
        
    Returns:
        File path with the appropriate extension
    """
    _, ext = os.path.splitext(file_path)
    if ext.lower() in VIDEO_EXTENSIONS:
        return file_path
    else:
        return file_path + default_ext

def get_output_path(input_path, suffix=None, default_ext='.mp4'):
    """
    Generate an output path based on the input path, preserving the extension.
    
    Args:
        input_path: Path to the input file
        suffix: Suffix to add before the extension (e.g., '_trimmed')
        default_ext: Default extension to use if none is found in the input path
        
    Returns:
        Output path with the appropriate extension
    """
    base, ext = os.path.splitext(input_path)
    
    # If the input has a valid video extension, use it
    if ext.lower() in VIDEO_EXTENSIONS:
        output_ext = ext
    else:
        output_ext = default_ext
    
    # Add suffix if provided
    if suffix:
        return f"{base}{suffix}{output_ext}"
    else:
        return f"{base}{output_ext}"

def is_video_file(file_path):
    """
    Check if a file is a video file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a video file, False otherwise
    """
    ext = get_file_extension(file_path)
    return ext.lower() in VIDEO_EXTENSIONS 