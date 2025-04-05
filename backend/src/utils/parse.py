def seconds_to_timestamp(total_seconds: float | str | int) -> str:
    """
    Convert seconds to timestamp format (HH:MM:SS).
    
    Args:
        seconds: Time in seconds (can be float)
        
    Returns:
        Formatted timestamp string (HH:MM:SS)
        
    Examples:
        >>> seconds_to_timestamp(3661.5)
        '01:01:01'
        >>> seconds_to_timestamp(70)
        '00:01:10'
    """
    total_seconds = int(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert timestamp (HH:MM:SS) to seconds.
    
    Args:
        timestamp: Time in HH:MM:SS format
        
    Returns:
        Time in seconds as float
        
    Raises:
        ValueError: If timestamp format is invalid
        
    Examples:
        >>> timestamp_to_seconds('01:01:01')
        3661.0
        >>> timestamp_to_seconds('00:01:10')
        70.0
    """
    try:
        # Split timestamp into hours, minutes, seconds
        parts = timestamp.split(':')
        
        if len(parts) != 3:
            raise ValueError("Timestamp must be in HH:MM:SS format")
            
        hours, minutes, seconds = map(int, parts)
        
        # Convert to seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        return int(total_seconds)
        
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format. {str(e)}")