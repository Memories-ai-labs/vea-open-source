from pydantic import BaseModel, RootModel, Field
from typing import List, Literal


class Timestamp(BaseModel):
    """Structured timestamp to avoid LLM format confusion.

    Instead of asking LLM for "HH:MM:SS,mmm" strings (which causes confusion),
    we ask for explicit numeric fields.
    """
    hours: int = Field(ge=0, le=23, description="Hours (0-23)")
    minutes: int = Field(ge=0, le=59, description="Minutes (0-59)")
    seconds: int = Field(ge=0, le=59, description="Seconds (0-59)")
    milliseconds: int = Field(default=0, ge=0, le=999, description="Milliseconds (0-999)")

    def to_string(self) -> str:
        """Convert to HH:MM:SS,mmm format string."""
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},{self.milliseconds:03d}"

    def to_seconds(self) -> float:
        """Convert to total seconds."""
        return self.hours * 3600 + self.minutes * 60 + self.seconds + self.milliseconds / 1000

    @classmethod
    def from_seconds(cls, total_seconds: float) -> "Timestamp":
        """Create Timestamp from total seconds."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return cls(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


def convert_timestamp_to_string(ts) -> str:
    """Convert structured timestamp (dict, Timestamp object, or string) to HH:MM:SS,mmm string.

    This is the canonical helper for converting LLM-returned structured timestamps
    back to string format for downstream processing.
    """
    if isinstance(ts, str):
        return ts
    if isinstance(ts, dict):
        hours = ts.get("hours", 0)
        minutes = ts.get("minutes", 0)
        seconds = ts.get("seconds", 0)
        milliseconds = ts.get("milliseconds", 0)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    if hasattr(ts, "to_string"):
        return ts.to_string()
    return str(ts)


class CroppingResponse(BaseModel):
    frame_id: str
    crop_center_x: float
    crop_center_y: float

class CropModeResponse(BaseModel):
    clip_id: int
    mode: Literal["general", "character_tracking"]

class GeneralCropCenterResponse(BaseModel):
    crop_center_x: float  # between 0 and 1
    crop_center_y: float  # between 0 and 1


class RefinedClipTimestamps(BaseModel):
    id: int
    refined_start: Timestamp
    refined_end: Timestamp

class ChosenMusic(BaseModel):
    id: str 
    title: str 

class ChosenMusicResponse(RootModel[List[ChosenMusic]]):
    """LLM response containing a ranked list of music choices."""

    def __iter__(self):  # allow direct iteration over items
        return iter(self.root)

    def __len__(self):  # len(response)
        return len(self.root)

    def __getitem__(self, item):  # indexing support
        return self.root[item]
