from pydantic import BaseModel
from typing import List

class SegmentTimestamps(BaseModel):
    start: str  # HH:MM:SS
    end: str    # HH:MM:SS

class SectionScreenplay(BaseModel):
    segment: SegmentTimestamps
    screenplay: str

class FinalScreenplay(BaseModel):
    screenplay: str
