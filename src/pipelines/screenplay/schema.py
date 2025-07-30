from pydantic import BaseModel, RootModel
from typing import List

class SegmentTimestamp(BaseModel):
    start: str  # HH:MM:SS
    end: str    # HH:MM:SS

class SegmentTimestamps(RootModel[List[SegmentTimestamp]]):
    pass

class SectionScreenplay(BaseModel):
    segment: SegmentTimestamp
    screenplay: str

class FinalScreenplay(BaseModel):
    screenplay: str
