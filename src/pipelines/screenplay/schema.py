from pydantic import BaseModel, RootModel
from typing import List

from src.pipelines.common.schema import Timestamp


class SegmentTimestamp(BaseModel):
    start: Timestamp
    end: Timestamp


class SegmentTimestamps(RootModel[List[SegmentTimestamp]]):
    pass


class SectionScreenplay(BaseModel):
    segment: SegmentTimestamp
    screenplay: str


class FinalScreenplay(BaseModel):
    screenplay: str
