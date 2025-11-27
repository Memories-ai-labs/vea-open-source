from pydantic import BaseModel, RootModel
from typing import List, Optional

from src.pipelines.common.schema import Timestamp


class SupportingClip(BaseModel):
    start: Timestamp
    end: Timestamp


class ShortsPlan(BaseModel):
    short_index: int
    description: str
    start: Timestamp
    end: Timestamp
    supporting_clips: Optional[List[SupportingClip]] = None


class ShortsPlans(RootModel[List[ShortsPlan]]):
    pass