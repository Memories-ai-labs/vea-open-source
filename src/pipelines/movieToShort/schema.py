from pydantic import BaseModel
from typing import List, Optional


# class ShortsPlan(BaseModel):
#     short_index: int
#     description: str
#     start: str
#     end: str

class ShortsPlan(BaseModel):
    short_index: int
    description: str
    start: str
    end: str
    supporting_clips: Optional[List[str]] = None  # List of start/end timestamps in HH:MM:SS format