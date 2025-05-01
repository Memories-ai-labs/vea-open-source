from typing import List
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    description: str

class ClipSummary(BaseModel):
    segment_summary: str
    characters: List[Character]
    key_events: List[str]
    notes: List[str]
