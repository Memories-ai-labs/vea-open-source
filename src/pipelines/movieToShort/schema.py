from pydantic import BaseModel
from typing import List, Optional


class ShortsPlan(BaseModel):
    short_index: int
    description: str
    start: str
    end: str