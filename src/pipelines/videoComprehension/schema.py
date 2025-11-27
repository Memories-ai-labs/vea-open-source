from typing import List
from pydantic import BaseModel, RootModel, Field


class RecapSentence(BaseModel):
    sentence_text: str
    segment_num: int


class RecapSentences(RootModel[List[RecapSentence]]):
    pass


class SceneTimestamp(BaseModel):
    """Structured timestamp for scenes (MM:SS format, relative to segment start)."""
    minutes: int = Field(ge=0, le=59, description="Minutes (0-59)")
    seconds: int = Field(ge=0, le=59, description="Seconds (0-59)")

    def to_string(self) -> str:
        """Convert to MM:SS format string."""
        return f"{self.minutes:02d}:{self.seconds:02d}"


class Scene(BaseModel):
    start_timestamp: SceneTimestamp
    end_timestamp: SceneTimestamp
    scene_description: str


class Scenes(RootModel[List[Scene]]):
    pass