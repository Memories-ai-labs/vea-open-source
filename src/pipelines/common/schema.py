from enum import Enum
from google import genai
from pydantic import BaseModel, RootModel
from typing import Optional, List

class CroppingResponse(BaseModel):
    frame_id: str
    crop_center_x: float
    crop_center_y: float

# from pydantic import BaseModel
from typing import Literal

class CropModeResponse(BaseModel):
    clip_id: int
    mode: Literal["general", "character_tracking"]

class GeneralCropCenterResponse(BaseModel):
    crop_center_x: float  # between 0 and 1
    crop_center_y: float  # between 0 and 1


# class CroppingResponse(BaseModel):
#     crop_center_x: float
#     crop_center_y: float

class RefinedClipTimestamps(BaseModel):
    id: int
    refined_start: str  # HH:MM:SS
    refined_end: str    # HH:MM:SS

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
