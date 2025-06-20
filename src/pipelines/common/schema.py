from enum import Enum
from google import genai
from pydantic import BaseModel
from typing import Optional

class CroppingResponse(BaseModel):
    frame_id: str
    crop_center_x: float
    crop_center_y: float

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
