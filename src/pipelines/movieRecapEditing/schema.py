from typing import List
from pydantic import BaseModel

class ChosenClip(BaseModel):
    id: str
    start_timestamp: str  # Start time of scene (HH:MM:SS)
    end_timestamp: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene
    corresponding_summary_sentence : str

class ChosenMusic(BaseModel):
    id: str 
    title: str 