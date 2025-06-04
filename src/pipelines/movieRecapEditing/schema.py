from typing import List
from pydantic import BaseModel

class ChosenClip(BaseModel):
    id: int
    start: str  # Start time of scene (HH:MM:SS)
    end: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene
    narration : str

class ChosenMusic(BaseModel):
    id: str 
    title: str 

class RecapSentence(BaseModel):
    sentence_text: str 
    segment_num: int 