from typing import List
from pydantic import BaseModel

class RecapSentence(BaseModel):
    sentence_text: str 
    segment_num: int 

class Scene(BaseModel):
    start_timestamp: str  # Start time of scene (HH:MM:SS)
    end_timestamp: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene

class ArtisticSegment(BaseModel):
    start_timestamp: str
    end_timestamp: str
    visual_elements: str
    audio_elements: str
