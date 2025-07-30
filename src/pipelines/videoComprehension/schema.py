from typing import List
from pydantic import BaseModel, RootModel

class RecapSentence(BaseModel):
    sentence_text: str 
    segment_num: int 

class RecapSentences(RootModel[List[RecapSentence]]):
    pass

class Scene(BaseModel):
    start_timestamp: str  # Start time of scene (HH:MM:SS)
    end_timestamp: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene

class Scenes(RootModel[List[Scene]]):
    pass