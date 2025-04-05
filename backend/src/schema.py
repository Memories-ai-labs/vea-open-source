from typing import TypedDict, Optional, List
from pydantic import BaseModel

class DialogueLine(BaseModel):
    Name: str  # Character's name
    Speech: str  # What they said

class Scene(BaseModel):
    start_timestamp: str  # Start time of scene (HH:MM:SS)
    end_timestamp: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene

class ChosenClip(BaseModel):
    id: str
    start_timestamp: str  # Start time of scene (HH:MM:SS)
    end_timestamp: str  # End time of scene (HH:MM:SS)
    scene_description: str  # Brief description of the action in the scene
    corresponding_summary_sentence : str