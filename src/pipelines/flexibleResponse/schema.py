from enum import Enum
from google import genai
from pydantic import BaseModel
from typing import Optional


class ResponseForm(Enum):
    TEXT_ONLY = "text_only"
    TEXT_AND_EVIDENCE = "text_and_evidence"

class EvidenceClip(BaseModel):
    file_name: str
    start: str               # e.g., "00:12:03"
    end: str                 # e.g., "00:13:45"
    description: str         # visual scene description
    reason: str              # why this clip supports the response

class ClipPriority(str, Enum):
    NARRATION = "narration"      # prioritize narration voiceover
    CLIP_AUDIO = "clip_audio"    # prioritize original clip audio (e.g., interviews)
    CLIP_VIDEO = "clip_video"    # use the whole video uncut (e.g., sports play)

class ChosenClip(BaseModel):
    id: int
    file_name: str
    start: str
    end: str
    narration: str
    priority: ClipPriority = ClipPriority.NARRATION  # defau