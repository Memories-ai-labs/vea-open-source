from enum import Enum
from pydantic import BaseModel, RootModel
from typing import List

from src.pipelines.common.schema import Timestamp


class ResponseForm(Enum):
    TEXT_ONLY = "text_only"
    TEXT_AND_EVIDENCE = "text_and_evidence"


class EvidenceClip(BaseModel):
    file_name: str
    start: Timestamp
    end: Timestamp
    description: str         # visual scene description
    reason: str              # why this clip supports the response


class EvidenceClips(RootModel[List[EvidenceClip]]):
    pass


class ClipPriority(str, Enum):
    NARRATION = "narration"      # prioritize narration voiceover
    CLIP_AUDIO = "clip_audio"    # prioritize original clip audio (e.g., interviews)
    CLIP_VIDEO = "clip_video"    # use the whole video uncut (e.g., sports play)


class ChosenClip(BaseModel):
    id: int
    file_name: str
    start: Timestamp
    end: Timestamp
    narration: str
    priority: ClipPriority = ClipPriority.NARRATION


class ChosenClips(RootModel[List[ChosenClip]]):
    pass