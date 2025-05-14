from enum import Enum
from google import genai
from pydantic import BaseModel


class ResponseForm(Enum):
    TEXT_ONLY = "text_only"
    TEXT_AND_EVIDENCE = "text_and_evidence"

class EvidenceClip(BaseModel):
    start: str               # e.g., "00:12:03"
    end: str                 # e.g., "00:13:45"
    description: str         # visual scene description
    reason: str              # why this clip supports the response
