# src/schema.py

from pydantic import BaseModel
from typing import List, Optional



class MovieRecapRequest(BaseModel):
    blob_path: str
    user_prompt: Optional[str] = None
    user_context: Optional[str] = None
    output_language: Optional[str] = "English"


class MovieRecapResponse(BaseModel):
    """
    Response schema after video editing.
    """
    message: str
    url: str


class MovieFile(BaseModel):
    """
    Schema representing a movie file in GCS.
    """
    name: str
    blob_path: str

class MovieIndexRequest(BaseModel):
    """
    Request schema for video editing.
    """
    blob_path: str


class MovieIndexResponse(BaseModel):
    """
    Response schema after video editing.
    """
    message: str

class FlexibleResponseRequest(BaseModel):
    """
    Request schema for flexible response from media.
    """
    blob_path: str
    prompt: str
    video_response: bool


class FlexibleResponseResult(BaseModel):
    """
    Output from flexible response pipeline.
    """
    response: str
    response_type: str  # one of: 'text', 'text_with_clips', 'video'
    evidence_paths: List[str]
