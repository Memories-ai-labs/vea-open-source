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
