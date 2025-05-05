# src/schema.py

from pydantic import BaseModel


class EditRequest(BaseModel):
    """
    Request schema for video editing.
    """
    blob_path: str


class EditResponse(BaseModel):
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

class SummaryRequest(BaseModel):
    """
    Request schema for video editing.
    """
    blob_path: str


class SummaryResponse(BaseModel):
    """
    Response schema after video editing.
    """
    message: str
    summary: str