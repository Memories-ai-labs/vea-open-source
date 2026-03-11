# src/schema.py

from pydantic import BaseModel
from typing import List, Optional


# --- V2 Schemas ---

class V2IndexRequest(BaseModel):
    project_name: str
    source_dir: Optional[str] = None  # if omitted, auto-detects from workspace footage/ dir
    start_fresh: bool = False

class V2IndexResponse(BaseModel):
    project_name: str
    video_nos: List[str]
    gist: str
    status: str

class V2PlanRequest(BaseModel):
    project_name: str
    prompt: str
    target_duration_seconds: float = 120.0
    max_iterations: int = 5

class V2GenerateFcpxmlRequest(BaseModel):
    project_name: str

class V2GenerateFcpxmlResponse(BaseModel):
    project_name: str
    fcpxml_path: str

class V2NarrationRequest(BaseModel):
    project_name: str
    override_script: Optional[str] = None

class V2MusicRequest(BaseModel):
    project_name: str
    mood: Optional[str] = None
    prompt: Optional[str] = None

class V2CropRequest(BaseModel):
    project_name: str
    aspect_ratio: float = 0.5625

class V2RenderRequest(BaseModel):
    project_name: str
    quality: str = "preview"

class V2RenderResponse(BaseModel):
    project_name: str
    output_path: str

class V2ResolveStatusResponse(BaseModel):
    running: bool
    version: Optional[str] = None
    studio: bool
    error: Optional[str] = None
