# src/schema.py

from pydantic import BaseModel
from typing import List, Optional
from src.pipelines.v1_legacy.qualityAnalysis.schema import QualityAssessmentResult

class MovieFile(BaseModel):
    """
    Schema representing a movie file in GCS.
    """
    name: str
    blob_path: str
class IndexRequest(BaseModel):
    blob_path: str
    start_fresh: Optional[bool] = False  # Re-process everything, including re-upload to Memories.ai


class IndexResponse(BaseModel):
    message: str

class FlexibleResponseRequest(BaseModel):
    """
    Request schema for flexible response from media.
    """
    blob_path: str
    prompt: str
    video_response: bool = False
    original_audio: bool = True
    music: bool = True
    narration: bool = True
    aspect_ratio: float = 0
    subtitles:bool = True
    snap_to_beat: bool = False
    output_path: str = None


class FlexibleResponseResult(BaseModel):
    """
    Output from flexible response pipeline.
    """
    response: str
    response_type: str  # one of: 'text', 'text_with_clips', 'video'
    evidence_paths: List[str]
    run_id: str

class ShortsRequest(BaseModel):
    blob_path: str

class ShortsResponse(BaseModel):
    shorts: List[dict]

class ScreenplayRequest(BaseModel):
    blob_path: str

class ScreenplayResponse(BaseModel):
    message: str
    output_path: str

class QualityAssessmentRequest(BaseModel):
    blob_path: str
    ground_truth: str
    user_prompt: str
    
class QualityAssessmentResponse(BaseModel):
    message: str
    result: QualityAssessmentResult


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