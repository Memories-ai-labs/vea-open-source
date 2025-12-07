# src/schema.py

from pydantic import BaseModel
from typing import List, Optional
from src.pipelines.qualityAnalysis.schema import QualityAssessmentResult

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