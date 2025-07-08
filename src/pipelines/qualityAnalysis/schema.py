from pydantic import BaseModel, Field
from typing import Dict


class BrandSafetyScores(BaseModel):
    drug_use: int = Field(..., ge=1, le=5)
    violence: int = Field(..., ge=1, le=5)
    language: int = Field(..., ge=1, le=5)
    sexual_content: int = Field(..., ge=1, le=5)
    overall: int = Field(..., ge=1, le=5)


class QualityAssessmentResult(BaseModel):
    plot_fidelity: int = Field(..., ge=1, le=5)
    user_prompt_alignment: int = Field(..., ge=1, le=5)
    video_quality: int = Field(..., ge=1, le=5)
    brand_safety: BrandSafetyScores
