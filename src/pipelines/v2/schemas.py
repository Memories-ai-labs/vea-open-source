"""V2 pipeline schemas — internal data models for the agentic editing pipeline."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Session state (stored in session.json as plain JSON via dataclasses.asdict)
# ---------------------------------------------------------------------------

@dataclass
class VideoEntry:
    video_no: str
    video_name: str
    source_path: str
    duration_seconds: Optional[float] = None

@dataclass
class PlanningState:
    iteration_count: int = 0
    user_prompts: List[str] = field(default_factory=list)
    target_duration_seconds: float = 120.0

@dataclass
class SessionData:
    project_name: str
    created_at: str
    updated_at: str
    status: str  # "indexed" | "planning" | "fcpxml_ready" | "rendered"
    videos: List[VideoEntry] = field(default_factory=list)
    gist: str = ""
    memories_session_id: Optional[str] = None
    planning: PlanningState = field(default_factory=PlanningState)
    version: str = "2.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionData":
        videos = [VideoEntry(**v) for v in d.get("videos", [])]
        planning_raw = d.get("planning", {})
        planning = PlanningState(
            iteration_count=planning_raw.get("iteration_count", 0),
            user_prompts=planning_raw.get("user_prompts", []),
            target_duration_seconds=planning_raw.get("target_duration_seconds", 120.0),
        )
        return cls(
            project_name=d["project_name"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            status=d["status"],
            videos=videos,
            gist=d.get("gist", ""),
            memories_session_id=d.get("memories_session_id"),
            planning=planning,
            version=d.get("version", "2.0"),
        )


# ---------------------------------------------------------------------------
# Planning loop schemas (Gemini structured output targets — use Pydantic)
# ---------------------------------------------------------------------------

class ChatTool(BaseModel):
    question: str
    purpose: str

class SearchTool(BaseModel):
    query: str
    purpose: str
    target_duration_sec: float = 5.0

class ToolCallPlan(BaseModel):
    reasoning: str
    chat_calls: List[ChatTool] = []
    search_calls: List[SearchTool] = []
    should_stop: bool = False

class RetrievedClip(BaseModel):
    video_no: str
    video_name: str
    source_path: str
    start_seconds: float
    end_seconds: float
    score: float
    description: str = ""
    shot_query: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

class Shot(BaseModel):
    id: str
    purpose: str
    search_query: str
    retrieved_clip: Optional[RetrievedClip] = None
    narration: Optional[str] = None
    priority: Literal["narration", "clip_audio", "clip_video"] = "narration"
    duration_seconds: float = 5.0

class Storyboard(BaseModel):
    iteration: int = 0
    target_duration_seconds: float = 120.0
    theme: str = ""
    narrative_arc: str = ""
    shots: List[Shot] = []
    open_questions: List[str] = []
    notes: str = ""
