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
    gist: str = ""  # multi-paragraph editorial overview for this specific video

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
        videos = [VideoEntry(
            video_no=v["video_no"],
            video_name=v["video_name"],
            source_path=v["source_path"],
            duration_seconds=v.get("duration_seconds"),
            gist=v.get("gist", ""),
        ) for v in d.get("videos", [])]
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


# ---------------------------------------------------------------------------
# Edit Decision format — JSON representation of a complete video edit.
# The LLM produces this; deterministic code compiles it to FCPXML.
# The dashboard reads this same JSON to render the timeline view.
# ---------------------------------------------------------------------------

class TimelineSettings(BaseModel):
    """Timeline-level metadata."""
    name: str = "VEA Edit"
    fps: float = 24.0
    width: int = 1920
    height: int = 1080

class TransformSettings(BaseModel):
    """Crop/reframe transform for a clip (FCPXML adjust-transform)."""
    scale_x: float = 1.0
    scale_y: float = 1.0
    position_x: float = 0.0
    position_y: float = 0.0
    rotation: float = 0.0

class SpeedChange(BaseModel):
    """Constant speed change. rate=0.5 → half speed, rate=2.0 → double speed."""
    rate: float = 1.0

class TransitionSpec(BaseModel):
    """Transition placed after a clip (between it and the next clip)."""
    type: Literal["cross-dissolve", "fade-in", "fade-out"] = "cross-dissolve"
    duration_seconds: float = 0.5

class ClipDecision(BaseModel):
    """A single clip on the primary storyline (spine). Ordered by list index."""
    id: str
    source_file: str            # filename of the source video
    source_path: str = ""       # full path (resolved at compile time if empty)
    source_start: float         # in-point in seconds
    source_end: float           # out-point in seconds
    label: str = ""             # human-readable description
    description: str = ""       # brief content description (what's in this clip)
    gain_db: Optional[float] = None
    speed: Optional[SpeedChange] = None
    transform: Optional[TransformSettings] = None
    transition_after: Optional[TransitionSpec] = None

class NarrationSegment(BaseModel):
    """A narration audio segment placed at a specific timeline position."""
    file: str                   # path to narration audio file
    timeline_offset: float      # where on the timeline to place it (seconds)
    start: float = 0.0          # in-point within the narration file
    duration: float             # duration of this segment
    gain_db: float = 0.0

class MusicTrack(BaseModel):
    """Background music spanning the timeline."""
    file: str
    start: float = 0.0         # in-point within the music file
    duration: float = 0.0      # 0 = use full timeline duration
    gain_db: float = -12.0

class TextOverlay(BaseModel):
    """A title/text overlay at a specific timeline position."""
    text: str
    timeline_offset: float     # seconds into timeline
    duration: float            # seconds
    lane: int = 1              # positive = above spine
    font_size: int = 72

class EditDecision(BaseModel):
    """
    Complete edit decision — the contract between the LLM and the FCPXML compiler.

    The LLM fills this out with creative decisions (which clips, what order,
    transitions, narration, music). Deterministic code compiles it to valid
    FCPXML 1.10. The dashboard reads the same JSON for the timeline view.
    """
    timeline: TimelineSettings = TimelineSettings()
    clips: List[ClipDecision] = []
    transitions: List[TransitionSpec] = []  # kept for backward compat; prefer clip.transition_after
    narration: List[NarrationSegment] = []
    music: Optional[MusicTrack] = None
    titles: List[TextOverlay] = []


class RefinedTimestamps(BaseModel):
    """Gemini's structured output for clip timestamp refinement."""
    new_start: float          # refined in-point in seconds (absolute in source video)
    new_end: float            # refined out-point in seconds (absolute in source video)
    reasoning: str            # why these timestamps were chosen
    focus_type: str = ""      # "visual" | "dialogue" | "audio" — what drove the decision
