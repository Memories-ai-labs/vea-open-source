"""V2 pipeline schemas — internal data models for the agentic editing pipeline."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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
    indexed_at: Optional[str] = None  # ISO 8601 UTC timestamp of when this video was indexed

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
            indexed_at=v.get("indexed_at"),
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

class ShotCropResult(BaseModel):
    """Per-shot crop result within a clip."""
    source_start: float
    source_end: float
    transform: TransformSettings

class MultiShotCropResult(BaseModel):
    """Result of multi-shot crop analysis on a clip."""
    shots: List[ShotCropResult]
    content_bounds: Optional[dict] = None

class SpeedChange(BaseModel):
    """Constant speed change. rate=0.5 → half speed, rate=2.0 → double speed."""
    rate: float = 1.0

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
    measured_loudness_lufs: Optional[float] = Field(
        default=None,
        description="Measured integrated loudness in LUFS. Set automatically during rendering. "
                    "Use this to reason about gain_db adjustments.",
    )
    speed: Optional[SpeedChange] = None
    transform: Optional[TransformSettings] = None
    transform_mode: Literal["fit", "custom", "saliency"] = "fit"
    shot_transforms: Optional[List[ShotCropResult]] = Field(
        default=None,
        description="Per-shot transforms for multi-shot clips. When present, the FCPXML compiler "
                    "emits one asset-clip per shot with different adjust-transform values. "
                    "The top-level 'transform' field is used as fallback / single-shot value.",
    )
    source_width: int = 1920
    source_height: int = 1080
    track: int = Field(default=1, description="Video track number (1=V1, 2=V2, etc.)")
    timeline_offset: Optional[float] = Field(default=None, description="Absolute timeline position in seconds (track 2+ free placement; track 1 uses sequential array order)")

class NarrationSegment(BaseModel):
    """A narration audio segment placed at a specific timeline position.

    ``track`` is a UI-level organizational lane (A1, A2, A3, ...). The FCPXML
    compiler and FFmpeg renderer ignore it — it just lets the dashboard place
    segments on visually distinct audio rows.
    """
    file: str                   # path to narration audio file
    timeline_offset: float      # where on the timeline to place it (seconds)
    start: float = 0.0          # in-point within the narration file
    duration: float             # duration of this segment
    gain_db: float = 0.0
    measured_loudness_lufs: Optional[float] = None
    track: int = 1              # audio lane (1 = A1, 2 = A2, ...)

class MusicTrack(BaseModel):
    """Background music spanning the timeline.

    gain_db is interpreted by the renderer as an OFFSET from the target LUFS
    (-18 LUFS for music). Default 0 means "play at target loudness."

    ``track`` is a UI-level organizational lane; the renderer ignores it.
    """
    file: str
    start: float = 0.0         # in-point within the music file
    duration: float = 0.0      # 0 = use full timeline duration
    gain_db: float = 0.0       # offset from -18 LUFS target
    measured_loudness_lufs: Optional[float] = None
    track: int = 2             # audio lane (defaults to A2 so music sits below narration)

class TextOverlay(BaseModel):
    """A title/text overlay at a specific timeline position."""
    text: str
    timeline_offset: float     # seconds into timeline
    duration: float            # seconds
    lane: int = 1              # positive = above spine
    font_size: int = 72
    style: str = "title"       # "title" = centered graphic, "subtitle" = bottom caption
    position: str = "center"   # "center", "bottom", "top"

class EditDecision(BaseModel):
    """
    Complete edit decision — the contract between the LLM and the FCPXML compiler.

    The LLM fills this out with creative decisions (which clips, what order,
    narration, music). Deterministic code compiles it to valid FCPXML 1.10.
    The dashboard reads the same JSON for the timeline view.
    """
    timeline: TimelineSettings = TimelineSettings()
    clips: List[ClipDecision] = []
    narration: List[NarrationSegment] = []
    music: Optional[MusicTrack] = None
    titles: List[TextOverlay] = []


class RefinedTimestamps(BaseModel):
    """Gemini's structured output for clip timestamp refinement.

    IMPORTANT: new_start and new_end are OFFSETS in seconds from the beginning
    of the provided video excerpt (starting at 0), NOT absolute/global timestamps
    from the original source video.
    """
    new_start: float = Field(
        ge=0,
        description="Refined in-point as seconds from the START of the provided video excerpt (0-based). Must be >= 0.",
    )
    new_end: float = Field(
        ge=0,
        description="Refined out-point as seconds from the START of the provided video excerpt (0-based). Must be > new_start.",
    )
    reasoning: str = Field(
        description="Brief explanation of why these specific timestamps were chosen.",
    )
    focus_type: str = Field(
        default="",
        description="What primarily drove the cut point decision: 'dialogue', 'visual', or 'audio'.",
    )
    speech_truncated_start: bool = Field(
        default=False,
        description="True if a sentence or word is cut off at the very START of the video excerpt (speaker mid-sentence at 0:00).",
    )
    speech_truncated_end: bool = Field(
        default=False,
        description="True if a sentence or word is cut off at the very END of the video excerpt (speaker mid-sentence at the last frame).",
    )
