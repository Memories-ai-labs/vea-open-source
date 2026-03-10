# VEA v2 — Detailed Implementation Plan

This document is the authoritative implementation guide for building the v2 pipeline described in `architecture-v2.md`. It covers every file to create, modify, or delete, with precise descriptions of what each file should contain.

---

## 1. Repo Structure After v2

```
vea-open-source/
├── src/
│   ├── app.py                          # MODIFY: add v2 endpoints + WebSocket
│   ├── schema.py                       # MODIFY: add v2 request/response schemas
│   ├── config.py                       # MODIFY: add WORKSPACES_DIR
│   └── pipelines/
│       ├── common/                     # UNCHANGED (v1 still uses all of these)
│       │   ├── audio_processing.py
│       │   ├── dynamic_cropping.py     # KEEP — reused by v2 crop pipeline
│       │   ├── edit_video_response.py
│       │   ├── fcpxml_exporter.py      # KEEP — reference + v1
│       │   ├── generate_narration_audio.py  # KEEP — reused by v2
│       │   ├── generate_subtitles.py
│       │   ├── metadata_helpers.py
│       │   ├── music_selection.py      # KEEP — reused by v2
│       │   ├── refine_clip_timestamps.py
│       │   ├── schema.py
│       │   └── timeline_constructor.py
│       │
│       ├── videoComprehension/         # UNCHANGED (v1 /index endpoint)
│       ├── flexibleResponse/           # UNCHANGED (v1 /flexible_respond endpoint)
│       ├── movieToShort/               # UNCHANGED
│       ├── screenplay/                 # UNCHANGED
│       ├── qualityAnalysis/            # UNCHANGED
│       │
│       └── v2/                         # NEW — all v2 pipeline code
│           ├── __init__.py
│           ├── workspace.py            # WorkspaceManager (session state)
│           ├── schemas.py              # All v2 Pydantic models
│           ├── comprehension/
│           │   ├── __init__.py
│           │   └── lightweight_comprehension.py
│           ├── planning/
│           │   ├── __init__.py
│           │   ├── planning_prompts.py
│           │   ├── iterative_planning_loop.py
│           │   └── clip_postprocess.py
│           ├── fcpxml/
│           │   ├── __init__.py
│           │   ├── fcpxml_scaffold.py  # Programmatic valid baseline generator
│           │   ├── fcpxml_compiler.py  # 3-layer validator + autofix
│           │   └── fcpxml_agent.py     # Hybrid: scaffold → LLM enhance → compile → correct
│           ├── narration/
│           │   ├── __init__.py
│           │   └── narration_pipeline.py
│           ├── music/
│           │   ├── __init__.py
│           │   └── music_pipeline.py
│           └── cropping/
│               ├── __init__.py
│               └── crop_pipeline.py
│
├── lib/
│   ├── llm/
│   │   ├── MemoriesAiManager.py        # MODIFY: add session_id to chat()
│   │   └── GeminiGenaiManager.py       # UNCHANGED
│   ├── oss/                            # UNCHANGED
│   └── utils/
│       ├── media.py                    # UNCHANGED
│       ├── metrics_collector.py        # UNCHANGED
│       ├── vinet_setup.py              # UNCHANGED
│       ├── resolve_setup.py            # NEW — health check + env validation
│       └── resolve_render.py           # NEW — DaVinci Resolve render workflow
│
├── dashboard/                          # NEW — React SPA
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/
│       │   └── session.ts              # WebSocket hook + REST calls
│       ├── components/
│       │   ├── PlanningMonitor.tsx     # Phase A: 4-panel planning view
│       │   ├── KnowledgePanel.tsx      # Gist + accumulated Q&A context
│       │   ├── StoryboardPanel.tsx     # Current storyboard shots
│       │   ├── ToolCallFeed.tsx        # Live tool call stream
│       │   ├── FootageMap.tsx          # Per-file timeline access visualization
│       │   ├── TimelineView.tsx        # Phase B: FCPXML timeline
│       │   ├── TrackRow.tsx            # Single track in timeline
│       │   ├── RenderControls.tsx      # Preview/Render buttons + progress
│       │   └── InjectBar.tsx           # Pause/inject input UI
│       ├── hooks/
│       │   ├── useSession.ts           # WebSocket connection management
│       │   └── useFcpxml.ts            # Parse FCPXML → timeline data
│       └── types/
│           └── session.ts              # TypeScript types matching server WS protocol
│
├── context/
│   ├── fcpxml_formatting_guide.md      # UNCHANGED — injected into FCPXML agent prompt
│   └── fcpxml_1_10.dtd                 # NEW — Apple's official DTD for xmllint validation
│
├── data/
│   ├── videos/                         # UNCHANGED
│   ├── indexing/                       # UNCHANGED (v1)
│   ├── outputs/                        # UNCHANGED (v1)
│   └── workspaces/                     # NEW — v2 session state (gitignored)
│       └── {project_name}/
│           ├── session.json
│           ├── context.md
│           ├── storyboard.json
│           ├── clips.json
│           ├── iterations/
│           ├── media/                  # Symlinks to source footage
│           ├── narration/
│           ├── music/
│           ├── fcpxml/
│           └── renders/
│
├── config.example.json                 # MODIFY: add workspaces_dir
├── pyproject.toml                      # MODIFY: add pydavinci
└── .gitignore                          # MODIFY: add workspaces/, debug_output/, renders/
```

---

## 2. Files to DELETE

These files have no current callers and add noise:

| File | Reason |
|------|--------|
| `src/pipelines/common/debug_crop.py` | Debug utility, not imported anywhere in production code |
| `debug_output/` (directory) | Runtime artifacts — add `debug_output/` to `.gitignore` instead |
| `failed_recaps.txt` | Runtime artifact at repo root |
| `quiz_predictions.txt` | Runtime artifact at repo root |

**Do not delete** any v1 pipeline files — `videoComprehension/`, `flexibleResponse/`, etc. are still wired to live v1 endpoints.

---

## 3. Files to MODIFY

### 3.1 `src/config.py`

Add:
```python
# v2 workspace directory
WORKSPACES_DIR = Path(_config.get("paths", {}).get("workspaces_dir", "data/workspaces"))

# v2 API prefix
V2_API_PREFIX = "/video-edit/v2"
```

Add `WORKSPACES_DIR` to `ensure_local_directories()`.

### 3.2 `src/schema.py`

Add v2 request/response schemas at the bottom. Keep all existing v1 schemas untouched.

```python
# --- V2 Schemas ---

class V2IndexRequest(BaseModel):
    project_name: str
    source_dir: str          # e.g. "data/videos/googleio/"
    start_fresh: bool = False

class V2IndexResponse(BaseModel):
    project_name: str
    video_nos: List[str]
    gist: str
    status: str              # "ready"

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
    override_script: Optional[str] = None  # User can supply narration text directly

class V2MusicRequest(BaseModel):
    project_name: str
    mood: Optional[str] = None   # e.g. "inspirational", "dramatic"
    prompt: Optional[str] = None # Natural language: "add upbeat background music"

class V2CropRequest(BaseModel):
    project_name: str
    aspect_ratio: float = 0.5625  # 9:16 default

class V2RenderRequest(BaseModel):
    project_name: str
    quality: str = "preview"  # "preview" | "final"

class V2RenderResponse(BaseModel):
    project_name: str
    output_path: str

class V2ResolveStatusResponse(BaseModel):
    running: bool
    version: Optional[str]
    studio: bool
    error: Optional[str]
```

### 3.3 `src/app.py`

Add to imports:
```python
from src.config import V2_API_PREFIX, WORKSPACES_DIR
from src.pipelines.v2.comprehension.lightweight_comprehension import LightweightComprehension
from src.pipelines.v2.planning.iterative_planning_loop import IterativePlanningLoop
from src.pipelines.v2.fcpxml.fcpxml_agent import FcpxmlAgent
from src.pipelines.v2.narration.narration_pipeline import V2NarrationPipeline
from src.pipelines.v2.music.music_pipeline import V2MusicPipeline
from src.pipelines.v2.cropping.crop_pipeline import V2CropPipeline
from lib.utils.resolve_render import ResolveRenderer
from lib.utils.resolve_setup import check_resolve_status
from fastapi import WebSocket, WebSocketDisconnect, BackgroundTasks
```

Add session event bus (module-level):
```python
# Per-session event queues for WebSocket broadcasting
_session_queues: Dict[str, asyncio.Queue] = {}
# Per-session pause events (set = paused, clear = running)
_session_pause_events: Dict[str, asyncio.Event] = {}
# Per-session inject queues (user prompts injected mid-loop)
_session_inject_queues: Dict[str, asyncio.Queue] = {}

async def v2_broadcast(project_name: str, event: dict):
    """Broadcast an event to the WebSocket client for a session."""
    if project_name in _session_queues:
        await _session_queues[project_name].put(event)
```

Add v2 endpoints:
- `WS  {V2_API_PREFIX}/session/{project_name}` — bidirectional session channel
- `POST {V2_API_PREFIX}/index`
- `POST {V2_API_PREFIX}/plan`     (starts background task, streams via WS)
- `POST {V2_API_PREFIX}/generate_fcpxml`
- `POST {V2_API_PREFIX}/narration`
- `POST {V2_API_PREFIX}/music`
- `POST {V2_API_PREFIX}/crop`
- `POST {V2_API_PREFIX}/render`   (streams progress via WS)
- `GET  {V2_API_PREFIX}/resolve/status`

The WebSocket endpoint handles three client message types: `pause`, `inject` (with `prompt` field), and `resume`. The planning loop checks the pause event at the top of each iteration and drains the inject queue before proceeding.

### 3.4 `lib/llm/MemoriesAiManager.py`

Add `session_id: Optional[str] = None` parameter to `chat()`. When provided, pass it as `"sessionId"` in the request body so Memories.ai maintains conversation continuity across calls within the same planning loop. The session ID is stored in `session.json` and reused on resume.

Also add:
```python
async def get_video_status(self, video_no: str) -> str:
    """Returns 'PARSE', 'UNPARSE', or 'NOT_FOUND'."""
```

This is used by the comprehension session cache check.

### 3.5 `pyproject.toml`

Add:
```
"pydavinci>=0.3.0",
```

### 3.6 `config.example.json`

Add to `paths` section:
```json
"workspaces_dir": "data/workspaces"
```

### 3.7 `.gitignore`

Add:
```
debug_output/
data/workspaces/
failed_recaps.txt
quiz_predictions.txt
```

---

## 4. Files to CREATE

### 4.1 `src/pipelines/v2/__init__.py`
Empty.

### 4.2 `src/pipelines/v2/schemas.py`

All v2 pipeline-internal Pydantic models (separate from the API-facing schemas in `src/schema.py`).

**Models to define:**

```python
# Session state
@dataclass
class VideoEntry:
    video_no: str
    video_name: str
    source_path: str
    duration_seconds: Optional[float]

@dataclass
class SessionData:
    version: str = "2.0"
    project_name: str
    created_at: str
    updated_at: str
    status: str          # "indexed" | "planning" | "fcpxml_ready" | "rendered"
    videos: List[VideoEntry]
    gist: str
    memories_session_id: Optional[str]
    planning: PlanningState

@dataclass
class PlanningState:
    iteration_count: int = 0
    user_prompts: List[str]
    target_duration_seconds: float

# Planning loop schemas (Gemini structured output targets)
class ChatTool(BaseModel):
    question: str
    purpose: str

class SearchTool(BaseModel):
    query: str
    purpose: str
    target_duration_sec: float

class ToolCallPlan(BaseModel):
    reasoning: str
    chat_calls: List[ChatTool]
    search_calls: List[SearchTool]
    should_stop: bool

class RetrievedClip(BaseModel):
    video_no: str
    video_name: str
    source_path: str
    start_seconds: float
    end_seconds: float
    score: float
    description: str
    shot_query: str

class Shot(BaseModel):
    id: str
    purpose: str
    search_query: str
    retrieved_clip: Optional[RetrievedClip]
    narration: Optional[str]
    priority: Literal["narration", "clip_audio", "clip_video"] = "narration"
    duration_seconds: float

class Storyboard(BaseModel):
    iteration: int
    target_duration_seconds: float
    theme: str
    narrative_arc: str
    shots: List[Shot]
    open_questions: List[str]
    notes: str
```

### 4.3 `src/pipelines/v2/workspace.py`

`WorkspaceManager` class — single object that owns all file I/O for a session.

**Key methods:**
```python
class WorkspaceManager:
    def __init__(self, project_name: str, workspaces_dir: Path): ...

    # Lifecycle
    def exists(self) -> bool
    def create(self) -> None          # mkdir structure
    def load_session(self) -> SessionData
    def save_session(self, session: SessionData) -> None

    # Planning state
    def load_storyboard(self) -> Optional[Storyboard]
    def save_storyboard(self, sb: Storyboard) -> None
    def load_clips(self) -> List[RetrievedClip]
    def save_clips(self, clips: List[RetrievedClip]) -> None
    def append_context(self, text: str) -> None    # append to context.md
    def load_context(self) -> str
    def save_iteration_snapshot(self, iteration: int,
                                 tool_plan: ToolCallPlan,
                                 storyboard: Storyboard) -> None

    # Media
    def ensure_media_symlinks(self, source_paths: List[str]) -> None
    def get_media_dir(self) -> Path

    # Outputs
    def get_fcpxml_path(self, version: int) -> Path
    def get_final_fcpxml_path(self) -> Path
    def get_narration_path(self) -> Path
    def get_music_path(self) -> Path
    def get_render_path(self, quality: str) -> Path
```

Workspace directory is `data/workspaces/{project_name}/`. All paths are computed relative to it. `save_session` updates `updated_at` and `status` automatically.

### 4.4 `src/pipelines/v2/comprehension/lightweight_comprehension.py`

```python
class LightweightComprehension:
    """
    Phase 1: Upload video to Memories.ai (or reuse existing video_no),
    get a broad gist via Chat API, save session.json.
    """
    def __init__(self, project_name: str, source_dir: str,
                 memories: MemoriesAiManager, workspace: WorkspaceManager): ...

    async def run(self, start_fresh: bool = False) -> SessionData:
        # 1. If not start_fresh: check workspace for existing session
        #    - If session exists with valid video_nos, verify status via get_video_status()
        #    - If all PARSE, skip upload and return loaded session
        # 2. Find all video files in source_dir
        # 3. For each video: upload_video_file() → video_no
        # 4. wait_for_ready() for each (in parallel with asyncio.gather)
        # 5. Single Chat API call for gist (all video_nos together)
        # 6. Save session.json, return SessionData
```

**GIST_PROMPT** (in `planning_prompts.py`, imported here):
```
Give me a broad overview of this video content: what it's about, who the
main speakers or subjects are, what major topics or events are covered,
the overall tone and pacing, and roughly what happens during different
time periods. Be comprehensive but concise.
```

### 4.5 `src/pipelines/v2/planning/planning_prompts.py`

Module-level string constants for all prompts. **No logic here — only strings.**

```python
GIST_PROMPT: str = "..."

MEMORIES_AI_TOOL_DOCS: str = """
--- MEMORIES.AI TOOLS ---
[full tool documentation as defined in architecture-v2.md §5.3]
"""

DECIDE_TOOL_CALLS_SYSTEM: str = """
You are a video editor planning an edit. You have two tools: the Memories.ai
Chat API (for understanding) and Search API (for clip retrieval).
Review the current storyboard, identify what information or footage is still
missing, and output the tool calls needed. Do NOT update the storyboard yet.

{MEMORIES_AI_TOOL_DOCS}
"""

DECIDE_TOOL_CALLS_USER: str = """
USER PROMPT: {user_prompt}

VIDEO GIST:
{gist}

ACCUMULATED CONTEXT:
{accumulated_context}

CURRENT STORYBOARD (iteration {iteration} of {max_iterations}):
{storyboard_json}

RETRIEVED CLIPS SO FAR:
{clips_summary}

{last_iteration_warning}

Output a ToolCallPlan. Set should_stop=true if the storyboard is complete.
"""

UPDATE_STORYBOARD_SYSTEM: str = """
You are a video editor. Using all available evidence, produce an updated
storyboard that best fulfills the user's prompt.
- Assign the best retrieved clip to each shot.
- Write narration text for each shot (if priority is "narration").
- Arrange shots for narrative coherence, not just chronological order.
- Set priority="clip_audio" for interview/speech moments worth preserving.
- Leave open_questions for shots that still need better clips.
"""

UPDATE_STORYBOARD_USER: str = """
USER PROMPT: {user_prompt}

VIDEO GIST:
{gist}

ACCUMULATED CONTEXT (includes evidence from this iteration):
{accumulated_context}

PREVIOUS STORYBOARD:
{storyboard_json}

ALL RETRIEVED CLIPS:
{clips_detail}

Output a complete updated Storyboard.
"""

GENERATE_FCPXML_SYSTEM: str = """
You are a video editing engineer. Generate a valid FCPXML 1.10 file from
the provided storyboard and clip list.

{FCPXML_FORMATTING_GUIDE}
"""

GENERATE_FCPXML_USER: str = """
Generate FCPXML 1.10 for this storyboard.

TARGET FORMAT: 1920x1080, {fps}fps, tcStart=3600/1s
MEDIA DIRECTORY: {media_dir}

STORYBOARD:
{storyboard_json}

CLIPS (with source paths and timestamps):
{clips_detail}

NARRATION AUDIO: {narration_path}
MUSIC AUDIO: {music_path}

Rules:
- All time values as rational fractions with 's' suffix (e.g. 48/24s)
- Add cross dissolve transitions (12 frames) between primary clips
- Narration audio: lane="-1", audioRole="music.narration"
- Music audio: lane="-2", audioRole="music", adjust-volume="-12dB"
- asset src paths: file:///absolute/path
- Never use float seconds (0.5s is wrong; use 12/24s)
"""

NARRATION_SCRIPT_SYSTEM: str = "..."
MUSIC_MOOD_SYSTEM: str = "..."
```

When `GENERATE_FCPXML_SYSTEM` is built, load `context/fcpxml_formatting_guide.md` from disk and substitute into `{FCPXML_FORMATTING_GUIDE}`. This file is the single source of truth for FCPXML rules.

### 4.6 `src/pipelines/v2/planning/clip_postprocess.py`

Pure functions, no LLM calls, no I/O.

```python
def apply_score_threshold(clips: List[RetrievedClip], min_score: float = 0.3) -> List[RetrievedClip]
def merge_overlapping_clips(clips: List[RetrievedClip], threshold: float = 0.5) -> List[RetrievedClip]
def enforce_temporal_diversity(clips: List[RetrievedClip],
                                window_seconds: float = 120.0,
                                max_fraction: float = 0.3) -> List[RetrievedClip]
def enforce_minimum_gap(clips: List[RetrievedClip], min_gap: float = 2.0) -> List[RetrievedClip]

def postprocess_clips(clips: List[RetrievedClip]) -> List[RetrievedClip]:
    """Apply all filters in order."""
    clips = apply_score_threshold(clips)
    clips = merge_overlapping_clips(clips)
    clips = enforce_temporal_diversity(clips)
    clips = enforce_minimum_gap(clips)
    return clips

def merge_clip_lists(existing: List[RetrievedClip],
                     new: List[RetrievedClip]) -> List[RetrievedClip]:
    """Merge two clip lists, deduplicating by (video_no, start, end)."""
```

### 4.7 `src/pipelines/v2/planning/iterative_planning_loop.py`

The main planning loop. Receives a `broadcast` callback so it can emit events without knowing about WebSockets.

```python
class IterativePlanningLoop:
    def __init__(self,
                 workspace: WorkspaceManager,
                 memories: MemoriesAiManager,
                 gemini: GeminiGenaiManager,
                 broadcast: Callable[[dict], Coroutine],
                 pause_event: asyncio.Event,
                 inject_queue: asyncio.Queue): ...

    async def run(self,
                  user_prompt: str,
                  target_duration_seconds: float,
                  max_iterations: int = 5) -> Storyboard:
        # Load existing state (session continuity)
        session = self.workspace.load_session()
        accumulated_context = self.workspace.load_context()
        current_storyboard = self.workspace.load_storyboard()  # None on first run
        clips_so_far = self.workspace.load_clips()

        video_nos = [v.video_no for v in session.videos]

        for iteration in range(max_iterations):
            # --- Check pause ---
            if self.pause_event.is_set():
                injected = await self.inject_queue.get()
                accumulated_context += f"\n\n[USER at iter {iteration}]: {injected}"
                self.pause_event.clear()

            await self.broadcast({"type": "iteration_start",
                                   "iteration": iteration,
                                   "max": max_iterations})

            # --- Gemini Call A: decide tool calls ---
            tool_plan: ToolCallPlan = await self._decide_tool_calls(
                user_prompt, session.gist, accumulated_context,
                current_storyboard, clips_so_far, iteration, max_iterations
            )
            await self.broadcast({"type": "tool_plan",
                                   "iteration": iteration,
                                   "reasoning": tool_plan.reasoning,
                                   "chat_calls": [c.dict() for c in tool_plan.chat_calls],
                                   "search_calls": [s.dict() for s in tool_plan.search_calls]})

            if tool_plan.should_stop:
                break

            # --- Execute tools in parallel ---
            chat_results, search_results = await asyncio.gather(
                self._run_chat_calls(tool_plan.chat_calls, video_nos, session),
                self._run_search_calls(tool_plan.search_calls, video_nos)
            )
            await self.broadcast({"type": "tool_results",
                                   "iteration": iteration,
                                   "clips_found": sum(len(r) for r in search_results)})

            # --- Update context and clips ---
            new_evidence = self._format_evidence(tool_plan, chat_results, search_results)
            accumulated_context += f"\n\n--- Iteration {iteration} Evidence ---\n{new_evidence}"
            flat_new_clips = [clip for batch in search_results for clip in batch]
            flat_new_clips = postprocess_clips(flat_new_clips)
            clips_so_far = merge_clip_lists(clips_so_far, flat_new_clips)

            # --- Gemini Call B: update storyboard ---
            current_storyboard = await self._update_storyboard(
                user_prompt, session.gist, accumulated_context,
                current_storyboard, clips_so_far, iteration
            )
            await self.broadcast({"type": "storyboard_updated",
                                   "iteration": iteration,
                                   "storyboard": current_storyboard.dict()})

            # --- Persist state ---
            self.workspace.append_context(
                f"\n\n--- Iteration {iteration} Evidence ---\n{new_evidence}"
            )
            self.workspace.save_storyboard(current_storyboard)
            self.workspace.save_clips(clips_so_far)
            self.workspace.save_iteration_snapshot(iteration, tool_plan, current_storyboard)

        await self.broadcast({"type": "planning_complete",
                               "storyboard": current_storyboard.dict()})
        return current_storyboard

    async def _decide_tool_calls(self, ...) -> ToolCallPlan:
        # GeminiGenaiManager.LLM_request() with ToolCallPlan schema
        # Uses DECIDE_TOOL_CALLS_SYSTEM + DECIDE_TOOL_CALLS_USER from planning_prompts

    async def _update_storyboard(self, ...) -> Storyboard:
        # GeminiGenaiManager.LLM_request() with Storyboard schema
        # Uses UPDATE_STORYBOARD_SYSTEM + UPDATE_STORYBOARD_USER from planning_prompts

    async def _run_chat_calls(self, calls: List[ChatTool], video_nos: List[str],
                               session: SessionData) -> List[str]:
        # asyncio.gather of memories.chat() calls
        # Pass session.memories_session_id for continuity
        # On first call, save returned session_id back to session

    async def _run_search_calls(self, calls: List[SearchTool],
                                 video_nos: List[str]) -> List[List[RetrievedClip]]:
        # asyncio.gather of memories.search() calls
        # Convert search results to RetrievedClip objects
        # Look up source_path from session.videos by video_no
```

### 4.8 `src/pipelines/v2/fcpxml/fcpxml_compiler.py`

The FCPXML compiler is the core reliability layer between LLM output and a valid importable file. It combines three layers of checking and can auto-fix a subset of common errors before falling back to LLM correction.

```python
class FcpxmlCompiler:
    """
    Three-layer FCPXML validation and auto-fix pipeline:

    Layer 1 — Structural (Python xml.etree): well-formed XML, required attributes,
               ref resolution, time value format
    Layer 2 — DTD (xmllint): validates against Apple's official FCPXML 1.10 DTD
    Layer 3 — Semantic (Python): time math correctness, file existence,
               spine continuity, duration consistency

    Also provides auto-fix for common LLM mistakes before asking Gemini to retry.
    """

    DTD_PATH = Path("context/fcpxml_1_10.dtd")  # Apple DTD, checked into repo
    XMLLINT_AVAILABLE = shutil.which("xmllint") is not None  # macOS: yes, Linux: apt install libxml2-utils

    def compile(self, xml_str: str, media_dir: Path) -> CompileResult:
        """
        Returns CompileResult(valid: bool, errors: List[str], warnings: List[str])
        Runs all three layers. Stops at first layer that has errors.
        """

    def autofix(self, xml_str: str) -> str:
        """
        Attempt to fix the most common LLM FCPXML mistakes without LLM involvement:
        - Float seconds → rational fractions (e.g. '2.0s' → '48/24s')
        - Missing 's' suffix on time values
        - Duplicate id= attributes (append suffix)
        - Unclosed tags (via minidom re-serialization)
        - Incorrect version attribute (normalize to '1.10')
        Returns the fixed XML string. Does NOT fix structural/semantic errors.
        """

@dataclass
class CompileResult:
    valid: bool
    errors: List[str]      # Blocking — must fix before import
    warnings: List[str]    # Non-blocking — import may still work
    layer: str             # Which layer caught the first error: "structural" | "dtd" | "semantic"
```

**Getting Apple's FCPXML DTD:**

Apple publishes FCPXML DTDs in Final Cut Pro itself. Extract with:
```bash
# macOS with FCP installed:
find /Applications/Final\ Cut\ Pro.app -name "*.dtd" 2>/dev/null
# Typically at: .../Contents/Resources/fcpxml_1_10.dtd
# Copy to: context/fcpxml_1_10.dtd  (check into repo)
```

Without FCP, community-maintained DTDs are available at:
`https://github.com/CommandPost/FCPCafe/tree/main/docs/fcpxml`

**xmllint DTD validation (Layer 2):**
```python
result = subprocess.run(
    ["xmllint", "--dtdvalid", str(self.DTD_PATH), "--noout", "-"],
    input=xml_str.encode(),
    capture_output=True
)
# stderr contains validation errors in standard format
```

**Semantic checks (Layer 3):**
- All `asset` `media-rep src` paths resolve to existing files under `media_dir`
- Spine `asset-clip` offsets are monotonically non-decreasing
- No clip extends beyond its parent asset's duration
- Sequence `duration` attribute matches sum of spine clip durations
- Rational fractions are mathematically valid (denominator ≠ 0)
- All audio lanes have valid `role` attributes

### 4.9 `src/pipelines/v2/fcpxml/fcpxml_scaffold.py`

The scaffold generates a **guaranteed-valid baseline FCPXML** from the storyboard programmatically — same approach as the existing `fcpxml_exporter.py` but adapted to v2 `Storyboard` + `RetrievedClip` inputs. The LLM then edits this scaffold rather than generating from scratch.

```python
class FcpxmlScaffold:
    """
    Programmatic FCPXML generator. Produces a simple but guaranteed-valid
    FCPXML 1.10 from a Storyboard. No transitions, no effects, no audio mixing —
    just correct spine with asset-clips, proper time math, and valid resources.

    This is the starting point that FcpxmlAgent hands to Gemini for enhancement.
    Adapts logic from src/pipelines/common/fcpxml_exporter.py.
    """

    def generate(self, storyboard: Storyboard, clips: List[RetrievedClip],
                 media_dir: Path, fps: int = 24) -> str:
        """
        Returns a valid FCPXML 1.10 string with:
        - resources: one <format> + one <asset> per unique source file
        - spine: one <asset-clip> per shot with correct start/duration/offset
        - NO transitions, effects, audio lanes, narration, or music yet
        - asset names set to source filename (for crop pipeline correlation)
        """
```

### 4.10 `src/pipelines/v2/fcpxml/fcpxml_agent.py`

The agent uses a **hybrid strategy**: programmatic scaffold → LLM enhancement → compiler validation → LLM correction loop. This gives the LLM a valid starting point and structured feedback when it makes mistakes.

```python
class FcpxmlAgent:
    """
    Hybrid FCPXML generation:
    1. Scaffold: generate guaranteed-valid baseline programmatically
    2. Enhance: LLM adds transitions, audio mixing, effects on top of scaffold
    3. Compile: run FcpxmlCompiler — autofix trivial errors, report the rest
    4. Correct: if errors remain, feed them back to LLM (up to MAX_ITERATIONS)

    The LLM never generates from scratch — it always edits a valid XML document.
    This dramatically reduces the surface area for errors.
    """
    MAX_ENHANCE_ITERATIONS = 3

    def __init__(self, workspace: WorkspaceManager,
                 gemini: GeminiGenaiManager,
                 scaffold: FcpxmlScaffold,
                 compiler: FcpxmlCompiler): ...

    async def run(self) -> Path:
        storyboard = self.workspace.load_storyboard()
        clips = self.workspace.load_clips()
        self.workspace.ensure_media_symlinks([c.source_path for c in clips])
        media_dir = self.workspace.get_media_dir()

        # Step 1: Generate valid baseline
        baseline_xml = self.scaffold.generate(storyboard, clips, media_dir)
        # Sanity check — scaffold should always produce valid output
        assert not self.compiler.compile(baseline_xml, media_dir).errors, \
            "Scaffold produced invalid FCPXML — bug in scaffold code"

        # Step 2: LLM enhancement loop
        current_xml = baseline_xml
        for i in range(self.MAX_ENHANCE_ITERATIONS):
            enhanced_xml = await self._enhance(current_xml, storyboard, clips,
                                                media_dir, i)

            # Step 3: Compile
            result = self.compiler.compile(enhanced_xml, media_dir)

            if result.valid:
                current_xml = enhanced_xml
                break

            # Autofix trivial issues (float seconds, missing 's' suffix, etc.)
            fixed_xml = self.compiler.autofix(enhanced_xml)
            result2 = self.compiler.compile(fixed_xml, media_dir)

            if result2.valid:
                current_xml = fixed_xml
                break

            # Feed structured errors back to LLM for correction
            current_xml = fixed_xml  # give LLM the partially-fixed version
            # errors injected into next enhance() call

        # If still invalid after all iterations, fall back to scaffold
        final_result = self.compiler.compile(current_xml, media_dir)
        if not final_result.valid:
            logger.warning("LLM enhancement produced invalid FCPXML after all "
                           "iterations — falling back to scaffold baseline")
            current_xml = baseline_xml

        output_path = self.workspace.get_fcpxml_path(version=i + 1)
        output_path.write_text(current_xml)
        self.workspace.get_final_fcpxml_path().write_text(current_xml)
        return output_path

    async def _enhance(self, current_xml: str, storyboard: Storyboard,
                       clips: List[RetrievedClip], media_dir: Path,
                       iteration: int, errors: List[str] = None) -> str:
        """
        LLM call: given the current FCPXML (valid scaffold or previous attempt),
        add enhancements or fix errors. Returns raw XML string.

        On first call (iteration=0): add transitions, audio lanes, narration/music
        On subsequent calls: fix the provided compiler errors
        """
        # Uses GeminiGenaiManager.LLM_request() with text output (not structured JSON)
        # Response is extracted from the text — strip markdown code fences if present
```

**Enhancement prompt (first pass):**
```
You are editing a valid FCPXML 1.10 document. The document below is structurally
correct. Your job is to ENHANCE it by adding:
1. Cross-dissolve transitions (12 frames) between spine clips
2. Narration audio track: lane="-1", role="music.narration", src={narration_path}
3. Music audio track: lane="-2", role="music", adjust-volume="-12dB", src={music_path}
4. Subtitles as <caption> elements under each asset-clip if narration is present

RULES (do not break these — the document is currently valid):
{key rules from fcpxml_formatting_guide.md, condensed}

CURRENT FCPXML:
{current_xml}

Return ONLY the complete updated FCPXML document. No explanation, no markdown fences.
```

**Correction prompt (subsequent passes):**
```
The FCPXML you produced has validation errors. Fix them.

COMPILER ERRORS:
{errors joined by newline}

CURRENT (INVALID) FCPXML:
{current_xml}

Return ONLY the corrected complete FCPXML document. No explanation, no markdown fences.
```

**Key design decisions:**
- LLM always edits existing XML, never generates from scratch → smaller surface area for errors
- Scaffold guarantees correct time math and resource IDs — the hardest parts for LLM to get right
- Autofix handles the most common LLM mistake (float seconds) without burning an LLM call
- Fallback to scaffold ensures we always produce something importable, even if plain
- Compiler errors are passed as structured text so LLM understands exactly what to fix
- `GeminiGenaiManager.LLM_request()` needs a raw text output mode — add `schema=None` path that returns `response.text` directly (likely already works, just needs verification)

### 4.10 `src/pipelines/v2/narration/narration_pipeline.py`

Wraps existing `src/pipelines/common/generate_narration_audio.py`. On-demand, called after storyboard is ready.

```python
class V2NarrationPipeline:
    """
    Generates TTS narration from storyboard shot narration text.
    Writes narration.mp3 to workspace, then triggers FCPXML regeneration.
    """
    def __init__(self, workspace: WorkspaceManager,
                 gemini: GeminiGenaiManager,
                 fcpxml_agent: FcpxmlAgent): ...

    async def run(self, override_script: Optional[str] = None) -> Path:
        storyboard = self.workspace.load_storyboard()

        # Build narration script from storyboard shots
        # If override_script provided, use that instead
        script = override_script or self._extract_script(storyboard)

        # Sync check: estimate speaking time vs clip duration
        # If any shot's narration would take >2x the clip duration, ask Gemini to shorten
        script = await self._check_and_trim_narration(script, storyboard)

        # Language detection (same logic as existing pipeline)
        # Call ElevenLabs (English) or Minimax (Chinese)
        audio_path = await self._generate_tts(script)
        self.workspace.get_narration_path().parent.mkdir(exist_ok=True)

        # Regenerate FCPXML with narration embedded
        await self.fcpxml_agent.run()
        return audio_path
```

### 4.11 `src/pipelines/v2/music/music_pipeline.py`

Wraps existing `src/pipelines/common/music_selection.py`.

```python
class V2MusicPipeline:
    """
    Selects and downloads background music. On-demand after storyboard.
    Writes track.mp3 to workspace, triggers FCPXML regeneration.
    """
    async def run(self, mood: Optional[str] = None,
                  prompt: Optional[str] = None) -> Path:
        storyboard = self.workspace.load_storyboard()

        # Derive mood from storyboard theme if not provided
        if not mood and not prompt:
            mood = await self._infer_mood(storyboard)

        # Use existing music_selection.py logic (Soundstripe)
        track_path = await select_and_download_music(mood, storyboard.target_duration_seconds)

        # Save to workspace
        # Regenerate FCPXML with music embedded
        await self.fcpxml_agent.run()
        return track_path
```

### 4.12 `src/pipelines/v2/cropping/crop_pipeline.py`

Dynamic cropping as an on-demand post-plan tool. Instead of re-encoding the video, it injects `<adjust-transform>` elements into the FCPXML, instructing the NLE (FCP/Resolve) to perform the crop at render time.

```python
class V2CropPipeline:
    """
    On-demand dynamic cropping. Analyzes each clip with ViNet saliency model
    to find the salient region, then injects <adjust-transform> crop parameters
    into the FCPXML — no re-encoding.

    For 9:16 output (aspect_ratio=0.5625):
    - For each clip, run ViNet to find horizontal center of saliency
    - Compute x-offset needed to center the salient region in a 9:16 crop
    - Inject into FCPXML as:
        <adjust-transform position="{x_offset} 0" scale="{scale} {scale}"/>
    """
    def __init__(self, workspace: WorkspaceManager,
                 dynamic_cropper,   # existing DynamicCropping class
                 fcpxml_agent: FcpxmlAgent): ...

    async def run(self, aspect_ratio: float = 0.5625) -> Path:
        fcpxml_path = self.workspace.get_final_fcpxml_path()
        clips = self.workspace.load_clips()

        # For each clip, compute saliency-based crop transform
        crop_params = {}  # clip_id -> (x_offset, scale)
        for clip in clips:
            params = await self._analyze_clip(clip, aspect_ratio)
            crop_params[clip.video_no + clip.start_seconds] = params

        # Inject <adjust-transform> into FCPXML at matching asset-clip elements
        updated_xml = self._inject_transforms(fcpxml_path.read_text(), crop_params)

        # Save as new version
        cropped_path = self.workspace.get_fcpxml_path(version="cropped")
        cropped_path.write_text(updated_xml)
        self.workspace.get_final_fcpxml_path().write_text(updated_xml)
        return cropped_path
```

**Key insight:** The crop is expressed as an FCPXML transform, not a pixel operation. The source footage stays untouched. FCP or Resolve applies the crop at render time using the `<adjust-transform>` values.

### 4.13 `lib/utils/resolve_setup.py`

```python
REQUIRED_ENV_VARS = [
    "RESOLVE_SCRIPT_API",
    "RESOLVE_SCRIPT_LIB",
]

def check_resolve_status() -> dict:
    """
    Returns: {"running": bool, "version": str|None, "studio": bool, "error": str|None}
    - Checks env vars are set
    - Attempts to import DaVinciResolveScript
    - Attempts to get resolve handle
    - Checks if Studio (not Free) by trying GetProjectManager
    """

def print_setup_instructions():
    """
    Prints shell commands needed to configure Resolve on macOS and Linux.
    Called when check_resolve_status() fails.
    """

MACOS_ENV_SETUP = """
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

# Start Resolve headless:
/Applications/DaVinci\\ Resolve/DaVinci\\ Resolve.app/Contents/MacOS/resolve -nogui &
"""

LINUX_ENV_SETUP = """
# Start virtual display (required even in -nogui mode):
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

export RESOLVE_SCRIPT_API="/opt/resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/opt/resolve/libs/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

DISPLAY=:99 /opt/resolve/bin/resolve -nogui &
"""
```

### 4.14 `lib/utils/resolve_render.py`

```python
class ResolveRenderer:
    """
    Wraps DaVinci Resolve Python scripting API.
    Resolve must already be running as a -nogui daemon.
    Uses pydavinci if available, falls back to raw DaVinciResolveScript.
    """

    RENDER_PRESETS = {
        "preview": {"format": "mp4",  "codec": "H264",      "suffix": "_preview"},
        "final":   {"format": "mov",  "codec": "ProRes422",  "suffix": "_final"},
    }

    def __init__(self):
        self._resolve = self._connect()

    def _connect(self):
        """Import DaVinciResolveScript and get resolve handle. Raises if not available."""

    async def render(self,
                     fcpxml_path: str,
                     media_dir: str,
                     output_path: str,
                     quality: str = "preview",
                     progress_callback: Optional[Callable] = None) -> str:
        """
        Full render workflow:
        1. Create temporary project
        2. ImportTimelineFromFile(fcpxml_path, sourceClipsPath=media_dir)
        3. SetCurrentTimeline
        4. Configure render settings (format, codec, output path)
        5. AddRenderJob + StartRendering
        6. Poll IsRenderingInProgress(), call progress_callback(pct) each tick
        7. DeleteAllRenderJobs + DeleteProject (cleanup)
        8. Return output_path

        Note: Resolve does not return the output path from its API.
        We reconstruct it from TargetDir + CustomName + extension.
        """

    def is_available(self) -> bool:
        """Returns False if Resolve is not running or not Studio."""
```

### 4.15 `dashboard/` — React SPA

**Stack:** Vite + React 18 + TypeScript + Tailwind CSS + Zustand

**`dashboard/src/api/session.ts`**
```typescript
// WebSocket connection with auto-reconnect
// REST wrappers for all /v2/* endpoints
export function useSession(projectName: string): {
    connect: () => void
    disconnect: () => void
    pause: () => void
    inject: (prompt: string) => void
    resume: () => void
    events: ServerMessage[]
    status: SessionStatus
}
```

**`dashboard/src/hooks/useFcpxml.ts`**
```typescript
// Parses FCPXML string using browser DOMParser
// Returns timeline data: { tracks: Track[], duration: number }
// Track: { id, name, clips: Clip[], type: 'video'|'audio' }
// Clip: { id, assetName, start, duration, offset, lane }
```

**`dashboard/src/components/PlanningMonitor.tsx`**
- 4-panel layout using CSS Grid
- Left: `KnowledgePanel` — scrollable text rendering of gist + `context.md` content
- Center: `StoryboardPanel` — shot cards with status (has clip / searching / empty)
- Right: `ToolCallFeed` — live log of tool calls, color-coded (Chat = blue, Search = green)
- Bottom: `FootageMap` — one horizontal bar per source file, filled regions = accessed timestamps
- Top bar: iteration progress, session status, Pause button

**`dashboard/src/components/TimelineView.tsx`**
- Renders tracks from `useFcpxml()` output
- SVG-based: each clip = rect with label, width proportional to duration
- Video tracks above, audio tracks below a center line
- Click clip = show details (file, timestamps, narration text)
- Horizontal scroll for long timelines

**`dashboard/src/components/RenderControls.tsx`**
- "Preview Render" and "Final Render" buttons
- "Add Narration" and "Add Music" buttons
- "Apply Crop (9:16)" button
- Inline progress bar during render (updates from WS events)
- Download FCPXML link once generated

**`dashboard/src/components/InjectBar.tsx`**
- Shown only when paused
- Text input + Submit button
- Sends `{ type: "inject", prompt: "..." }` over WebSocket

---

## 5. New API Endpoints (detailed)

### `WS /video-edit/v2/session/{project_name}`

Bidirectional. See WebSocket message protocol in `architecture-v2.md §9.3`.

Server-side implementation:
```python
@app.websocket(f"{V2_API_PREFIX}/session/{{project_name}}")
async def v2_session_ws(websocket: WebSocket, project_name: str):
    await websocket.accept()
    queue = asyncio.Queue()
    pause_event = asyncio.Event()
    inject_queue = asyncio.Queue()

    _session_queues[project_name] = queue
    _session_pause_events[project_name] = pause_event
    _session_inject_queues[project_name] = inject_queue

    async def sender():
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    async def receiver():
        async for data in websocket.iter_json():
            if data["type"] == "pause":
                pause_event.set()
            elif data["type"] == "inject":
                await inject_queue.put(data["prompt"])
                pause_event.clear()
            elif data["type"] == "resume":
                pause_event.clear()

    try:
        await asyncio.gather(sender(), receiver())
    except WebSocketDisconnect:
        pass
    finally:
        _session_queues.pop(project_name, None)
        _session_pause_events.pop(project_name, None)
        _session_inject_queues.pop(project_name, None)
```

### `POST /video-edit/v2/plan`

Starts planning as a background task, returns immediately. Progress streams via WebSocket.

```python
@app.post(f"{V2_API_PREFIX}/plan")
async def v2_plan(request: V2PlanRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        run_planning_loop,
        project_name=request.project_name,
        prompt=request.prompt,
        target_duration_seconds=request.target_duration_seconds,
        max_iterations=request.max_iterations,
    )
    return {"status": "started", "project_name": request.project_name}
```

The `run_planning_loop` function creates the workspace, loop, and broadcasts events via `v2_broadcast`.

---

## 6. Implementation Order

### Phase 0 — Foundation (do first, everything depends on this)
1. `src/config.py` — add `WORKSPACES_DIR`, `V2_API_PREFIX`
2. `src/pipelines/v2/schemas.py` — all data models
3. `src/pipelines/v2/workspace.py` — `WorkspaceManager`
4. `lib/llm/MemoriesAiManager.py` — add `session_id` to `chat()`, add `get_video_status()`
5. `lib/utils/resolve_setup.py` — health check
6. `lib/utils/resolve_render.py` — render workflow
7. `pyproject.toml` — add `pydavinci`
8. `.gitignore` — add workspaces, debug_output
9. **Delete:** `src/pipelines/common/debug_crop.py`, `failed_recaps.txt`, `quiz_predictions.txt`

### Phase 1 — Comprehension
10. `src/pipelines/v2/planning/planning_prompts.py` — `GIST_PROMPT` (rest filled later)
11. `src/pipelines/v2/comprehension/lightweight_comprehension.py`
12. `src/schema.py` — add `V2IndexRequest`, `V2IndexResponse`
13. `src/app.py` — add `POST /v2/index`

### Phase 2 — Planning Loop
14. `src/pipelines/v2/planning/planning_prompts.py` — add planning prompts
15. `src/pipelines/v2/planning/clip_postprocess.py`
16. `src/pipelines/v2/planning/iterative_planning_loop.py`
17. `src/schema.py` — add `V2PlanRequest`
18. `src/app.py` — add `WS /v2/session/{project_name}`, `POST /v2/plan`

### Phase 3 — FCPXML Generation
19. Obtain Apple FCPXML 1.10 DTD → save to `context/fcpxml_1_10.dtd` (see §4.8)
20. `src/pipelines/v2/fcpxml/fcpxml_scaffold.py` — programmatic baseline (adapt from `fcpxml_exporter.py`)
21. `src/pipelines/v2/fcpxml/fcpxml_compiler.py` — 3-layer validator + autofix
22. `src/pipelines/v2/planning/planning_prompts.py` — add `GENERATE_FCPXML_*` prompts
23. `src/pipelines/v2/fcpxml/fcpxml_agent.py` — hybrid enhance → compile → correct loop
24. `src/schema.py` — add `V2GenerateFcpxmlRequest/Response`
25. `src/app.py` — add `POST /v2/generate_fcpxml`
26. **Experiment:** test LLM FCPXML enhancement quality on real storyboard; tune prompts and compiler error messages based on actual failure modes

### Phase 4 — On-Demand Tools
24. `src/pipelines/v2/narration/narration_pipeline.py`
25. `src/pipelines/v2/music/music_pipeline.py`
26. `src/pipelines/v2/cropping/crop_pipeline.py`
27. `src/schema.py` — add narration/music/crop/render schemas
28. `src/app.py` — add `/v2/narration`, `/v2/music`, `/v2/crop`

### Phase 5 — Resolve Render
29. `src/app.py` — add `POST /v2/render`, `GET /v2/resolve/status`

### Phase 6 — Dashboard
30. `dashboard/` — scaffold with `npm create vite`
31. Core layout + WebSocket hook
32. PlanningMonitor panels
33. TimelineView + FCPXML parser
34. RenderControls

---

## 7. Dependencies Summary

| Package | Reason | Already in pyproject.toml? |
|---------|--------|---------------------------|
| `pydavinci>=0.3.0` | DaVinci Resolve Python scripting wrapper | No — add |
| `websockets` | WebSocket support (fastapi[standard] includes this) | Yes |
| `fastapi[standard]` | WebSocket support built in | Yes (fastapi==0.115.12) |
| All others | Already present | Yes |

No other new Python dependencies needed — `aiohttp`, `asyncio`, `pydantic`, `tenacity` are all already present.

For the dashboard:
```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "zustand": "^4",
    "tailwindcss": "^3"
  },
  "devDependencies": {
    "vite": "^5",
    "typescript": "^5",
    "@types/react": "^18"
  }
}
```
