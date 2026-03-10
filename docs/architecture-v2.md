# VEA Architecture v2 — Agentic Video Editing with FCPXML + DaVinci Resolve

## Status: Design Document (not yet implemented)

This document defines the next-generation VEA architecture. It replaces the current pipeline (heavy upfront scene-by-scene indexing → moviepy/ffmpeg rendering) with a lightweight, iterative, agent-driven approach where understanding and planning happen on-demand, and the edit is expressed as FCPXML rendered via DaVinci Resolve.

---

## 1. Core Principles

### 1.1 No Heavy Upfront Indexing
The current system generates `scenes.json`, `story.json`, `people.txt` etc. via a slow multi-pass segment analysis before any response can be generated. This is expensive, slow, and produces a static index that may not be useful for every prompt.

**v2 approach:** Upload the video to Memories.ai once, ask for a broad gist, and stop. All further understanding is acquired on-demand as the planning loop determines it needs more information.

### 1.2 Iterative Planning (Not Linear)
The current system is a fixed linear pipeline: plan questions → gather info → plan shots → retrieve → narrate. If a retrieved clip is wrong or information is missing, there is no recovery.

**v2 approach:** A Python while-loop (up to 4–5 iterations) where Gemini reviews the current storyboard, decides what it still needs to know, calls Memories.ai tools to get it, and updates the plan. Each iteration expands the accumulated context.

### 1.3 FCPXML is the Edit
The current system assembles videos with moviepy/ffmpeg at render time, exporting FCPXML as a side product. This limits quality, flexibility, and editability.

**v2 approach:** FCPXML is the primary output. A dedicated FCPXML-generation agent converts the finalized storyboard into a valid FCPXML 1.10 file. Source footage is referenced directly — no re-encoding. The user gets a file they can open in Final Cut Pro or render via DaVinci Resolve.

### 1.4 DaVinci Resolve as Render Engine
For programmatic preview and final render, DaVinci Resolve (Studio, required) is used via its Python scripting API. Resolve runs as a persistent `-nogui` daemon; scripts connect to it via IPC.

### 1.5 Session Continuity
All intermediate outputs — gist, accumulated context, storyboard iterations, retrieved clips, FCPXML drafts, rendered previews — are persisted to a workspace directory. Re-running on the same workspace resumes from where it left off rather than starting over. This applies to both CLI and web dashboard access.

### 1.6 CLI and Dashboard are Peers
Both the web dashboard and CLI invoke the same backend API. There is no feature available in one that is not in the other. The dashboard provides a richer real-time view; the CLI is headless-compatible.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│  ┌──────────────────┐          ┌──────────────────────────────┐ │
│  │   Web Dashboard  │          │           CLI                │ │
│  │  (React SPA)     │          │  (FastAPI client / curl)     │ │
│  └────────┬─────────┘          └──────────────┬───────────────┘ │
│           │ WebSocket + REST                   │ REST            │
└───────────┼────────────────────────────────────┼─────────────────┘
            │                                    │
┌───────────▼────────────────────────────────────▼─────────────────┐
│                      FastAPI Backend (src/app.py)                  │
│                                                                    │
│  POST /v2/index           →  Comprehension (lightweight)           │
│  POST /v2/plan            →  Iterative Planning Loop               │
│  POST /v2/generate_fcpxml →  FCPXML Generation Agent               │
│  POST /v2/narration       →  Narration/Music (on-demand)           │
│  POST /v2/render          →  DaVinci Resolve render                │
│  WS   /v2/session/{id}    →  Live updates, pause/inject            │
└──────────┬─────────────────────────┬────────────────┬─────────────┘
           │                         │                │
     ┌─────▼──────┐          ┌───────▼──────┐  ┌─────▼──────────┐
     │ Memories.ai│          │   Gemini     │  │DaVinci Resolve │
     │  Chat API  │          │ (Vertex AI)  │  │ Python Script  │
     │  Search API│          │              │  │  (-nogui IPC)  │
     └────────────┘          └──────────────┘  └────────────────┘
```

---

## 3. Workspace / Session Model

Every project has a **workspace directory** under `data/workspaces/{project_name}/`. The workspace is the single source of truth for all session state.

### 3.1 Directory Structure

```
data/workspaces/{project_name}/
├── session.json              # Core session metadata (video_nos, gist, status)
├── context.md                # Accumulated Q&A context — append-only, grows each iteration
├── storyboard.json           # Current editing plan (latest iteration)
├── iterations/               # Snapshots of each planning iteration
│   ├── iter_0_initial.json
│   ├── iter_1_storyboard.json
│   ├── iter_2_storyboard.json
│   └── ...
├── clips.json                # All retrieved clips (video_no, timestamps, descriptions, scores)
├── media/                    # Symlinks to source footage in data/videos/{project_name}/
│   ├── video1.mp4 -> ../../videos/{project_name}/video1.mp4
│   └── ...
├── narration/
│   ├── narration.mp3         # Generated TTS audio
│   └── narration_script.txt  # Script used to generate it
├── music/
│   └── track.mp3             # Selected background music
├── fcpxml/
│   ├── edit_v1.fcpxml        # FCPXML drafts (one per generation iteration)
│   ├── edit_v2.fcpxml
│   └── edit_final.fcpxml     # Accepted/current FCPXML
└── renders/
    ├── preview.mp4           # Low-quality preview render
    └── final.mp4             # Full-quality final render
```

### 3.2 session.json Schema

```json
{
  "version": "2.0",
  "project_name": "googleio",
  "created_at": "2026-03-10T14:00:00Z",
  "updated_at": "2026-03-10T15:30:00Z",
  "status": "planning | fcpxml_ready | rendered",

  "videos": [
    {
      "video_no": "abc123",
      "video_name": "keynote.mp4",
      "source_path": "data/videos/googleio/keynote.mp4",
      "duration_seconds": 6960.0
    }
  ],

  "gist": "A 116-minute Google I/O 2025 keynote covering...",

  "planning": {
    "iteration_count": 3,
    "user_prompts": [
      "Create a 2-minute highlight reel of the most exciting AI announcements"
    ],
    "target_duration_seconds": 120
  },

  "memories_session_id": "optional-chat-session-id-for-continuity"
}
```

### 3.3 storyboard.json Schema

```json
{
  "iteration": 3,
  "target_duration_seconds": 120,
  "theme": "AI announcements from Google I/O 2025",
  "narrative_arc": "Opening excitement → core announcements → closing vision",

  "shots": [
    {
      "id": "shot_001",
      "purpose": "Opening hook — packed auditorium, energy",
      "search_query": "large audience cheering at Google I/O keynote opening",
      "retrieved_clip": {
        "video_no": "abc123",
        "video_name": "keynote.mp4",
        "start_seconds": 42.0,
        "end_seconds": 56.0,
        "description": "Packed auditorium, crowd applauding as presenter walks on stage",
        "score": 0.91
      },
      "narration": "At Google I/O 2025, the biggest AI announcements of the year were about to unfold.",
      "priority": "narration",
      "duration_seconds": 14.0
    }
  ],

  "open_questions": [
    "What specific demo moments showed the Gemini multimodal capability?"
  ],

  "notes": "Need more coverage of the Project Astra demo — only one clip found so far."
}
```

---

## 4. Phase 1 — Comprehension (Lightweight)

**Goal:** Get just enough understanding of the footage to start planning. No scene-by-scene analysis.

### 4.1 Steps

1. **Check for existing session** — if `session.json` exists in the workspace and contains valid `video_no`(s), skip upload entirely.
2. **Upload** — call `MemoriesAiManager.upload_video_file()` (or `upload_video_url()` for GCS paths) to get `video_no`.
3. **Wait for indexing** — poll `wait_for_ready()` until status is `PARSE`. Timeout: 3600s.
4. **Get gist** — single Chat API call: `"Give me a broad overview of this video: what it's about, who appears, what major topics are covered, and roughly what time periods cover what content."`
5. **Save session** — write `session.json` with `video_no`, gist, and metadata.

### 4.2 Session Cache Logic

```python
# Pseudo-code
if workspace_exists and session.json has valid video_no:
    if memories.get_video_status(video_no) == "PARSE":
        logger.info("Resuming existing session, skipping upload")
        return load_session()

# Otherwise upload and index
video_no = await memories.upload_video_file(source_path)
await memories.wait_for_ready(video_no, timeout=3600)
gist = await memories.chat(video_no, GIST_PROMPT)
save_session(video_no, gist)
```

### 4.3 API Endpoint

```
POST /v2/index
Body: { "project_name": "googleio", "start_fresh": false }
Response: { "session_id": "googleio", "gist": "...", "status": "ready" }
```

---

## 5. Phase 2 — Iterative Planning Loop

**Goal:** Build a complete, high-quality storyboard by iteratively querying the video and refining the plan. Max 4–5 iterations.

### 5.1 Schemas

Each Gemini call in the loop is focused on a single task. The loop uses three distinct structured output schemas:

```python
# --- Call A: Decide what information is needed ---
class ChatTool(BaseModel):
    question: str        # Question to ask Memories.ai Chat API
    purpose: str         # Why this is needed for the plan

class SearchTool(BaseModel):
    query: str                   # Semantic search query
    purpose: str                 # What shot this fills
    target_duration_sec: float   # Desired clip length

class ToolCallPlan(BaseModel):
    reasoning: str               # Text analysis of current gaps (free-form)
    chat_calls: List[ChatTool]   # Questions to ask about the video
    search_calls: List[SearchTool]  # Clips to search for
    should_stop: bool            # True = plan is complete, skip tool calls

# --- Call B: Update storyboard with new evidence ---
class Shot(BaseModel):
    id: str
    purpose: str
    search_query: str
    retrieved_clip: Optional[RetrievedClip]
    narration: Optional[str]
    priority: Literal["narration", "clip_audio", "clip_video"]
    duration_seconds: float

class Storyboard(BaseModel):
    iteration: int
    target_duration_seconds: float
    theme: str
    narrative_arc: str
    shots: List[Shot]
    open_questions: List[str]   # Remaining uncertainties, carried to next iter
    notes: str
```

### 5.2 Loop Structure

Each iteration makes **two focused Gemini calls**: one to decide what tools to call (text reasoning + structured tool list), and one to update the storyboard after seeing the results. This mirrors the pattern used elsewhere in the codebase (e.g., `plan_questions.py` + `gather_information.py` + `plan_shots.py` as separate focused calls).

```python
MAX_ITERATIONS = 5
iteration = 0

while iteration < MAX_ITERATIONS:
    # --- Check for pause signal from dashboard ---
    if pause_event.is_set():
        user_input = await wait_for_user_input()
        accumulated_context += f"\n\n[USER INPUT at iteration {iteration}]: {user_input}"
        pause_event.clear()

    # --- Gemini Call A: Decide what tools to call ---
    # Input:  user_prompt + gist + accumulated_context + current_storyboard
    # Output: ToolCallPlan (reasoning + list of chat/search calls)
    tool_plan = await decide_tool_calls(
        user_prompt=user_prompt,
        gist=session.gist,
        context=accumulated_context,
        storyboard=current_storyboard,
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
    )

    await ws_broadcast({"type": "tool_plan", "iteration": iteration,
                        "reasoning": tool_plan.reasoning,
                        "calls": tool_plan.chat_calls + tool_plan.search_calls})

    if tool_plan.should_stop:
        break

    # --- Execute tool calls in parallel ---
    chat_results, search_results = await asyncio.gather(
        asyncio.gather(*[
            memories.chat(video_nos, call.question, session_id=session.memories_session_id)
            for call in tool_plan.chat_calls
        ]),
        asyncio.gather(*[
            memories.search(call.query, video_nos, top_k=5)
            for call in tool_plan.search_calls
        ]),
    )

    await ws_broadcast({"type": "tool_results", "iteration": iteration,
                        "chat": format_chat_results(chat_results),
                        "clips_found": len(flatten(search_results))})

    # Append all new evidence to context
    new_evidence = format_evidence(tool_plan, chat_results, search_results)
    accumulated_context += f"\n\n--- Iteration {iteration} Evidence ---\n{new_evidence}"
    clips_so_far = merge_and_deduplicate(clips_so_far, search_results)

    # --- Gemini Call B: Update storyboard with new evidence ---
    # Input:  user_prompt + gist + full accumulated_context (now includes new evidence)
    #         + current_storyboard + all retrieved clips
    # Output: Storyboard (structured, replaces current_storyboard)
    current_storyboard = await update_storyboard(
        user_prompt=user_prompt,
        gist=session.gist,
        context=accumulated_context,
        storyboard=current_storyboard,
        all_clips=clips_so_far,
        iteration=iteration,
    )

    await ws_broadcast({"type": "storyboard_updated", "iteration": iteration,
                        "storyboard": current_storyboard})

    save_iteration_snapshot(iteration, tool_plan, current_storyboard)
    iteration += 1

save_storyboard(current_storyboard)
save_clips(clips_so_far)
save_context(accumulated_context)
```

### 5.3 Gemini Call A — Decide Tool Calls

**Task:** Given the current state, identify what information is still missing and produce a concrete list of Chat questions and Search queries to fill the gaps.

**Output schema:** `ToolCallPlan`

**Memories.ai Tool Documentation (injected into system prompt)**

Gemini must understand what each Memories.ai tool does, what inputs it expects, and what it returns. This is injected as a static block in the Call A system prompt:

```
MEMORIES.AI TOOLS AVAILABLE:

--- CHAT API ---
Use for: Understanding what happens in the video, who appears, what is said,
when events occur, narrative context, relationships between moments.
Input: A natural language question about the video.
Returns: A free-text answer synthesized from the video's indexed content.
         May include rough timestamp references (treat as approximate guides,
         not frame-accurate — use Search API for precise clip retrieval).

Good Chat questions:
  "What are the top 3 most energetic or surprising moments in this video?"
  "Who are the main speakers and when do they appear?"
  "Describe what happens during the demo of [feature name]."
  "What does the presenter say about [topic] and approximately when?"
  "What is the emotional tone of the opening 5 minutes?"
  "Are there any moments where the audience reacts strongly?"

Bad Chat questions (too vague or mismatched to video content):
  "What is this video about?" (use the gist instead)
  "Give me a clip from 00:42" (use Search API for retrieval)
  "List all scenes" (too broad, use for specific information gaps only)

--- SEARCH API ---
Use for: Finding specific clips matching a visual or audio description.
         Returns clips with accurate start/end timestamps and relevance scores.
Input: A short, concrete semantic query describing what should be seen or heard.
       Write as if describing what a viewer would observe, not an abstract concept.
Returns: List of clips with video_no, start_seconds, end_seconds, score, description.
         Timestamps are accurate and suitable for FCPXML use.

Good Search queries:
  "presenter standing at podium on stage in front of large audience"
  "phone screen showing AI assistant responding to spoken question"
  "two people laughing and reacting with surprise"
  "aerial drone shot of city skyline at night"
  "close-up of hands typing on laptop keyboard"
  "product demo showing new feature being demonstrated on screen"

Bad Search queries (too abstract, too long, or narrative-level):
  "the most exciting moment" (not visual/concrete)
  "AI announcement that surprised everyone" (too abstract)
  "when the presenter talks about the future of AI and what it means" (too long)

Use Chat API to understand what you're looking for.
Use Search API to actually retrieve the clip once you know what it looks like.
```

**Prompt structure:**
```
SYSTEM: You are a video editor planning an edit. You have two tools:
the Memories.ai Chat API (for understanding) and Search API (for clip retrieval).
Review the current storyboard, identify gaps, and output the tool calls needed
to fill them. Do NOT update the storyboard — only produce the tool call list.

{MEMORIES_AI_TOOL_DOCS}  ← static block above, always included

USER PROMPT: {user_prompt}
VIDEO GIST: {gist}
ACCUMULATED CONTEXT: {accumulated_context}
CURRENT STORYBOARD: {storyboard_json}
RETRIEVED CLIPS SO FAR: {clips_summary}

This is iteration {n} of {max}.
{"This is your last iteration — only call tools if truly critical." if n == max-1 else ""}

Output a ToolCallPlan. Set should_stop=true if the storyboard is already
complete and no further information is needed.
```

> **Implementation note:** `MEMORIES_AI_TOOL_DOCS` should live in
> `src/pipelines/v2/planning/planning_prompts.py` as a module-level constant,
> separate from the dynamic parts of the prompt. Update it if the Memories.ai
> API changes (e.g., new endpoints, updated response format).

### 5.4 Gemini Call B — Update Storyboard

**Task:** Given all accumulated context (including the new evidence just retrieved), produce an updated storyboard. Assign retrieved clips to shots, refine narration, update the narrative arc.

**Output schema:** `Storyboard`

**Prompt structure:**
```
SYSTEM: You are a video editor. Using all available context and the retrieved
clips, produce an updated storyboard. Assign the best available clip to each
shot. Write narration text for each shot. Arrange shots for narrative coherence.

USER PROMPT: {user_prompt}
VIDEO GIST: {gist}
ALL CONTEXT (including new evidence): {accumulated_context}
PREVIOUS STORYBOARD: {storyboard_json}
ALL RETRIEVED CLIPS: {clips_detail}

Output a complete updated Storyboard. For shots where no good clip was found,
keep them in open_questions for the next iteration.
```

### 5.5 Clip Anti-Clustering (carried forward from v1)

After each search batch, apply:
- **Score threshold:** drop clips with score < 0.3
- **Overlap merge:** merge clips with >50% time overlap, keep higher-scored
- **Temporal diversity:** if >30% of clips cluster in a 2-minute window, prune to top-scored
- **Minimum gap:** enforce 2+ second gap between consecutive clips from the same file

### 5.6 Session Continuity

On resume (workspace already has `storyboard.json`, `clips.json`, `context.md`):
- Load existing state into loop
- If new user prompt provided, inject it as context and continue iterating
- If `iteration_count` >= MAX_ITERATIONS, start a fresh planning round from the existing storyboard

### 5.7 API Endpoint

```
POST /v2/plan
Body: {
  "project_name": "googleio",
  "prompt": "Create a 2-minute highlight reel of AI announcements",
  "target_duration_seconds": 120,
  "max_iterations": 5
}
Response: stream of { "type": "iteration" | "complete", ... }
```

---

## 6. Phase 3 — FCPXML Generation

**Goal:** Convert the finalized storyboard + retrieved clips into a valid, importable FCPXML 1.10 file. Source footage is referenced directly (no re-encoding).

### 6.1 Media Preparation

Before FCPXML generation, prepare the workspace `media/` directory:

```python
for clip in storyboard.shots:
    source = clip.retrieved_clip.source_path
    link = workspace / "media" / Path(source).name
    if not link.exists():
        link.symlink_to(Path(source).resolve())
```

FCPXML asset paths will be `file:///absolute/path/to/media/video.mp4`. Using symlinks keeps the media co-located without copying large files.

### 6.2 FCPXML Generation Agent

A dedicated Gemini call with the full `fcpxml_formatting_guide.md` as system context:

```python
fcpxml_prompt = f"""
Generate a valid FCPXML 1.10 file for the following storyboard.

REQUIREMENTS:
- Target format: 1920x1080, {fps}fps
- tcStart: 3600/1s (01:00:00:00)
- All time values as rational fractions (e.g., 48/24s for 2 seconds at 24fps)
- Asset paths relative to media directory: {media_dir}
- Include audio role="dialogue" on primary clips
- If narration audio is provided, add it as lane="-1" (below spine) with role="music.narration"
- If music audio is provided, add it as lane="-2" with role="music" and adjust-volume="-12dB"
- Add a cross dissolve transition (12 frames) between each primary clip
- Subtitles as <caption> elements if narration text is provided

STORYBOARD:
{storyboard_json}

CLIPS:
{clips_detail}

NARRATION AUDIO: {narration_path or "none"}
MUSIC AUDIO: {music_path or "none"}
"""
```

### 6.3 FCPXML Self-Review Loop

After initial generation, run up to 3 review passes:

```python
MAX_FCPXML_ITERATIONS = 3

for i in range(MAX_FCPXML_ITERATIONS):
    fcpxml_str = await gemini.generate_fcpxml(prompt)

    # Structural validation
    errors = validate_fcpxml(fcpxml_str)  # Check refs, time math, required attrs

    if not errors:
        break

    # Feed errors back to Gemini for correction
    prompt += f"\n\nFIX THESE ERRORS:\n" + "\n".join(errors)

save_fcpxml(fcpxml_str, version=i+1)
```

### 6.4 FCPXML Validation Checks

```python
def validate_fcpxml(xml_str: str) -> List[str]:
    errors = []
    root = parse_xml(xml_str)

    # All ref= values must resolve to a resource id=
    # All time values must have 's' suffix
    # duration must be > 0 on all clips
    # offset must be monotonically increasing on spine
    # asset media-rep src must point to an existing file
    # Total spine duration must match sequence duration attribute

    return errors
```

### 6.5 API Endpoint

```
POST /v2/generate_fcpxml
Body: { "project_name": "googleio" }
Response: { "fcpxml_path": "data/workspaces/googleio/fcpxml/edit_final.fcpxml" }
```

---

## 7. Phase 4 — Narration and Music (On-Demand)

Narration and music are not part of the planning loop. They are requested after the plan is formed, either by natural language prompt or by button press on the dashboard.

### 7.1 Narration

**Trigger:** `POST /v2/narration` or user types "add voiceover" in dashboard

**Steps:**
1. Extract narration text from `storyboard.shots[*].narration`
2. If narration text is missing/thin, ask Gemini to write it from context and shot descriptions
3. Send to ElevenLabs TTS (English) or Minimax (Chinese) — language detected from storyboard
4. Save audio to `workspace/narration/narration.mp3`
5. Regenerate FCPXML to embed narration audio (lane="-1", role="music.narration")

**Sync check:** Each narration segment is estimated at ~140 words/minute speaking rate. If a segment's estimated speaking time exceeds the clip duration by more than 2x, Gemini is asked to shorten it before TTS.

### 7.2 Music

**Trigger:** `POST /v2/music` or user types "add background music" in dashboard

**Steps:**
1. Gemini selects a mood/genre based on storyboard theme (or user specifies directly)
2. Query Soundstripe API for matching track
3. Trim/loop to match total storyboard duration
4. Save to `workspace/music/track.mp3`
5. Regenerate FCPXML to embed music (lane="-2", role="music", volume="-12dB")
6. Optionally: snap-to-beat for clip transitions (carried forward from v1)

### 7.3 Removing/Swapping

Both narration and music can be removed or swapped at any time:
- Remove: regenerate FCPXML without the relevant audio lane
- Swap: repeat the generation step, then regenerate FCPXML

---

## 8. Phase 5 — DaVinci Resolve Rendering

### 8.1 Setup Requirements

**License:** DaVinci Resolve **Studio** is required. The free version does not support external Python scripting via IPC.

**macOS (development):**
```bash
# Add to shell profile or launchd plist
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

# Launch Resolve headless (can be a launchd daemon)
/Applications/DaVinci\ Resolve/DaVinci\ Resolve.app/Contents/MacOS/resolve -nogui &
```

**Linux (production):**
```bash
# Virtual framebuffer required even in -nogui mode
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# GPU drivers must be installed
# Resolve must be installed: .run installer from Blackmagic

DISPLAY=:99 /opt/resolve/bin/resolve -nogui &
```

**Setup script:** `lib/utils/resolve_setup.py` — checks env vars, verifies connection, validates Studio license.

### 8.2 Render Workflow

```python
import DaVinciResolveScript as dvr_script  # or pydavinci

async def render_with_resolve(
    fcpxml_path: str,
    media_dir: str,
    output_path: str,
    quality: Literal["preview", "final"] = "preview"
) -> str:
    resolve = dvr_script.scriptapp("Resolve")
    pm = resolve.GetProjectManager()

    project_name = f"vea_{uuid4().hex[:8]}"
    project = pm.CreateProject(project_name)
    media_pool = project.GetMediaPool()

    # Import FCPXML
    timeline = media_pool.ImportTimelineFromFile(fcpxml_path, {
        "timelineName": "VEA Edit",
        "importSourceClips": True,
        "sourceClipsPath": media_dir,
    })
    project.SetCurrentTimeline(timeline)

    # Configure render
    if quality == "preview":
        project.LoadRenderPreset("H.264 Master")   # fast, compressed
        project.SetRenderSettings({
            "TargetDir": str(Path(output_path).parent),
            "CustomName": Path(output_path).stem,
            "ExportVideo": True,
            "ExportAudio": True,
        })
    else:  # final
        project.LoadRenderPreset("ProRes 422")     # high quality
        project.SetRenderSettings({
            "TargetDir": str(Path(output_path).parent),
            "CustomName": Path(output_path).stem,
        })

    job_id = project.AddRenderJob()
    project.StartRendering(job_id)

    while project.IsRenderingInProgress():
        status = project.GetRenderJobStatus(job_id)
        progress = status.get("CompletionPercentage", 0)
        await ws_broadcast({"type": "render_progress", "progress": progress})
        await asyncio.sleep(2)

    # Cleanup project
    project.DeleteAllRenderJobs()
    pm.DeleteProject(project_name)

    return output_path
```

### 8.3 Render Presets

| Quality | Codec | Use Case |
|---------|-------|----------|
| `preview` | H.264, 720p | Dashboard preview, quick check |
| `final` | ProRes 422 or H.264 Master, 1080p+ | Delivery |

### 8.4 API Endpoint

```
POST /v2/render
Body: { "project_name": "googleio", "quality": "preview" }
Response: stream of { "type": "progress", "pct": 42 } ... { "type": "done", "path": "..." }
```

### 8.5 Health Check

```
GET /v2/resolve/status
Response: { "running": true, "version": "19.1", "studio": true }
```

---

## 9. Web Dashboard

### 9.1 Architecture

- **Backend:** FastAPI (existing) extended with WebSocket endpoint and new v2 routes
- **Frontend:** React SPA (served as static files from FastAPI, or standalone dev server)
- **Real-time:** WebSocket connection per session for live updates

### 9.2 Dashboard Phases

**Phase A — Planning Monitor** (active during planning loop)

| Panel | Content |
|-------|---------|
| Top left | Accumulated knowledge: gist + Q&A context, rendered as readable text |
| Top right | Current storyboard: structured view of shots, narration, clips assigned |
| Bottom left | Tool call feed: live stream of "Asked: ...", "Searched for: ...", "Found clip at 00:42–01:10" |
| Bottom right | Footage access map: visual bar per source file showing which time ranges have been accessed |
| Top bar | Current iteration (e.g., "Iteration 3 / 5"), status, and **Pause** button |

**Pause flow:** User clicks Pause → dashboard sends `{"type": "pause"}` over WebSocket → backend sets a `pause_event` flag → planning loop checks flag at top of each iteration → loop suspends, backend sends `{"type": "paused"}` → dashboard shows text input for user prompt injection → user submits → backend resumes.

**Phase B — Timeline View** (active once FCPXML is generated)

- Timeline visualization parsed from FCPXML: clips displayed as blocks on tracks with durations and labels
- Audio tracks visible (narration, music) as separate lanes
- Buttons: **Preview Render**, **Final Render**, **Add Narration**, **Add Music**, **Pause & Edit**
- Render progress shown inline as a progress bar

**Phase C — Post-Render**

- Embedded video player for preview renders
- Download button for FCPXML and final render
- Option to continue iterating (re-enters Phase A)

### 9.3 WebSocket Message Protocol

```typescript
// Server → Client
type ServerMessage =
  | { type: "iteration_start"; iteration: number; max: number }
  | { type: "tool_call"; tool: "chat" | "search"; query: string; purpose: string }
  | { type: "tool_result"; tool: "chat" | "search"; result: string; clips?: Clip[] }
  | { type: "storyboard_updated"; storyboard: Storyboard }
  | { type: "iteration_complete"; iteration: number; summary: string }
  | { type: "planning_complete"; storyboard: Storyboard }
  | { type: "fcpxml_ready"; path: string }
  | { type: "render_progress"; pct: number }
  | { type: "render_complete"; path: string }
  | { type: "paused" }
  | { type: "error"; message: string }

// Client → Server
type ClientMessage =
  | { type: "pause" }
  | { type: "inject"; prompt: string }
  | { type: "resume" }
```

---

## 10. CLI Usage

All operations are also available via CLI using the same API:

```bash
# Step 1: Index (lightweight — just upload + gist)
curl -X POST http://localhost:8001/v2/index \
  -d '{"project_name": "googleio"}'

# Step 2: Plan (streams iteration updates)
curl -X POST http://localhost:8001/v2/plan \
  -d '{
    "project_name": "googleio",
    "prompt": "Create a 2-minute highlight reel of the AI announcements",
    "target_duration_seconds": 120
  }'

# Step 3: Generate FCPXML
curl -X POST http://localhost:8001/v2/generate_fcpxml \
  -d '{"project_name": "googleio"}'

# Optional: Add narration
curl -X POST http://localhost:8001/v2/narration \
  -d '{"project_name": "googleio"}'

# Optional: Add music
curl -X POST http://localhost:8001/v2/music \
  -d '{"project_name": "googleio", "mood": "inspirational"}'

# Step 4: Render
curl -X POST http://localhost:8001/v2/render \
  -d '{"project_name": "googleio", "quality": "preview"}'

# Check Resolve status
curl http://localhost:8001/v2/resolve/status
```

---

## 11. Key Differences from v1

| Aspect | v1 (current) | v2 (this doc) |
|--------|-------------|---------------|
| Comprehension | 3-pass segment analysis (slow, ~20min) | Single Chat API call for gist (~30s) |
| Understanding | Static `scenes.json`, `story.json` built upfront | On-demand via Chat + Search during planning |
| Planning | Linear 6-step pipeline, no recovery | Iterative while-loop, Gemini directs tool calls |
| Edit format | Python-assembled video (moviepy/ffmpeg) | FCPXML 1.10 referencing source footage directly |
| Render engine | ffmpeg | DaVinci Resolve (Studio) via Python IPC |
| Narration timing | Fixed in pipeline | On-demand, after plan is approved |
| Music | Fixed in pipeline | On-demand, after plan is approved |
| Session continuity | None (each run starts fresh) | Full workspace persistence |
| User control | Submit prompt, wait for output | Pause/inject at any planning iteration |
| Output | `.mp4` + `.fcpxml` (side product) | `.fcpxml` (primary) + optional `.mp4` render |

---

## 12. Known Constraints and Open Issues

### DaVinci Resolve
- **Studio license required** — external Python scripting does not work with the free version
- **FCPXML max version:** Resolve supports up to 1.10; FCP supports up to 1.13. Author to 1.10 for compatibility with both
- **Resolve must be pre-running** as `-nogui` daemon — cannot be spawned on-demand per request
- **Linux headless:** requires virtual display (`Xvfb`) + GPU drivers even in `-nogui` mode
- **No output path returned from render API** — must reconstruct from configured `TargetDir` + `CustomName` + extension
- **No parallelism** — single Resolve instance per machine; concurrent render requests must be queued

### Memories.ai Chat API
- Using a `session_id` for continuity across Chat API calls within the same planning session avoids re-sending full video context each call
- Chat API timestamps in responses are sometimes approximate — use Search API for clip retrieval (not Chat API references) for accurate timestamps

### FCPXML Generation
- Gemini must be given the full `context/fcpxml_formatting_guide.md` on each generation call
- LLMs commonly produce float seconds (e.g., `2.0s`) instead of rational fractions (`48/24s`) — the validation pass must catch and correct this
- Connected clips, retiming, and multicam are advanced features that may require human review

### Web Dashboard
- The pause/inject mechanism requires WebSocket connection to remain open during the entire planning loop — handle reconnects gracefully
- Timeline visualization requires a lightweight FCPXML parser in JavaScript — consider using an existing XML parser with custom rendering logic

---

## 13. Implementation Roadmap

### Phase 0 — Foundation (Prerequisites)
- [ ] `lib/utils/resolve_setup.py` — Resolve health check + setup instructions
- [ ] FCPXML validator (`validate_fcpxml()`)
- [ ] Workspace manager (`WorkspaceManager` class — create, load, save all state)
- [ ] Update `MemoriesAiManager` to support `session_id` continuity across Chat calls

### Phase 1 — New Comprehension Endpoint
- [ ] `POST /v2/index` endpoint
- [ ] Session cache logic (skip re-upload if video_no valid)
- [ ] Gist extraction prompt

### Phase 2 — Iterative Planning Loop
- [ ] `POST /v2/plan` endpoint (streaming)
- [ ] Planning loop with Gemini tool-call structured output
- [ ] Pause/inject via WebSocket
- [ ] Iteration snapshot saving
- [ ] Clip anti-clustering post-processing

### Phase 3 — FCPXML Generation
- [ ] `POST /v2/generate_fcpxml` endpoint
- [ ] Gemini FCPXML generation with guide as context
- [ ] Self-review loop (up to 3 passes)
- [ ] FCPXML validator

### Phase 4 — Narration and Music
- [ ] `POST /v2/narration` endpoint (reuse existing ElevenLabs logic)
- [ ] `POST /v2/music` endpoint (reuse existing Soundstripe logic)
- [ ] FCPXML re-generation with audio lanes

### Phase 5 — DaVinci Resolve Render
- [ ] `POST /v2/render` endpoint
- [ ] `GET /v2/resolve/status` endpoint
- [ ] Resolve render script (`lib/utils/resolve_render.py`)
- [ ] Render queue (serialize concurrent requests)

### Phase 6 — Web Dashboard
- [ ] WebSocket endpoint (`WS /v2/session/{project_name}`)
- [ ] React SPA: Planning Monitor view
- [ ] React SPA: Timeline view (FCPXML visualization)
- [ ] React SPA: Post-render view

---

## 14. Files To Create (New)

```
src/
└── pipelines/
    └── v2/
        ├── comprehension/
        │   └── lightweight_comprehension.py   # Phase 1
        ├── planning/
        │   ├── iterative_planning_loop.py     # Phase 2 loop
        │   ├── planning_prompts.py            # Gemini prompts
        │   └── clip_postprocess.py            # Anti-clustering
        ├── fcpxml/
        │   ├── fcpxml_agent.py                # Phase 3 generation
        │   └── fcpxml_validator.py            # Structural validation
        └── workspace.py                       # Workspace manager

lib/
└── utils/
    ├── resolve_setup.py                       # Resolve health check
    └── resolve_render.py                      # Render workflow

dashboard/                                     # React SPA (Phase 6)
├── src/
│   ├── components/
│   │   ├── PlanningMonitor.tsx
│   │   ├── StoryboardView.tsx
│   │   ├── ToolCallFeed.tsx
│   │   ├── FootageAccessMap.tsx
│   │   ├── TimelineView.tsx
│   │   └── RenderControls.tsx
│   └── hooks/
│       └── useSession.ts                      # WebSocket hook

context/
└── fcpxml_formatting_guide.md                 # Already exists — used as Gemini context
```

---

## 15. References

- `context/fcpxml_formatting_guide.md` — FCPXML 1.10 authoring guide (Gemini context for generation)
- [Memories.ai API Docs](https://api-tools.memories.ai/platform/introduction)
- [DaVinci Resolve Scripting API (unofficial)](https://deric.github.io/DaVinciResolve-API-Docs/)
- [pydavinci Python wrapper](https://github.com/pedrolabonia/pydavinci)
- [FCPXML DTD Reference (Apple)](https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference)
- `xvu-infrastructure/docs/memories-ai-migration-plan.md` — prior internal migration plan (reference for clip anti-clustering logic and MaviQuerier patterns)
