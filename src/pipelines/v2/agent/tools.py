"""Tool declarations and implementations for the agentic editing session."""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai.types import FunctionDeclaration, Tool

from src.pipelines.v2.agent.scratchpad import ScratchpadManager

logger = logging.getLogger(__name__)

# ── Gemini function declarations ──────────────────────────────────────────────

TOOL_DECLARATIONS = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="ask_memories",
            description=(
                "Ask a natural-language question about the indexed video footage. "
                "Memories.ai has watched every frame and can answer questions about "
                "content, people, dialogue, visuals, timing, and structure. "
                "Returns a text answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the videos.",
                    },
                },
                "required": ["question"],
            },
        ),
        FunctionDeclaration(
            name="search_footage",
            description=(
                "Search for specific video clips matching a query. "
                "Returns clips with video_name, start/end timestamps, "
                "relevance score, and description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the clip you want.",
                    },
                    "target_duration_seconds": {
                        "type": "number",
                        "description": "Desired clip length in seconds. Default 5.",
                    },
                },
                "required": ["query"],
            },
        ),
        FunctionDeclaration(
            name="update_scratchpad",
            description=(
                "Modify one of your 4 persistent scratchpads. These survive the "
                "sliding window — they are your ONLY durable memory. "
                "Names: comprehension, creative_direction, planning, fcpxml. "
                "Operations: replace (overwrite), append (add to end), prepend (add to start)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["comprehension", "creative_direction", "planning", "fcpxml"],
                        "description": "Which scratchpad to update.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["replace", "append", "prepend"],
                        "description": "How to apply the content.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text to write.",
                    },
                },
                "required": ["name", "operation", "content"],
            },
        ),
        FunctionDeclaration(
            name="generate_fcpxml",
            description=(
                "Generate a Final Cut Pro XML timeline from a complete edit decision. "
                "Call this when the edit plan is finalized and clips are assigned. "
                "Provide the edit as a JSON string. The system compiles it deterministically "
                "to valid FCPXML 1.10."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "edit_decision_json": {
                        "type": "string",
                        "description": (
                            "A JSON string representing the full edit decision. Schema: "
                            '{"timeline": {"name": str, "fps": number, "width": int, "height": int}, '
                            '"clips": [{"id": str, "source_file": str, "source_start": number, '
                            '"source_end": number, "label": str, "description": str, "gain_db": number, '
                            '"speed": {"rate": number}, '
                            '"transition_after": {"type": "cross-dissolve"|"fade-in"|"fade-out", "duration_seconds": number}}], '
                            '"narration": [{"file": str, "timeline_offset": number, "start": number, '
                            '"duration": number, "gain_db": number}], '
                            '"music": {"file": str, "start": number, "duration": number, "gain_db": number}, '
                            '"titles": [{"text": str, "timeline_offset": number, "duration": number, "font_size": int}]}. '
                            "Only clips is required. timeline defaults to 24fps 1920x1080."
                        ),
                    },
                },
                "required": ["edit_decision_json"],
            },
        ),
        FunctionDeclaration(
            name="refine_clip_timestamps",
            description=(
                "Refine the in/out points of a clip to find the best segment within a larger range. "
                "Use this after search_footage returns a broad segment — this tool trims the source video, "
                "transcribes dialogue via Memories.ai, sends both the video and transcript to Gemini, "
                "and returns optimized start/end timestamps. Works for both dialogue-heavy and visual clips. "
                "The tool automatically decides whether to focus on visuals, dialogue, or audio cues."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Filename of the source video (must be in the footage directory).",
                    },
                    "source_start": {
                        "type": "number",
                        "description": "Current in-point in seconds (start of the broad segment).",
                    },
                    "source_end": {
                        "type": "number",
                        "description": "Current out-point in seconds (end of the broad segment).",
                    },
                    "target_duration": {
                        "type": "number",
                        "description": "Desired clip duration in seconds. The refined clip should be close to this length.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What makes a good clip here — e.g. 'Find the moment where the speaker announces the product' "
                            "or 'Best visual of the crowd reacting' or 'The clearest explanation of the feature'."
                        ),
                    },
                    "clip_description": {
                        "type": "string",
                        "description": "Brief description of what this clip is supposed to show in the edit.",
                    },
                },
                "required": ["source_file", "source_start", "source_end", "target_duration", "prompt"],
            },
        ),
        FunctionDeclaration(
            name="generate_narration",
            description=(
                "Generate narration voiceover audio from a script. Takes the narration script text "
                "and produces a single narration.mp3 file in the workspace. Call this ONLY after "
                "an edit plan exists (so you know clip durations). The user must have requested "
                "narration — do NOT call this unprompted. Returns the file path, duration, and a "
                "transcript with per-sentence timestamps ({text, start, end}). Use the transcript "
                "to align clips to the narration — adjust clip order and durations so visuals "
                "match what's being narrated at each moment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": (
                            "The full narration script to convert to speech. Write it as natural "
                            "spoken text — no stage directions, no shot labels. Pace at ~140 words/minute. "
                            "Use '...' for pauses between sections."
                        ),
                    },
                },
                "required": ["script"],
            },
        ),
        FunctionDeclaration(
            name="select_music",
            description=(
                "Search for and download a background music track. Fetches candidate tracks from "
                "the music library, uses an LLM to pick the best match based on your prompt, and "
                "downloads it. Returns the file path, track name, and duration. Use this when the "
                "user wants background music in their edit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Describe the ideal music — mood, energy, genre, instruments, tempo. "
                            "Be specific: 'upbeat electronic with synths, 120bpm, energetic but not aggressive' "
                            "is better than just 'upbeat'."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        FunctionDeclaration(
            name="message_user",
            description=(
                "Send a visible message to the user in the chat interface. "
                "Use this to share findings, propose plans, ask questions, or report progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to show to the user.",
                    },
                },
                "required": ["message"],
            },
        ),
    ]
)


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tool calls from Gemini and returns results.

    Each tool call emits events over the event_callback so the frontend
    can show live status.
    """

    def __init__(
        self,
        memories_manager,
        gemini_manager,
        workspace,
        scratchpads: ScratchpadManager,
        video_nos: List[str],
        event_callback=None,
    ):
        self.memories = memories_manager
        self.gemini = gemini_manager
        self.workspace = workspace
        self.scratchpads = scratchpads
        self.video_nos = video_nos
        self._emit = event_callback or (lambda *a, **kw: None)

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result dict."""
        try:
            if tool_name == "ask_memories":
                return await self._ask_memories(args)
            elif tool_name == "search_footage":
                return await self._search_footage(args)
            elif tool_name == "update_scratchpad":
                return self._update_scratchpad(args)
            elif tool_name == "generate_fcpxml":
                return await self._generate_fcpxml(args)
            elif tool_name == "refine_clip_timestamps":
                return await self._refine_clip_timestamps(args)
            elif tool_name == "generate_narration":
                return await self._generate_narration(args)
            elif tool_name == "select_music":
                return await self._select_music(args)
            elif tool_name == "message_user":
                return self._message_user(args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"[AGENT] Tool {tool_name} failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ── Individual tool implementations ───────────────────────────────────

    async def _ask_memories(self, args: Dict) -> Dict:
        question = args.get("question", "")
        logger.info(f"[AGENT] ask_memories: {question[:100]}")

        response = await self.memories.chat(
            video_nos=self.video_nos,
            prompt=question,
        )

        result = {
            "answer": response.text,
            "reference_count": len(response.ref_items),
        }

        # Include timestamped references if available
        if response.references:
            result["references"] = [
                {
                    "video_name": ref.video_name,
                    "timestamp": ref.timestamp,
                    "description": ref.description,
                }
                for ref in response.references[:10]  # cap at 10
            ]

        return result

    async def _search_footage(self, args: Dict) -> Dict:
        query = args.get("query", "")
        target_duration = args.get("target_duration_seconds", 5.0)
        logger.info(f"[AGENT] search_footage: {query[:100]}")

        raw_results = await self.memories.search(
            query,
            search_type="BY_CLIP",
            video_nos=self.video_nos,
        )

        # raw_results is either a list of dicts or a dict with an "items" key
        items = []
        if isinstance(raw_results, list):
            items = raw_results
        elif isinstance(raw_results, dict):
            items = raw_results.get("items", [])

        clips = []
        for item in items[:15]:  # cap results
            if not isinstance(item, dict):
                continue
            video_no = item.get("videoNo", "")
            start = float(item.get("startTime", 0))
            end_raw = item.get("endTime")
            end = float(end_raw) if end_raw else start + target_duration
            # Use target_duration if the returned clip is too short
            if end - start < target_duration:
                end = start + target_duration
            score = float(item.get("score", 0))

            clips.append({
                "video_no": video_no,
                "video_name": item.get("videoName", video_no),
                "start_seconds": start,
                "end_seconds": end,
                "score": score,
            })

        return {"clips": clips, "count": len(clips), "query": query}

    def _update_scratchpad(self, args: Dict) -> Dict:
        name = args.get("name", "")
        operation = args.get("operation", "replace")
        content = args.get("content", "")
        logger.info(f"[AGENT] update_scratchpad: {name} ({operation}, {len(content)} chars)")
        return self.scratchpads.update(name, operation, content)

    async def _generate_fcpxml(self, args: Dict) -> Dict:
        import json
        from src.pipelines.v2.schemas import EditDecision
        from src.pipelines.v2.fcpxml.edit_compiler import compile_edit_decision

        # Parse the JSON string from the tool call
        raw_json = args.get("edit_decision_json", "{}")
        try:
            edit_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}

        logger.info(f"[AGENT] generate_fcpxml: {len(edit_data.get('clips', []))} clips")

        # Validate and parse the EditDecision
        try:
            edit = EditDecision.model_validate(edit_data)
        except Exception as e:
            return {"error": f"Invalid EditDecision: {e}"}

        # Resolve source_path for clips that only have source_file
        footage_dir = self.workspace.get_footage_dir()
        footage_files = self.workspace.scan_footage() if footage_dir.is_dir() else []
        for clip in edit.clips:
            if not clip.source_path and clip.source_file:
                # Match by exact filename or substring
                for fp in footage_files:
                    if fp.name == clip.source_file or clip.source_file in fp.name or fp.name in clip.source_file:
                        clip.source_path = str(fp)
                        break

        # Save the EditDecision JSON for dashboard / debugging
        json_path = self.workspace.root / "fcpxml" / "edit_decision.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(edit.model_dump(), f, indent=2)

        # Compile to FCPXML
        fcpxml_path = str(self.workspace.get_fcpxml_path(version=1))
        try:
            output = compile_edit_decision(edit, fcpxml_path)
        except Exception as e:
            logger.error(f"[AGENT] FCPXML compilation failed: {e}", exc_info=True)
            return {
                "error": f"FCPXML compilation failed: {e}",
                "edit_decision_saved": str(json_path),
            }

        return {
            "status": "compiled",
            "fcpxml_path": output,
            "edit_decision_path": str(json_path),
            "clip_count": len(edit.clips),
            "narration_count": len(edit.narration),
            "has_music": edit.music is not None,
            "title_count": len(edit.titles),
        }

    def _message_user(self, args: Dict) -> Dict:
        message = args.get("message", "")
        # The event callback handles sending this to the frontend
        return {"delivered": True, "length": len(message)}

    async def _generate_narration(self, args: Dict) -> Dict:
        """Generate narration audio from a script using ElevenLabs TTS."""
        import os

        script = args.get("script", "")
        if not script.strip():
            return {"error": "Script is empty"}

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            return {"error": "ELEVENLABS_API_KEY not set — narration unavailable"}

        logger.info(f"[AGENT] generate_narration: {len(script)} chars")

        await self._emit("refine_progress", {
            "step": "generating_narration",
            "message": "Generating voiceover with ElevenLabs...",
        })

        # Save the script
        script_path = self.workspace.root / "narration" / "narration_script.txt"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        # Generate audio
        audio_path = self.workspace.root / "narration" / "narration.mp3"
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: _tts_sync(script, str(audio_path), api_key),
            )
        except Exception as e:
            return {"error": f"TTS generation failed: {e}"}

        # Get duration
        duration = await self._get_audio_duration(str(audio_path))

        # Build per-sentence transcript with estimated timestamps.
        # Pro-rate by word count so the agent can align clips to narration.
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script.strip()) if s.strip()]
        word_counts = [len(s.split()) for s in sentences]
        total_words = sum(word_counts) or 1
        transcript = []
        cursor = 0.0
        for sent, wc in zip(sentences, word_counts):
            sent_dur = duration * (wc / total_words)
            transcript.append({
                "text": sent,
                "start": round(cursor, 2),
                "end": round(cursor + sent_dur, 2),
                "word_count": wc,
            })
            cursor += sent_dur

        return {
            "status": "generated",
            "narration_path": str(audio_path),
            "script_length": len(script),
            "duration_seconds": duration,
            "word_count": len(script.split()),
            "transcript": transcript,
        }

    async def _select_music(self, args: Dict) -> Dict:
        """Fetch tracks from Soundstripe, use Gemini to pick best match, download it."""
        import os
        import requests

        prompt = args.get("prompt", "")
        if not prompt.strip():
            return {"error": "Prompt is empty"}

        soundstripe_key = os.environ.get("SOUNDSTRIPE_KEY")
        if not soundstripe_key:
            return {"error": "SOUNDSTRIPE_KEY not set — music selection unavailable"}

        logger.info(f"[AGENT] select_music: {prompt[:100]}")

        # Step 1: Fetch tracks from Soundstripe
        await self._emit("refine_progress", {
            "step": "fetching_tracks",
            "message": "Fetching tracks from music library...",
        })

        loop = asyncio.get_event_loop()
        tracks = await loop.run_in_executor(
            None, lambda: _fetch_soundstripe_tracks(soundstripe_key)
        )

        if not tracks:
            return {"error": "No tracks found from music library"}

        # Step 2: Build descriptions for Gemini to choose from
        track_descriptions = []
        for i, t in enumerate(tracks[:50]):  # Cap at 50 for context
            attrs = t.get("attributes", {})
            name = attrs.get("title", f"Track {i+1}")
            tags = attrs.get("tags", {})
            mood = ", ".join(tags.get("mood", []))
            genre = ", ".join(tags.get("genre", []))
            energy = attrs.get("energy", "")
            desc = attrs.get("description", "")
            bpm = attrs.get("bpm", "")
            track_descriptions.append(
                f"[{i}] {name} | mood: {mood} | genre: {genre} | "
                f"energy: {energy} | bpm: {bpm} | {desc[:100]}"
            )

        await self._emit("refine_progress", {
            "step": "selecting_track",
            "message": f"Choosing best track from {len(track_descriptions)} candidates...",
        })

        # Step 3: Ask Gemini to pick the best track
        selection_prompt = (
            f"You are selecting background music for a video edit.\n\n"
            f"## What the editor wants\n{prompt}\n\n"
            f"## Available tracks\n" + "\n".join(track_descriptions) + "\n\n"
            f"## Instructions\n"
            f"Return ONLY the index number (e.g. '3') of the single best matching track. "
            f"Nothing else — just the number."
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.gemini.LLM_request([selection_prompt], schema=None),
            )
            selected_idx = int(str(result).strip())
        except (ValueError, TypeError):
            # Fallback: pick first track
            logger.warning("[AGENT] Could not parse Gemini track selection, using first track")
            selected_idx = 0

        if selected_idx < 0 or selected_idx >= len(tracks[:50]):
            selected_idx = 0

        selected_track = tracks[selected_idx]
        track_attrs = selected_track.get("attributes", {})
        track_name = track_attrs.get("title", track_attrs.get("name", "Unknown"))

        # Step 4: Download the track
        await self._emit("refine_progress", {
            "step": "downloading_track",
            "message": f"Downloading: {track_name}...",
        })

        music_path = self.workspace.root / "music" / "track.mp3"
        music_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = await loop.run_in_executor(
            None,
            lambda: _download_soundstripe_track(selected_track, str(music_path)),
        )

        if not downloaded:
            return {"error": f"Failed to download track: {track_name}"}

        # Get duration
        duration = await self._get_audio_duration(str(music_path))

        tags = track_attrs.get("tags", {})
        return {
            "status": "downloaded",
            "music_path": str(music_path),
            "track_name": track_name,
            "mood": ", ".join(tags.get("mood", [])),
            "genre": ", ".join(tags.get("genre", [])),
            "duration_seconds": duration,
        }

    async def _get_audio_duration(self, path: str) -> Optional[float]:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ]
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, check=True),
            )
            return round(float(result.stdout.strip()), 2)
        except Exception:
            return None

    async def _refine_clip_timestamps(self, args: Dict) -> Dict:
        """
        Refine clip timestamps by sending both video and transcription to Gemini.

        Steps:
        1. Resolve source file path from footage directory
        2. Extract the video segment using ffmpeg
        3. Downsample it for Gemini (lower res, reasonable fps)
        4. Get dialogue transcription from Memories.ai
        5. Send video + transcription + prompt to Gemini with structured output
        6. Return refined start/end timestamps
        """
        from src.pipelines.v2.schemas import RefinedTimestamps

        source_file = args.get("source_file", "")
        source_start = float(args.get("source_start", 0))
        source_end = float(args.get("source_end", 0))
        target_duration = float(args.get("target_duration", 5.0))
        prompt = args.get("prompt", "")
        clip_description = args.get("clip_description", "")

        segment_duration = source_end - source_start
        if segment_duration <= 0:
            return {"error": "source_end must be greater than source_start"}

        # Resolve the source file path
        footage_dir = self.workspace.get_footage_dir()
        source_path = None
        if footage_dir.is_dir():
            for fp in self.workspace.scan_footage():
                if fp.name == source_file or source_file in fp.name or fp.name in source_file:
                    source_path = fp
                    break

        if not source_path or not source_path.exists():
            return {"error": f"Source file not found in footage: {source_file}"}

        logger.info(
            f"[AGENT] refine_clip_timestamps: {source_file} "
            f"[{source_start:.1f}–{source_end:.1f}] target={target_duration:.1f}s"
        )

        # Work in a temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="vea_refine_"))
        try:
            # Step 1: Extract the video segment
            await self._emit("refine_progress", {
                "step": "extracting",
                "message": f"Extracting {segment_duration:.1f}s segment...",
                "source_file": source_file,
            })
            segment_path = tmp_dir / "segment.mp4"
            extract_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(source_start),
                "-i", str(source_path),
                "-t", str(segment_duration),
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                str(segment_path),
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(extract_cmd, check=True)
            )

            # Step 2: Downsample for Gemini (480p, 2fps — enough for visual analysis)
            await self._emit("refine_progress", {
                "step": "downsampling",
                "message": "Downsampling for analysis...",
                "source_file": source_file,
            })
            downsampled_path = tmp_dir / "downsampled.mp4"
            downsample_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(segment_path),
                "-vf", "fps=2,scale=-2:480",
                "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "64k",
                str(downsampled_path),
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(downsample_cmd, check=True)
            )

            # Step 3: Fetch audio transcript from Memories.ai for the clip range
            clip_transcript = ""
            transcript_segments = []
            try:
                # Get full transcript for the video, filter to our time range
                video_no = self.video_nos[0] if self.video_nos else None
                if video_no:
                    all_segments = await self.memories.get_audio_transcription(video_no)
                    # Filter to segments overlapping our clip range
                    for seg in all_segments:
                        seg_start = float(seg.get("startTime", 0))
                        seg_end = float(seg.get("endTime", 0))
                        if seg_end >= source_start and seg_start <= source_end:
                            transcript_segments.append({
                                "text": seg.get("content", "").strip(),
                                "start": seg_start,
                                "end": seg_end,
                            })
                    if transcript_segments:
                        clip_transcript = "\n".join(
                            f"[{s['start']:.0f}s–{s['end']:.0f}s] {s['text']}"
                            for s in transcript_segments
                        )
                        logger.info(f"[AGENT] Got {len(transcript_segments)} transcript segments for clip")
            except Exception as e:
                logger.warning(f"[AGENT] Could not fetch transcript: {e}")

            # Step 4: Send to Gemini for analysis (Gemini watches the video + listens to audio directly)
            await self._emit("refine_progress", {
                "step": "analyzing",
                "message": "Gemini analyzing video + audio...",
                "source_file": source_file,
            })

            # Build the Gemini refinement prompt
            refinement_prompt = self._build_refinement_prompt(
                source_start=source_start,
                source_end=source_end,
                target_duration=target_duration,
                prompt=prompt,
                clip_description=clip_description,
                transcript=clip_transcript,
            )

            # Step 5: Call Gemini with video + prompt, structured output
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini.LLM_request(
                    prompt_contents=[downsampled_path, refinement_prompt],
                    schema=RefinedTimestamps,
                    context="refine_clip_timestamps",
                ),
            )

            # Parse result
            if isinstance(result, RefinedTimestamps):
                refined = result
            elif isinstance(result, dict):
                refined = RefinedTimestamps(**result)
            else:
                return {"error": f"Unexpected Gemini response type: {type(result)}"}

            # Gemini returns offsets relative to the trimmed video (0 to segment_duration).
            # Convert back to absolute source timestamps.
            new_start = source_start + max(0, min(refined.new_start, segment_duration))
            new_end = source_start + max(0, min(refined.new_end, segment_duration))
            # Ensure minimum clip length and correct order
            if new_end <= new_start:
                new_end = min(new_start + target_duration, source_end)
            new_end = min(new_end, source_end)

            logger.info(
                f"[AGENT] Refined: [{source_start:.1f}–{source_end:.1f}] → "
                f"[{new_start:.1f}–{new_end:.1f}] ({new_end - new_start:.1f}s) "
                f"focus={refined.focus_type}"
            )

            result = {
                "original_start": source_start,
                "original_end": source_end,
                "new_start": new_start,
                "new_end": new_end,
                "duration": new_end - new_start,
                "reasoning": refined.reasoning,
                "focus_type": refined.focus_type,
            }

            # Include transcript if available — helps agent align clips with narration
            if transcript_segments:
                result["transcript"] = transcript_segments

            return result

        finally:
            # Clean up temp files
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    @staticmethod
    def _build_refinement_prompt(
        source_start: float,
        source_end: float,
        target_duration: float,
        prompt: str,
        clip_description: str,
        transcript: str = "",
    ) -> str:
        """Build the Gemini prompt for timestamp refinement."""
        segment_duration = source_end - source_start

        transcript_section = ""
        if transcript:
            transcript_section = f"""
## Audio transcript (from Memories.ai)

The following is the dialogue/speech transcription for this segment. Timestamps are absolute
(relative to the full source video, not this excerpt). The excerpt starts at {source_start:.1f}s.

{transcript}

CRITICAL: Use the transcript to find natural speech boundaries. NEVER cut in the middle of a
sentence or word. Start just before a sentence begins and end after it completes with a
natural pause. The transcript timestamps help you align what you hear with precise timing.
"""

        return f"""\
You are a professional video editor refining clip timestamps. You are watching a video segment
and must choose the best start and end points for a clip.

## Context

You are watching a {segment_duration:.1f}-second video excerpt. The video you see starts at 0:00
and ends at {segment_duration:.1f}s. You need to find the best ~{target_duration:.1f}s window
within this segment.

IMPORTANT: Return your timestamps as offsets from the START of this video (0 to {segment_duration:.1f}).
For example, if the best moment starts 5 seconds into this video, return new_start=5.0.

## What this clip is for
{clip_description or "(No description provided)"}

## What to look for
{prompt}
{transcript_section}
## Instructions

Watch the video AND listen to the audio carefully. Use both to decide the best cut points.

- **Dialogue-heavy clips**: Listen for complete sentences and natural speech boundaries.
  Don't cut mid-sentence. Include brief pauses before/after key statements.

- **Visual clips** (b-roll, reactions, scenery): Prioritize visual composition, movement
  peaks, and natural motion boundaries. Cut on action — start when movement begins,
  end when it resolves.

- **Audio-driven clips** (music, applause, sound effects): Align cuts with audio beats,
  applause peaks, or natural audio transitions.

## Output

Return `new_start` and `new_end` as seconds from the beginning of this video (0 to {segment_duration:.1f}).
The clip duration should be close to {target_duration:.1f}s but can vary if a natural boundary
makes it slightly shorter or longer.

Set `focus_type` to "dialogue", "visual", or "audio" based on what primarily drove your decision."""


# ── Standalone helpers for narration & music ──────────────────────────────────


def _tts_sync(text: str, output_path: str, api_key: str) -> None:
    """Blocking ElevenLabs TTS call."""
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2_5",
    )
    with open(output_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)


def _fetch_soundstripe_tracks(api_key: str, page_count: int = 3) -> list:
    """Fetch tracks from Soundstripe API with audio_files sideloaded."""
    import requests
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/vnd.api+json",
    }
    all_tracks = []
    # Map audio_file id → mp3 URL from sideloaded includes
    audio_urls: dict = {}
    for page in range(1, page_count + 1):
        try:
            resp = requests.get(
                "https://api.soundstripe.com/v1/songs",
                headers=headers,
                params={
                    "page[size]": 50,
                    "page[number]": page,
                    "include": "audio_files",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                break
            body = resp.json()
            # Collect audio_file URLs from the `included` sideload
            for inc in body.get("included", []):
                if inc.get("type") == "audio_files":
                    versions = inc.get("attributes", {}).get("versions", {})
                    mp3_url = versions.get("mp3")
                    if mp3_url:
                        audio_urls[inc["id"]] = mp3_url
            all_tracks.extend(body.get("data", []))
        except Exception as e:
            logger.warning(f"[MUSIC] Soundstripe fetch error: {e}")
            break

    # Attach the resolved mp3 URL to each track for easy download later
    for track in all_tracks:
        audio_rels = track.get("relationships", {}).get("audio_files", {}).get("data", [])
        for af in audio_rels:
            url = audio_urls.get(af.get("id"))
            if url:
                track["_mp3_url"] = url
                break

    return all_tracks


def _download_soundstripe_track(track: dict, output_path: str) -> bool:
    """Download the mp3 for the selected track."""
    import requests
    url = track.get("_mp3_url")
    if not url:
        logger.warning(f"[MUSIC] No download URL found for track {track.get('id')}")
        return False

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"[MUSIC] Download failed: {e}")
        return False
