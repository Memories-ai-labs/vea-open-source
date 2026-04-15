"""Tool executor for the agentic editing session."""

import asyncio
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.v2.agent.scratchpad import ScratchpadManager
from src.pipelines.v2.agent.tool_definitions import TOOL_DECLARATIONS  # noqa: F401 — re-exported
from src.pipelines.v2.agent.tool_helpers import (
    generate_music_track,
    stt_word_timestamps,
    tts_sync,
    tts_with_timestamps,
)
from src.pipelines.v2.workspace import _atomic_write_json

logger = logging.getLogger(__name__)


def _refine_debug_log(workspace, entry: dict):
    """Append a JSON line to the workspace's refine debug log."""
    try:
        log_dir = workspace.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "refine_debug.jsonl"
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[AGENT] Failed to write refine debug log: {e}")


def _detect_scene_cuts(video_path: str, start_sec: float, end_sec: float) -> List[float]:
    """Detect shot boundaries in the ``[start_sec, end_sec]`` window of ``video_path``.

    Returns a list of cut timestamps in seconds, **relative to start_sec**. A "cut"
    is the start of the 2nd+ detected scene. An empty list means the window is one
    continuous shot (or detection failed). Uses PySceneDetect's ContentDetector with
    the same thresholds as ``src/pipelines/common/dynamic_cropping.py``.

    Called synchronously; callers should wrap in ``run_in_executor`` since
    PySceneDetect reads frames sequentially.
    """
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(
            video_path,
            ContentDetector(threshold=27.0, min_scene_len=12),
            start_time=float(start_sec),
            end_time=float(end_sec),
        )
    except Exception as e:
        logger.warning(f"[REFINE] Scene detection failed for {video_path}: {e}")
        return []

    cuts: List[float] = []
    for i, scene in enumerate(scenes):
        if i == 0:
            continue  # first scene starts at window start — not a real cut
        scene_start, _ = scene
        rel = scene_start.get_seconds() - start_sec
        if rel > 0.05:  # drop near-zero artifacts
            cuts.append(round(rel, 2))
    return cuts


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
        video_llm=None,
    ):
        self.memories = memories_manager
        # ``gemini`` is the main text LLM (may be Claude/GPT via OpenRouter).
        # ``video_llm`` is used for tasks that need native video input
        # (refine_clip_timestamps, verify_preview). If omitted, falls back to
        # ``gemini`` so older callers keep working.
        self.gemini = gemini_manager
        self.video_llm = video_llm or gemini_manager
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
            elif tool_name == "generate_subtitles":
                return await self._generate_subtitles(args)
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

        # Fetch transcript once so we can snap endpoints to sentence boundaries
        # and include dialogue context in results
        transcript_segments = await self._get_transcript_segments()

        clips = []
        for item in items[:15]:  # cap results
            if not isinstance(item, dict):
                continue
            video_no = item.get("videoNo", "")
            start = float(item.get("startTime", 0))
            end_raw = item.get("endTime")
            end = float(end_raw) if end_raw else start + target_duration

            # If clip is shorter than target, extend to nearest sentence boundary
            if end - start < target_duration:
                desired_end = start + target_duration
                end = self._snap_to_sentence_boundary(
                    desired_end, transcript_segments, direction="forward"
                )

            score = float(item.get("score", 0))

            # Slice transcript to this clip's time range so the agent can see
            # what dialogue exists before deciding to refine.
            # Use generous overlap: include any sentence that overlaps the clip
            # window at all, even if it starts well before or ends well after.
            # This catches sentences that straddle the clip boundary.
            clip_transcript = []
            for seg in transcript_segments:
                seg_text = seg.get("text", "").strip()
                # Skip empty or metadata-only segments
                if not seg_text or len(seg_text) < 3:
                    continue
                if seg_text.lower().startswith("transcription by"):
                    continue
                # Include if any part of the sentence overlaps the clip range
                # (sentence ends after clip starts AND sentence starts before clip ends)
                if seg["end"] > start and seg["start"] < end:
                    clip_transcript.append({
                        "text": seg_text,
                        "start": round(seg["start"], 1),
                        "end": round(seg["end"], 1),
                    })

            clip_data = {
                "video_no": video_no,
                "video_name": item.get("videoName", video_no),
                "start_seconds": start,
                "end_seconds": end,
                "score": score,
            }
            if clip_transcript:
                clip_data["transcript"] = clip_transcript
            clips.append(clip_data)

        return {"clips": clips, "count": len(clips), "query": query}

    async def _get_transcript_segments(self) -> List[Dict]:
        """Fetch and cache the full audio transcript for the first video."""
        if not hasattr(self, '_cached_transcript'):
            self._cached_transcript = []
            try:
                video_no = self.video_nos[0] if self.video_nos else None
                if video_no:
                    all_segs = await self.memories.get_audio_transcription(video_no)
                    self._cached_transcript = [
                        {
                            "text": seg.get("content", "").strip(),
                            "start": float(seg.get("startTime", 0)),
                            "end": float(seg.get("endTime", 0)),
                        }
                        for seg in all_segs
                    ]
            except Exception as e:
                logger.warning(f"[AGENT] Could not fetch transcript for caching: {e}")
        return self._cached_transcript

    @staticmethod
    def _snap_to_sentence_boundary(
        timestamp: float,
        transcript_segments: List[Dict],
        direction: str = "forward",
        max_drift: float = 3.0,
    ) -> float:
        """Snap a timestamp to the nearest sentence boundary.

        direction="forward": find the end of the sentence at or after timestamp
        direction="backward": find the start of the sentence at or before timestamp
        """
        if not transcript_segments:
            return timestamp

        best = timestamp
        best_dist = max_drift + 1

        for seg in transcript_segments:
            if direction == "forward":
                boundary = seg["end"]
                if boundary >= timestamp and boundary - timestamp <= max_drift:
                    dist = boundary - timestamp
                    if dist < best_dist:
                        best = boundary + 0.3  # small buffer after sentence
                        best_dist = dist
            else:  # backward
                boundary = seg["start"]
                if boundary <= timestamp and timestamp - boundary <= max_drift:
                    dist = timestamp - boundary
                    if dist < best_dist:
                        best = max(0, boundary - 0.2)  # small buffer before sentence
                        best_dist = dist

        return best

    def _update_scratchpad(self, args: Dict) -> Dict:
        name = args.get("name", "")
        operation = args.get("operation", "replace")
        content = args.get("content", "")
        logger.info(f"[AGENT] update_scratchpad: {name} ({operation}, {len(content)} chars)")
        return self.scratchpads.update(name, operation, content)

    async def _generate_fcpxml(self, args: Dict) -> Dict:
        import json
        from pathlib import Path
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
        generated_dir = self.workspace.root / "generated"
        generated_files = list(generated_dir.glob("*.mp4")) if generated_dir.is_dir() else []
        all_sources = footage_files + generated_files
        for clip in edit.clips:
            if not clip.source_path and clip.source_file:
                # Match by exact filename or substring
                for fp in all_sources:
                    if fp.name == clip.source_file or clip.source_file in fp.name or fp.name in clip.source_file:
                        clip.source_path = str(fp)
                        break

        # Validate clip source ranges against actual file durations.
        # Catches the agent fabricating timestamps past file end (a real bug we've seen).
        # Probe each unique source file once via ffprobe and cache.
        import subprocess
        file_durations: Dict[str, float] = {}
        def _probe_dur(path: str) -> float:
            if path in file_durations:
                return file_durations[path]
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", path],
                    capture_output=True, text=True, timeout=10,
                )
                d = float(out.stdout.strip()) if out.stdout.strip() else 0.0
            except Exception:
                d = 0.0
            file_durations[path] = d
            return d

        validation_errors: List[str] = []
        valid_clips = []
        for clip in edit.clips:
            if not clip.source_path:
                # Can't validate without a path; let it through and let the renderer fail
                valid_clips.append(clip)
                continue
            actual_dur = _probe_dur(clip.source_path)
            if actual_dur <= 0:
                valid_clips.append(clip)
                continue
            if clip.source_start >= actual_dur:
                validation_errors.append(
                    f"❌ Clip `{clip.id}` source_start={clip.source_start}s is past the end of "
                    f"`{clip.source_file}` (duration {actual_dur}s). Drop this clip or pick a "
                    f"different timestamp from search_footage."
                )
                continue  # drop the clip — completely out of bounds
            if clip.source_end > actual_dur:
                # Clamp source_end to file duration; warn but allow
                old_end = clip.source_end
                clip.source_end = round(actual_dur, 3)
                logger.warning(
                    f"[AGENT] Clamped clip {clip.id} source_end {old_end} → {clip.source_end} "
                    f"(file {clip.source_file} is only {actual_dur}s)"
                )
                validation_errors.append(
                    f"⚠️ Clip `{clip.id}` source_end={old_end}s was clamped to {clip.source_end}s "
                    f"(file `{clip.source_file}` is only {actual_dur}s)."
                )
            if clip.source_end <= clip.source_start:
                validation_errors.append(
                    f"❌ Clip `{clip.id}` has source_end ({clip.source_end}) <= source_start "
                    f"({clip.source_start}) after clamping. Drop this clip."
                )
                continue
            valid_clips.append(clip)

        if validation_errors:
            edit.clips = valid_clips
            logger.warning(f"[AGENT] generate_fcpxml: {len(validation_errors)} validation issues")

        # Validate narration segment boundaries against the persisted word grid
        # (saved by generate_narration). Prevents the LLM from picking mid-word
        # start/end times that produce choppy playback.
        if edit.narration:
            words_path = self.workspace.root / "narration" / "words.json"
            if words_path.exists():
                try:
                    word_grid = json.loads(words_path.read_text())
                    starts = {round(float(w["start"]), 3) for w in word_grid}
                    ends = {round(float(w["end"]), 3) for w in word_grid}
                    eps = 0.05  # tolerate up to 50ms of rounding slop
                    for seg in edit.narration:
                        s = round(float(seg.start), 3)
                        e = round(float(seg.start + seg.duration), 3)
                        start_ok = any(abs(s - ws) <= eps for ws in starts)
                        end_ok = any(abs(e - we) <= eps for we in ends)
                        if not (start_ok and end_ok):
                            validation_errors.append(
                                f"⚠️ Narration segment at {seg.timeline_offset:.2f}s "
                                f"[{s:.3f}–{e:.3f}] does not align to word boundaries. "
                                f"Snap to a nearby word start/end from the words array to "
                                f"avoid mid-word cuts."
                            )
                except Exception as e:
                    logger.warning(f"[AGENT] Narration boundary check skipped: {e}")

        # Note: auto beat-snap used to run here. It's now advisory — the agent
        # gets beat times + tempo back from select_music and decides itself
        # which clip boundaries to align with beats. This keeps editorial
        # control (e.g. deliberate off-beat tension cuts) in the model's hands
        # instead of silently shifting every clip's source_end by up to 0.5s.
        beat_metadata = None

        # Save the EditDecision JSON for dashboard / debugging
        json_path = self.workspace.root / "fcpxml" / "edit_decision.json"
        _atomic_write_json(json_path, edit.model_dump())

        # Compile to FCPXML
        fcpxml_path = str(self.workspace.get_fcpxml_path(version=1))
        try:
            output = compile_edit_decision(edit, fcpxml_path, workspace_root=self.workspace.root)
        except Exception as e:
            logger.error(f"[AGENT] FCPXML compilation failed: {e}", exc_info=True)
            return {
                "error": f"FCPXML compilation failed: {e}",
                "edit_decision_saved": str(json_path),
            }

        # Include loudness data from previous measurements if available
        loudness_info = []
        for c in edit.clips:
            if c.measured_loudness_lufs is not None:
                loudness_info.append(
                    f"{c.id} ({c.label}): {c.measured_loudness_lufs} LUFS, gain_db={c.gain_db}"
                )

        result = {
            "status": "compiled",
            "fcpxml_path": output,
            "edit_decision_path": str(json_path),
            "clip_count": len(edit.clips),
            "narration_count": len(edit.narration),
            "has_music": edit.music is not None,
            "title_count": len(edit.titles),
        }
        if loudness_info:
            result["loudness_measurements"] = loudness_info
            if edit.music and edit.music.measured_loudness_lufs is not None:
                result["music_loudness"] = f"{edit.music.measured_loudness_lufs} LUFS, gain_db={edit.music.gain_db}"
        if beat_metadata:
            result["beat_sync"] = {
                "tempo_bpm": beat_metadata.get("tempo_bpm", 0),
                "clips_snapped": beat_metadata.get("snapped", 0),
                "total_beats": beat_metadata.get("beats_detected", 0),
            }
        if validation_errors:
            result["validation_warnings"] = validation_errors
            result["status"] = "compiled_with_warnings"
        return result

    async def _verify_preview(self, args: Dict) -> Dict:
        """Watch the rendered preview and return a professional video critique."""
        import shutil

        focus = args.get("focus", "")

        # Find latest render
        render_dir = self.workspace.root / "renders"
        render_path = render_dir / "preview.mp4"
        if not render_path.exists():
            # Try any mp4 in renders
            mp4s = sorted(render_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if render_dir.exists() else []
            if mp4s:
                render_path = mp4s[0]
            else:
                return {"error": "No rendered preview found. Generate FCPXML and render first."}

        logger.info(f"[AGENT] verify_preview: analyzing {render_path} (focus: {focus or 'general'})")
        await self._emit("tool_progress", {
            "tool": "verify_preview",
            "step": "preparing",
            "message": "Preparing preview for analysis...",
        })

        # Downsample for cost efficiency (2fps, 480p, CRF 30)
        tmp_dir = Path(tempfile.mkdtemp(prefix="vea_verify_"))
        downsampled = tmp_dir / "verify_preview.mp4"

        try:
            ds_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(render_path),
                "-vf", "fps=2,scale=-2:480",
                "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "64k",
                str(downsampled),
            ]
            proc = await asyncio.create_subprocess_exec(
                *ds_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(f"[AGENT] verify_preview downsample failed: {stderr.decode()}")
                # Fall back to original file
                downsampled = render_path

            await self._emit("tool_progress", {
                "tool": "verify_preview",
                "step": "analyzing",
                "message": "Gemini is watching the preview...",
            })

            # Load current edit decision for context
            edit_context = ""
            ed_path = self.workspace.root / "fcpxml" / "edit_decision.json"
            if ed_path.exists():
                try:
                    with open(ed_path) as f:
                        ed_data = json.load(f)
                    clip_summary = []
                    for i, c in enumerate(ed_data.get("clips", [])):
                        dur = c.get("source_end", 0) - c.get("source_start", 0)
                        clip_summary.append(
                            f"  {i+1}. {c.get('label', c.get('id', '?'))} "
                            f"({dur:.1f}s) — {c.get('description', 'no description')}"
                        )
                    edit_context = (
                        f"\n## Current edit structure ({len(clip_summary)} clips)\n"
                        + "\n".join(clip_summary)
                    )
                except Exception:
                    pass

            focus_section = ""
            if focus:
                focus_section = f"\n## Specific focus requested\n{focus}\n"

            verification_prompt = f"""\
You are a professional video editor reviewing a rendered edit. Watch the attached video \
carefully — both visuals AND audio — and provide a detailed critique.

## Evaluation dimensions
- **Pacing & rhythm**: Does the edit flow naturally? Are clips too long/short?
- **Transitions**: Are cuts smooth? Do they feel motivated or jarring?
- **Visual coherence**: Do shots work together? Any awkward juxtapositions?
- **Audio mix**: Is dialogue/narration clear? Music balanced? Any level issues?
- **Narrative flow**: Does it tell a coherent story? Are there dead spots?
- **Technical quality**: Artifacts, color shifts, audio glitches, sync issues?
{focus_section}{edit_context}

## Response format

Structure your critique as:

**STRENGTHS** — What's working well (be specific with timestamps)

**ISSUES** — Problems ranked by severity. For each:
- Timestamp or clip number
- What's wrong
- Suggested fix (e.g. "shorten clip 3 by 2s", "add dissolve between clips 2-3")

**PRIORITY FIXES** — Top 3 changes that would most improve the edit

Be concise but specific. Reference timestamps (MM:SS) when possible."""

            loop = asyncio.get_event_loop()
            critique = await loop.run_in_executor(
                None,
                lambda: self.video_llm.LLM_request(
                    prompt_contents=[Path(str(downsampled)), verification_prompt],
                    schema=None,
                    context="verify_preview",
                ),
            )

            logger.info(f"[AGENT] verify_preview complete: {len(str(critique))} chars")

            return {
                "status": "reviewed",
                "render_file": render_path.name,
                "critique": str(critique),
            }

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    async def _generate_subtitles(self, args: Dict) -> Dict:
        """Generate subtitles by transcribing original audio from each clip."""
        import os
        import shutil

        max_words = args.get("max_words_per_line", 8)
        font_size = args.get("font_size", 48)

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            return {"error": "ELEVENLABS_API_KEY not set — subtitles unavailable"}

        # Load current edit decision
        json_path = self.workspace.root / "fcpxml" / "edit_decision.json"
        if not json_path.exists():
            return {"error": "No edit decision found. Generate an edit first."}

        from src.pipelines.v2.schemas import EditDecision, TextOverlay
        edit = EditDecision.model_validate_json(json_path.read_text())

        if not edit.clips:
            return {"error": "Edit has no clips."}

        await self._emit("refine_progress", {
            "step": "generating_subtitles",
            "message": "Transcribing clip audio for subtitles...",
        })

        # Compute timeline offset for each clip
        clip_tl_offsets = []
        cursor = 0.0
        for clip in edit.clips:
            clip_tl_offsets.append(cursor)
            cursor += (clip.source_end - clip.source_start)

        subtitles: list = []
        tmp_dir = tempfile.mkdtemp(prefix="vea_stt_")

        try:
            for ci, clip in enumerate(edit.clips):
                tl_offset = clip_tl_offsets[ci]
                clip_dur = clip.source_end - clip.source_start

                # Find source footage file
                footage_dir = self.workspace.root / "footage"
                src_path = footage_dir / clip.source_file
                if not src_path.exists():
                    logger.warning(f"[SUBTITLES] Source not found: {src_path}")
                    continue

                await self._emit("refine_progress", {
                    "step": "transcribing_clip",
                    "message": f"Transcribing clip {ci + 1}/{len(edit.clips)}: {clip.label or clip.source_file}",
                })

                # Extract audio segment from source
                audio_out = Path(tmp_dir) / f"clip_{ci}.wav"
                cmd = [
                    "ffmpeg", "-y", "-ss", str(clip.source_start),
                    "-t", str(clip_dur), "-i", str(src_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(audio_out),
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

                if not audio_out.exists() or audio_out.stat().st_size < 1000:
                    logger.warning(f"[SUBTITLES] Audio extraction failed for clip {ci}")
                    continue

                # Run STT
                try:
                    loop = asyncio.get_event_loop()
                    words = await loop.run_in_executor(
                        None,
                        lambda: stt_word_timestamps(str(audio_out), api_key),
                    )
                except Exception as e:
                    logger.warning(f"[SUBTITLES] STT failed for clip {ci}: {e}")
                    continue

                if not words:
                    continue

                # Group words into subtitle lines
                line_words: list = []
                line_start = 0.0
                for w in words:
                    if w.get("type") != "word":
                        continue
                    if not line_words:
                        line_start = w["start"]
                    line_words.append(w)

                    if len(line_words) >= max_words:
                        line_text = " ".join(lw["text"] for lw in line_words)
                        line_end = line_words[-1]["end"]
                        subtitles.append(TextOverlay(
                            text=line_text,
                            timeline_offset=tl_offset + line_start,
                            duration=max(0.3, line_end - line_start),
                            lane=2,
                            font_size=font_size,
                            style="subtitle",
                            position="bottom",
                        ))
                        line_words = []

                # Flush remaining words
                if line_words:
                    line_text = " ".join(lw["text"] for lw in line_words)
                    line_end = line_words[-1]["end"]
                    subtitles.append(TextOverlay(
                        text=line_text,
                        timeline_offset=tl_offset + line_start,
                        duration=max(0.3, line_end - line_start),
                        lane=2,
                        font_size=font_size,
                        style="subtitle",
                        position="bottom",
                    ))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Replace existing subtitles, keep title-style overlays
        edit.titles = [t for t in edit.titles if t.style != "subtitle"] + subtitles

        # Save updated edit decision
        _atomic_write_json(json_path, edit.model_dump())

        # Emit updated edit decision to dashboard
        await self._emit("edit_decision", edit.model_dump())

        logger.info(f"[SUBTITLES] Generated {len(subtitles)} subtitle lines for {len(edit.clips)} clips")

        return {
            "status": "generated",
            "subtitle_count": len(subtitles),
            "clips_processed": len(edit.clips),
        }

    async def _generate_content(self, args: Dict) -> Dict:
        """Generate an AI video clip using Veo via Vertex AI."""
        from src.pipelines.v2.video_generation.video_gen_pipeline import generate_video_async

        prompt = args.get("prompt", "")
        if not prompt.strip():
            return {"error": "Prompt is empty"}

        duration = args.get("duration", 8)
        aspect_ratio = args.get("aspect_ratio", "16:9")
        name = args.get("name", "generated")
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        output_path = self.workspace.get_generated_video_path(safe_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[AGENT] generate_content: prompt={prompt[:80]}... duration={duration}s")

        async def progress_cb(step, message):
            await self._emit("tool_progress", {
                "tool": "generate_content",
                "step": step,
                "message": message,
            })

        result = await generate_video_async(
            prompt=prompt,
            output_path=str(output_path),
            duration=duration,
            aspect_ratio=aspect_ratio,
            progress_callback=progress_cb,
        )

        if "error" in result:
            logger.warning(f"[AGENT] generate_content failed: {result['error']}")
            return result

        # Get video info for the edit decision
        file_path = result["file_path"]
        rel_path = str(Path(file_path).relative_to(self.workspace.root.parent.parent))

        logger.info(
            f"[AGENT] generate_content complete: {file_path} "
            f"({result.get('generation_time_seconds', '?')}s generation time)"
        )

        return {
            "status": "complete",
            "file_path": file_path,
            "source_file": Path(file_path).name,
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "generation_time_seconds": result.get("generation_time_seconds"),
            "usage_hint": (
                f"Use source_file='{Path(file_path).name}' in the edit decision. "
                f"The file is at {file_path} and will be resolved automatically."
            ),
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

        # Generate audio + word-level timestamps in one call
        audio_path = self.workspace.root / "narration" / "narration.mp3"
        try:
            loop = asyncio.get_event_loop()
            words = await loop.run_in_executor(
                None,
                lambda: tts_with_timestamps(script, str(audio_path), api_key),
            )
        except Exception as e:
            logger.warning(f"[AGENT] TTS-with-timestamps failed: {e}, falling back to plain TTS")
            try:
                await loop.run_in_executor(
                    None, lambda: tts_sync(script, str(audio_path), api_key)
                )
                words = []
            except Exception as e2:
                return {"error": f"TTS generation failed: {e2}"}

        # Get audio duration (from file, not from word timestamps which may have rounding)
        duration = await self._get_audio_duration(str(audio_path))

        # Build per-sentence transcript from REAL word timestamps. Sentences are
        # delimited by words ending in . ? ! (marked with is_sentence_end).
        transcript = []
        if words:
            sent_words = []
            sent_start: Optional[float] = None
            for w in words:
                if sent_start is None:
                    sent_start = w["start"]
                sent_words.append(w["text"])
                if w["is_sentence_end"]:
                    transcript.append({
                        "text": " ".join(sent_words),
                        "start": round(sent_start, 3),
                        "end": round(w["end"], 3),
                        "word_count": len(sent_words),
                    })
                    sent_words = []
                    sent_start = None
            # Trailing sentence without terminal punctuation
            if sent_words and sent_start is not None:
                transcript.append({
                    "text": " ".join(sent_words),
                    "start": round(sent_start, 3),
                    "end": round(words[-1]["end"], 3),
                    "word_count": len(sent_words),
                })
        else:
            # Fallback: if word timestamps unavailable, use proportional estimation
            # (legacy behavior — agent should know these are estimates)
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script.strip()) if s.strip()]
            word_counts = [len(s.split()) for s in sentences]
            total_words = sum(word_counts) or 1
            cursor = 0.0
            for sent, wc in zip(sentences, word_counts):
                sent_dur = duration * (wc / total_words)
                transcript.append({
                    "text": sent,
                    "start": round(cursor, 2),
                    "end": round(cursor + sent_dur, 2),
                    "word_count": wc,
                    "_estimated": True,
                })
                cursor += sent_dur

        result = {
            "status": "generated",
            "narration_path": str(audio_path),
            "script_length": len(script),
            "duration_seconds": duration,
            "word_count": len(script.split()),
            "transcript": transcript,
        }
        if words:
            result["words"] = words  # full word-level timing available
            result["timestamps_source"] = "elevenlabs_alignment"
            # Persist the word grid so generate_fcpxml can validate narration
            # segment boundaries against real word starts/ends — prevents the
            # LLM from inventing timestamps that cut mid-word.
            try:
                words_path = self.workspace.root / "narration" / "words.json"
                _atomic_write_json(words_path, words)
            except Exception as e:
                logger.warning(f"[AGENT] Could not persist narration words.json: {e}")
        else:
            result["timestamps_source"] = "estimated_proportional"
        logger.info(
            f"[AGENT] Narration generated: {duration:.2f}s, {len(transcript)} sentences, "
            f"{len(words)} words ({result['timestamps_source']})"
        )
        return result

    async def _select_music(self, args: Dict) -> Dict:
        """Generate a background music track using Google Lyria 3 Pro via OpenRouter."""
        import os

        prompt = args.get("prompt", "")
        if not prompt.strip():
            return {"error": "Prompt is empty"}

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            return {"error": "OPENROUTER_API_KEY not set — music generation unavailable"}

        logger.info(f"[AGENT] select_music: {prompt[:100]}")

        await self._emit("refine_progress", {
            "step": "generating_music",
            "message": "Generating music with Lyria 3...",
        })

        music_path = self.workspace.root / "music" / "track.mp3"
        music_path.parent.mkdir(parents=True, exist_ok=True)

        duration_seconds = args.get("duration_seconds", 120)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_music_track(
                api_key=openrouter_key,
                prompt=prompt,
                output_path=str(music_path),
                duration_seconds=int(float(duration_seconds)),
            ),
        )

        if not result.get("success"):
            return {
                "error": result.get("error", "Music generation failed"),
            }

        # Get actual duration via ffprobe
        duration = await self._get_audio_duration(str(music_path))

        # Detect beats so the agent can align its own cut points. Returned
        # as a plain list of seconds + a BPM number — no auto-snap, the
        # agent reads these and decides which clip boundaries to land on a
        # beat. Beat detection can be slow on long tracks, so we cap the
        # returned array to something the prompt can actually fit.
        beats_list: List[float] = []
        tempo_bpm = 0.0
        try:
            from src.pipelines.v2.music.beat_sync import detect_beats
            import numpy as np

            def _detect():
                import librosa
                y, sr = librosa.load(str(music_path), sr=22050)
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                return float(tempo), beat_times

            tempo_bpm, beat_times = await loop.run_in_executor(None, _detect)
            beats_list = [round(float(t), 3) for t in beat_times]
            logger.info(f"[MUSIC] Detected {len(beats_list)} beats at {tempo_bpm:.1f} BPM")
        except Exception as e:
            logger.warning(f"[MUSIC] Beat detection failed (non-fatal): {e}")

        result_payload: Dict[str, Any] = {
            "status": "generated",
            "music_path": str(music_path),
            "duration_seconds": duration,
            "prompt_used": prompt,
        }
        if beats_list:
            # Trim to ~200 beats (more than enough for a 2-3 min edit) so the
            # tool result doesn't explode the context window.
            truncated = beats_list[:200]
            result_payload["tempo_bpm"] = round(tempo_bpm, 1)
            result_payload["beats"] = truncated
            result_payload["beats_note"] = (
                f"{len(beats_list)} beats detected at {tempo_bpm:.1f} BPM "
                f"(first {len(truncated)} returned). To cut on a beat, set a "
                f"clip's source_end so its timeline end (sum of prior durations + "
                f"this clip's duration) equals a value in this array. Beat cuts "
                f"are an editorial choice — align when it serves the edit, break "
                f"intentionally when you want dissonance or emphasis."
            )
        return result_payload

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

    # Padding (seconds) added on each side of the search window before refinement.
    # Gives Gemini room to find complete sentences that overlap the edges.
    REFINE_PAD_SECONDS = 5.0

    async def _refine_clip_timestamps(self, args: Dict) -> Dict:
        """
        Refine clip timestamps by sending both video and transcription to Gemini.

        Steps:
        1. Resolve source file path from footage directory
        2. Add padding buffer around the search window
        3. Extract the padded video segment using ffmpeg
        4. Downsample it for Gemini (lower res, reasonable fps)
        5. Get dialogue transcription from Memories.ai (video-relative timestamps)
        6. Send video + transcription + prompt to Gemini with structured output
        7. If Gemini detects truncated speech, auto-retry with a wider window
        8. Return refined start/end timestamps
        """
        return await self._refine_clip_timestamps_inner(args, retry=0)

    async def _refine_clip_timestamps_inner(self, args: Dict, retry: int = 0) -> Dict:
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

        # Add padding buffer so Gemini can see complete sentences at the edges
        pad = self.REFINE_PAD_SECONDS
        padded_start = max(0, source_start - pad)
        padded_end = source_end + pad
        padded_duration = padded_end - padded_start
        # Offset from padded_start to the original region of interest
        roi_offset = source_start - padded_start
        roi_end_offset = roi_offset + segment_duration

        logger.info(
            f"[AGENT] refine_clip_timestamps: {source_file} "
            f"[{source_start:.1f}–{source_end:.1f}] padded=[{padded_start:.1f}–{padded_end:.1f}] "
            f"target={target_duration:.1f}s retry={retry}"
        )

        # Work in a temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="vea_refine_"))
        try:
            # Step 1: Extract + downsample in one pass (re-encode for frame-accurate cuts)
            # Using -c copy would cut on keyframes, causing a timing offset between
            # the requested padded_start and the actual file start. Re-encoding ensures
            # the output starts exactly at padded_start so Gemini/STT offsets map correctly.
            await self._emit("refine_progress", {
                "step": "extracting",
                "message": f"Extracting {padded_duration:.1f}s segment (with {pad:.0f}s padding)...",
                "source_file": source_file,
            })
            segment_path = tmp_dir / "segment.mp4"
            downsampled_path = tmp_dir / "downsampled.mp4"
            extract_downsample_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(padded_start),
                "-i", str(source_path),
                "-t", str(padded_duration),
                "-vf", "fps=2,scale=-2:480",
                "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k",
                str(downsampled_path),
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(extract_downsample_cmd, check=True)
            )
            # Also extract a frame-accurate audio-only file for STT
            # (re-encode ensures exact start; audio-only is fast)
            segment_path = downsampled_path  # reuse for debug logging
            audio_segment_path = tmp_dir / "segment_audio.m4a"
            audio_extract_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(padded_start),
                "-i", str(source_path),
                "-t", str(padded_duration),
                "-vn", "-c:a", "aac", "-b:a", "128k",
                str(audio_segment_path),
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(audio_extract_cmd, check=True)
            )

            # Save segment to workspace logs for manual inspection
            try:
                debug_dir = self.workspace.root / "logs" / "refine_clips"
                debug_dir.mkdir(parents=True, exist_ok=True)
                clip_label = f"{source_start:.0f}_{source_end:.0f}"
                import shutil as _shutil
                _shutil.copy2(str(segment_path), str(debug_dir / f"segment_{clip_label}.mp4"))
            except Exception:
                pass

            # Step 3: Transcribe segment audio with ElevenLabs STT (word-level timestamps)
            clip_transcript = ""
            transcript_segments = []
            try:
                import os as _os
                el_key = _os.environ.get("ELEVENLABS_API_KEY", "")
                if el_key:
                    # Use the frame-accurate audio segment extracted earlier
                    words = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: stt_word_timestamps(str(audio_segment_path), el_key)
                    )
                    logger.info(f"[AGENT] ElevenLabs STT: {len(words)} words transcribed")

                    # Words have timestamps relative to the extracted segment (0-based).
                    # Group into sentence-like segments and also keep word-level detail.
                    for w in words:
                        transcript_segments.append({
                            "text": w["text"],
                            "start": padded_start + w["start"],   # absolute
                            "end": padded_start + w["end"],       # absolute
                            "rel_start": w["start"],              # relative to video excerpt
                            "rel_end": w["end"],                  # relative to video excerpt
                        })

                    if transcript_segments:
                        clip_transcript = "\n".join(
                            f"[{s['rel_start']:.2f}s–{s['rel_end']:.2f}s] {s['text']}"
                            for s in transcript_segments
                        )
                else:
                    logger.warning("[AGENT] ELEVENLABS_API_KEY not set — skipping STT for refinement")
            except Exception as e:
                logger.warning(f"[AGENT] Could not transcribe segment: {e}")

            # Step 3b: Detect scene cuts in the padded window. Runs on the original
            # source (not the downsampled file — 2 fps is too coarse for ContentDetector's
            # 12-frame minimum). Gemini uses these to avoid sloppy edits near a cut.
            scene_cuts = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _detect_scene_cuts(str(source_path), padded_start, padded_end),
            )
            if scene_cuts:
                logger.info(f"[AGENT] Scene cuts in padded window: {scene_cuts}")

            # Step 4: Send to Gemini for analysis
            await self._emit("refine_progress", {
                "step": "analyzing",
                "message": "Gemini analyzing video + audio...",
                "source_file": source_file,
            })

            refinement_prompt = self._build_refinement_prompt(
                padded_start=padded_start,
                padded_duration=padded_duration,
                roi_offset=roi_offset,
                roi_end_offset=roi_end_offset,
                target_duration=target_duration,
                prompt=prompt,
                clip_description=clip_description,
                transcript=clip_transcript,
                scene_cuts=scene_cuts,
            )

            # Debug: log the full request before sending to Gemini
            video_file_size = downsampled_path.stat().st_size if downsampled_path.exists() else 0
            _refine_debug_log(self.workspace, {
                "event": "gemini_request",
                "source_file": source_file,
                "retry": retry,
                "source_start": source_start,
                "source_end": source_end,
                "padded_start": padded_start,
                "padded_end": padded_end,
                "padded_duration": padded_duration,
                "roi_offset": roi_offset,
                "roi_end_offset": roi_end_offset,
                "target_duration": target_duration,
                "clip_description": clip_description,
                "agent_prompt": prompt,
                "video_file_size_bytes": video_file_size,
                "transcript_word_count": len(transcript_segments),
                "transcript_text": clip_transcript,
                "full_gemini_prompt": refinement_prompt,
            })

            # Step 5a: Gemini reasoning pass — free-text analysis with video
            reasoning_text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.video_llm.LLM_request(
                    prompt_contents=[downsampled_path, refinement_prompt],
                    schema=None,  # free-text reasoning
                    context="refine_clip_timestamps_reasoning",
                ),
            )

            _refine_debug_log(self.workspace, {
                "event": "gemini_reasoning",
                "source_file": source_file,
                "retry": retry,
                "reasoning_text": reasoning_text,
            })

            # Step 5b: Structured output pass — use reasoning to produce timestamps
            structured_prompt = (
                f"{reasoning_text}\n\n"
                "Based on your analysis above, you have identified the best cut points for this clip.\n\n"
                "Now convert your reasoning into structured JSON output. Return:\n"
                "- `new_start` and `new_end` as seconds from the beginning of the video "
                f"(0 to {padded_duration:.1f})\n"
                "- `reasoning`: a brief summary of why you chose these timestamps\n"
                "- `focus_type`: \"dialogue\", \"visual\", or \"audio\"\n"
                "- `speech_truncated_start`: true if speech is cut off at the start of the video\n"
                "- `speech_truncated_end`: true if speech is cut off at the end of the video\n"
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.video_llm.LLM_request(
                    prompt_contents=[structured_prompt],
                    schema=RefinedTimestamps,
                    context="refine_clip_timestamps_structured",
                ),
            )

            if isinstance(result, RefinedTimestamps):
                refined = result
            elif isinstance(result, dict):
                refined = RefinedTimestamps(**result)
            else:
                _refine_debug_log(self.workspace, {
                    "event": "gemini_error",
                    "source_file": source_file,
                    "error": f"Unexpected response type: {type(result)}",
                    "raw_result": str(result)[:2000],
                })
                return {"error": f"Unexpected Gemini response type: {type(result)}"}

            # Debug: log Gemini's raw response (before timestamp conversion)
            _refine_debug_log(self.workspace, {
                "event": "gemini_response",
                "source_file": source_file,
                "retry": retry,
                "gemini_new_start_offset": refined.new_start,
                "gemini_new_end_offset": refined.new_end,
                "gemini_duration": refined.new_end - refined.new_start,
                "reasoning": refined.reasoning,
                "focus_type": refined.focus_type,
                "speech_truncated_start": refined.speech_truncated_start,
                "speech_truncated_end": refined.speech_truncated_end,
                "note": f"These are offsets from 0 in the {padded_duration:.1f}s padded video. "
                        f"Add padded_start={padded_start:.1f} to get absolute timestamps.",
            })

            # Gemini returns offsets relative to the padded video (0 to padded_duration).
            # Convert back to absolute source timestamps.
            new_start = padded_start + max(0, min(refined.new_start, padded_duration))
            new_end = padded_start + max(0, min(refined.new_end, padded_duration))
            if new_end <= new_start:
                new_end = min(new_start + target_duration, padded_end)

            # Debug: log the final converted timestamps
            _refine_debug_log(self.workspace, {
                "event": "timestamp_conversion",
                "source_file": source_file,
                "retry": retry,
                "input_range": f"[{source_start:.1f}–{source_end:.1f}]",
                "padded_range": f"[{padded_start:.1f}–{padded_end:.1f}]",
                "gemini_offsets": f"[{refined.new_start:.2f}–{refined.new_end:.2f}]",
                "absolute_output": f"[{new_start:.2f}–{new_end:.2f}]",
                "output_duration": round(new_end - new_start, 2),
                "target_duration": target_duration,
                "duration_delta": round((new_end - new_start) - target_duration, 2),
            })

            logger.info(
                f"[AGENT] Refined: [{source_start:.1f}–{source_end:.1f}] → "
                f"[{new_start:.1f}–{new_end:.1f}] ({new_end - new_start:.1f}s) "
                f"focus={refined.focus_type} "
                f"truncated_start={refined.speech_truncated_start} "
                f"truncated_end={refined.speech_truncated_end}"
            )

            # Step 6: Auto-retry with wider window if speech is truncated (max 1 retry)
            if retry == 0 and (refined.speech_truncated_start or refined.speech_truncated_end):
                wider_start = source_start - (pad * 2 if refined.speech_truncated_start else pad)
                wider_end = source_end + (pad * 2 if refined.speech_truncated_end else pad)
                wider_start = max(0, wider_start)
                _refine_debug_log(self.workspace, {
                    "event": "truncation_retry",
                    "source_file": source_file,
                    "truncated_start": refined.speech_truncated_start,
                    "truncated_end": refined.speech_truncated_end,
                    "original_window": f"[{source_start:.1f}–{source_end:.1f}]",
                    "wider_window": f"[{wider_start:.1f}–{wider_end:.1f}]",
                })
                logger.info(
                    f"[AGENT] Speech truncated — retrying with wider window "
                    f"[{wider_start:.1f}–{wider_end:.1f}]"
                )
                wider_args = dict(args)
                wider_args["source_start"] = wider_start
                wider_args["source_end"] = wider_end
                return await self._refine_clip_timestamps_inner(wider_args, retry=1)

            result_dict = {
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
                result_dict["transcript"] = [
                    {"text": s["text"], "start": s["start"], "end": s["end"]}
                    for s in transcript_segments
                ]

            return result_dict

        finally:
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    @staticmethod
    def _build_refinement_prompt(
        padded_start: float,
        padded_duration: float,
        roi_offset: float,
        roi_end_offset: float,
        target_duration: float,
        prompt: str,
        clip_description: str,
        transcript: str = "",
        scene_cuts: Optional[List[float]] = None,
    ) -> str:
        """Build the Gemini prompt for timestamp refinement.

        All timestamps in the prompt are relative to the video excerpt (0 to padded_duration).
        The "region of interest" (roi_offset to roi_end_offset) is where the search result
        pointed, but Gemini is free to choose timestamps anywhere in the padded window.
        """
        transcript_section = ""
        if transcript:
            transcript_section = f"""
## Audio transcript (word-level) — AUTHORITATIVE for speech clips

The following is a WORD-LEVEL transcription with frame-accurate timestamps from speech-to-text.
Each line is one word with its exact start and end time. Words ending in `.` `?` `!` mark
sentence boundaries.

{transcript}

For speech-heavy clips, these timestamps are MORE RELIABLE than the video for choosing cut points.
Pick your new_start/new_end by reading the transcript and finding complete sentences — do NOT
rely on visual cues to decide where speech starts or ends.
"""

        scene_cuts_section = ""
        if scene_cuts:
            cuts_str = ", ".join(f"{c:.2f}s" for c in scene_cuts)
            scene_cuts_section = f"""
## Scene cut boundaries (from PySceneDetect)

Shot changes in this video excerpt occur at: {cuts_str}

When choosing new_start / new_end, respect these boundaries:
- **Land EXACTLY on a cut** (offset 0 from a boundary) — ideal. You're riding the
  existing edit.
- **Stay ≥0.5s away** from any cut — also fine. Gives the incoming shot room to breathe.
- **Avoid the 0.01s-0.50s band** on either side of a cut — this feels like a sloppy
  mistake, as if you nearly caught the old edit but missed.

**Dialogue continuity overrides this rule.** If the transcript says the sentence you
need starts at T=7.35s and there's a cut at T=7.00s, cut at 7.35s (word boundary) even
though that's inside the "sloppy" band — never cut mid-word or drop speech to satisfy
a scene-cut margin.
"""

        return f"""\
You are a professional video editor refining clip timestamps. You are watching a video segment
and must choose the best start and end points for a clip.

## Context

You are watching a {padded_duration:.1f}-second video excerpt. The video starts at 0:00 and ends
at {padded_duration:.1f}s.

The search result pointed to the region from {roi_offset:.1f}s to {roi_end_offset:.1f}s in this
video. Focus your search around that area, but you CAN choose timestamps outside it if needed
to capture a complete sentence or natural boundary.

IMPORTANT: Return your timestamps as offsets from the START of this video (0 to {padded_duration:.1f}).
Your ideal clip duration is ~{target_duration:.1f}s.

## What this clip is for
{clip_description or "(No description provided)"}

## What to look for
{prompt}
{transcript_section}{scene_cuts_section}
## Instructions

First, determine if this clip is **speech-heavy** (someone talking) or **visual** (b-roll, reactions, scenery).

**If speech-heavy (dialogue/presentation):** The transcript above is your PRIMARY source for
cut points — NOT the video. The word-level timestamps are frame-accurate from STT. Your job:
1. Read the transcript and find the sentence that best matches what this clip needs.
2. Set `new_start` to the timestamp of the FIRST word of that sentence (minus ~0.15s for breathing room).
3. Set `new_end` to the timestamp AFTER the LAST word of that sentence (plus ~0.2s for natural tail).
4. NEVER cut mid-word or mid-sentence. If the best content spans multiple sentences, include them all.
5. Use the video only to confirm the speaker is on screen — do NOT let visual cuts override transcript boundaries.

**If visual (b-roll, reactions, scenery):** Prioritize visual composition, movement peaks,
and natural motion boundaries. Cut on action. Transcript is secondary.

**If audio-driven (music, applause, sound effects):** Align cuts with audio beats,
applause peaks, or natural audio transitions.

## Truncation detection

If you notice that a sentence or word is cut off at the very START of the video (the speaker
is mid-sentence at 0:00), set `speech_truncated_start` to true.

If you notice that a sentence or word is cut off at the very END of the video (the speaker
is mid-sentence at {padded_duration:.1f}s), set `speech_truncated_end` to true.

## Output (all fields are REQUIRED)

- `new_start` (number) — seconds from the beginning of this video (0 to {padded_duration:.1f})
- `new_end` (number) — seconds from the beginning of this video (must be > new_start, ≤ {padded_duration:.1f})
- `reasoning` (string) — brief explanation of why you chose these cut points
- `focus_type` (string) — "dialogue" | "visual" | "audio" (what primarily drove your decision)
- `speech_truncated_start` (boolean) — see Truncation detection above
- `speech_truncated_end` (boolean) — see Truncation detection above"""
