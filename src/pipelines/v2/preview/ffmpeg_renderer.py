"""
FFmpeg renderer — EditDecision → MP4 without DaVinci Resolve.

Two quality modes:
  - ``draft`` : 480p, ultrafast, crf 28.  Default — runs after every
                generate_fcpxml call so the user sees a preview quickly.
  - ``full``  : timeline-native resolution, preset slow, crf 18, 192k audio.
                Produced on demand when the user toggles full-resolution
                playback in the dashboard's ffmpeg tab.

Feature coverage — matches what the agent can express in EditDecision:
  - V1 spine: sequencing, source in/out, per-clip gain, speed change,
              transform (scale/pan/rotation), multi-shot crops.
  - V2+:      overlay / picture-in-picture clips at their timeline_offset,
              scaled and positioned via transform.
  - Audio:    narration + music mix with LUFS auto-gain, music fade in/out.
  - Titles:   centered / top / bottom, subtitle box, fade-in/out alpha.

Pipeline:
  Pass 1 (parallel)    — extract + process each clip to a temp file.
  Pass 2 (sequential)  — concat spine clips.
  Pass 3 (sequential)  — overlay V2+ clips, mix audio, draw titles.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from src.pipelines.v2.schemas import (
    ClipDecision,
    EditDecision,
    MusicTrack,
    NarrationSegment,
    ShotCropResult,
    TextOverlay,
    TransformSettings,
)

logger = logging.getLogger(__name__)

# Per-render stderr log file. Set by ``render_ffmpeg_preview`` for the
# duration of one render pass; ``_run_ffmpeg`` appends its command +
# exit code + stderr to the path pointed at here. Contextvar keeps this
# scoped to the current asyncio task so two parallel renders don't mix
# their output.
_render_log_path: ContextVar[Optional[Path]] = ContextVar(
    "vea_ffmpeg_render_log", default=None
)

QualityMode = Literal["draft", "full"]

# Encoder + output settings per quality mode. Draft keeps the render cheap
# enough for the agent loop to spawn after every generate_fcpxml call;
# full matches what DaVinci would produce at native timeline resolution.
QUALITY_PRESETS: Dict[str, Dict[str, object]] = {
    "draft": {
        "height": 480,                 # scale to 480p, width from aspect
        "preset": "ultrafast",
        "crf": 28,
        "audio_bitrate": "128k",
        "scale_flags": "bilinear",
    },
    "full": {
        "height": None,                # use timeline native
        "preset": "slow",
        "crf": 18,
        "audio_bitrate": "192k",
        "scale_flags": "lanczos",
    },
}

_drawtext_available: Optional[bool] = None
_system_font_file: Optional[str] = None

def _has_drawtext() -> bool:
    """Check if FFmpeg was built with the drawtext filter (requires libfreetype)."""
    global _drawtext_available
    if _drawtext_available is not None:
        return _drawtext_available
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5,
        )
        _drawtext_available = "drawtext" in result.stdout
    except Exception:
        _drawtext_available = False
    if not _drawtext_available:
        logger.info("[PREVIEW] drawtext filter not available in this FFmpeg build")
    return _drawtext_available


def _system_font() -> Optional[str]:
    """Locate a sans-serif font file suitable for drawtext.

    FFmpeg's drawtext can pick up a default font via fontconfig, but on
    sandboxed / minimal systems that lookup silently fails. We prefer an
    explicit fontfile so the title reliably renders on every machine.
    Returns None if nothing workable is found; drawtext then relies on its
    own default.
    """
    global _system_font_file
    if _system_font_file is not None:
        return _system_font_file or None  # empty string means "already searched, none found"

    candidates: List[str] = []
    sysname = platform.system()
    if sysname == "Darwin":
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    elif sysname == "Linux":
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    elif sysname == "Windows":
        candidates = [r"C:\Windows\Fonts\Arial.ttf", r"C:\Windows\Fonts\arial.ttf"]

    for c in candidates:
        if Path(c).exists():
            _system_font_file = c
            return c
    _system_font_file = ""
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def render_ffmpeg_preview(
    edit: EditDecision,
    footage_dir: Path,
    output_path: Path,
    quality: QualityMode = "draft",
    resolution: Optional[int] = None,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
) -> str:
    """
    Render EditDecision to MP4 via FFmpeg.

    Args:
        edit: Complete edit decision with clips, narration, music, titles
        footage_dir: Directory containing source footage files
        output_path: Where to write the output MP4
        quality: "draft" (480p, ultrafast) or "full" (timeline-native, slow)
        resolution: Deprecated override — explicit output height in pixels.
            When set, overrides the quality preset's height.
        progress_callback: async callback receiving 0-100 progress

    Returns:
        str path to the rendered MP4
    """
    if not edit.clips:
        raise ValueError("No clips in edit decision")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality mode: {quality!r} (expected 'draft' or 'full')")
    preset = QUALITY_PRESETS[quality]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If we're inside an active workspace log scope, open a fresh per-render
    # stderr file. Every _run_ffmpeg call during this render appends to it.
    from src.pipelines.v2.logging_setup import _active_workspace, open_render_log
    _ws = _active_workspace.get()
    _log_token = None
    if _ws is not None:
        render_log = open_render_log(_ws, f"ffmpeg-{quality}")
        _log_token = _render_log_path.set(render_log)
        logger.info(f"[RENDER] stderr → {render_log}")

    tl_w = edit.timeline.width
    tl_h = edit.timeline.height
    tl_fps = edit.timeline.fps
    if tl_w <= 0 or tl_h <= 0:
        # Timeline dims come straight from the LLM-supplied EditDecision with no
        # schema constraint; 0 would ZeroDivisionError at the out_w / title-scale
        # math below. Fail with a clear message (the caller treats render errors
        # as non-fatal and surfaces them to the agent). Matches resolve_render.
        raise ValueError(
            f"Invalid timeline dimensions {tl_w}x{tl_h}; width and height must be > 0."
        )

    target_h = resolution if resolution is not None else preset["height"]
    if target_h is None:
        out_w, out_h = tl_w, tl_h
    else:
        out_h = int(target_h)
        out_w = int(tl_w * out_h / tl_h)
    # H.264 requires even dimensions
    out_w += out_w % 2
    out_h += out_h % 2

    spine_clips = [c for c in edit.clips if c.track == 1]
    overlay_clips = [c for c in edit.clips if c.track > 1]
    if not spine_clips:
        raise ValueError("No track-1 clips in edit decision")

    logger.info(
        f"[RENDER] quality={quality} out={out_w}x{out_h}@{tl_fps}fps "
        f"spine={len(spine_clips)} overlays={len(overlay_clips)} "
        f"narration={len(edit.narration)} music={'yes' if edit.music else 'no'} "
        f"titles={len(edit.titles)}"
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="vea_render_"))

    try:
        # ── Pass 1: extract + process each V1 clip in parallel ────────────
        total_dur = sum(_clip_duration(c) for c in spine_clips)
        completed_dur = 0.0

        async def process_clip(
            idx: int, clip: ClipDecision, prefix: str = "clip",
        ) -> Path:
            nonlocal completed_dur
            out_file = tmpdir / f"{prefix}_{idx:03d}.mp4"
            source = _resolve_source(clip, footage_dir)
            if not source.exists():
                logger.warning(f"[RENDER] Source not found: {source}, skipping clip {clip.id}")
                dur = _clip_duration(clip)
                await _run_ffmpeg([
                    "-f", "lavfi", "-i", f"color=black:s={out_w}x{out_h}:d={dur:.3f}:r={tl_fps}",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
                    "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", str(preset["preset"]), "-crf", str(preset["crf"]),
                    "-c:a", "aac", "-b:a", str(preset["audio_bitrate"]),
                    str(out_file),
                ])
                return out_file

            segments = _get_clip_segments(clip, tl_w, tl_h)
            if len(segments) == 1:
                await _render_single_segment(
                    source, segments[0], clip, out_file,
                    out_w, out_h, tl_fps, preset,
                )
            else:
                seg_files = []
                for si, seg in enumerate(segments):
                    seg_file = tmpdir / f"{prefix}_{idx:03d}_shot_{si:02d}.mp4"
                    await _render_single_segment(
                        source, seg, clip, seg_file,
                        out_w, out_h, tl_fps, preset,
                    )
                    seg_files.append(seg_file)
                await _concat_files(seg_files, out_file)

            if prefix == "clip":
                completed_dur += _clip_duration(clip)
                if progress_callback:
                    pct = min(70.0, (completed_dur / total_dur) * 70.0)
                    await progress_callback(pct)
            return out_file

        clip_tasks = [process_clip(i, c) for i, c in enumerate(spine_clips)]
        clip_files = await asyncio.gather(*clip_tasks)
        valid_clip_files: List[Path] = [f for f in clip_files if f.exists()]
        if not valid_clip_files:
            raise RuntimeError("All clips failed to render")

        # Upper-track clips — processed once, composited later via overlay.
        overlay_tasks = [
            process_clip(i, c, prefix="overlay")
            for i, c in enumerate(overlay_clips)
        ]
        overlay_files = await asyncio.gather(*overlay_tasks) if overlay_tasks else []

        if progress_callback:
            await progress_callback(70.0)

        # ── Pass 2: concat spine ──────────────────────────────────────────
        concat_video = tmpdir / "concat.mp4"
        await _concat_files(valid_clip_files, concat_video)

        if progress_callback:
            await progress_callback(85.0)

        # ── Pass 3: overlay V2+, mix audio, draw titles ───────────────────
        await _mix_and_finalize(
            concat_video, edit, footage_dir, output_path,
            out_w, out_h, tl_fps, spine_clips,
            overlay_clips, overlay_files, preset,
        )

        if progress_callback:
            await progress_callback(100.0)

        logger.info(f"[RENDER] FFmpeg {quality} render complete: {output_path}")
        return str(output_path)

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        if _log_token is not None:
            try:
                _render_log_path.reset(_log_token)
            except ValueError:
                pass


async def extract_frame(
    edit: EditDecision,
    footage_dir: Path,
    timeline_time: float,
    output_width: int = 854,
    output_height: int = 480,
) -> Optional[bytes]:
    """
    Extract a single frame at a given timeline position.

    Maps timeline time → source clip + source time, extracts frame with
    transforms applied, returns JPEG bytes.
    """
    track1_clips = [c for c in edit.clips if c.track == 1]
    if not track1_clips:
        return None

    # Find which clip contains this timeline time
    current_pos = 0.0
    target_clip = None
    clip_start_in_timeline = 0.0

    for clip in track1_clips:
        dur = _clip_duration(clip)
        if current_pos + dur > timeline_time:
            target_clip = clip
            clip_start_in_timeline = current_pos
            break
        current_pos += dur

    if target_clip is None:
        # Past the end — use last frame of last clip
        target_clip = track1_clips[-1]
        clip_start_in_timeline = sum(_clip_duration(c) for c in track1_clips[:-1])
        timeline_time = clip_start_in_timeline + _clip_duration(target_clip) - 0.04

    # Map timeline position to source time
    offset_in_clip = timeline_time - clip_start_in_timeline
    speed = target_clip.speed.rate if target_clip.speed else 1.0
    source_time = target_clip.source_start + (offset_in_clip * speed)
    source_time = max(target_clip.source_start, min(source_time, target_clip.source_end - 0.04))

    source = _resolve_source(target_clip, footage_dir)
    if not source.exists():
        return None

    # Determine transform for this position (multi-shot aware)
    tl_w = edit.timeline.width
    tl_h = edit.timeline.height
    transform = _get_transform_at_time(target_clip, source_time, tl_w, tl_h)

    # Build crop/scale filter
    vf = _build_transform_filter(
        transform, target_clip.source_width, target_clip.source_height,
        tl_w, tl_h, output_width, output_height,
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{source_time:.3f}",
        "-i", str(source),
        "-vframes", "1",
        "-vf", vf,
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "5",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning(f"[PREVIEW] Frame extraction failed: {stderr.decode()[-200:]}")
        return None

    return stdout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clip_duration(clip: ClipDecision) -> float:
    """Effective duration after speed change."""
    dur = clip.source_end - clip.source_start
    if clip.speed and clip.speed.rate > 0:
        dur /= clip.speed.rate
    return max(0.0, dur)


def _resolve_source(clip: ClipDecision, footage_dir: Path) -> Path:
    """Resolve clip source to an absolute path."""
    if clip.source_path and Path(clip.source_path).exists():
        return Path(clip.source_path)
    # Try footage_dir / source_file
    p = footage_dir / clip.source_file
    if p.exists():
        return p
    # Try source_file as-is (may be absolute or relative to CWD)
    p2 = Path(clip.source_file)
    if p2.exists():
        return p2
    # Try generated/ subdirectory (AI-generated content)
    workspace_root = footage_dir.parent
    p3 = workspace_root / "generated" / clip.source_file
    if p3.exists():
        return p3
    return footage_dir / clip.source_file  # return expected path even if missing


ClipSegment = Tuple[float, float, TransformSettings]  # source_start, source_end, transform


def _get_clip_segments(
    clip: ClipDecision, tl_w: int, tl_h: int,
) -> List[ClipSegment]:
    """Split clip into segments based on shot_transforms or return single segment."""
    if clip.shot_transforms and len(clip.shot_transforms) > 1:
        segments = []
        for shot in clip.shot_transforms:
            t = shot.transform or TransformSettings()
            segments.append((shot.source_start, shot.source_end, t))
        return segments

    # Single segment with clip-level transform
    t = clip.transform or TransformSettings()
    return [(clip.source_start, clip.source_end, t)]


def _get_transform_at_time(
    clip: ClipDecision, source_time: float,
    tl_w: int, tl_h: int,
) -> TransformSettings:
    """Get the active transform for a specific source time (multi-shot aware)."""
    if clip.shot_transforms:
        for shot in clip.shot_transforms:
            if shot.source_start <= source_time <= shot.source_end:
                return shot.transform or TransformSettings()
    return clip.transform or TransformSettings()


def _build_transform_filter(
    transform: TransformSettings,
    src_w: int, src_h: int,
    tl_w: int, tl_h: int,
    out_w: int, out_h: int,
    scale_flags: str = "bilinear",
) -> str:
    """Convert TransformSettings to an FFmpeg crop+scale+rotate filter string.

    Pipeline per clip:
      1. Honour SAR — ``scale=iw*sar:ih,setsar=1`` so anamorphic sources
         (e.g. 1920x1080 cinemascope stored with SAR≠1) display at correct
         aspect. Without this step, ffmpeg renders the raw pixel grid
         directly and a 2.40:1 source squeezes vertically, diverging from
         how Resolve (which honours SAR) renders the same FCPXML.
      2. Crop (if the agent specified a non-default scale_x/y/position).
      3. Scale to the output canvas. For the fit case, preserve aspect and
         crop the excess so we fully fill the frame — matches Resolve's
         default "Scale to Frame Size" behaviour.
      4. Rotate (if transform.rotation is set).
    """
    sx = transform.scale_x or 1.0
    sy = transform.scale_y or 1.0
    rot_deg = getattr(transform, "rotation", 0.0) or 0.0

    # Step 1 — SAR normalization (always applied; no-op for square pixels).
    chain = "scale=iw*sar:ih,setsar=1"

    default_scale = tl_w / max(src_w, 1)
    is_fit = (
        (abs(sx - default_scale) < 0.02 or abs(sx - 1.0) < 0.02)
        and abs(transform.position_x) < 1
        and abs(transform.position_y) < 1
    )
    if is_fit:
        # Scale-to-fit with aspect preservation, then pad black bars to fill
        # the canvas. This matches Resolve's default "Spatial Conform: Fit"
        # behaviour for FCPXML imports — all source content stays visible;
        # mismatched aspects get letterbox or pillarbox bars.
        chain += (
            f",scale={out_w}:{out_h}:"
            f"force_original_aspect_ratio=decrease:flags={scale_flags},"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    else:
        # Agent-specified crop/pan inside the SAR-corrected source.
        crop_frac_w = min(1.0, (tl_w / sx) / max(src_w, 1))
        crop_frac_h = min(1.0, (tl_h / sy) / max(src_h, 1))
        norm = 19.2
        off_frac_x = (transform.position_x * norm) / max(src_w * sx, 1)
        off_frac_y = (transform.position_y * norm) / max(src_h * sy, 1)
        cw = f"iw*{crop_frac_w:.4f}"
        ch = f"ih*{crop_frac_h:.4f}"
        cx = f"(iw-iw*{crop_frac_w:.4f})/2-iw*{off_frac_x:.4f}"
        cy = f"(ih-ih*{crop_frac_h:.4f})/2+ih*{off_frac_y:.4f}"
        chain += (
            f",crop={cw}:{ch}:{cx}:{cy},"
            f"scale={out_w}:{out_h}:flags={scale_flags}"
        )

    if abs(rot_deg) > 0.01:
        rad_expr = f"{rot_deg}*PI/180"
        chain += f",rotate={rad_expr}:ow={out_w}:oh={out_h}:c=black@0"

    return chain


async def _render_single_segment(
    source: Path,
    segment: ClipSegment,
    clip: ClipDecision,
    out_file: Path,
    out_w: int, out_h: int,
    fps: float,
    preset: Dict[str, object],
) -> None:
    """Render a single clip segment (one transform) to a temp file."""
    src_start, src_end, transform = segment
    # Frame-snap to the source file's frame rate so ffmpeg -ss/-to land on
    # real frames. Probe once per source file; if probe fails, fall back to
    # the timeline fps (still better than raw decimal seconds).
    from src.pipelines.v2.timing import probe_fps as _probe_fps, snap_to_frame as _snap
    src_fps = _probe_fps(source) or fps
    if src_fps and src_fps > 0:
        src_start = _snap(src_start, src_fps)
        src_end = _snap(src_end, src_fps)
    duration = src_end - src_start
    speed_rate = clip.speed.rate if clip.speed else 1.0
    if speed_rate <= 0:
        # An LLM can emit a non-positive speed.rate (e.g. misreading "freeze
        # frame"). setpts=PTS/0 would error and _atempo_chain would loop
        # forever. Treat as no speed change, matching edit_compiler's clamp.
        speed_rate = 1.0
    scale_flags = str(preset.get("scale_flags", "bilinear"))

    vf_parts = [
        _build_transform_filter(
            transform, clip.source_width, clip.source_height,
            1920, 1080,
            out_w, out_h,
            scale_flags=scale_flags,
        )
    ]
    if abs(speed_rate - 1.0) > 0.01:
        vf_parts.append(f"setpts=PTS/{speed_rate}")
    vf = ",".join(vf_parts)

    af_parts: List[str] = []
    if clip.gain_db is not None and clip.gain_db != 0:
        af_parts.append(f"volume={clip.gain_db}dB")
    if abs(speed_rate - 1.0) > 0.01:
        af_parts.extend(_atempo_chain(speed_rate))
    af = ",".join(af_parts) if af_parts else None

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{src_start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source),
        "-vf", vf,
    ]
    if af:
        cmd.extend(["-af", af])
    cmd.extend([
        "-c:v", "libx264",
        "-preset", str(preset["preset"]),
        "-crf", str(preset["crf"]),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", str(preset["audio_bitrate"]),
        "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_file),
    ])

    await _run_ffmpeg(cmd)


def _atempo_chain(rate: float) -> List[str]:
    """Build atempo filter chain for a given speed rate (each atempo limited to 0.5-2.0)."""
    # Defensive: the loops below never terminate for a non-positive rate.
    # Callers clamp speed_rate to 1.0, but guard here too in case of new callers.
    if rate <= 0:
        return []
    filters = []
    remaining = rate
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.01:
        filters.append(f"atempo={remaining:.4f}")
    return filters


async def _concat_files(files: List[Path], output: Path) -> None:
    """Concatenate multiple MP4 files using ffmpeg concat demuxer."""
    if len(files) == 1:
        shutil.copy2(files[0], output)
        return

    list_file = output.parent / f"{output.stem}_list.txt"
    with open(list_file, "w") as f:
        for fp in files:
            f.write(f"file '{fp}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output),
    ]
    await _run_ffmpeg(cmd)
    list_file.unlink(missing_ok=True)


async def _mix_and_finalize(
    video_file: Path,
    edit: EditDecision,
    footage_dir: Path,
    output: Path,
    out_w: int, out_h: int,
    fps: float,
    spine_clips: List[ClipDecision],
    overlay_clips: List[ClipDecision],
    overlay_files: List[Path],
    preset: Dict[str, object],
) -> None:
    """Mix narration + music, composite V2+ overlays, draw titles.

    Title z-policy: titles are always drawn LAST, so they render on top of
    every video layer. The FCPXML compiler mirrors this by bumping title
    lanes into a reserved high range, keeping draft and final visually
    identical regardless of what the agent set as ``title.lane``.
    """
    has_narration = bool(edit.narration)
    has_music = edit.music is not None
    has_titles = bool(edit.titles)
    has_overlays = bool(overlay_clips) and any(f.exists() for f in overlay_files)

    if not has_narration and not has_music and not has_titles and not has_overlays:
        shutil.copy2(video_file, output)
        return

    total_dur = sum(_clip_duration(c) for c in spine_clips)
    inputs = ["-i", str(video_file)]
    input_idx = 1  # [0] is the spine
    filter_parts: List[str] = []
    audio_streams = ["[0:a]"]

    # Auto-target loudness for music/narration. The agent has been bad at writing
    # gain_db math, so we hardcode targets and compute the actual dB at render time.
    # The agent's gain_db is treated as an OFFSET from the target (so gain_db=0 means
    # "play at target", gain_db=-3 means "3 dB below target", etc.).
    #
    # Music target drops when narration is present so the voice sits clearly
    # on top. -35 LUFS for music vs -16 LUFS for narration = 19 dB separation —
    # normal "bed music" range for voice-over. Without the drop, music at -18
    # is only 2 dB under narration and drowns it.
    TARGET_NARRATION_LUFS = -16.0
    TARGET_MUSIC_LUFS = -35.0 if has_narration else -18.0

    def _compute_auto_gain(measured_lufs: Optional[float], target_lufs: float, agent_offset_db: float) -> float:
        if measured_lufs is None:
            return agent_offset_db  # no measurement yet — use literal
        gain = (target_lufs + agent_offset_db) - measured_lufs
        # Clamp to sane range to avoid extreme adjustments
        return max(-30.0, min(20.0, round(gain, 1)))

    # ── Upper-track (V2+) overlay clips ─────────────────────────────────
    # Processed first so their audio joins the mix alongside narration/music.
    # The video overlay filter is added here but operates on [0:v] at composite
    # time; filter_complex ignores the order of filter_parts since nodes link
    # by named label, so adding them early is safe.
    overlay_video_ops: List[Tuple[str, str]] = []  # (prep_filter, overlay_filter)
    if has_overlays:
        for oi, (ovc, of) in enumerate(zip(overlay_clips, overlay_files)):
            if not of.exists():
                continue
            inputs.extend(["-i", str(of)])
            ovly_idx = input_idx
            input_idx += 1

            t = ovc.transform or TransformSettings()
            sx = max(0.01, min(1.0, float(t.scale_x or 1.0)))
            sy = max(0.01, min(1.0, float(t.scale_y or 1.0)))
            is_fullscreen = abs(sx - 1.0) < 0.02 and abs(sy - 1.0) < 0.02
            scaled_w = int(out_w * sx); scaled_w += scaled_w % 2
            scaled_h = int(out_h * sy); scaled_h += scaled_h % 2

            norm = 19.2
            pos_x_px = float(t.position_x or 0.0) * norm
            pos_y_px = float(t.position_y or 0.0) * norm
            ox = f"(W-w)/2+{pos_x_px:.1f}" if not is_fullscreen else "0"
            oy = f"(H-h)/2+{pos_y_px:.1f}" if not is_fullscreen else "0"

            start_t = float(ovc.timeline_offset or 0.0)
            dur = _clip_duration(ovc)
            end_t = start_t + dur

            prep_label = f"[ovprep{oi}]"
            if is_fullscreen:
                prep_filter = (
                    f"[{ovly_idx}:v]setpts=PTS-STARTPTS+{start_t:.3f}/TB{prep_label}"
                )
            else:
                prep_filter = (
                    f"[{ovly_idx}:v]scale={scaled_w}:{scaled_h},"
                    f"setpts=PTS-STARTPTS+{start_t:.3f}/TB{prep_label}"
                )
            out_label = f"[vov{oi}]"
            overlay_filter = (
                f"__PREV__{prep_label}overlay=x={ox}:y={oy}:"
                f"enable='between(t\\,{start_t:.3f}\\,{end_t:.3f})'{out_label}"
            )
            overlay_video_ops.append((prep_filter, overlay_filter))

            # Audio: trim to clip duration, apply gain, delay into timeline.
            clip_gain = float(ovc.gain_db or 0.0)
            delay_ms = int(start_t * 1000)
            gain_f = f"volume={clip_gain}dB," if clip_gain != 0 else ""
            filter_parts.append(
                f"[{ovly_idx}:a]atrim=duration={dur:.3f},asetpts=PTS-STARTPTS,"
                f"{gain_f}adelay={delay_ms}|{delay_ms}[ova{oi}]"
            )
            audio_streams.append(f"[ova{oi}]")

    # Narration segments
    if has_narration:
        for seg in edit.narration:
            nar_path = _resolve_audio_file(seg.file, footage_dir)
            if not nar_path.exists():
                logger.warning(f"[PREVIEW] Narration file not found: {nar_path}")
                continue
            inputs.extend(["-i", str(nar_path)])
            delay_ms = int(seg.timeline_offset * 1000)
            actual_gain = _compute_auto_gain(
                getattr(seg, "measured_loudness_lufs", None),
                TARGET_NARRATION_LUFS,
                seg.gain_db,
            )
            logger.info(
                f"[PREVIEW] Narration auto-gain: agent_offset={seg.gain_db:+g}dB "
                f"measured={getattr(seg, 'measured_loudness_lufs', '?')} → applied={actual_gain:+g}dB"
            )
            gain = f"volume={actual_gain}dB," if actual_gain != 0 else ""
            filter_parts.append(
                f"[{input_idx}:a]atrim={seg.start:.3f}:duration={seg.duration:.3f},"
                f"asetpts=PTS-STARTPTS,{gain}"
                f"adelay={delay_ms}|{delay_ms}[nar{input_idx}]"
            )
            audio_streams.append(f"[nar{input_idx}]")
            input_idx += 1

    # Music
    if has_music and edit.music:
        mus = edit.music
        mus_path = _resolve_audio_file(mus.file, footage_dir)
        if not mus_path.exists():
            logger.warning(f"[PREVIEW] Music file not found: {mus.file} (resolved to {mus_path})")
        if mus_path.exists():
            inputs.extend(["-i", str(mus_path)])
            mus_dur = mus.duration if mus.duration > 0 else total_dur
            actual_gain = _compute_auto_gain(
                getattr(mus, "measured_loudness_lufs", None),
                TARGET_MUSIC_LUFS,
                mus.gain_db,
            )
            logger.info(
                f"[PREVIEW] Music auto-gain: agent_offset={mus.gain_db:+g}dB "
                f"measured={getattr(mus, 'measured_loudness_lufs', '?')} → applied={actual_gain:+g}dB"
            )
            gain = f"volume={actual_gain}dB," if actual_gain != 0 else ""
            # Symmetric fades — fade-in prevents the click that otherwise lands
            # on the first sample of the music bed; fade-out avoids an abrupt
            # cutoff when the music is longer than needed.
            fade_in_dur = min(0.3, max(0.05, mus_dur * 0.1))
            fade_out_dur = min(2.0, max(0.0, mus_dur * 0.5))
            fade_out_start = max(0.0, mus_dur - fade_out_dur)
            filter_parts.append(
                f"[{input_idx}:a]atrim={mus.start:.3f}:duration={mus_dur:.3f},"
                f"asetpts=PTS-STARTPTS,{gain}"
                f"afade=t=in:st=0:d={fade_in_dur:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_dur:.3f}[mus]"
            )
            audio_streams.append("[mus]")
            input_idx += 1

    # Audio mixing
    if len(audio_streams) > 1:
        mix_inputs = "".join(audio_streams)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(audio_streams)}:"
            f"duration=first:dropout_transition=2:normalize=0[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = "0:a"

    # ── Video composition: spine → overlays → titles ───────────────────
    video_map = "0:v"

    # Commit the pending overlay chain (video side) now. prep filters run on
    # their own [:v] input; the overlay filter chains ``[prev_v] + [prep]`` →
    # new label, cascaded so the last overlay becomes video_map.
    if overlay_video_ops:
        prev_v = "[0:v]"
        for prep_filter, overlay_filter in overlay_video_ops:
            filter_parts.append(prep_filter)
            filter_parts.append(overlay_filter.replace("__PREV__", prev_v))
            # The overlay filter's output label is the trailing [vov{oi}]
            prev_v = overlay_filter.rsplit("[", 1)[-1]
            prev_v = f"[{prev_v}"
        video_map = prev_v

    # Text overlays — requires FFmpeg built with --enable-libfreetype.
    # Titles draw last so they sit above every video layer (z-policy).
    txt_files: List[Path] = []
    fontfile = _system_font()
    if has_titles and _has_drawtext():
        title_in = video_map if video_map != "0:v" else "[0:v]"
        # Route through a null filter if we're starting from the raw input so
        # we can chain drawtexts reliably.
        if title_in == "[0:v]":
            filter_parts.append(f"[0:v]null[vtxtin]")
            title_in = "[vtxtin]"
        vf_chain = title_in
        for ti, title in enumerate(edit.titles):
            scaled_size = max(12, int(title.font_size * out_h / edit.timeline.height))
            start_t = title.timeline_offset
            end_t = start_t + title.duration
            is_subtitle = getattr(title, "style", "title") == "subtitle"
            pos = getattr(title, "position", "center")
            if pos == "bottom":
                y_expr = "h-text_h-h*0.06"
            elif pos == "top":
                y_expr = "h*0.06"
            else:
                y_expr = "(h-text_h)/2"
            box_opts = ":box=1:boxcolor=black@0.5:boxborderw=6" if is_subtitle else ""

            # Fade-in/out via alpha expression. Cap at 25% of duration so very
            # short titles don't spend all their time fading. Subtitles fade
            # a bit faster so captions feel snappy.
            fade = min(0.4 if not is_subtitle else 0.15, title.duration * 0.25)
            alpha_expr = (
                f"'if(lt(t,{start_t:.3f}+{fade:.3f}),"
                f"(t-{start_t:.3f})/{fade:.3f},"
                f"if(gt(t,{end_t:.3f}-{fade:.3f}),"
                f"({end_t:.3f}-t)/{fade:.3f},1))'"
            )

            import tempfile as _tf
            txt_file = Path(_tf.mktemp(suffix=".txt", dir=str(output.parent)))
            txt_file.write_text(title.text, encoding="utf-8")
            txt_files.append(txt_file)
            txt_path_escaped = str(txt_file).replace("'", "\\'")
            font_opt = f":fontfile='{fontfile}'" if fontfile else ""
            vf_chain += (
                f"drawtext=textfile='{txt_path_escaped}'{font_opt}:"
                f"fontsize={scaled_size}:fontcolor=white:"
                f"x=(w-text_w)/2:y={y_expr}{box_opts}:"
                f"alpha={alpha_expr}:"
                f"enable='between(t\\,{start_t:.3f}\\,{end_t:.3f})'"
            )
            if ti < len(edit.titles) - 1:
                vf_chain += ","
        vf_chain += "[vout]"
        filter_parts.append(vf_chain)
        video_map = "[vout]"
    elif has_titles:
        logger.warning("[PREVIEW] drawtext filter not available — skipping text overlays. "
                       "Install FFmpeg with --enable-libfreetype for subtitle support.")

    cmd = ["ffmpeg", "-y"] + inputs

    logger.info(
        f"[RENDER] Mix pass: filter_parts={len(filter_parts)} "
        f"video_map={video_map} audio_map={audio_map}"
    )
    if filter_parts:
        for i, fp in enumerate(filter_parts):
            logger.debug(f"[RENDER]   filter_part[{i}]: {fp[:160]}")
        if video_map == "0:v":
            filter_parts.insert(0, "[0:v]null[vpass]")
            video_map = "[vpass]"
        filter_complex = ";\n".join(filter_parts)
        cmd.extend(["-filter_complex", filter_complex])

    cmd.extend([
        "-map", video_map,
        "-map", audio_map,
        "-c:v", "libx264",
        "-preset", str(preset["preset"]),
        "-crf", str(preset["crf"]),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", str(preset["audio_bitrate"]),
        "-t", f"{total_dur:.3f}",
        "-movflags", "+faststart",
        str(output),
    ])

    try:
        await _run_ffmpeg(cmd, timeout=600.0)
    finally:
        for tf in txt_files:
            try:
                tf.unlink(missing_ok=True)
            except OSError:
                pass


def _resolve_audio_file(file_path: str, footage_dir: Path) -> Path:
    """Resolve an audio file path (narration/music)."""
    p = Path(file_path)
    if p.exists():
        return p
    # Try relative to workspace root (footage_dir parent)
    workspace_root = footage_dir.parent
    p2 = workspace_root / file_path
    if p2.exists():
        return p2
    # Try common subdirectories (music/, narration/)
    for subdir in ("music", "narration"):
        p3 = workspace_root / subdir / Path(file_path).name
        if p3.exists():
            return p3
    return p


async def _run_ffmpeg(cmd: List[str], timeout: float = 120.0) -> None:
    """Run an ffmpeg command and raise on failure.

    When a per-render log path is bound via ``_render_log_path`` (set by
    ``render_ffmpeg_preview``), appends the command, exit code, and full
    stderr there so a crash leaves a replayable trace on disk.
    """
    if cmd[0] != "ffmpeg":
        cmd = ["ffmpeg"] + cmd

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        _append_render_log(cmd, None, b"timed out")
        raise RuntimeError(f"FFmpeg timed out after {timeout}s")

    full_err_bytes = stderr or b""
    _append_render_log(cmd, proc.returncode, full_err_bytes)

    if proc.returncode != 0:
        full_err = full_err_bytes.decode(errors="replace") if full_err_bytes else "(no stderr)"
        logger.debug(f"[RENDER] FFmpeg stderr:\n{full_err}")
        err_text = full_err[-500:]
        raise RuntimeError(f"FFmpeg failed (exit {proc.returncode}): {err_text}")


def _append_render_log(cmd: List[str], returncode: Optional[int], stderr: bytes) -> None:
    """Append one sub-ffmpeg invocation to the active render log, if any."""
    path = _render_log_path.get()
    if path is None:
        return
    try:
        from datetime import datetime, timezone
        header = (
            f"\n===== {datetime.now(timezone.utc).isoformat()} "
            f"exit={returncode if returncode is not None else 'timeout'} =====\n"
        )
        body = "$ " + " ".join(cmd) + "\n"
        body += stderr.decode(errors="replace") if stderr else ""
        with open(path, "a") as f:
            f.write(header + body)
    except Exception:
        pass
