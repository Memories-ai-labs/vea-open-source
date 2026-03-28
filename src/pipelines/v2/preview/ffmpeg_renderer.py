"""
FFmpeg preview renderer — EditDecision → MP4 without DaVinci Resolve.

Produces a fast 480p draft preview using FFmpeg filter graphs.
Supports: clip concatenation, transforms/crops, audio mixing, speed changes,
text overlays, and multi-shot crop transforms.

Two-pass approach:
  Pass 1 (parallel): Extract + process each clip to temp files at 480p
  Pass 2 (sequential): Concatenate clips + mix audio → final MP4
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def render_ffmpeg_preview(
    edit: EditDecision,
    footage_dir: Path,
    output_path: Path,
    resolution: int = 480,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
) -> str:
    """
    Render EditDecision to MP4 via FFmpeg.

    Args:
        edit: Complete edit decision with clips, narration, music, titles
        footage_dir: Directory containing source footage files
        output_path: Where to write the output MP4
        resolution: Output height in pixels (width scales proportionally)
        progress_callback: async callback receiving 0-100 progress

    Returns:
        str path to the rendered MP4
    """
    if not edit.clips:
        raise ValueError("No clips in edit decision")

    # Ensure ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tl_w = edit.timeline.width
    tl_h = edit.timeline.height
    tl_fps = edit.timeline.fps

    # Output dimensions (preserve aspect ratio, scale to target height)
    out_h = resolution
    out_w = int(tl_w * out_h / tl_h)
    # Ensure even dimensions for H.264
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    track1_clips = [c for c in edit.clips if c.track == 1]
    if not track1_clips:
        raise ValueError("No track-1 clips in edit decision")

    tmpdir = Path(tempfile.mkdtemp(prefix="vea_preview_"))

    try:
        # ── Pass 1: Extract + process each clip in parallel ──
        total_dur = sum(_clip_duration(c) for c in track1_clips)
        completed_dur = 0.0

        async def process_clip(idx: int, clip: ClipDecision) -> Path:
            nonlocal completed_dur
            out_file = tmpdir / f"clip_{idx:03d}.mp4"
            source = _resolve_source(clip, footage_dir)
            if not source.exists():
                logger.warning(f"[PREVIEW] Source not found: {source}, skipping clip {clip.id}")
                # Create a black placeholder
                dur = _clip_duration(clip)
                await _run_ffmpeg([
                    "-f", "lavfi", "-i", f"color=black:s={out_w}x{out_h}:d={dur:.3f}:r={tl_fps}",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
                    "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-c:a", "aac", "-b:a", "128k",
                    str(out_file),
                ])
                return out_file

            # Build clip segments (multi-shot or single)
            segments = _get_clip_segments(clip, tl_w, tl_h)
            if len(segments) == 1:
                await _render_single_segment(
                    source, segments[0], clip, out_file,
                    out_w, out_h, tl_fps,
                )
            else:
                # Multi-shot: render each segment then concat
                seg_files = []
                for si, seg in enumerate(segments):
                    seg_file = tmpdir / f"clip_{idx:03d}_shot_{si:02d}.mp4"
                    await _render_single_segment(
                        source, seg, clip, seg_file,
                        out_w, out_h, tl_fps,
                    )
                    seg_files.append(seg_file)
                await _concat_files(seg_files, out_file)

            completed_dur += _clip_duration(clip)
            if progress_callback:
                pct = min(80.0, (completed_dur / total_dur) * 80.0)
                await progress_callback(pct)
            return out_file

        clip_tasks = [process_clip(i, c) for i, c in enumerate(track1_clips)]
        clip_files = await asyncio.gather(*clip_tasks)

        # Filter out any clips that failed
        valid_clip_files = [f for f in clip_files if f.exists()]
        if not valid_clip_files:
            raise RuntimeError("All clips failed to render")

        if progress_callback:
            await progress_callback(80.0)

        # ── Pass 2: Concatenate clips + mix audio ──
        concat_video = tmpdir / "concat.mp4"
        await _concat_files(valid_clip_files, concat_video)

        if progress_callback:
            await progress_callback(85.0)

        # ── Pass 3: Mix in narration + music + titles ──
        await _mix_audio_and_titles(
            concat_video, edit, footage_dir, output_path,
            out_w, out_h, tl_fps, track1_clips,
        )

        if progress_callback:
            await progress_callback(100.0)

        logger.info(f"[PREVIEW] FFmpeg render complete: {output_path}")
        return str(output_path)

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
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
) -> str:
    """Convert TransformSettings to an FFmpeg crop+scale filter string."""
    sx = transform.scale_x or 1.0
    sy = transform.scale_y or 1.0

    # If scale is ~1.0 and position is ~0, just scale (no crop needed)
    if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01 and abs(transform.position_x) < 1 and abs(transform.position_y) < 1:
        return f"scale={out_w}:{out_h}:flags=bilinear"

    # Compute crop region in source pixels
    crop_w = tl_w / sx
    crop_h = tl_h / sy
    crop_x = (src_w / 2.0) - (crop_w / 2.0) - (transform.position_x / sx)
    crop_y = (src_h / 2.0) - (crop_h / 2.0) + (transform.position_y / sy)

    # Clamp to source bounds
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)
    crop_x = max(0, min(crop_x, src_w - crop_w))
    crop_y = max(0, min(crop_y, src_h - crop_h))

    return (
        f"crop={int(crop_w)}:{int(crop_h)}:{int(crop_x)}:{int(crop_y)},"
        f"scale={out_w}:{out_h}:flags=bilinear"
    )


async def _render_single_segment(
    source: Path,
    segment: ClipSegment,
    clip: ClipDecision,
    out_file: Path,
    out_w: int, out_h: int,
    fps: float,
) -> None:
    """Render a single clip segment (one transform) to a temp file."""
    src_start, src_end, transform = segment
    duration = src_end - src_start
    speed_rate = clip.speed.rate if clip.speed else 1.0

    # Video filter chain
    vf_parts = []

    # Transform (crop + scale)
    vf_parts.append(_build_transform_filter(
        transform, clip.source_width, clip.source_height,
        1920, 1080,  # Use full HD for crop math, we scale to out_w/out_h
        out_w, out_h,
    ))

    # Speed change
    if abs(speed_rate - 1.0) > 0.01:
        vf_parts.append(f"setpts=PTS/{speed_rate}")

    vf = ",".join(vf_parts)

    # Audio filter chain
    af_parts = []
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_file),
    ])

    await _run_ffmpeg(cmd)


def _atempo_chain(rate: float) -> List[str]:
    """Build atempo filter chain for a given speed rate (each atempo limited to 0.5-2.0)."""
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

    # Write concat list
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


async def _mix_audio_and_titles(
    video_file: Path,
    edit: EditDecision,
    footage_dir: Path,
    output: Path,
    out_w: int, out_h: int,
    fps: float,
    track1_clips: List[ClipDecision],
) -> None:
    """Mix narration + music audio and overlay titles onto concatenated video."""
    has_narration = bool(edit.narration)
    has_music = edit.music is not None
    has_titles = bool(edit.titles)

    # If nothing to mix, just copy
    if not has_narration and not has_music and not has_titles:
        shutil.copy2(video_file, output)
        return

    total_dur = sum(_clip_duration(c) for c in track1_clips)
    inputs = ["-i", str(video_file)]
    input_idx = 1  # [0] is the video
    filter_parts = []
    audio_streams = ["[0:a]"]

    # Narration segments
    if has_narration:
        for seg in edit.narration:
            nar_path = _resolve_audio_file(seg.file, footage_dir)
            if not nar_path.exists():
                logger.warning(f"[PREVIEW] Narration file not found: {nar_path}")
                continue
            inputs.extend(["-i", str(nar_path)])
            delay_ms = int(seg.timeline_offset * 1000)
            gain = f"volume={seg.gain_db}dB," if seg.gain_db != 0 else ""
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
        if mus_path.exists():
            inputs.extend(["-i", str(mus_path)])
            mus_dur = mus.duration if mus.duration > 0 else total_dur
            gain = f"volume={mus.gain_db}dB," if mus.gain_db != 0 else ""
            filter_parts.append(
                f"[{input_idx}:a]atrim={mus.start:.3f}:duration={mus_dur:.3f},"
                f"asetpts=PTS-STARTPTS,{gain}"
                f"afade=t=out:st={max(0, mus_dur - 2):.3f}:d=2[mus]"
            )
            audio_streams.append("[mus]")
            input_idx += 1

    # Audio mixing
    if len(audio_streams) > 1:
        mix_inputs = "".join(audio_streams)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(audio_streams)}:"
            f"duration=first:dropout_transition=2[aout]"
        )
        audio_map = "[aout]"
    else:
        audio_map = "0:a"

    # Text overlays
    video_map = "[0:v]"
    if has_titles:
        vf_chain = "[0:v]"
        for ti, title in enumerate(edit.titles):
            # Scale font size to output resolution
            scaled_size = max(12, int(title.font_size * out_h / edit.timeline.height))
            escaped = title.text.replace("'", "\\'").replace(":", "\\:")
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
            vf_chain += (
                f"drawtext=text='{escaped}':"
                f"fontsize={scaled_size}:fontcolor=white:"
                f"x=(w-text_w)/2:y={y_expr}{box_opts}:"
                f"enable='between(t\\,{start_t:.3f}\\,{end_t:.3f})'"
            )
            if ti < len(edit.titles) - 1:
                vf_chain += ","
        vf_chain += "[vout]"
        filter_parts.append(vf_chain)
        video_map = "[vout]"

    cmd = ["ffmpeg", "-y"] + inputs

    if filter_parts:
        filter_complex = ";\n".join(filter_parts)
        cmd.extend(["-filter_complex", filter_complex])

    cmd.extend([
        "-map", video_map,
        "-map", audio_map,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k",
        "-t", f"{total_dur:.3f}",
        "-movflags", "+faststart",
        str(output),
    ])

    await _run_ffmpeg(cmd)


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
    return p


async def _run_ffmpeg(cmd: List[str], timeout: float = 120.0) -> None:
    """Run an ffmpeg command and raise on failure."""
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
        raise RuntimeError(f"FFmpeg timed out after {timeout}s")

    if proc.returncode != 0:
        err_text = stderr.decode()[-500:] if stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed (exit {proc.returncode}): {err_text}")
