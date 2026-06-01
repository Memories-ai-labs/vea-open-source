"""
V2 Cropping Pipeline — on-demand dynamic reframing via FCPXML adjust-transform.

Matches v1 cropping logic exactly:
  1. Shot detection (PySceneDetect) to find scene cuts within a clip
  2. Content region detection (letterbox masking)
  3. Per-shot saliency via ViNet with stride-based overlapping windows
  4. Per-shot FCPXML adjust-transform (multiple sub-clips if multi-shot)

No video re-encoding — DaVinci Resolve applies transforms at render time.
"""
from __future__ import annotations

import asyncio
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, fromstring, tostring, ParseError

from src.pipelines.common.fcpxml_exporter import _format_fraction_seconds, _indent
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

CLIP_LEN = 32
STRIDE = 32
MAX_SALIENCY_FPS = 8


# ---------------------------------------------------------------------------
# Public API: crop a single clip with multi-shot support
# ---------------------------------------------------------------------------

async def crop_single_clip(
    source_path: str,
    start_sec: float,
    end_sec: float,
    tl_w: int,
    tl_h: int,
    src_w: int = 1920,
    src_h: int = 1080,
    progress_cb=None,
) -> "MultiShotCropResult":
    """
    Run full v1-equivalent cropping pipeline on a clip:
    shot detection → content masking → per-shot saliency → per-shot transforms.

    Returns MultiShotCropResult with one ShotCropResult per detected shot.
    If only one shot is detected, the list has one element (backward compatible).
    """
    from src.pipelines.v2.schemas import MultiShotCropResult, ShotCropResult, TransformSettings

    aspect_ratio = tl_w / max(tl_h, 1)

    if progress_cb:
        await progress_cb("detecting_shots")

    loop = asyncio.get_event_loop()
    try:
        shot_results = await loop.run_in_executor(
            None,
            lambda: _run_multishot_saliency(
                source_path, start_sec, end_sec,
                src_w, src_h, aspect_ratio,
                tl_w=tl_w, tl_h=tl_h,
            ),
        )
    except Exception as e:
        logger.warning(f"[CROP] Multi-shot saliency failed for {source_path}: {e}, using center")
        scale, pos_x, pos_y = _compute_fcpxml_transform(0.5, 0.5, src_w, src_h, aspect_ratio, tl_w=tl_w, tl_h=tl_h)
        shot_results = [(start_sec, end_sec, (scale, pos_x, pos_y), {"x": 0.5, "y": 0.5})]

    if progress_cb:
        await progress_cb("complete")

    shots = []
    for (shot_start, shot_end, (scale, pos_x, pos_y), center_norm) in shot_results:
        shots.append(ShotCropResult(
            source_start=shot_start,
            source_end=shot_end,
            transform=TransformSettings(
                scale_x=scale,
                scale_y=scale,
                position_x=pos_x,
                position_y=pos_y,
                rotation=0.0,
            ),
        ))

    if not shots:
        scale, pos_x, pos_y = _compute_fcpxml_transform(0.5, 0.5, src_w, src_h, aspect_ratio, tl_w=tl_w, tl_h=tl_h)
        shots = [ShotCropResult(
            source_start=start_sec,
            source_end=end_sec,
            transform=TransformSettings(
                scale_x=scale, scale_y=scale,
                position_x=pos_x, position_y=pos_y,
                rotation=0.0,
            ),
        )]

    return MultiShotCropResult(shots=shots)


# ---------------------------------------------------------------------------
# Core saliency pipeline (mirrors v1 DynamicCropping._process_clip)
# ---------------------------------------------------------------------------

def _run_multishot_saliency(
    source_path: str,
    start_sec: float,
    end_sec: float,
    src_w: int,
    src_h: int,
    aspect_ratio: float,
    tl_w: int = 0,
    tl_h: int = 0,
) -> List[Tuple[float, float, Tuple[float, float, float], dict]]:
    """
    Full v1-equivalent pipeline:
    1. Open clip segment
    2. Detect content region (letterbox masking)
    3. Detect shot boundaries (PySceneDetect)
    4. Per-shot saliency with stride-based windows
    5. Return list of (abs_start, abs_end, (scale, px, py), center_norm) per shot
    """
    import cv2
    import numpy as np
    import torch
    from moviepy import VideoFileClip
    from lib.utils.vinet_setup import load_vinet
    from src.pipelines.common.dynamic_cropping import (
        ShotDetector,
        detect_content_region,
        find_shot_center,
    )

    backend = load_vinet()
    input_width, input_height = backend.input_size
    clip_len = backend.clip_len

    # Open the clip segment
    full_video = VideoFileClip(source_path)
    # Clamp end to video duration
    vid_dur = full_video.duration or end_sec
    clamped_start = max(0.0, min(start_sec, vid_dur - 0.1))
    clamped_end = min(end_sec, vid_dur)
    if clamped_end <= clamped_start:
        full_video.close()
        return [(start_sec, end_sec,
                 _compute_fcpxml_transform(0.5, 0.5, src_w, src_h, aspect_ratio, tl_w=tl_w, tl_h=tl_h),
                 {"x": 0.5, "y": 0.5})]

    clip = full_video.subclipped(clamped_start, clamped_end)
    fps = clip.fps or 24.0
    total_frames = max(1, int(math.ceil(clip.duration * fps)))

    # ── Step 1: Content region detection (letterbox masking) ──
    # Matches v1 DynamicCropping._get_content_bounds (lines 282-286)
    sample_count = min(10, total_frames)
    sample_times = np.linspace(0, clip.duration, num=sample_count, endpoint=False)
    sample_frames = [clip.get_frame(float(t)) for t in sample_times]
    y1, y2, x1, x2 = detect_content_region(sample_frames)
    logger.info(f"[CROP] Content region: y=[{y1},{y2}] x=[{x1},{x2}] of {sample_frames[0].shape[:2]}")

    # ── Step 2: Shot detection (PySceneDetect) ──
    # Matches v1 DynamicCropping._process_clip (lines 522-546)
    try:
        shots_frames = ShotDetector.detect(clip, fps, total_frames)
    except Exception as e:
        logger.warning(f"[CROP] Shot detection failed: {e}")
        shots_frames = []

    # Ensure at least one shot covering the whole clip
    if not shots_frames:
        shots_frames = [(0, total_frames)]

    # Normalize shots (v1 lines 533-546)
    normalized = []
    cursor = 0
    for s_start, s_end in shots_frames:
        s_start = max(cursor, min(s_start, total_frames - 1))
        s_end = max(s_start + 1, min(s_end, total_frames))
        normalized.append((s_start, s_end))
        cursor = s_end
        if cursor >= total_frames:
            break
    if not normalized:
        normalized = [(0, total_frames)]
    elif normalized[-1][1] < total_frames:
        normalized.append((normalized[-1][1], total_frames))

    logger.info(f"[CROP] Detected {len(normalized)} shot(s) in {clip.duration:.1f}s clip")

    # ── Step 3: Per-shot saliency (matches v1 _compute_shot_centers) ──
    results = []

    for shot_start_frame, shot_end_frame in normalized:
        if shot_end_frame <= shot_start_frame:
            continue

        shot_start_time = shot_start_frame / fps
        shot_end_time = shot_end_frame / fps

        # Clamp to clip duration (v1 lines 352-356)
        if clip.duration is not None:
            margin = 0.1
            shot_start_time = min(shot_start_time, max(0.0, clip.duration - margin))
            shot_end_time = min(shot_end_time, clip.duration - margin)

        if shot_end_time <= shot_start_time:
            continue

        # Extract frames for this shot with content masking (v1 lines 362-375)
        shot_frames = []
        try:
            shot_clip = clip.subclipped(shot_start_time, shot_end_time)
            frame_limit = shot_end_frame - shot_start_frame

            for idx, frame in enumerate(shot_clip.iter_frames(fps=fps, dtype="uint8")):
                if idx >= frame_limit:
                    break
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Apply content mask (ignore letterboxing) — v1 lines 372-374
                masked = np.zeros_like(frame_bgr)
                masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
                resized = cv2.resize(masked, (input_width, input_height))
                shot_frames.append(resized)

            shot_clip.close()
        except Exception as e:
            logger.warning(f"[CROP] Failed to process shot {shot_start_time:.1f}s-{shot_end_time:.1f}s: {e}")
            # Fallback to center
            abs_start = clamped_start + shot_start_time
            abs_end = clamped_start + shot_end_time
            results.append((abs_start, abs_end,
                           _compute_fcpxml_transform(0.5, 0.5, src_w, src_h, aspect_ratio, tl_w=tl_w, tl_h=tl_h),
                           {"x": 0.5, "y": 0.5}))
            continue

        # Compute saliency with stride-based windows (v1 _compute_saliency, lines 299-327)
        saliency = _compute_saliency_strided(shot_frames, backend, clip_len, STRIDE)

        if saliency is None or saliency.ndim != 3:
            center_norm_x, center_norm_y = 0.5, 0.5
        else:
            cx_px, cy_px = find_shot_center(saliency)
            center_norm_x = cx_px / max(saliency.shape[2], 1)
            center_norm_y = cy_px / max(saliency.shape[1], 1)

        # Convert to FCPXML transform
        scale, pos_x, pos_y = _compute_fcpxml_transform(
            center_norm_x, center_norm_y, src_w, src_h, aspect_ratio,
            tl_w=tl_w, tl_h=tl_h,
        )

        # Convert to absolute source times
        abs_start = clamped_start + shot_start_time
        abs_end = clamped_start + shot_end_time

        results.append((abs_start, abs_end, (scale, pos_x, pos_y),
                        {"x": float(center_norm_x), "y": float(center_norm_y)}))

        logger.info(
            f"[CROP] Shot [{abs_start:.1f}-{abs_end:.1f}s] center=({center_norm_x:.3f},{center_norm_y:.3f}) "
            f"pos=({pos_x:.2f},{pos_y:.2f})"
        )

    clip.close()
    full_video.close()

    return results


def _compute_saliency_strided(
    frames: list,
    backend,
    clip_len: int,
    stride: int,
) -> "Optional[np.ndarray]":
    """
    Stride-based saliency computation matching v1 _compute_saliency (lines 299-327).
    Processes frames in overlapping windows of clip_len with given stride.
    """
    import numpy as np
    import torch

    if not frames:
        return None

    # Preprocess all frames to tensors
    input_width, input_height = backend.input_size
    preprocessed = []
    for frame in frames:
        # frames are already BGR resized from the extraction step
        rgb = frame[..., ::-1].copy()  # BGR to RGB
        preprocessed.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))

    chunks = []
    total = len(preprocessed)
    start = 0

    while start < total:
        end = min(start + clip_len, total)
        batch = list(preprocessed[start:end])
        valid = end - start

        if valid <= 0:
            break

        # Pad batch if needed (v1 lines 317-318)
        if valid < clip_len:
            batch.extend([batch[-1]] * (clip_len - valid))

        stacked = np.stack(batch, axis=1)[None]
        tensor = torch.from_numpy(stacked)
        saliency = backend.predict(tensor).squeeze(0).squeeze(1).numpy()
        chunks.append(saliency[:valid].astype(np.float16))
        start += stride

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# FCPXML transform math (unchanged from original)
# ---------------------------------------------------------------------------

def _compute_fcpxml_transform(
    center_x: float, center_y: float, src_w: int, src_h: int, aspect_ratio: float,
    tl_w: int = 0, tl_h: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute scale and position for FCPXML adjust-transform to crop to aspect_ratio.
    Mirrors the math in fcpxml_exporter.py.

    For vertical targets (e.g. 9:16) from landscape sources, the scale must be
    large enough to fill the timeline frame — use max(tl_w/src_w, tl_h/src_h).
    """
    # If timeline dimensions supplied, compute scale to fill the frame
    if tl_w > 0 and tl_h > 0:
        scale = max(tl_w / max(src_w, 1), tl_h / max(src_h, 1))
    else:
        # Legacy fallback: height-based
        tgt_h = src_h
        scale = src_h / max(tgt_h, 1)

    scaled_w = src_w * scale
    scaled_h = src_h * scale
    # Target dimensions in source-pixel space
    tgt_w = int(src_h * aspect_ratio) if tl_w == 0 else tl_w
    tgt_h = src_h if tl_h == 0 else tl_h
    max_pan_x = max(0.0, (scaled_w - tgt_w) / 2.0)
    max_pan_y = max(0.0, (scaled_h - tgt_h) / 2.0)
    pos_x_px = (0.5 - center_x) * scaled_w
    pos_y_px = (center_y - 0.5) * scaled_h
    clamped_x = max(-max_pan_x, min(max_pan_x, pos_x_px))
    clamped_y = max(-max_pan_y, min(max_pan_y, pos_y_px))
    norm = 19.2
    return scale, clamped_x / norm, clamped_y / norm


def _default_transform(
    tl_w: int, tl_h: int, aspect_ratio: float,
    src_w: int = 1920, src_h: int = 1080,
) -> Tuple[float, float, float]:
    """Return a centered, no-pan transform."""
    scale, px, py = _compute_fcpxml_transform(0.5, 0.5, src_w, src_h, aspect_ratio, tl_w=tl_w, tl_h=tl_h)
    return scale, 0.0, 0.0


# ---------------------------------------------------------------------------
# CropPipeline (batch FCPXML injection — kept for backward compat)
# ---------------------------------------------------------------------------

class CropPipeline:
    """Injects dynamic crop transforms into an existing FCPXML."""

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    async def run(self, aspect_ratio: float = 0.5625) -> str:
        source_path = self._find_latest_fcpxml()
        if not source_path:
            raise ValueError("No FCPXML found in workspace.")

        xml_text = Path(source_path).read_text(encoding="utf-8")
        try:
            root = fromstring(xml_text)
        except ParseError as e:
            raise ValueError(f"Cannot parse FCPXML: {e}")

        fmt = root.find(".//format")
        if fmt is None:
            raise ValueError("No <format> element found in FCPXML")
        tl_w = int(fmt.get("width", "1920"))
        tl_h = int(fmt.get("height", "1080"))

        modified = await self._inject_transforms(root, tl_w, tl_h, aspect_ratio)

        next_ver = self._next_version()
        out_path = str(self.workspace.get_fcpxml_path(next_ver))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        _indent(root)
        xml_str = tostring(root, encoding="unicode")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE fcpxml>\n')
            f.write(xml_str)

        logger.info(f"[CROP] Injected transforms for {modified} clips → {out_path}")
        return out_path

    async def _inject_transforms(
        self, root: Element, tl_w: int, tl_h: int, aspect_ratio: float
    ) -> int:
        modified = 0
        tasks = []

        for clip in root.findall(".//spine/asset-clip"):
            if clip.find("adjust-transform") is not None:
                continue
            lane = clip.get("lane", "0")
            try:
                if int(lane) < 0:
                    continue
            except ValueError:
                pass
            tasks.append((clip, self._compute_transform(clip, tl_w, tl_h, aspect_ratio)))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (clip, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"[CROP] Saliency failed for {clip.get('name')}: {result}")
                scale, px, py = _default_transform(tl_w, tl_h, aspect_ratio)
            else:
                scale, px, py = result
            SubElement(
                clip, "adjust-transform",
                scale=f"{scale:.6f} {scale:.6f}",
                position=f"{px:.4f} {py:.4f}",
            )
            modified += 1

        return modified

    async def _compute_transform(
        self, clip: Element, tl_w: int, tl_h: int, aspect_ratio: float
    ) -> Tuple[float, float, float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _run_saliency_sync(clip, tl_w, tl_h, aspect_ratio),
        )

    def _find_latest_fcpxml(self) -> Optional[str]:
        fcpxml_dir = self.workspace.root / "fcpxml"
        best_ver = 0
        best_path = None
        for f in fcpxml_dir.glob("edit_v*.fcpxml"):
            try:
                ver = int(f.stem.replace("edit_v", ""))
                if ver > best_ver:
                    best_ver = ver
                    best_path = str(f)
            except ValueError:
                pass
        if best_path is None:
            final = fcpxml_dir / "edit_final.fcpxml"
            if final.exists():
                return str(final)
        return best_path

    def _next_version(self) -> int:
        fcpxml_dir = self.workspace.root / "fcpxml"
        v = 1
        while (fcpxml_dir / f"edit_v{v}.fcpxml").exists():
            v += 1
        return v


# ---------------------------------------------------------------------------
# Legacy helpers (used by CropPipeline batch mode)
# ---------------------------------------------------------------------------

def _run_saliency_sync(
    clip: Element, tl_w: int, tl_h: int, aspect_ratio: float
) -> Tuple[float, float, float]:
    """Run saliency on a clip element from FCPXML. Falls back to center."""
    try:
        clip_name = clip.get("name", "")
        src_file = _find_source_for_clip(clip_name)
        if not src_file:
            return _default_transform(tl_w, tl_h, aspect_ratio)

        start_str = clip.get("start", "0s")
        dur_str = clip.get("duration", "1/24s")
        start_sec = float(_parse_time(start_str))
        dur_sec = float(_parse_time(dur_str))

        # Probe the TRUE source dimensions. Passing tl_w/tl_h into the
        # src_w/src_h slots (as this call previously did) collapses the saliency
        # transform to an identity no-op — defeating the crop entirely.
        src_w, src_h = _probe_dimensions(src_file)

        results = _run_multishot_saliency(
            src_file, start_sec, start_sec + dur_sec,
            src_w, src_h, aspect_ratio,
            tl_w=tl_w, tl_h=tl_h,
        )
        if results:
            return results[0][2]  # First shot's transform
        return _default_transform(tl_w, tl_h, aspect_ratio)

    except Exception as e:
        logger.debug(f"[CROP] Saliency unavailable, using center: {e}")
        return _default_transform(tl_w, tl_h, aspect_ratio)


def _probe_dimensions(source_path: str) -> Tuple[int, int]:
    """Best-effort (width, height) of a source video, for the batch crop path.

    Falls back to 1920x1080 if probing fails (cv2 is already a cropping dep).
    """
    try:
        import cv2
        cap = cv2.VideoCapture(source_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception as e:  # pragma: no cover - probe is best-effort
        logger.debug(f"[CROP] dimension probe failed for {source_path}: {e}")
    return 1920, 1080


def _find_source_for_clip(clip_name: str) -> Optional[str]:
    import glob as _glob
    candidates = _glob.glob(f"**/{clip_name}", recursive=True)
    for c in candidates:
        p = Path(c)
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            return str(p)
    return None


def _parse_time(val: str) -> Fraction:
    val = val.strip()
    if val == "0s":
        return Fraction(0)
    if val.endswith("s"):
        return Fraction(val[:-1])
    return Fraction(0)
