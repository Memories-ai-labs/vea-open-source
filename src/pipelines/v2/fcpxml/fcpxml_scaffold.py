"""
FCPXML Scaffold Generator — Phase 3 of the VEA v2 pipeline.

Produces a guaranteed-valid FCPXML 1.10 document from a Storyboard.
This is the "safe baseline" that the LLM agent enhances.

Design:
- Uses only the math utilities from fcpxml_exporter (no v1 pipeline dependencies).
- Clips are placed on the spine in storyboard order.
- Audio tracks (narration/music) added only if paths are provided.
- The output is structurally valid and can be imported into FCP/DaVinci Resolve as-is.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

from src.pipelines.common.fcpxml_exporter import (
    _file_uri,
    _format_fraction_seconds,
    _fps_to_fraction,
    _fraction_from_seconds,
    _indent,
    _quantize_duration_to_timeline,
)
from src.pipelines.v2.schemas import RetrievedClip, Shot, Storyboard

logger = logging.getLogger(__name__)

# Default timeline settings
DEFAULT_FPS = 24.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def build_scaffold(
    storyboard: Storyboard,
    clips_by_id: Dict[str, RetrievedClip],
    *,
    output_path: str,
    frame_rate: float = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    project_name: str = "VEA v2 Edit",
    narration_path: Optional[str] = None,
    narration_duration: float = 0.0,
    music_path: Optional[str] = None,
    music_duration: float = 0.0,
    music_gain_db: float = -12.0,
) -> str:
    """
    Generate a valid FCPXML 1.10 document from the storyboard.

    Args:
        storyboard: The planning output with shots.
        clips_by_id: Map from shot.id → RetrievedClip (from workspace clips.json).
        output_path: Where to write the .fcpxml file.
        narration_path: Optional path to narration audio file.
        narration_duration: Duration of narration file in seconds.
        music_path: Optional path to background music file.
        music_duration: Duration of music file in seconds.
        music_gain_db: Music volume adjustment in dB (default -12dB).

    Returns:
        The output_path on success.
    """
    fps_fraction = _fps_to_fraction(frame_rate)
    frame_duration_str = f"{fps_fraction.denominator}/{fps_fraction.numerator}s"

    # -------------------------------------------------------------------------
    # Root element
    # -------------------------------------------------------------------------
    fcpxml = Element("fcpxml", version="1.10")
    resources = SubElement(fcpxml, "resources")

    # Format resource
    fmt_id = "r1"
    SubElement(
        resources, "format",
        id=fmt_id,
        frameDuration=frame_duration_str,
        width=str(width),
        height=str(height),
        colorSpace="1-1-1 (Rec. 709)",
    )

    # -------------------------------------------------------------------------
    # Collect unique source video files from storyboard shots
    # -------------------------------------------------------------------------
    seen_sources: Dict[str, str] = {}  # source_path → asset_id
    asset_counter = 2

    for shot in storyboard.shots:
        clip = clips_by_id.get(shot.id)
        if clip is None or not clip.source_path:
            continue
        src = clip.source_path
        if src not in seen_sources:
            aid = f"r{asset_counter}"
            asset_counter += 1
            seen_sources[src] = aid
            _add_video_asset(resources, aid, src, clip)

    # Narration asset
    narration_aid = None
    if narration_path and narration_duration > 0:
        narration_aid = f"r{asset_counter}"
        asset_counter += 1
        _add_audio_asset(resources, narration_aid, narration_path, narration_duration)

    # Music asset
    music_aid = None
    if music_path and music_duration > 0:
        music_aid = f"r{asset_counter}"
        asset_counter += 1
        _add_audio_asset(resources, music_aid, music_path, music_duration, channels=2)

    # -------------------------------------------------------------------------
    # Library / Event / Project / Sequence / Spine
    # -------------------------------------------------------------------------
    library = SubElement(fcpxml, "library")
    event_el = SubElement(library, "event", name="VEA Export")
    project_el = SubElement(event_el, "project", name=project_name)
    sequence_el = SubElement(
        project_el, "sequence",
        format=fmt_id,
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = SubElement(sequence_el, "spine")

    timeline_frames = 0

    for shot in storyboard.shots:
        clip = clips_by_id.get(shot.id)
        if clip is None or not clip.source_path:
            logger.debug(f"[SCAFFOLD] Shot {shot.id!r} has no clip — skipping")
            continue

        asset_id = seen_sources.get(clip.source_path)
        if not asset_id:
            continue

        clip_duration_sec = clip.end_seconds - clip.start_seconds
        if clip_duration_sec <= 0:
            continue

        src_dur = _fraction_from_seconds(clip_duration_sec, fps_hint=frame_rate)
        tl_dur, tl_frames = _quantize_duration_to_timeline(src_dur, frame_rate)
        if tl_frames <= 0:
            continue

        offset = Fraction(timeline_frames, 1) / fps_fraction
        start = _fraction_from_seconds(clip.start_seconds, fps_hint=frame_rate)

        SubElement(
            spine, "asset-clip",
            name=_clip_name(shot, clip),
            ref=asset_id,
            offset=_format_fraction_seconds(offset),
            start=_format_fraction_seconds(start),
            duration=_format_fraction_seconds(tl_dur),
            format=fmt_id,
            enabled="1",
        )
        timeline_frames += tl_frames

    total_duration = Fraction(timeline_frames, 1) / fps_fraction
    sequence_el.set("duration", _format_fraction_seconds(total_duration))

    # -------------------------------------------------------------------------
    # Narration track (lane -1, attached to first clip)
    # -------------------------------------------------------------------------
    if narration_aid and narration_duration > 0 and len(spine) > 0:
        first_clip = spine[0]
        nar_dur = _fraction_from_seconds(narration_duration, fps_hint=frame_rate)
        nar_tl_dur, _ = _quantize_duration_to_timeline(nar_dur, frame_rate)
        SubElement(
            first_clip, "asset-clip",
            name=Path(narration_path).name,
            ref=narration_aid,
            lane="-1",
            offset="0s",
            start="0s",
            duration=_format_fraction_seconds(nar_tl_dur),
            role="dialogue",
            enabled="1",
        )

    # -------------------------------------------------------------------------
    # Music track (lane -2, under spine as gap clip)
    # -------------------------------------------------------------------------
    if music_aid and music_duration > 0:
        mus_dur = _fraction_from_seconds(music_duration, fps_hint=frame_rate)
        mus_tl_dur, _ = _quantize_duration_to_timeline(mus_dur, frame_rate)
        music_clip = SubElement(
            spine, "asset-clip",
            name=Path(music_path).name,
            ref=music_aid,
            lane="-2",
            offset="0s",
            start="0s",
            duration=_format_fraction_seconds(mus_tl_dur),
            role="music",
            enabled="1",
        )
        SubElement(music_clip, "adjust-volume", amount=f"{music_gain_db:.1f}dB")

    # -------------------------------------------------------------------------
    # Write to disk
    # -------------------------------------------------------------------------
    _indent(fcpxml)
    xml_str = tostring(fcpxml, encoding="unicode")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        f.write(xml_str)

    logger.info(f"[SCAFFOLD] Wrote {len(storyboard.shots)} shots → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_video_asset(
    resources: Element, asset_id: str, src_path: str, clip: RetrievedClip
) -> None:
    """Add a <asset> + <media-rep> for a video file."""
    # Asset duration = full source file duration (safe upper bound)
    # We use end_seconds of the clip as a conservative estimate
    duration_frac = _fraction_from_seconds(clip.end_seconds)
    uri = _file_uri(src_path)
    asset = SubElement(
        resources, "asset",
        id=asset_id,
        name=Path(src_path).name,
        start="0s",
        duration=_format_fraction_seconds(duration_frac),
        hasVideo="1",
        hasAudio="1",
        audioSources="1",
        audioChannels="2",
        audioRate="48000",
        src=uri,
    )
    SubElement(asset, "media-rep", kind="original-media", src=uri)


def _add_audio_asset(
    resources: Element,
    asset_id: str,
    src_path: str,
    duration_seconds: float,
    channels: int = 1,
) -> None:
    """Add a <asset> + <media-rep> for an audio-only file."""
    duration_frac = _fraction_from_seconds(duration_seconds)
    uri = _file_uri(src_path)
    asset = SubElement(
        resources, "asset",
        id=asset_id,
        name=Path(src_path).name,
        start="0s",
        duration=_format_fraction_seconds(duration_frac),
        hasVideo="0",
        hasAudio="1",
        audioSources="1",
        audioChannels=str(channels),
        audioRate="48000",
        src=uri,
    )
    SubElement(asset, "media-rep", kind="original-media", src=uri)


def _clip_name(shot: Shot, clip: RetrievedClip) -> str:
    name = shot.id or clip.video_name or "clip"
    return name[:64]  # FCPXML name length cap
