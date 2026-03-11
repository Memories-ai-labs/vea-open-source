"""
Deterministic FCPXML 1.10 compiler: EditDecision → valid FCPXML.

No LLM involvement. All creative decisions live in EditDecision (JSON);
this module only handles XML structure and rational-fraction time math.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

from src.pipelines.common.fcpxml_exporter import (
    _file_uri,
    _format_fraction_seconds,
    _fps_to_fraction,
    _fraction_from_seconds,
    _indent,
    _quantize_duration_to_timeline,
)
from src.pipelines.v2.schemas import (
    ClipDecision,
    EditDecision,
    MusicTrack,
    NarrationSegment,
    TextOverlay,
    TransitionSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_edit_decision(edit: EditDecision, output_path: str) -> str:
    """
    Compile an EditDecision to FCPXML 1.10 and write to disk.

    Returns the output_path on success.
    """
    fps = edit.timeline.fps
    fps_frac = _fps_to_fraction(fps)
    frame_dur_str = f"{fps_frac.denominator}/{fps_frac.numerator}s"

    # --- Root + resources ---
    fcpxml = Element("fcpxml", version="1.10")
    resources = SubElement(fcpxml, "resources")

    fmt_id = "r1"
    SubElement(
        resources, "format",
        id=fmt_id,
        frameDuration=frame_dur_str,
        width=str(edit.timeline.width),
        height=str(edit.timeline.height),
        colorSpace="1-1-1 (Rec. 709)",
    )

    # --- Register video assets ---
    asset_counter = 2
    source_asset_map: Dict[str, str] = {}  # source_file → asset_id
    source_max_end: Dict[str, float] = {}  # track max end for duration estimate

    for clip in edit.clips:
        key = clip.source_path or clip.source_file
        if key not in source_max_end or clip.source_end > source_max_end[key]:
            source_max_end[key] = clip.source_end

    for key, max_end in source_max_end.items():
        aid = f"r{asset_counter}"
        asset_counter += 1
        source_asset_map[key] = aid
        # Find a clip to get the file name
        name = Path(key).name
        _add_video_asset(resources, aid, key, max_end, name, fps)

    # --- Register narration assets ---
    narration_asset_map: Dict[str, str] = {}
    for seg in edit.narration:
        if seg.file not in narration_asset_map:
            aid = f"r{asset_counter}"
            asset_counter += 1
            narration_asset_map[seg.file] = aid
            est_dur = seg.start + seg.duration + 10.0  # safe overestimate
            _add_audio_asset(resources, aid, seg.file, est_dur)

    # --- Register music asset ---
    music_aid: Optional[str] = None
    if edit.music and edit.music.file:
        music_aid = f"r{asset_counter}"
        asset_counter += 1
        est_dur = edit.music.start + (edit.music.duration or 600.0) + 10.0
        _add_audio_asset(resources, music_aid, edit.music.file, est_dur, channels=2)

    # --- Register title effect ---
    title_effect_id: Optional[str] = None
    if edit.titles:
        title_effect_id = f"r{asset_counter}"
        asset_counter += 1
        SubElement(
            resources, "effect",
            id=title_effect_id,
            name="Basic Title",
            uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",
        )

    # --- Library / Event / Project / Sequence / Spine ---
    library = SubElement(fcpxml, "library")
    event_el = SubElement(library, "event", name="VEA Export")
    project_el = SubElement(event_el, "project", name=edit.timeline.name)
    sequence_el = SubElement(
        project_el, "sequence",
        format=fmt_id,
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = SubElement(sequence_el, "spine")

    # --- Build spine clips + transitions ---
    timeline_frames = 0
    clip_timeline_ranges: List[Tuple[int, int]] = []  # (start_frame, end_frame) per clip

    for i, clip in enumerate(edit.clips):
        key = clip.source_path or clip.source_file
        asset_id = source_asset_map.get(key)
        if not asset_id:
            logger.warning(f"[COMPILER] No asset for clip {clip.id} source={key}")
            continue

        src_dur_sec = clip.source_end - clip.source_start
        if src_dur_sec <= 0:
            continue

        # Apply speed change
        speed_rate = clip.speed.rate if clip.speed else 1.0
        if speed_rate <= 0:
            speed_rate = 1.0
        tl_dur_sec = src_dur_sec / speed_rate

        src_dur_frac = _fraction_from_seconds(tl_dur_sec, fps_hint=fps)
        tl_dur, tl_frames = _quantize_duration_to_timeline(src_dur_frac, fps)
        if tl_frames <= 0:
            continue

        offset = Fraction(timeline_frames, 1) / fps_frac
        start = _fraction_from_seconds(clip.source_start, fps_hint=fps)

        clip_el = SubElement(
            spine, "asset-clip",
            name=clip.label or clip.id,
            ref=asset_id,
            offset=_format_fraction_seconds(offset),
            start=_format_fraction_seconds(start),
            duration=_format_fraction_seconds(tl_dur),
            format=fmt_id,
            enabled="1",
        )

        # Speed remap via timeMap
        if clip.speed and clip.speed.rate != 1.0:
            _add_speed_remap(clip_el, src_dur_sec, tl_dur, fps_frac, clip.speed.rate)

        # Transform (crop/reframe)
        if clip.transform:
            t = clip.transform
            SubElement(
                clip_el, "adjust-transform",
                position=f"{t.position_x:.1f} {t.position_y:.1f}",
                scale=f"{t.scale_x:.4f} {t.scale_y:.4f}",
                rotation=f"{t.rotation:.1f}",
            )

        # Per-clip gain
        if clip.gain_db is not None:
            SubElement(clip_el, "adjust-volume", amount=f"{clip.gain_db:.1f}dB")

        clip_timeline_ranges.append((timeline_frames, timeline_frames + tl_frames))
        timeline_frames += tl_frames

        # NOTE: Transitions disabled for now — hard cuts only.
        # Cross-dissolve transitions cause audio overlap between adjacent clips.
        # TODO: Re-enable with proper audio crossfade handling.
        # if clip.transition_after and i < len(edit.clips) - 1:
        #     trans_dur = _fraction_from_seconds(clip.transition_after.duration_seconds, fps_hint=fps)
        #     _, trans_frames = _quantize_duration_to_timeline(trans_dur, fps)
        #     _add_transition(spine, clip.transition_after, fps_frac, fps)
        #     timeline_frames -= trans_frames

    total_duration = Fraction(timeline_frames, 1) / fps_frac
    sequence_el.set("duration", _format_fraction_seconds(total_duration))

    # --- Place narration segments (lane -1, attached to spine clips) ---
    spine_clips = [el for el in spine if el.tag == "asset-clip"]
    for seg in edit.narration:
        nar_aid = narration_asset_map.get(seg.file)
        if not nar_aid:
            continue
        parent_clip, parent_idx = _find_parent_clip(
            spine_clips, clip_timeline_ranges, seg.timeline_offset, fps_frac
        )
        if parent_clip is None:
            continue

        parent_start_sec = float(Fraction(clip_timeline_ranges[parent_idx][0], 1) / fps_frac)
        rel_offset = seg.timeline_offset - parent_start_sec

        nar_dur = _fraction_from_seconds(seg.duration, fps_hint=fps)
        nar_tl_dur, _ = _quantize_duration_to_timeline(nar_dur, fps)
        nar_start = _fraction_from_seconds(seg.start, fps_hint=fps)
        nar_offset = _fraction_from_seconds(rel_offset, fps_hint=fps)

        nar_el = SubElement(
            parent_clip, "asset-clip",
            name=Path(seg.file).stem,
            ref=nar_aid,
            lane="-1",
            offset=_format_fraction_seconds(nar_offset),
            start=_format_fraction_seconds(nar_start),
            duration=_format_fraction_seconds(nar_tl_dur),
            role="dialogue",
            enabled="1",
        )
        if seg.gain_db != 0:
            SubElement(nar_el, "adjust-volume", amount=f"{seg.gain_db:.1f}dB")

    # --- Place music (lane -2, attached to first spine clip) ---
    if edit.music and music_aid and spine_clips:
        mus = edit.music
        mus_dur_sec = mus.duration if mus.duration > 0 else float(total_duration)
        mus_dur = _fraction_from_seconds(mus_dur_sec, fps_hint=fps)
        mus_tl_dur, _ = _quantize_duration_to_timeline(mus_dur, fps)
        mus_start = _fraction_from_seconds(mus.start, fps_hint=fps)

        mus_el = SubElement(
            spine_clips[0], "asset-clip",
            name=Path(mus.file).stem,
            ref=music_aid,
            lane="-2",
            offset="0s",
            start=_format_fraction_seconds(mus_start),
            duration=_format_fraction_seconds(mus_tl_dur),
            role="music",
            enabled="1",
        )
        SubElement(mus_el, "adjust-volume", amount=f"{mus.gain_db:.1f}dB")

    # --- Place titles (positive lanes, attached to spine clips) ---
    for title in edit.titles:
        if not title_effect_id:
            break
        parent_clip, parent_idx = _find_parent_clip(
            spine_clips, clip_timeline_ranges, title.timeline_offset, fps_frac
        )
        if parent_clip is None:
            continue

        parent_start_sec = float(Fraction(clip_timeline_ranges[parent_idx][0], 1) / fps_frac)
        rel_offset = title.timeline_offset - parent_start_sec

        t_dur = _fraction_from_seconds(title.duration, fps_hint=fps)
        t_tl_dur, _ = _quantize_duration_to_timeline(t_dur, fps)
        t_offset = _fraction_from_seconds(rel_offset, fps_hint=fps)

        title_el = SubElement(
            parent_clip, "title",
            name=title.text[:32],
            ref=title_effect_id,
            lane=str(title.lane),
            offset=_format_fraction_seconds(t_offset),
            start="0s",
            duration=_format_fraction_seconds(t_tl_dur),
        )
        text_style_def = SubElement(title_el, "text-style-def", id=f"ts-{id(title)}")
        SubElement(
            text_style_def, "text-style",
            font="Helvetica Neue",
            fontSize=str(title.font_size),
            fontColor="1 1 1 1",
            alignment="center",
        )
        text_el = SubElement(title_el, "text")
        ts = SubElement(text_el, "text-style", ref=f"ts-{id(title)}")
        ts.text = title.text

    # --- Write to disk ---
    _indent(fcpxml)
    xml_str = tostring(fcpxml, encoding="unicode")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        f.write(xml_str)

    logger.info(
        f"[COMPILER] Wrote {len(edit.clips)} clips, "
        f"{len(edit.narration)} narration segs, "
        f"{'music' if edit.music else 'no music'}, "
        f"{len(edit.titles)} titles → {output_path}"
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_video_asset(
    resources: Element,
    asset_id: str,
    src_path: str,
    max_end: float,
    name: str,
    fps: float,
) -> None:
    duration_frac = _fraction_from_seconds(max_end + 10.0, fps_hint=fps)
    uri = _file_uri(src_path) if "/" in src_path else f"file:///media/{name}"
    asset = SubElement(
        resources, "asset",
        id=asset_id,
        name=name,
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
    duration_frac = _fraction_from_seconds(duration_seconds)
    uri = _file_uri(src_path) if "/" in src_path else f"file:///media/{Path(src_path).name}"
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


def _add_transition(
    spine: Element,
    spec: TransitionSpec,
    fps_frac: Fraction,
    fps: float,
) -> None:
    dur = _fraction_from_seconds(spec.duration_seconds, fps_hint=fps)
    tl_dur, _ = _quantize_duration_to_timeline(dur, fps)
    name_map = {
        "cross-dissolve": "Cross Dissolve",
        "fade-in": "Fade In",
        "fade-out": "Fade Out",
    }
    SubElement(
        spine, "transition",
        name=name_map.get(spec.type, "Cross Dissolve"),
        duration=_format_fraction_seconds(tl_dur),
    )


def _add_speed_remap(
    clip_el: Element,
    src_dur_sec: float,
    tl_dur: Fraction,
    fps_frac: Fraction,
    rate: float,
) -> None:
    """Add a <timeMap> for constant speed changes."""
    time_map = SubElement(clip_el, "timeMap")
    SubElement(time_map, "timept", time="0s", value="0s", interp="smooth2")
    src_end = _fraction_from_seconds(src_dur_sec)
    SubElement(
        time_map, "timept",
        time=_format_fraction_seconds(tl_dur),
        value=_format_fraction_seconds(src_end),
        interp="smooth2",
    )


def _find_parent_clip(
    spine_clips: List[Element],
    clip_ranges: List[Tuple[int, int]],
    timeline_offset_sec: float,
    fps_frac: Fraction,
) -> Tuple[Optional[Element], int]:
    """Find which spine clip contains the given timeline offset."""
    for i, (start_frame, end_frame) in enumerate(clip_ranges):
        start_sec = float(Fraction(start_frame, 1) / fps_frac)
        end_sec = float(Fraction(end_frame, 1) / fps_frac)
        if start_sec <= timeline_offset_sec < end_sec:
            return spine_clips[i], i
    # Fallback: attach to last clip
    if spine_clips:
        return spine_clips[-1], len(spine_clips) - 1
    return None, -1
