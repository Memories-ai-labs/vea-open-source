"FCPXML export helpers for multiround explainable outputs."

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring


_COMMON_FPS = (
    Fraction(24000, 1001),
    Fraction(30000, 1001),
    Fraction(60000, 1001),
    Fraction(12000, 1001),
    Fraction(24000, 1),
    Fraction(25, 1),
    Fraction(30, 1),
    Fraction(50, 1),
    Fraction(60, 1),
)


def _fps_to_fraction(fps: float) -> Fraction:
    if fps <= 0:
        return Fraction(0, 1)
    for frac in _COMMON_FPS:
        if abs(float(frac) - fps) < 1e-3:
            return frac
    # Use string conversion to avoid float precision surprises
    return Fraction(str(fps)).limit_denominator(1000)


def _seconds_to_fraction(seconds: float, max_denominator: int = 48000) -> Fraction:
    return Fraction(str(seconds)).limit_denominator(max_denominator)


def _format_fraction_seconds(value: Fraction) -> str:
    frac = value.limit_denominator(48000)
    return f"{frac.numerator}/{frac.denominator}s"



def _frames_to_seconds(frame_count: int, fps: float) -> Fraction:
    if frame_count <= 0 or fps <= 0:
        return Fraction(0, 1)
    fps_fraction = _fps_to_fraction(fps)
    if fps_fraction.numerator == 0:
        return Fraction(0, 1)
    return Fraction(frame_count, 1) / fps_fraction


def _quantize_duration_to_timeline(duration: Fraction, timeline_fps: float) -> tuple[Fraction, int]:
    if duration <= 0:
        return Fraction(0, 1), 0
    timeline_fps_fraction = _fps_to_fraction(timeline_fps)
    if timeline_fps_fraction.numerator == 0:
        return Fraction(0, 1), 0
    frames_fraction = duration * timeline_fps_fraction
    timeline_frames = int(frames_fraction + Fraction(1, 2))
    if timeline_frames <= 0:
        return Fraction(0, 1), 0
    quantized = Fraction(timeline_frames, 1) / timeline_fps_fraction
    return quantized, timeline_frames


def _fraction_from_seconds(seconds: float, fps_hint: float | None = None) -> Fraction:
    if fps_hint and fps_hint > 0:
        max_denominator = max(int(round(fps_hint)) * 1000, 1)
    else:
        max_denominator = 48000
    return _seconds_to_fraction(seconds, max_denominator=max_denominator)


def _file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _indent(elem: Element, level: int = 0) -> None:
    indent_str = "  "
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i



def export_fcpxml(
    *,
    timeline_metadata: List[dict],
    video_asset_map: Dict[str, str],
    narration_asset_map: Dict[str, str],
    music_asset_path: Optional[str],
    music_asset_name: Optional[str],
    music_duration: float,
    music_gain_db: Optional[float],
    output_path: str,
    frame_rate: float = 24.0,
    project_name: str = "VEA Edit",
    event_name: str = "VEA Export",
) -> str:
    if not timeline_metadata:
        raise ValueError("timeline_metadata is required to export FCPXML")

    width = 1080
    height = 1920
    for meta in timeline_metadata:
        crop = meta.get("crop") or {}
        output_size = crop.get("output_size") or []
        if len(output_size) == 2:
            width, height = int(output_size[0]), int(output_size[1])
            break

    fcpxml = Element("fcpxml", version="1.10")
    resources = SubElement(fcpxml, "resources")

    timeline_fps_fraction = _fps_to_fraction(frame_rate)
    if timeline_fps_fraction.numerator == 0:
        timeline_fps_fraction = Fraction(24, 1)
    frame_duration_value = f"{timeline_fps_fraction.denominator}/{timeline_fps_fraction.numerator}s"

    format_id = "f1"
    SubElement(
        resources,
        "format",
        id=format_id,
        frameDuration=frame_duration_value,
        width=str(width),
        height=str(height),
        colorSpace="1-1-1 (Rec. 709)",
    )

    asset_usage: Dict[str, Fraction] = {}
    for meta in timeline_metadata:
        source = meta.get("source", {}) or {}
        file_name = source.get("file_name")
        if not file_name:
            continue
        timings = meta.get("timings", {}) or {}
        fps_hint = source.get("fps") or meta.get("timeline", {}).get("frame_rate") or frame_rate
        source_end_fraction = _fraction_from_seconds(
            float(timings.get("source_end", timings.get("applied_end", 0.0))),
            fps_hint=fps_hint,
        )
        existing = asset_usage.get(file_name, Fraction(0, 1))
        if source_end_fraction > existing:
            asset_usage[file_name] = source_end_fraction

    asset_id_map: Dict[str, str] = {}
    for idx, (file_name, local_path) in enumerate(video_asset_map.items(), start=2):
        asset_id = f"a{idx}"
        asset_id_map[file_name] = asset_id
        asset_duration = asset_usage.get(file_name, Fraction(0, 1))
        asset_attrs = dict(
            id=asset_id,
            name=file_name,
            start="0s",
            duration=_format_fraction_seconds(asset_duration),
            hasVideo="1",
            hasAudio="1",
            audioSources="1",
            audioChannels="2",
            audioRate="48000",
            src=_file_uri(local_path),
        )
        SubElement(resources, "asset", asset_attrs).append(
            Element("media-rep", kind="original-media", src=_file_uri(local_path))
        )

    narration_asset_id_map: Dict[str, str] = {}
    next_index = len(video_asset_map) + 2
    for original_path, copied_path in narration_asset_map.items():
        asset_id = f"a{next_index}"
        next_index += 1
        narration_asset_id_map[original_path] = asset_id
        durations: List[Fraction] = []
        for meta in timeline_metadata:
            details = meta.get("audio", {}).get("details", {}) or {}
            if details.get("narration_path") != original_path:
                continue
            timeline_info = meta.get("timeline", {}) or {}
            clip_fps = timeline_info.get("frame_rate") or frame_rate
            frame_count = timeline_info.get("frame_count")
            if frame_count:
                durations.append(_frames_to_seconds(int(round(frame_count)), float(clip_fps)))
            else:
                durations.append(
                    _fraction_from_seconds(
                        float(timeline_info.get("duration", 0.0)),
                        fps_hint=clip_fps,
                    )
                )
        narration_duration = max(durations, default=Fraction(0, 1))
        SubElement(
            resources,
            "asset",
            {
                "id": asset_id,
                "name": Path(copied_path).name,
                "start": "0s",
                "duration": _format_fraction_seconds(narration_duration),
                "hasVideo": "0",
                "hasAudio": "1",
                "audioSources": "1",
                "audioChannels": "1",
                "audioRate": "48000",
                "src": _file_uri(copied_path),
            },
        ).append(
            Element("media-rep", kind="original-media", src=_file_uri(copied_path))
        )

    music_asset_id = None
    music_duration_fraction = Fraction(0, 1)
    if music_asset_path and music_asset_name and music_duration > 0:
        music_duration_fraction = _fraction_from_seconds(float(music_duration), fps_hint=frame_rate)
        music_asset_id = f"a{next_index}"
        next_index += 1
        SubElement(
            resources,
            "asset",
            {
                "id": music_asset_id,
                "name": music_asset_name,
                "start": "0s",
                "duration": _format_fraction_seconds(music_duration_fraction),
                "hasVideo": "0",
                "hasAudio": "1",
                "audioSources": "1",
                "audioChannels": "2",
                "audioRate": "48000",
                "src": _file_uri(music_asset_path),
            },
        ).append(
            Element("media-rep", kind="original-media", src=_file_uri(music_asset_path))
        )

    library = SubElement(fcpxml, "library")
    event_el = SubElement(library, "event", name=event_name)
    project_el = SubElement(event_el, "project", name=project_name)
    sequence_el = SubElement(project_el, "sequence", format=format_id, tcStart="0s", tcFormat="NDF")
    spine_el = SubElement(sequence_el, "spine")

    timeline_frames_accum = 0

    for meta in timeline_metadata:
        timeline_info = meta.get("timeline", {}) or {}
        timings = meta.get("timings", {}) or {}
        source = meta.get("source", {}) or {}
        audio_info = meta.get("audio", {}) or {}

        clip_frame_rate = float(
            timeline_info.get("frame_rate")
            or source.get("fps")
            or frame_rate
        )
        frame_count = timeline_info.get("frame_count")
        frame_count_int = int(round(frame_count)) if frame_count else None

        if frame_count_int and frame_count_int > 0:
            duration_source = _frames_to_seconds(frame_count_int, clip_frame_rate)
        else:
            duration_source = _fraction_from_seconds(
                float(timeline_info.get("duration", 0.0)),
                fps_hint=clip_frame_rate,
            )

        if duration_source <= 0:
            continue

        duration_timeline, duration_timeline_frames = _quantize_duration_to_timeline(
            duration_source,
            frame_rate,
        )
        if duration_timeline_frames <= 0:
            continue

        file_name = source.get("file_name")
        asset_id = asset_id_map.get(file_name)
        if not asset_id:
            timeline_frames_accum += duration_timeline_frames
            continue

        segment_offset_fraction = Fraction(timeline_frames_accum, 1) / timeline_fps_fraction
        source_start_fraction = _fraction_from_seconds(
            float(timings.get("source_start", 0.0)),
            fps_hint=clip_frame_rate,
        )
        source_end_fraction = _fraction_from_seconds(
            float(timings.get("source_end", timings.get("source_start", 0.0))),
            fps_hint=clip_frame_rate,
        )

        crop_meta = meta.get("crop") or {}
        shots = crop_meta.get("shots") or []

        details = audio_info.get("details", {}) or {}
        gain_db = details.get("gain_db")
        if gain_db is None and "gain_multiplier" in details:
            try:
                gain_db = 20.0 * math.log10(float(details["gain_multiplier"]))
            except (ValueError, TypeError):
                gain_db = None

        if len(shots) > 1:
            clip_fps_for_shots = float(crop_meta.get("fps", clip_frame_rate))
            clip_fps_fraction = _fps_to_fraction(clip_fps_for_shots)
            if clip_fps_fraction.numerator == 0:
                clip_fps_fraction = _fps_to_fraction(clip_frame_rate)

            processed_timeline_frames = 0
            for idx, shot in enumerate(shots):
                shot_start_frame = int(shot.get("start_frame", 0))
                shot_end_frame = int(shot.get("end_frame", 0))
                shot_frame_count = shot_end_frame - shot_start_frame
                if shot_frame_count <= 0:
                    continue

                shot_source_offset = Fraction(shot_start_frame, 1) / clip_fps_fraction
                shot_source_duration = Fraction(shot_frame_count, 1) / clip_fps_fraction

                if idx == len(shots) - 1:
                    shot_timeline_frames = duration_timeline_frames - processed_timeline_frames
                    shot_timeline_duration = Fraction(shot_timeline_frames, 1) / timeline_fps_fraction
                else:
                    shot_timeline_duration, shot_timeline_frames = _quantize_duration_to_timeline(
                        shot_source_duration,
                        frame_rate,
                    )
                if shot_timeline_frames <= 0:
                    continue

                shot_offset_frames = timeline_frames_accum + processed_timeline_frames
                offset_attr_str = _format_fraction_seconds(
                    Fraction(shot_offset_frames, 1) / timeline_fps_fraction
                )
                duration_attr_str = _format_fraction_seconds(shot_timeline_duration)
                start_attr_str = _format_fraction_seconds(source_start_fraction + shot_source_offset)

                shot_clip_attrs = {
                    "name": f"{file_name}_shot_{shot_start_frame}",
                    "offset": offset_attr_str,
                    "start": start_attr_str,
                    "duration": duration_attr_str,
                    "ref": asset_id,
                    "enabled": "1",
                    "format": format_id,
                }
                shot_clip_el = SubElement(spine_el, "asset-clip", shot_clip_attrs)

                shot_center = shot.get("center_norm", {"x": 0.5, "y": 0.5})
                center_x = float(shot_center.get("x", 0.5))
                center_y = float(shot_center.get("y", 0.5))
                # Use content_bounds if available (for letterbox/pillarbox removal)
                content_bounds = crop_meta.get("content_bounds")
                if content_bounds:
                    src_w = float(content_bounds["x2"] - content_bounds["x1"])
                    src_h = float(content_bounds["y2"] - content_bounds["y1"])
                else:
                    source_size = crop_meta.get("source_size") or [width, height]
                    src_w = float(source_size[0]) if source_size else float(width)
                    src_h = float(source_size[1]) if source_size else float(height)
                # FCPXML scale: timeline auto-fits width first, then scale is applied
                # Formula: (output_h × src_w) / (src_h × output_w)
                scale_value = (height * src_w) / (src_h * width) if src_w > 0 and src_h > 0 and width > 0 else 1.0
                # Calculate final dimensions after auto-fit + scale
                # Step 1: Auto-fit to timeline width, Step 2: Apply scale
                scaled_w = width * scale_value
                scaled_h = (src_h * width / src_w) * scale_value if src_w > 0 else height
                max_pan_x = max(0.0, (scaled_w - width) / 2.0)
                max_pan_y = max(0.0, (scaled_h - height) / 2.0)
                position_x_pixels = (0.5 - center_x) * scaled_w
                position_y_pixels = (center_y - 0.5) * scaled_h
                clamped_x = max(-max_pan_x, min(max_pan_x, position_x_pixels))
                clamped_y = max(-max_pan_y, min(max_pan_y, position_y_pixels))
                # Normalize position (hardcoded divisor calibrated for FCPXML coordinate system)
                normalization_divisor = 19.2
                final_pos_x = clamped_x / normalization_divisor
                final_pos_y = clamped_y / normalization_divisor
                SubElement(
                    shot_clip_el,
                    "adjust-transform",
                    scale=f"{scale_value:.6f} {scale_value:.6f}",
                    position=f"{final_pos_x:.4f} {final_pos_y:.4f}",
                )

                if gain_db is not None:
                    SubElement(shot_clip_el, "adjust-volume", amount=f"{gain_db:.2f}dB")

                processed_timeline_frames += shot_timeline_frames
        else:
            clip_attrs = {
                "name": file_name or "Segment",
                "offset": _format_fraction_seconds(segment_offset_fraction),
                "start": _format_fraction_seconds(source_start_fraction),
                "duration": _format_fraction_seconds(duration_timeline),
                "ref": asset_id,
                "enabled": "1",
                "format": format_id,
            }
            if meta.get("segment_type"):
                clip_attrs["role"] = str(meta.get("segment_type"))
            clip_el = SubElement(spine_el, "asset-clip", clip_attrs)

            if gain_db is not None:
                SubElement(clip_el, "adjust-volume", amount=f"{gain_db:.2f}dB")

            retime = meta.get("retime", {}) or {}
            if abs(float(retime.get("speed", 1.0)) - 1.0) > 1e-6 or retime.get("adjustments"):
                time_map = SubElement(clip_el, "timeMap")
                SubElement(
                    time_map,
                    "timept",
                    time="0s",
                    value=_format_fraction_seconds(source_start_fraction),
                )
                SubElement(
                    time_map,
                    "timept",
                    time=_format_fraction_seconds(duration_timeline),
                    value=_format_fraction_seconds(source_end_fraction),
                )

            if crop_meta:
                shots_data = crop_meta.get("shots") or []
                center = shots_data[0].get("center_norm") if shots_data else crop_meta.get("center_norm")
                if not center:
                    center = {"x": 0.5, "y": 0.5}
                center_x = float(center.get("x", 0.5))
                center_y = float(center.get("y", 0.5))
                # Use content_bounds if available (for letterbox/pillarbox removal)
                content_bounds = crop_meta.get("content_bounds")
                if content_bounds:
                    src_w = float(content_bounds["x2"] - content_bounds["x1"])
                    src_h = float(content_bounds["y2"] - content_bounds["y1"])
                else:
                    source_size = crop_meta.get("source_size") or [width, height]
                    src_w = float(source_size[0]) if source_size else float(width)
                    src_h = float(source_size[1]) if source_size else float(height)
                # FCPXML scale: timeline auto-fits width first, then scale is applied
                # Formula: (output_h × src_w) / (src_h × output_w)
                scale_value = (height * src_w) / (src_h * width) if src_w > 0 and src_h > 0 and width > 0 else 1.0
                # Calculate final dimensions after auto-fit + scale
                # Step 1: Auto-fit to timeline width, Step 2: Apply scale
                scaled_w = width * scale_value
                scaled_h = (src_h * width / src_w) * scale_value if src_w > 0 else height
                max_pan_x = max(0.0, (scaled_w - width) / 2.0)
                max_pan_y = max(0.0, (scaled_h - height) / 2.0)
                position_x_pixels = (0.5 - center_x) * scaled_w
                position_y_pixels = (center_y - 0.5) * scaled_h
                clamped_x = max(-max_pan_x, min(max_pan_x, position_x_pixels))
                clamped_y = max(-max_pan_y, min(max_pan_y, position_y_pixels))
                # Normalize position (hardcoded divisor calibrated for FCPXML coordinate system)
                normalization_divisor = 19.2
                final_pos_x = clamped_x / normalization_divisor
                final_pos_y = clamped_y / normalization_divisor
                SubElement(
                    clip_el,
                    "adjust-transform",
                    scale=f"{scale_value:.6f} {scale_value:.6f}",
                    position=f"{final_pos_x:.4f} {final_pos_y:.4f}",
                )

        narration_path = details.get("narration_path")
        if narration_path:
            audio_asset_id = narration_asset_id_map.get(narration_path)
            if audio_asset_id:
                narration_duration_val = float(details.get("narration_duration", float(duration_source)))
                narration_duration_fraction = _fraction_from_seconds(
                    narration_duration_val,
                    fps_hint=frame_rate,
                )
                narration_timeline_duration, _ = _quantize_duration_to_timeline(
                    narration_duration_fraction,
                    frame_rate,
                )
                narration_clip_el = SubElement(
                    spine_el,
                    "asset-clip",
                    {
                        "name": Path(narration_path).name,
                        "offset": _format_fraction_seconds(segment_offset_fraction),
                        "start": "0s",
                        "duration": _format_fraction_seconds(narration_timeline_duration),
                        "ref": audio_asset_id,
                        "enabled": "1",
                        "lane": "-1",
                        "role": "dialogue",
                    },
                )
                narration_gain = details.get("narration_gain_db")
                if narration_gain is None and "narration_gain_multiplier" in details:
                    try:
                        narration_gain = 20.0 * math.log10(float(details["narration_gain_multiplier"]))
                    except (ValueError, TypeError):
                        narration_gain = None
                if narration_gain is not None:
                    SubElement(narration_clip_el, "adjust-volume", amount=f"{narration_gain:.2f}dB")

        timeline_frames_accum += duration_timeline_frames

    if music_asset_id and music_duration_fraction > 0:
        music_timeline_duration, _ = _quantize_duration_to_timeline(
            music_duration_fraction,
            frame_rate,
        )
        music_clip = SubElement(
            spine_el,
            "asset-clip",
            {
                "name": music_asset_name or "Music",
                "offset": "0s",
                "start": "0s",
                "duration": _format_fraction_seconds(music_timeline_duration),
                "ref": music_asset_id,
                "enabled": "1",
                "lane": "-10",
                "role": "music",
            },
        )
        SubElement(
            music_clip,
            "adjust-volume",
            amount=f"{(music_gain_db if music_gain_db is not None else 0.0):.2f}dB",
        )

    sequence_el.set(
        "duration",
        _format_fraction_seconds(Fraction(timeline_frames_accum, 1) / timeline_fps_fraction),
    )

    _indent(fcpxml)
    output_path = str(output_path)
    xml_payload = tostring(fcpxml, encoding="unicode")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        f.write(xml_payload)
    return output_path
