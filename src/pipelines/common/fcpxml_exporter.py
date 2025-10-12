"""FCPXML export helpers for multiround explainable outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring


def _format_time(seconds: float, frame_rate: float) -> str:
    frames = round(float(seconds) * frame_rate)
    denom = int(round(frame_rate))
    return f"{frames}/{denom}s"


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

    # Determine timeline dimensions from metadata if available (default 1080x1920)
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

    format_id = "f1"
    SubElement(
        resources,
        "format",
        id=format_id,
        frameDuration=f"1/{int(frame_rate)}s",
        width=str(width),
        height=str(height),
        colorSpace="1-1-1 (Rec. 709)",
    )

    # Asset registry
    asset_usage: Dict[str, float] = {}
    for meta in timeline_metadata:
        source = meta.get("source", {})
        file_name = source.get("file_name")
        if not file_name:
            continue
        timings = meta.get("timings", {})
        source_end = float(timings.get("source_end", timings.get("applied_end", 0.0)))
        asset_usage[file_name] = max(asset_usage.get(file_name, 0.0), source_end)

    asset_id_map: Dict[str, str] = {}
    for idx, (file_name, local_path) in enumerate(video_asset_map.items(), start=2):
        asset_id = f"a{idx}"
        asset_id_map[file_name] = asset_id
        duration_value = _format_time(asset_usage.get(file_name, 0.0), frame_rate)
        SubElement(
            resources,
            "asset",
            id=asset_id,
            name=file_name,
            start="0s",
            duration=duration_value,
            hasVideo="1",
            hasAudio="1",
            format=format_id,
            audioSources="1",
            audioChannels="2",
            audioRate="48000",
            src=_file_uri(local_path),
        ).append(
            Element("media-rep", kind="original-media", src=_file_uri(local_path))
        )

    narration_asset_id_map: Dict[str, str] = {}
    next_index = len(video_asset_map) + 2
    for original_path, copied_path in narration_asset_map.items():
        asset_id = f"a{next_index}"
        next_index += 1
        narration_asset_id_map[original_path] = asset_id
        durations = [
            float(meta.get("timeline", {}).get("duration", 0.0))
            for meta in timeline_metadata
            if meta.get("audio", {}).get("details", {}).get("narration_path") == original_path
        ]
        narration_duration = _format_time(max(durations) if durations else 0.0, frame_rate)
        SubElement(
            resources,
            "asset",
            id=asset_id,
            name=Path(copied_path).name,
            start="0s",
            duration=narration_duration,
            hasVideo="0",
            hasAudio="1",
            audioSources="1",
            audioChannels="1",
            audioRate="48000",
            src=_file_uri(copied_path),
        ).append(
            Element("media-rep", kind="original-media", src=_file_uri(copied_path))
        )

    music_asset_id = None
    if music_asset_path and music_asset_name and music_duration > 0:
        music_asset_id = f"a{next_index}"
        next_index += 1
        SubElement(
            resources,
            "asset",
            id=music_asset_id,
            name=music_asset_name,
            start="0s",
            duration=_format_time(music_duration, frame_rate),
            hasVideo="0",
            hasAudio="1",
            audioSources="1",
            audioChannels="2",
            audioRate="48000",
            src=_file_uri(music_asset_path),
        ).append(
            Element("media-rep", kind="original-media", src=_file_uri(music_asset_path))
        )

    library = SubElement(fcpxml, "library")
    event_el = SubElement(library, "event", name=event_name)
    project_el = SubElement(event_el, "project", name=project_name)
    sequence_el = SubElement(project_el, "sequence", format=format_id, tcStart="0s", tcFormat="NDF")
    spine_el = SubElement(sequence_el, "spine")

    offset_seconds = 0.0
    for meta in timeline_metadata:
        timeline_info = meta.get("timeline", {})
        timings = meta.get("timings", {})
        source = meta.get("source", {})
        audio_info = meta.get("audio", {}) or {}

        duration = float(timeline_info.get("duration", 0.0))
        if duration <= 0:
            continue

        file_name = source.get("file_name")
        asset_id = asset_id_map.get(file_name)
        if not asset_id:
            offset_seconds += duration
            continue

        clip_attrs = {
            "name": file_name or "Segment",
            "offset": _format_time(offset_seconds, frame_rate),
            "start": _format_time(float(timings.get("source_start", 0.0)), frame_rate),
            "duration": _format_time(duration, frame_rate),
            "ref": asset_id,
            "enabled": "1",
            "format": format_id,
        }

        segment_type = meta.get("segment_type")
        if segment_type:
            clip_attrs["role"] = str(segment_type)

        clip_el = SubElement(spine_el, "asset-clip", clip_attrs)

        retime = meta.get("retime", {}) or {}
        speed = float(retime.get("speed", 1.0))
        if abs(speed - 1.0) > 1e-6 or retime.get("adjustments"):
            time_map = SubElement(clip_el, "timeMap")
            SubElement(time_map, "timept", time="0s", value=_format_time(float(timings.get("source_start", 0.0)), frame_rate))
            SubElement(
                time_map,
                "timept",
                time=_format_time(duration, frame_rate),
                value=_format_time(float(timings.get("source_end", timings.get("source_start", 0.0))), frame_rate),
            )

        crop_meta = meta.get("crop") or {}
        if crop_meta:
            shots = crop_meta.get("shots") or []
            center = shots[0]["center_norm"] if shots else {"x": 0.5, "y": 0.5}
            center_x = float(center.get("x", 0.5))
            center_y = float(center.get("y", 0.5))
            source_size = crop_meta.get("source_size") or [width, height]
            src_w = float(source_size[0]) or float(width)
            src_h = float(source_size[1]) or float(height)

            fit_scale = width / src_w if src_w else 1.0
            fill_scale = (height / src_h) / fit_scale if (src_h and fit_scale) else 1.0
            scale_value = fill_scale

            width_after = width * fill_scale
            max_offset_x = max(0.0, (width_after - width) / 2.0)
            position_x = (center_x - 0.5) * width_after
            position_x = max(-max_offset_x, min(max_offset_x, position_x))
            normalized_position_x = position_x / 19.2
            position_y = 0.0

            SubElement(
                clip_el,
                "adjust-transform",
                scale=f"{scale_value:.6f} {scale_value:.6f}",
                position=f"{normalized_position_x:.4f} {position_y:.4f}",
            )
            if len(shots) > 1:
                note = SubElement(clip_el, "note")
                note.text = json.dumps({"shots": shots})

        details = audio_info.get("details", {}) or {}
        gain_db = details.get("gain_db")
        if gain_db is None and "gain_multiplier" in details:
            try:
                gain_db = 20.0 * math.log10(float(details["gain_multiplier"]))
            except (ValueError, TypeError):
                gain_db = None
        if gain_db is not None:
            SubElement(
                clip_el,
                "adjust-audio",
                pan="0 0",
                gain=f"{gain_db:.2f}",
            )
        narration_path = details.get("narration_path")
        if narration_path:
            audio_asset_id = narration_asset_id_map.get(narration_path)
            if audio_asset_id:
                narration_duration_val = details.get("narration_duration")
                if narration_duration_val is None:
                    narration_duration_val = duration
                narration_clip_el = SubElement(
                    spine_el,
                    "asset-clip",
                    {
                        "name": Path(narration_path).name,
                        "offset": _format_time(offset_seconds, frame_rate),
                        "start": "0s",
                        "duration": _format_time(float(narration_duration_val), frame_rate),
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
                    SubElement(
                        narration_clip_el,
                        "adjust-audio",
                        pan="0 0",
                        gain=f"{narration_gain:.2f}",
                    )

        offset_seconds += duration

    if music_asset_id and music_duration > 0:
        music_clip = SubElement(
            spine_el,
            "asset-clip",
            {
                "name": music_asset_name or "Music",
                "offset": "0s",
                "start": "0s",
                "duration": _format_time(music_duration, frame_rate),
                "ref": music_asset_id,
                "enabled": "1",
                "lane": "-10",
                "role": "music",
            },
        )
        SubElement(
            music_clip,
            "adjust-audio",
            pan="0 0",
            gain=f"{(music_gain_db if music_gain_db is not None else 0.0):.2f}",
        )

    sequence_el.set("duration", _format_time(offset_seconds, frame_rate))

    _indent(fcpxml)
    output_path = str(output_path)
    xml_payload = tostring(fcpxml, encoding="unicode")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<!DOCTYPE fcpxml>\n")
        f.write(xml_payload)
    return output_path
