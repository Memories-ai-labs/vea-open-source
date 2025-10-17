from copy import deepcopy

def _build_segment_metadata(
    clip_info,
    start_sec: float,
    end_sec: float,
    *,
    segment_type: str = "primary",
    original_audio: bool = True,
) -> dict:
    metadata = {
        "clip_id": clip_info.get("id") if isinstance(clip_info, dict) else None,
        "segment_type": segment_type,
        "source": {
            "file_name": clip_info.get("file_name") if isinstance(clip_info, dict) else None,
            "cloud_storage_path": clip_info.get("cloud_storage_path") if isinstance(clip_info, dict) else None,
        },
        "timings": {
            "source_start": float(start_sec),
            "source_end": float(end_sec),
            "applied_start": float(start_sec),
            "applied_end": float(end_sec),
        },
        "timeline": {
            "duration": max(float(end_sec) - float(start_sec), 0.0),
            "frame_rate": None,
            "frame_count": None,
        },
        "retime": {
            "speed": 1.0,
            "adjustments": [],
        },
        "audio": {
            "original_audio_enabled": bool(original_audio),
            "mix": "original" if original_audio else "muted",
            "details": {},
        },
        "crop": None,
        "notes": [],
    }
    return metadata

def _attach_metadata(clip_obj, metadata):
    duration = float(clip_obj.duration or 0.0)
    metadata.setdefault("timeline", {})["duration"] = duration
    timings = metadata.setdefault("timings", {})
    applied_start = timings.get("applied_start", timings.get("source_start", 0.0))
    applied_end = applied_start + duration
    timings["applied_end"] = applied_end
    if not metadata.get("_lock_source_end"):
        timings["source_end"] = applied_end

    frame_rate = metadata.get("timeline", {}).get("frame_rate")
    if frame_rate:
        metadata["timeline"]["frame_count"] = int(round(duration * float(frame_rate)))
    else:
        metadata["timeline"].pop("frame_count", None)
    setattr(clip_obj, "_vea_metadata", metadata)
    return clip_obj

def _apply_clip_transform(clip_obj, metadata, transform_fn):
    transformed = transform_fn(clip_obj)
    return _attach_metadata(transformed, metadata)

def _record_retime_adjustment(
    metadata,
    *,
    speed_factor: float,
    reason: str,
    original_duration: float,
    target_duration: float,
) -> None:
    retime = metadata.setdefault("retime", {})
    retime.setdefault("adjustments", []).append(
        {
            "reason": reason,
            "speed_factor": float(speed_factor),
            "original_duration": float(original_duration),
            "target_duration": float(target_duration),
        }
    )
    retime["speed"] = float(retime.get("speed", 1.0) * speed_factor)
    timings = metadata.setdefault("timings", {})
    timings["source_end"] = timings.get("source_start", 0.0) + float(original_duration)
    metadata["_lock_source_end"] = True

def _record_audio_mix(metadata, *, mix_type: str, details: dict | None = None) -> None:
    audio = metadata.setdefault("audio", {})
    audio["mix"] = mix_type
    if details:
        audio.setdefault("details", {}).update(details)
    if mix_type == "muted":
        audio.setdefault("details", {})["gain_db"] = -100.0

def _append_note(metadata, note: str) -> None:
    metadata.setdefault("notes", []).append(note)

def _register_clip_edit(clip_dict, metadata) -> None:
    if isinstance(clip_dict, dict):
        clip_dict.setdefault("edits", []).append(deepcopy(metadata))
