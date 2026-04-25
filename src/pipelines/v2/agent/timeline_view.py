"""
Programmatic timeline view for the agent system prompt.

Builds a vertical, row-aligned table where each row is a time slice bounded by any
clip/narration/music/title boundary, and each column is a track. Cells in the same
row are simultaneously active. This gives Gemini exact temporal data without forcing
it to compute timeline offsets from sequential clip durations or rely on imprecise
visual analysis of the rendered preview.

Also detects concrete audio issues (narration overlapping dialogue, muted clips
without coverage, etc.) as a derived view.
"""
from __future__ import annotations

import json as _json
from typing import Optional


# ── Formatting helpers ─────────────────────────────────────────────────────

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_clip_cell(item: dict) -> str:
    """Format a clip block for a timeline table cell."""
    kind = item.get("kind", "clip")
    if kind == "title":
        return (
            f"📝 {item['id']} "
            f"({item['tl_start']:.2f}→{item['tl_end']:.2f}, {item['duration']:.2f}s)"
        )
    if kind == "narration":
        meta_bits = []
        if item.get("audio_start") is not None:
            meta_bits.append(f"audio:[{item['audio_start']:.2f}-{item['audio_end']:.2f}s]")
        if item.get("gain_db") is not None:
            meta_bits.append(f"gain={item['gain_db']:+g}")
        if item.get("lufs") is not None:
            meta_bits.append(f"meas={item['lufs']}LUFS")
        extent = f"({item['tl_start']:.2f}→{item['tl_end']:.2f}, {item['duration']:.2f}s)"
        return f"{item['id']} {extent}" + (" " + " ".join(meta_bits) if meta_bits else "")

    parts = [item["id"]]
    if item.get("label"):
        parts.append(_truncate(item["label"], 28))
    meta = []
    if item.get("gain_db") is not None:
        meta.append(f"gain={item['gain_db']:+g}dB")
    if item.get("lufs") is not None:
        meta.append(f"meas={item['lufs']}LUFS")
    if item.get("speed_rate") and abs(item["speed_rate"] - 1.0) > 0.001:
        meta.append(f"speed={item['speed_rate']:g}x")
    if item.get("source_start") is not None and item.get("source_end") is not None:
        meta.append(f"src=[{item['source_start']:.1f}-{item['source_end']:.1f}s]")
    if item.get("transform_mode") and item["transform_mode"] != "fit":
        meta.append(f"crop={item['transform_mode']}")
    extent = f"({item['tl_start']:.2f}→{item['tl_end']:.2f}, {item['duration']:.2f}s)"
    return " · ".join(parts) + " " + extent + (" " + " ".join(meta) if meta else "")


# ── Item extraction ────────────────────────────────────────────────────────

def _extract_items(ed: dict) -> dict:
    """
    Pull all timeline elements out of an EditDecision dict and compute their
    absolute timeline ranges. Returns a dict with v1_clips, overlay_clips,
    narrations, music, titles, and total_duration.
    """
    all_clips = ed.get("clips", []) or []
    narrations = ed.get("narration", []) or []
    music = ed.get("music")
    titles = ed.get("titles", []) or []

    v1_clips = []
    overlay_clips: dict[int, list] = {}  # track_num -> list
    t = 0.0
    for c in all_clips:
        track_num = c.get("track", 1) or 1
        speed_rate = (c.get("speed") or {}).get("rate", 1.0) or 1.0
        src_start = c.get("source_start", 0)
        src_end = c.get("source_end", 0)
        dur = (src_end - src_start) / speed_rate
        item = {
            "kind": "clip",
            "id": c.get("id", "?"),
            "label": c.get("label", ""),
            "duration": dur,
            "gain_db": c.get("gain_db"),
            "lufs": c.get("measured_loudness_lufs"),
            "source_start": src_start,
            "source_end": src_end,
            "speed_rate": speed_rate,
            "transform_mode": c.get("transform_mode"),
        }
        if track_num == 1:
            item["tl_start"] = t
            item["tl_end"] = t + dur
            t += dur
            v1_clips.append(item)
        else:
            tl_start = c.get("timeline_offset") or 0.0
            item["tl_start"] = tl_start
            item["tl_end"] = tl_start + dur
            overlay_clips.setdefault(track_num, []).append(item)

    total_dur = t

    # Titles render on V-tracks (lane=N maps to V(N+1), matching NLETimeline.tsx
    # and FCPXML "lane above spine" semantics). Merge them into overlay_clips
    # so they show up in the right column alongside overlay clips.
    for tt in titles:
        s = tt.get("timeline_offset", 0) or 0
        d = tt.get("duration", 0) or 0
        lane = tt.get("lane", 1) or 1
        v_track = max(1, lane) + 1
        title_item = {
            "kind": "title",
            "id": _truncate(tt.get("text", ""), 24),
            "label": "",
            "tl_start": s,
            "tl_end": s + d,
            "duration": d,
            "gain_db": None,
            "lufs": None,
        }
        overlay_clips.setdefault(v_track, []).append(title_item)

    narr_items = []
    for i, n in enumerate(narrations):
        s = n.get("timeline_offset", 0)
        d = n.get("duration", 0)
        audio_start = n.get("start", 0)
        narr_items.append({
            "kind": "narration",
            "id": f"narr_{i+1}",
            "label": "",
            "tl_start": s,
            "tl_end": s + d,
            "duration": d,
            "audio_start": audio_start,
            "audio_end": audio_start + d,
            "track": n.get("track", 1) or 1,
            "gain_db": n.get("gain_db"),
            "lufs": n.get("measured_loudness_lufs"),
        })

    music_item = None
    if music and music.get("file"):
        m_start = music.get("start", 0) or 0
        m_dur = music.get("duration", 0) or total_dur
        music_item = {
            "kind": "music",
            "id": "music",
            "label": _truncate((music.get("file") or "").split("/")[-1], 24),
            "tl_start": m_start,
            "tl_end": m_start + m_dur,
            "duration": m_dur,
            "track": music.get("track", 2) or 2,
            "gain_db": music.get("gain_db"),
            "lufs": music.get("measured_loudness_lufs"),
        }

    return {
        "v1_clips": v1_clips,
        "overlay_clips": overlay_clips,
        "narrations": narr_items,
        "music": music_item,
        "total_duration": total_dur,
    }


# ── Row-aligned table builder ──────────────────────────────────────────────

def _build_table(items: dict) -> list[str]:
    """Build the markdown table portion of the timeline view."""
    v1_clips = items["v1_clips"]
    overlay_clips = items["overlay_clips"]
    narr_items = items["narrations"]
    music_item = items["music"]
    total_dur = items["total_duration"]

    # Collect unique event times across all tracks
    event_times = {0.0}
    for grp in [v1_clips, narr_items] + list(overlay_clips.values()):
        for it in grp:
            event_times.add(round(it["tl_start"], 3))
            event_times.add(round(it["tl_end"], 3))
    if music_item:
        event_times.add(round(music_item["tl_start"], 3))
        event_times.add(round(music_item["tl_end"], 3))
    event_times = sorted(event_times)
    rows = [(a, b) for a, b in zip(event_times[:-1], event_times[1:]) if b - a > 0.001]

    # Define columns:
    #   V1 (spine), V2+ (overlays + titles merged via lane+1),
    #   one A-column per narration track actually used, plus music on its own track.
    columns = [("V1", v1_clips)]
    for tn in sorted(overlay_clips.keys()):
        columns.append((f"V{tn}", overlay_clips[tn]))

    # Narration: one column per distinct `track` so cross-track segments
    # don't silently share a lane in the view.
    narr_by_track: dict[int, list] = {}
    for n in narr_items:
        narr_by_track.setdefault(n.get("track", 1) or 1, []).append(n)
    for tn in sorted(narr_by_track.keys()):
        columns.append((f"A{tn} Narration", narr_by_track[tn]))

    if music_item:
        mt = music_item.get("track", 2) or 2
        columns.append((f"A{mt} Music", [music_item]))

    def _active_in(items_list, row_start, row_end):
        """Return every item active in [row_start, row_end). A V-column can
        hold multiple items at once (e.g. a title over an overlay clip), so
        we return a list and let the caller join them.
        """
        eps = 0.005
        return [
            it for it in items_list
            if it["tl_start"] < row_end - eps and it["tl_end"] > row_start + eps
        ]

    lines = []
    lines.append(f"**Total V1 duration: {total_dur:.2f}s**")
    lines.append("")
    header = ["Time range"] + [c[0] for c in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    prev_active_ids: list[set] = [set() for _ in columns]
    for row_start, row_end in rows:
        time_cell = f"{row_start:.2f} → {row_end:.2f}"
        cells = [time_cell]
        for i, (_, col_items) in enumerate(columns):
            active = _active_in(col_items, row_start, row_end)
            if not active:
                cells.append("—")
            else:
                rendered = []
                for it in active:
                    if it["id"] in prev_active_ids[i]:
                        rendered.append(f"↓ {it['id']}")
                    else:
                        rendered.append(_fmt_clip_cell(it))
                cells.append(" + ".join(rendered))
            prev_active_ids[i] = {it["id"] for it in active}
        lines.append("| " + " | ".join(cells) + " |")

    return lines


# ── Audio issue detection ──────────────────────────────────────────────────

def _detect_audio_issues(items: dict) -> list[str]:
    """Find concrete audio problems in the edit (narration overlap, wrong gain, etc.).

    NOTE: The thresholds below (<10%, ≥90%) must match the "Audio ducking" rules in
    the agent system prompt (src/pipelines/v2/agent/system_prompt.py). If you change
    one, update the other so the agent's manual reasoning and the automatic detection
    agree.
    """
    v1_clips = items["v1_clips"]
    narr_items = items["narrations"]
    music = items.get("music")
    total_dur = items.get("total_duration") or 0.0
    issues: list[str] = []

    # ── Per-V1-clip narration coverage checks ──────────────────────────────
    for c in v1_clips:
        if not narr_items:
            break
        cov_sec = 0.0
        for n in narr_items:
            ov_s = max(c["tl_start"], n["tl_start"])
            ov_e = min(c["tl_end"], n["tl_end"])
            if ov_e > ov_s:
                cov_sec += ov_e - ov_s
        cov_pct = (cov_sec / c["duration"] * 100) if c["duration"] > 0 else 0
        is_dialogue = "dialogue" in (c["label"] or "").lower() or "dialogue" in c["id"].lower()
        gain = c["gain_db"]

        if is_dialogue and cov_pct > 5:
            issues.append(
                f"❌ **Narration overlaps dialogue clip `{c['id']}`** "
                f"({cov_pct:.0f}% of {c['duration']:.2f}s). "
                f"Split the narration into two segments around this clip — use the per-sentence "
                f"transcript timecodes from generate_narration to find the right split point."
            )
        elif cov_pct >= 90 and gain != -96 and gain is not None:
            issues.append(
                f"⚠️ Clip `{c['id']}` is fully under narration but gain_db={gain} (should be -96)."
            )
        elif cov_pct < 10 and gain == -96 and not is_dialogue:
            issues.append(
                f"⚠️ Clip `{c['id']}` is muted (-96) but has only {cov_pct:.0f}% narration coverage. "
                f"Either restore nat sound (gain_db = -6 to -12) or extend narration to cover it."
            )

    # ── Narration-vs-narration overlap (same timeline, two voices at once) ─
    if narr_items:
        by_start = sorted(narr_items, key=lambda n: n["tl_start"])
        for a, b in zip(by_start, by_start[1:]):
            overlap = a["tl_end"] - b["tl_start"]
            if overlap > 0.01:  # ignore rounding slop
                issues.append(
                    f"❌ **Narration segments overlap by {overlap:.2f}s** on the timeline "
                    f"(`{a['id']}` ends at {a['tl_end']:.2f}s, `{b['id']}` starts at {b['tl_start']:.2f}s). "
                    f"Two narrations playing simultaneously = garbled speech. Push the later "
                    f"segment's `timeline_offset` to at least {a['tl_end']:.2f}, or shorten the "
                    f"earlier one's `duration`."
                )

    # ── Narration extending past the spine (compiler will silently drop) ──
    if narr_items and total_dur > 0:
        for n in narr_items:
            if n["tl_start"] > total_dur + 0.01:
                issues.append(
                    f"❌ Narration `{n['id']}` starts at {n['tl_start']:.2f}s, beyond the "
                    f"spine end ({total_dur:.2f}s). The FCPXML compiler will DROP this segment "
                    f"entirely. Either extend the spine, move the segment earlier, or remove it."
                )
            elif n["tl_end"] > total_dur + 0.5:
                issues.append(
                    f"⚠️ Narration `{n['id']}` ends at {n['tl_end']:.2f}s, past the spine end "
                    f"({total_dur:.2f}s). The renderer will clamp it to the spine — trim its "
                    f"`duration` to make the clamp explicit."
                )

    # ── Music tail / orphan checks ─────────────────────────────────────────
    if music and total_dur > 0:
        m_end = music["tl_end"]
        m_start = music["tl_start"]
        if m_end < total_dur - 0.5:
            issues.append(
                f"⚠️ Music ends at {m_end:.2f}s but the spine runs to {total_dur:.2f}s — "
                f"there's ~{total_dur - m_end:.1f}s of silence at the end. Extend the music "
                f"`duration` or accept the tail silence deliberately."
            )
        elif m_end > total_dur + 0.5:
            issues.append(
                f"⚠️ Music extends to {m_end:.2f}s, past the spine end ({total_dur:.2f}s). "
                f"Trim `music.duration` so it ends on or before the last clip."
            )
        if m_start > 0.5:
            issues.append(
                f"⚠️ Music starts at {m_start:.2f}s — first {m_start:.1f}s of the edit has no "
                f"music bed. Intentional, or move `music.start` to 0?"
            )

    return issues


# ── Public API ─────────────────────────────────────────────────────────────

def build_timeline_view(edit_decision_str: str) -> str:
    """
    Build a vertical timeline diagram showing what's active on each track at every
    time slice. Rows are bounded by event times (any clip/narration/music start or end);
    columns are tracks. Two clips on different tracks that share a start/end time are
    visually aligned in the same row.
    """
    if not edit_decision_str:
        return ""
    try:
        ed = _json.loads(edit_decision_str)
    except Exception:
        return ""

    items = _extract_items(ed)
    if not items["v1_clips"] and not items["narrations"] and not items["music"]:
        return ""

    lines = ["## Computed timeline view", ""]
    lines.append(
        "Programmatically computed from the current edit_decision. Each row is a time "
        "slice bounded by a clip/narration/music boundary; each column is a track. "
        "Cells in the same row are simultaneously active."
    )
    lines.append("")
    lines.extend(_build_table(items))

    issues = _detect_audio_issues(items)
    if issues:
        lines.append("")
        lines.append("## Audio issues detected")
        lines.append("")
        for iss in issues:
            lines.append(f"- {iss}")
        lines.append("")
        lines.append(
            "**To fix:** modify clip gain_db values OR split/move narration segments in the JSON, "
            "then call generate_fcpxml again. The analysis will recompute."
        )

    return "\n".join(lines)
