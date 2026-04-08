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
    parts = [item["id"]]
    if item.get("label"):
        parts.append(_truncate(item["label"], 28))
    meta = []
    if item.get("gain_db") is not None:
        meta.append(f"gain={item['gain_db']:+g}dB")
    if item.get("lufs") is not None:
        meta.append(f"meas={item['lufs']}LUFS")
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
        speed = (c.get("speed") or {}).get("rate", 1.0) or 1.0
        dur = (c.get("source_end", 0) - c.get("source_start", 0)) / speed
        item = {
            "id": c.get("id", "?"),
            "label": c.get("label", ""),
            "duration": dur,
            "gain_db": c.get("gain_db"),
            "lufs": c.get("measured_loudness_lufs"),
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

    narr_items = []
    for i, n in enumerate(narrations):
        s = n.get("timeline_offset", 0)
        d = n.get("duration", 0)
        narr_items.append({
            "id": f"narr_{i+1}",
            "label": "",
            "tl_start": s,
            "tl_end": s + d,
            "duration": d,
            "gain_db": n.get("gain_db"),
            "lufs": n.get("measured_loudness_lufs"),
        })

    music_item = None
    if music and music.get("file"):
        m_start = music.get("start", 0) or 0
        m_dur = music.get("duration", 0) or total_dur
        music_item = {
            "id": "music",
            "label": _truncate((music.get("file") or "").split("/")[-1], 24),
            "tl_start": m_start,
            "tl_end": m_start + m_dur,
            "duration": m_dur,
            "gain_db": music.get("gain_db"),
            "lufs": music.get("measured_loudness_lufs"),
        }

    title_items = []
    for tt in titles:
        s = tt.get("timeline_offset", 0) or 0
        d = tt.get("duration", 0) or 0
        title_items.append({
            "id": _truncate(tt.get("text", ""), 20),
            "label": "",
            "tl_start": s,
            "tl_end": s + d,
            "duration": d,
            "gain_db": None,
            "lufs": None,
        })

    return {
        "v1_clips": v1_clips,
        "overlay_clips": overlay_clips,
        "narrations": narr_items,
        "music": music_item,
        "titles": title_items,
        "total_duration": total_dur,
    }


# ── Row-aligned table builder ──────────────────────────────────────────────

def _build_table(items: dict) -> list[str]:
    """Build the markdown table portion of the timeline view."""
    v1_clips = items["v1_clips"]
    overlay_clips = items["overlay_clips"]
    narr_items = items["narrations"]
    music_item = items["music"]
    title_items = items["titles"]
    total_dur = items["total_duration"]

    # Collect unique event times across all tracks
    event_times = {0.0}
    for grp in [v1_clips, narr_items, title_items] + list(overlay_clips.values()):
        for it in grp:
            event_times.add(round(it["tl_start"], 3))
            event_times.add(round(it["tl_end"], 3))
    if music_item:
        event_times.add(round(music_item["tl_start"], 3))
        event_times.add(round(music_item["tl_end"], 3))
    event_times = sorted(event_times)
    rows = [(a, b) for a, b in zip(event_times[:-1], event_times[1:]) if b - a > 0.001]

    # Define columns
    columns = [("V1", v1_clips)]
    for tn in sorted(overlay_clips.keys()):
        columns.append((f"V{tn}", overlay_clips[tn]))
    if title_items:
        columns.append(("T1 Titles", title_items))
    if narr_items:
        columns.append(("A1 Narration", narr_items))
    if music_item:
        columns.append(("A2 Music", [music_item]))

    def _active_in(items_list, row_start, row_end):
        eps = 0.005
        for it in items_list:
            if it["tl_start"] < row_end - eps and it["tl_end"] > row_start + eps:
                return it
        return None

    lines = []
    lines.append(f"**Total V1 duration: {total_dur:.2f}s**")
    lines.append("")
    header = ["Time range"] + [c[0] for c in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    prev_active = [None] * len(columns)
    for row_start, row_end in rows:
        time_cell = f"{row_start:.2f} → {row_end:.2f}"
        cells = [time_cell]
        for i, (_, col_items) in enumerate(columns):
            it = _active_in(col_items, row_start, row_end)
            if it is None:
                cells.append("—")
            else:
                if prev_active[i] is not None and prev_active[i]["id"] == it["id"]:
                    # Continuation: show id only so the model knows what's still playing
                    cells.append(f"↓ {it['id']}")
                else:
                    cells.append(_fmt_clip_cell(it))
            prev_active[i] = it
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
    if not narr_items:
        return []

    issues = []
    for c in v1_clips:
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
