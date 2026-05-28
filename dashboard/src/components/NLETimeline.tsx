import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type {
  EditDecision,
  EditDecisionClip,
} from '../hooks/useAgentChat';

// ─── Types ───────────────────────────────────────────────────────────────────

interface NLETimelineProps {
  editDecision: EditDecision | null;
  playheadTime?: number;
  selectedClipId?: string | null;
  cropStatuses?: Record<string, { status: string; step?: string }>;
  onSeek?: (time: number) => void;
  /** Commit a change: persists to server + history. Use on drag-end/click. */
  onEditDecisionChange?: (updated: EditDecision) => void;
  /** Live preview during a drag: updates local state only (no server, no history).
   *  If omitted, drags fall back to ``onEditDecisionChange`` per pointer-move. */
  onEditDecisionPreview?: (updated: EditDecision) => void;
  onClipSelect?: (clipId: string | null) => void;
}

// Discriminator so drag handlers know which schema field to write back to.
// - 'v1'        : primary spine video clip (ordering matters, no timeline_offset)
// - 'v-overlay' : overlay video clip on V2+, free-placed via timeline_offset
// - 'title'     : TextOverlay on a V-track lane (above the spine)
// - 'narration' : NarrationSegment on an A-track lane
// - 'music'     : MusicTrack on an A-track lane
type ItemKind = 'v1' | 'v-overlay' | 'title' | 'narration' | 'music';

// Which visual family a track belongs to. Cross-family drags are blocked.
type TrackFamily = 'video' | 'audio';

interface TrackItem {
  id: string;
  kind: ItemKind;
  sourceIdx: number;   // index into edit.clips / edit.narration / edit.titles (0 for music)
  tlStart: number;
  tlEnd: number;
  label: string;
  sublabel?: string;
  color: string;
  borderColor: string;
  gainDb?: number | null;
  tooltip: string[];
  type: 'video' | 'audio' | 'title';
}

interface Track {
  id: string;                  // "V1", "V2", "A1", "A2", ...
  name: string;
  family: TrackFamily;
  trackNum: number;            // integer extracted from id
  type: 'video' | 'audio' | 'title';
  height: number;
  color: string;
  items: TrackItem[];
}

// Drag modes
type DragMode = 'move' | 'retrim-left' | 'retrim-right' | null;

interface DragState {
  mode: DragMode;
  sourceFamily: TrackFamily;   // 'video' or 'audio' — destination must match
  sourceTrackNum: number;      // track the item started on
  kind: ItemKind;              // which schema array to write back to
  sourceIdx: number;           // index in that array
  itemId: string;
  startX: number;
  startY: number;
  ghostOffset: number;         // px offset from item left edge to pointer
  // For move — updated during drag
  targetTrackNum: number;      // track the item is hovering over
  ghostLeft: number;           // px position of ghost in timeline
  timelineOffset: number;      // absolute time position where item would land
  // For retrim — snapshot the fields we'll compute deltas against
  originalStart: number;       // source_start (clip) / start (narr/music) / 0 (title)
  originalEnd: number;         // source_end (clip) / start+duration (narr/music/title)
  originalTimelineOffset: number | undefined;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const TRACK_HEADER_W = 52;
const RULER_H = 22;
const VIDEO_TRACK_H = 48;
const AUDIO_TRACK_H = 34;
const TRACK_BORDER = 1;
const MIN_CLIP_W = 22;
const BASE_PX_PER_SEC = 18;
const RETRIM_HANDLE_W = 5;
const DRAG_DEADZONE = 4;

const V1 = { color: '#60d5c8', border: '#88e0d8', bg: 'rgba(96,213,200,' };
const T1 = { color: '#d79bb5', border: '#e4b4ca', bg: 'rgba(215,155,181,' };
const A1 = { color: '#8ad27e', border: '#a8e09e', bg: 'rgba(138,210,126,' };
const A2 = { color: '#f39a55', border: '#f5b47a', bg: 'rgba(243,154,85,' };

const CLIP_PALETTE = [
  { color: '#60d5c8', border: '#88e0d8' },
  { color: '#d79bb5', border: '#e4b4ca' },
  { color: '#8ad27e', border: '#a8e09e' },
  { color: '#f39a55', border: '#f5b47a' },
  { color: '#f1bf63', border: '#f5d08a' },
  { color: '#ff7d6f', border: '#ff9f95' },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtShort(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${pad(s)}`;
}

function pad(n: number): string {
  return n.toString().padStart(2, '0');
}

function baseName(path: string): string {
  const name = path.split('/').pop() || path;
  return name.replace(/\.[^/.]+$/, '').replace(/^user_media_\w+_/, '');
}

function waveformPath(width: number, height: number, seed: number): string {
  const segs = Math.max(4, Math.floor(width / 3));
  const mid = height / 2;
  let d = `M 0 ${mid}`;
  for (let i = 0; i <= segs; i++) {
    const x = (i / segs) * width;
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    const amp = (hash - Math.floor(hash)) * 0.85 + 0.15;
    d += ` L ${x.toFixed(1)} ${(mid - amp * (mid - 2)).toFixed(1)}`;
  }
  for (let i = segs; i >= 0; i--) {
    const x = (i / segs) * width;
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    const amp = (hash - Math.floor(hash)) * 0.85 + 0.15;
    d += ` L ${x.toFixed(1)} ${(mid + amp * (mid - 2)).toFixed(1)}`;
  }
  return d + ' Z';
}

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function clipTooltip(c: EditDecisionClip): string[] {
  return [
    c.label || c.id,
    c.description || '',
    `Source: ${baseName(c.source_file)}`,
    `In: ${fmtShort(c.source_start)} → Out: ${fmtShort(c.source_end)}`,
    c.gain_db != null ? `Gain: ${c.gain_db}dB` : '',
    c.speed && c.speed.rate !== 1 ? `Speed: ${c.speed.rate}×` : '',
    (c.track ?? 1) > 1 && c.timeline_offset != null ? `Offset: ${fmtShort(c.timeline_offset)}` : '',
  ].filter(Boolean);
}

// ─── Edit-decision mutation helpers ─────────────────────────────────────────
// Each drag produces a candidate next EditDecision. Keeping the math out of
// the component's body keeps the drag callbacks small and each kind's rule
// localized.

const MIN_DUR = 0.1;   // never let something trim below 100ms

function applyRetrim(
  ed: EditDecision,
  drag: DragState,
  deltaSec: number,
): EditDecision | null {
  switch (drag.kind) {
    case 'v1':
    case 'v-overlay': {
      const clips = [...ed.clips];
      const c = { ...clips[drag.sourceIdx] };
      if (!c) return null;
      if (drag.mode === 'retrim-left') {
        c.source_start = Math.max(0, Math.min(drag.originalStart + deltaSec, c.source_end - MIN_DUR));
      } else {
        c.source_end = Math.max(c.source_start + MIN_DUR, drag.originalEnd + deltaSec);
      }
      clips[drag.sourceIdx] = c;
      return { ...ed, clips };
    }
    case 'title': {
      const titles = [...(ed.titles ?? [])];
      const t = { ...titles[drag.sourceIdx] };
      if (!t) return null;
      if (drag.mode === 'retrim-left') {
        // Move the left edge: shift timeline_offset + shrink duration
        const origStartTL = drag.originalTimelineOffset ?? t.timeline_offset;
        const origEndTL = origStartTL + drag.originalEnd; // originalEnd = original duration
        const newStart = Math.max(0, Math.min(origStartTL + deltaSec, origEndTL - MIN_DUR));
        t.timeline_offset = newStart;
        t.duration = origEndTL - newStart;
      } else {
        t.duration = Math.max(MIN_DUR, drag.originalEnd + deltaSec);
      }
      titles[drag.sourceIdx] = t;
      return { ...ed, titles };
    }
    case 'narration': {
      const narr = [...(ed.narration ?? [])];
      const n = { ...narr[drag.sourceIdx] };
      if (!n) return null;
      if (drag.mode === 'retrim-left') {
        // Left edge: shift in-point AND timeline_offset together, shrink duration.
        // Clamp so we don't run off either end of the audio file.
        const origStart = drag.originalStart;
        const origEnd = drag.originalEnd;
        const origTL = drag.originalTimelineOffset ?? n.timeline_offset;
        const newStart = Math.max(0, Math.min(origStart + deltaSec, origEnd - MIN_DUR));
        const shift = newStart - origStart;
        n.start = newStart;
        n.duration = origEnd - newStart;
        n.timeline_offset = Math.max(0, origTL + shift);
      } else {
        // Right edge: grow/shrink duration. Don't grow past audio file length
        // if we know it (we don't here — the renderer will clamp at play time).
        const newEnd = Math.max(n.start + MIN_DUR, drag.originalEnd + deltaSec);
        n.duration = newEnd - n.start;
      }
      narr[drag.sourceIdx] = n;
      return { ...ed, narration: narr };
    }
    case 'music': {
      const m = ed.music ? { ...ed.music } : null;
      if (!m) return null;
      if (drag.mode === 'retrim-left') {
        // Music's "position" on the timeline is tracked only via start + duration
        // (it doesn't have a timeline_offset field), so left-trim both advances
        // the in-point and shrinks the duration by the same amount.
        const newStart = Math.max(0, Math.min((drag.originalStart) + deltaSec, drag.originalEnd - MIN_DUR));
        const shift = newStart - drag.originalStart;
        m.start = newStart;
        m.duration = Math.max(MIN_DUR, (drag.originalEnd - drag.originalStart) - shift);
      } else {
        m.duration = Math.max(MIN_DUR, (drag.originalEnd - drag.originalStart) + deltaSec);
      }
      return { ...ed, music: m };
    }
  }
  return null;
}


// ─── Track builder ──────────────────────────────────────────────────────────

function buildTracks(ed: EditDecision): { tracks: Track[]; totalDuration: number } {
  const tracks: Track[] = [];
  let totalDur = 0;

  // ── Video family: V-tracks contain clips AND titles. Titles sit above clips
  // on a V-lane above the spine (per their `lane` field). We merge them so a
  // single family ('video') covers every visual layer.
  //
  // Collect per-track items in a dict keyed by V-track number, then emit
  // in NLE order — V1 at the bottom, V2/V3/… stacked upward. This matches
  // DaVinci/Premiere/FCP: higher track number = visually on top = drawn above.
  const videoBuckets = new Map<number, TrackItem[]>();

  // Pass 1 — V1 is the primary spine: clips are sequential. Compute timeline
  // positions from array order, not timeline_offset.
  const v1Items: TrackItem[] = [];
  let v1Offset = 0;
  ed.clips.forEach((c, idx) => {
    if ((c.track ?? 1) !== 1) return;
    const dur = Math.max(0.5, (c.source_end - c.source_start) / (c.speed?.rate ?? 1));
    const start = v1Offset;
    v1Offset += dur;
    const pal = CLIP_PALETTE[idx % CLIP_PALETTE.length];
    v1Items.push({
      id: c.id, kind: 'v1', sourceIdx: idx,
      tlStart: start, tlEnd: v1Offset,
      label: c.label || c.id, sublabel: baseName(c.source_file),
      color: pal.color, borderColor: pal.border, gainDb: c.gain_db,
      tooltip: clipTooltip(c), type: 'video',
    });
  });
  if (v1Items.length > 0) {
    videoBuckets.set(1, v1Items);
    totalDur = Math.max(totalDur, v1Offset);
  }

  // Pass 2 — V2+ overlay clips: free-placed via timeline_offset
  ed.clips.forEach((c, idx) => {
    const trackNum = c.track ?? 1;
    if (trackNum === 1) return;
    const dur = Math.max(0.5, (c.source_end - c.source_start) / (c.speed?.rate ?? 1));
    const start = c.timeline_offset ?? 0;
    const pal = CLIP_PALETTE[idx % CLIP_PALETTE.length];
    const item: TrackItem = {
      id: c.id, kind: 'v-overlay', sourceIdx: idx,
      tlStart: start, tlEnd: start + dur,
      label: c.label || c.id, sublabel: baseName(c.source_file),
      color: pal.color, borderColor: pal.border, gainDb: c.gain_db,
      tooltip: clipTooltip(c), type: 'video',
    };
    if (!videoBuckets.has(trackNum)) videoBuckets.set(trackNum, []);
    videoBuckets.get(trackNum)!.push(item);
    totalDur = Math.max(totalDur, start + dur);
  });

  // Pass 3 — Titles render on V-tracks per their `lane` field (default 1).
  // FCPXML lane numbering is "above the spine", so lane=1 → V2, lane=2 → V3.
  // V1 stays dedicated to the spine (main video clips).
  (ed.titles ?? []).forEach((t, idx) => {
    const laneNum = Math.max(1, t.lane ?? 1);
    const vTrack = laneNum + 1;
    const item: TrackItem = {
      id: `title-${idx}`, kind: 'title', sourceIdx: idx,
      tlStart: t.timeline_offset,
      tlEnd: t.timeline_offset + t.duration,
      label: t.text.length > 20 ? t.text.slice(0, 19) + '…' : t.text,
      color: T1.color, borderColor: T1.border,
      tooltip: [`"${t.text}"`, `Duration: ${t.duration}s`, `Lane V${vTrack}`],
      type: 'title',
    };
    if (!videoBuckets.has(vTrack)) videoBuckets.set(vTrack, []);
    videoBuckets.get(vTrack)!.push(item);
    totalDur = Math.max(totalDur, t.timeline_offset + t.duration);
  });

  // Emit V-tracks in descending order so V1 ends up at the visual BOTTOM
  // (rendered last in the track column, NLE-style). tracks.map() below
  // renders top-to-bottom, so the first emitted track is drawn at the top.
  for (const trackNum of [...videoBuckets.keys()].sort((a, b) => b - a)) {
    const id = `V${trackNum}`;
    tracks.push({
      id, name: id, family: 'video', trackNum,
      type: 'video', height: VIDEO_TRACK_H, color: V1.color,
      items: videoBuckets.get(trackNum)!,
    });
  }

  // ── Audio family: A-tracks contain narration segments and the music clip,
  // bucketed by their (optional) track field. Defaults preserve the old
  // layout: narration → A1, music → A2.
  const audioBuckets = new Map<number, TrackItem[]>();

  (ed.narration ?? []).forEach((n, idx) => {
    const trackNum = Math.max(1, n.track ?? 1);
    const item: TrackItem = {
      id: `narr-${idx}`, kind: 'narration', sourceIdx: idx,
      tlStart: n.timeline_offset,
      tlEnd: n.timeline_offset + n.duration,
      label: 'Narration', sublabel: baseName(n.file),
      color: A1.color, borderColor: A1.border, gainDb: n.gain_db,
      tooltip: [
        'Narration',
        `File: ${baseName(n.file)}`,
        `Duration: ${n.duration.toFixed(1)}s`,
        `Gain: ${n.gain_db}dB`,
        `Lane A${trackNum}`,
      ],
      type: 'audio',
    };
    if (!audioBuckets.has(trackNum)) audioBuckets.set(trackNum, []);
    audioBuckets.get(trackNum)!.push(item);
    totalDur = Math.max(totalDur, n.timeline_offset + n.duration);
  });

  if (ed.music) {
    const m = ed.music;
    const dur = m.duration > 0 ? m.duration : Math.max(totalDur, 10);
    const trackNum = Math.max(1, m.track ?? 2);
    const item: TrackItem = {
      id: 'music-0', kind: 'music', sourceIdx: 0,
      tlStart: m.start ?? 0,
      tlEnd: (m.start ?? 0) + dur,
      label: 'Music', sublabel: baseName(m.file),
      color: A2.color, borderColor: A2.border, gainDb: m.gain_db,
      tooltip: [
        'Music', `File: ${baseName(m.file)}`,
        `Gain: ${m.gain_db}dB`, `Duration: ${dur.toFixed(1)}s`,
        `Lane A${trackNum}`,
      ],
      type: 'audio',
    };
    if (!audioBuckets.has(trackNum)) audioBuckets.set(trackNum, []);
    audioBuckets.get(trackNum)!.push(item);
    totalDur = Math.max(totalDur, (m.start ?? 0) + dur);
  }

  // Emit A-tracks in ascending order
  for (const trackNum of [...audioBuckets.keys()].sort((a, b) => a - b)) {
    const id = `A${trackNum}`;
    tracks.push({
      id, name: id, family: 'audio', trackNum,
      type: 'audio', height: AUDIO_TRACK_H,
      color: trackNum === 1 ? A1.color : A2.color,
      items: audioBuckets.get(trackNum)!,
    });
  }

  return { tracks, totalDuration: totalDur };
}

// ─── ClipItem sub-component ──────────────────────────────────────────────────

interface ClipItemProps {
  item: TrackItem;
  track: Track;
  trackIdx: number;
  itemIdx: number;
  left: number;
  width: number;
  isHovered: boolean;
  isSelected: boolean;
  isCropping: boolean;
  isDragging: boolean;
  onHover: (trackIdx: number, itemIdx: number, e: React.MouseEvent) => void;
  onMove: (e: React.MouseEvent) => void;
  onLeave: () => void;
  onDragStart: (e: React.PointerEvent, item: TrackItem, edge: 'left' | 'right' | 'body') => void;
  onSelect?: (clipId: string) => void;
}

function ClipItem({
  item, track, trackIdx, itemIdx, left, width,
  isHovered, isSelected, isCropping, isDragging,
  onHover, onMove, onLeave, onDragStart, onSelect,
}: ClipItemProps) {
  const isAudioTrack = track.type === 'audio';
  const isTitleTrack = track.type === 'title';
  // All item kinds now support trim + move.
  const canDrag = true;
  const canSelect = true;

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (!canDrag) return;
    // Determine if we're on a retrim edge
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const relX = e.clientX - rect.left;
    if (relX <= RETRIM_HANDLE_W) {
      onDragStart(e, item, 'left');
    } else if (relX >= rect.width - RETRIM_HANDLE_W) {
      onDragStart(e, item, 'right');
    } else {
      onDragStart(e, item, 'body');
    }
  }, [canDrag, item, onDragStart]);

  // Determine cursor based on hover position
  const [cursor, setCursor] = useState<string>('default');
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    onMove(e);
    if (!canDrag) {
      if (canSelect) setCursor('pointer');
      return;
    }
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const relX = e.clientX - rect.left;
    if (relX <= RETRIM_HANDLE_W || relX >= rect.width - RETRIM_HANDLE_W) {
      setCursor('ew-resize');
    } else {
      setCursor('grab');
    }
  }, [canDrag, canSelect, onMove]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (!onSelect) return;
    if (canSelect) {
      e.stopPropagation();
      onSelect(item.id);
    }
  }, [onSelect, canSelect, item.id]);

  return (
    <div
      title={item.label + (item.sublabel ? ` — ${item.sublabel}` : '')}
      onMouseEnter={(e) => onHover(trackIdx, itemIdx, e)}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => { onLeave(); setCursor('default'); }}
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      style={{
        position: 'absolute',
        left,
        top: 2,
        width,
        height: track.height - 4,
        borderRadius: '3px',
        overflow: 'hidden',
        cursor: canDrag || canSelect ? cursor : 'default',
        background: isTitleTrack
          ? `linear-gradient(180deg, ${item.color}44, ${item.color}22)`
          : `linear-gradient(180deg, ${item.color}38, ${item.color}18, ${item.color}08)`,
        border: isSelected
          ? `1px solid ${item.borderColor}cc`
          : `1px solid ${isHovered ? `${item.borderColor}88` : `${item.color}44`}`,
        transition: isDragging ? 'none' : 'border-color 0.12s, box-shadow 0.12s',
        boxShadow: isSelected
          ? `inset 0 1px 0 ${item.color}55, 0 0 12px ${item.color}44, 0 0 4px ${item.color}22`
          : isHovered
            ? `inset 0 1px 0 ${item.color}55, 0 0 8px ${item.color}18`
            : `inset 0 1px 0 ${item.color}33`,
        opacity: isDragging ? 0.4 : 1,
        userSelect: 'none',
        touchAction: 'none',
      }}
    >
      {/* Retrim handle indicators (only video clips) */}
      {canDrag && (
        <>
          <div style={{
            position: 'absolute', left: 0, top: 0, width: RETRIM_HANDLE_W,
            height: '100%', zIndex: 3,
          }} />
          <div style={{
            position: 'absolute', right: 0, top: 0, width: RETRIM_HANDLE_W,
            height: '100%', zIndex: 3,
          }} />
          {/* Visual retrim indicators on hover */}
          {isHovered && (
            <>
              <div style={{
                position: 'absolute', left: 1, top: '20%', width: 2, height: '60%',
                background: `${item.color}88`, borderRadius: 1, zIndex: 4, pointerEvents: 'none',
              }} />
              <div style={{
                position: 'absolute', right: 1, top: '20%', width: 2, height: '60%',
                background: `${item.color}88`, borderRadius: 1, zIndex: 4, pointerEvents: 'none',
              }} />
            </>
          )}
        </>
      )}

      {/* Top color bar (video & title clips) */}
      {!isAudioTrack && (
        <div
          style={{
            height: '3px',
            background: `linear-gradient(90deg, ${item.color}aa, ${item.color}66)`,
            flexShrink: 0,
          }}
        />
      )}

      {/* Waveform for audio tracks */}
      {isAudioTrack && width > 30 && (
        <svg
          width={width}
          height={track.height - 4}
          style={{ position: 'absolute', left: 0, top: 0, display: 'block' }}
          viewBox={`0 0 ${width} ${track.height - 4}`}
          preserveAspectRatio="none"
        >
          <path
            d={waveformPath(width, track.height - 4, hashStr(item.id))}
            fill={`${item.color}25`}
            stroke={`${item.color}40`}
            strokeWidth="0.5"
          />
        </svg>
      )}

      {/* Content labels */}
      <div
        style={{
          position: 'relative',
          zIndex: 1,
          padding: isAudioTrack ? '0 5px' : '2px 5px 0',
          display: 'flex',
          flexDirection: 'column',
          gap: '0px',
          height: '100%',
          justifyContent: isAudioTrack ? 'center' : 'flex-start',
        }}
      >
        {width > 28 && (
          <span
            style={{
              fontSize: isTitleTrack ? '8px' : '9px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: isHovered ? 'var(--text-primary)' : 'var(--text-secondary)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              lineHeight: 1.3,
              transition: 'color 0.12s',
              fontStyle: isTitleTrack ? 'italic' : 'normal',
              pointerEvents: 'none',
            }}
          >
            {item.label}
          </span>
        )}
        {item.sublabel && width > 80 && !isTitleTrack && (
          <span
            style={{
              fontSize: '7px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              lineHeight: 1.2,
              opacity: 0.7,
              pointerEvents: 'none',
            }}
          >
            {item.sublabel}
          </span>
        )}
      </div>

      {/* Gain badge */}
      {item.gainDb != null && item.gainDb !== 0 && width > 50 && (
        <div
          style={{
            position: 'absolute',
            right: 4,
            top: 3,
            fontSize: '7px',
            fontFamily: 'var(--font-mono)',
            color:
              item.gainDb < -12
                ? 'rgba(255,140,124,0.8)'
                : 'rgba(255,255,255,0.45)',
            background: 'rgba(0,0,0,0.5)',
            borderRadius: '2px',
            padding: '1px 3px',
            zIndex: 2,
            pointerEvents: 'none',
          }}
        >
          {item.gainDb > 0 ? '+' : ''}
          {item.gainDb}dB
        </div>
      )}

      {/* Crop progress shimmer */}
      {isCropping && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            bottom: 0,
            width: '100%',
            height: '3px',
            background: `linear-gradient(90deg, transparent, ${item.color}88, transparent)`,
            backgroundSize: '200% 100%',
            animation: 'shimmer 1.5s infinite linear',
            zIndex: 5,
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export function NLETimeline({ editDecision, playheadTime = 0, selectedClipId, cropStatuses, onSeek, onEditDecisionChange, onEditDecisionPreview, onClipSelect }: NLETimelineProps) {
  // During a drag, stream updates through the preview channel (no server, no
  // history). On pointer-up we commit once with onEditDecisionChange. Falls
  // back to the commit channel if no preview callback is provided.
  const livePreview = onEditDecisionPreview ?? onEditDecisionChange;
  const scrollRef = useRef<HTMLDivElement>(null);
  const [hoveredItem, setHoveredItem] = useState<{ trackIdx: number; itemIdx: number } | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const isPanning = useRef(false);
  const panStartX = useRef(0);
  const panScrollLeft = useRef(0);

  // Drag state (using refs for performance during pointer moves)
  const dragRef = useRef<DragState | null>(null);
  const [dragVisual, setDragVisual] = useState<{
    mode: DragMode;
    itemId: string;
    ghostLeft: number;
    targetTrackNum: number;
    sourceTrackNum: number;
    sourceFamily: TrackFamily;
  } | null>(null);

  // Drag-to-pan (only when not dragging a clip)
  const onPanPointerDown = useCallback((e: React.PointerEvent) => {
    if (dragRef.current) return;
    const el = scrollRef.current;
    if (!el) return;
    isPanning.current = true;
    panStartX.current = e.clientX;
    panScrollLeft.current = el.scrollLeft;
    el.style.cursor = 'grabbing';
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPanPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isPanning.current || !scrollRef.current) return;
    scrollRef.current.scrollLeft = panScrollLeft.current - (e.clientX - panStartX.current);
  }, []);

  const onPanPointerUp = useCallback(() => {
    isPanning.current = false;
    if (scrollRef.current) scrollRef.current.style.cursor = 'grab';
  }, []);

  // Ctrl+wheel → zoom
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const d = e.deltaY > 0 ? -0.15 : 0.15;
        setZoom((z) => Math.max(0.25, Math.min(4, z + d * z)));
      }
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, []);

  // Build tracks
  const { tracks, totalDuration, timelineWidth, pxPerSec } = useMemo(() => {
    if (!editDecision || editDecision.clips.length === 0) {
      return { tracks: [] as Track[], totalDuration: 0, timelineWidth: 0, pxPerSec: BASE_PX_PER_SEC };
    }
    const { tracks, totalDuration } = buildTracks(editDecision);
    const pps = BASE_PX_PER_SEC * zoom;
    const tlW = Math.max(totalDuration * pps, 400);
    return { tracks, totalDuration, timelineWidth: tlW, pxPerSec: pps };
  }, [editDecision, zoom]);

  // Ruler ticks
  const rulerTicks = useMemo(() => {
    if (totalDuration <= 0) return [];
    let interval = 1;
    if (pxPerSec < 8) interval = 30;
    else if (pxPerSec < 15) interval = 10;
    else if (pxPerSec < 30) interval = 5;
    else if (pxPerSec < 60) interval = 2;

    const ticks: { x: number; label: string; major: boolean }[] = [];
    for (let t = 0; t <= totalDuration + 0.01; t += interval) {
      const x = t * pxPerSec;
      const major = t % (interval * 2) === 0 || t === 0;
      ticks.push({ x, label: fmtShort(t), major });
    }
    return ticks;
  }, [totalDuration, pxPerSec]);

  const handleHover = useCallback(
    (trackIdx: number, itemIdx: number, e: React.MouseEvent) => {
      if (dragRef.current) return; // don't show tooltips while dragging
      setHoveredItem({ trackIdx, itemIdx });
      setTooltipPos({ x: e.clientX, y: e.clientY });
    },
    [],
  );

  const handleMove = useCallback((e: React.MouseEvent) => {
    if (dragRef.current) return;
    setTooltipPos({ x: e.clientX, y: e.clientY });
  }, []);

  const handleLeave = useCallback(() => {
    setHoveredItem(null);
  }, []);

  // ─── Ruler seek ────────────────────────────────────────────────────────
  const seekingRef = useRef(false);
  const [isSeeking, setIsSeeking] = useState(false);
  const resolveSeekTime = useCallback((clientX: number): number | null => {
    const el = scrollRef.current;
    if (!el || totalDuration <= 0) return null;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left + el.scrollLeft;
    return Math.max(0, Math.min(totalDuration, x / pxPerSec));
  }, [totalDuration, pxPerSec]);

  const onRulerPointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation();
    const t = resolveSeekTime(e.clientX);
    if (t != null && onSeek) { onSeek(t); }
    seekingRef.current = true;
    setIsSeeking(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [resolveSeekTime, onSeek]);

  const onRulerPointerMove = useCallback((e: React.PointerEvent) => {
    if (!seekingRef.current) return;
    const t = resolveSeekTime(e.clientX);
    if (t != null && onSeek) { onSeek(t); }
  }, [resolveSeekTime, onSeek]);

  const onRulerPointerUp = useCallback(() => {
    seekingRef.current = false;
    setIsSeeking(false);
  }, []);

  // ─── Clip drag (reorder / retrim) ─────────────────────────────────────

  const handleClipDragStart = useCallback((e: React.PointerEvent, item: TrackItem, edge: 'left' | 'right' | 'body') => {
    if (!editDecision || !onEditDecisionChange) return;
    e.stopPropagation();
    e.preventDefault();

    // Resolve the current source record from its array + track this item is on
    // based on the item's kind. Each branch also records the family so the
    // move handler can block cross-family drops (V → A or A → V).
    let family: TrackFamily;
    let sourceTrackNum: number;
    let originalStart: number;
    let originalEnd: number;
    let originalTimelineOffset: number | undefined;

    switch (item.kind) {
      case 'v1':
      case 'v-overlay': {
        const clip = editDecision.clips[item.sourceIdx];
        if (!clip) return;
        family = 'video';
        sourceTrackNum = clip.track ?? 1;
        originalStart = clip.source_start;
        originalEnd = clip.source_end;
        originalTimelineOffset = clip.timeline_offset;
        break;
      }
      case 'title': {
        const t = (editDecision.titles ?? [])[item.sourceIdx];
        if (!t) return;
        family = 'video';  // titles live on V-tracks above the spine
        // lane=1 (first overlay in FCPXML) renders on V2; lane=N → V(N+1).
        sourceTrackNum = Math.max(1, t.lane ?? 1) + 1;
        originalStart = 0;  // titles don't have a source in-point
        originalEnd = t.duration;
        originalTimelineOffset = t.timeline_offset;
        break;
      }
      case 'narration': {
        const n = (editDecision.narration ?? [])[item.sourceIdx];
        if (!n) return;
        family = 'audio';
        sourceTrackNum = Math.max(1, n.track ?? 1);
        originalStart = n.start;
        originalEnd = n.start + n.duration;
        originalTimelineOffset = n.timeline_offset;
        break;
      }
      case 'music': {
        const m = editDecision.music;
        if (!m) return;
        family = 'audio';
        sourceTrackNum = Math.max(1, m.track ?? 2);
        originalStart = m.start ?? 0;
        originalEnd = (m.start ?? 0) + (m.duration || (item.tlEnd - item.tlStart));
        originalTimelineOffset = item.tlStart;  // music has no timeline_offset field
        break;
      }
    }

    const mode: DragMode = edge === 'body' ? 'move' : edge === 'left' ? 'retrim-left' : 'retrim-right';

    // Calculate ghost offset for move
    const el = scrollRef.current;
    let ghostOffset = 0;
    if (el && mode === 'move') {
      const rect = el.getBoundingClientRect();
      const scrollX = el.scrollLeft;
      ghostOffset = (e.clientX - rect.left + scrollX) - item.tlStart * pxPerSec;
    }

    dragRef.current = {
      mode,
      sourceFamily: family,
      sourceTrackNum,
      kind: item.kind,
      sourceIdx: item.sourceIdx,
      itemId: item.id,
      startX: e.clientX,
      startY: e.clientY,
      ghostOffset,
      targetTrackNum: sourceTrackNum,
      ghostLeft: item.tlStart * pxPerSec,
      timelineOffset: item.tlStart,
      originalStart,
      originalEnd,
      originalTimelineOffset,
    };

    const target = scrollRef.current || (e.target as HTMLElement);
    target.setPointerCapture(e.pointerId);
    setHoveredItem(null);
  }, [editDecision, onEditDecisionChange, pxPerSec]);

  // Resolve pointer Y to a destination track for the given family.
  // Returns the track number inside that family (1-indexed). If the pointer
  // is below the last track of the family, returns max+1 (new track). If
  // above the first track of the family, returns 1.
  const resolveTrackAtY = useCallback((clientY: number, family: TrackFamily): number => {
    const el = scrollRef.current;
    if (!el) return 1;
    const familyTracks = tracks.filter(t => t.family === family);
    if (familyTracks.length === 0) return 1;

    for (const t of familyTracks) {
      const trackEl = el.querySelector(`[data-track-id="${t.id}"]`);
      if (trackEl) {
        const rect = trackEl.getBoundingClientRect();
        if (clientY >= rect.top && clientY <= rect.bottom) {
          return t.trackNum;
        }
      }
    }

    // Pointer outside every family track. Pick above-all vs below-all based
    // on the first and last track's bounding rects — note "first" here is the
    // *visually topmost* track, which for video is the HIGHEST trackNum (V3
    // is drawn above V1, NLE-style). For audio the order is still ascending.
    const firstEl = el.querySelector(`[data-track-id="${familyTracks[0].id}"]`);
    const lastEl = el.querySelector(`[data-track-id="${familyTracks[familyTracks.length - 1].id}"]`);
    const maxNum = Math.max(...familyTracks.map(t => t.trackNum));
    const minNum = Math.min(...familyTracks.map(t => t.trackNum));
    // Dragging ABOVE the visual top: create a new higher-numbered track.
    if (firstEl && clientY < firstEl.getBoundingClientRect().top) {
      return family === 'video' ? maxNum + 1 : minNum;
    }
    // Dragging BELOW the visual bottom: create a new lower-numbered track
    // for audio, or snap to V1 for video (can't go lower than V1).
    if (lastEl && clientY > lastEl.getBoundingClientRect().bottom) {
      return family === 'video' ? minNum : maxNum + 1;
    }
    return familyTracks[0].trackNum;
  }, [tracks]);

  // Global pointer move for drag operations
  const handleDragPointerMove = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || !editDecision) {
      onPanPointerMove(e);
      return;
    }

    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;

    // Deadzone
    if (Math.abs(dx) < DRAG_DEADZONE && Math.abs(dy) < DRAG_DEADZONE) return;

    if (drag.mode === 'move') {
      const el = scrollRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const scrollX = el.scrollLeft;
      const pointerXInTimeline = e.clientX - rect.left + scrollX;
      const ghostLeft = Math.max(0, pointerXInTimeline - drag.ghostOffset);
      const timelineOffset = Math.max(0, ghostLeft / pxPerSec);

      // Constrain destination track to the item's family (V → A or A → V
      // is not allowed; the drop snaps back to the source track).
      const targetTrackNum = resolveTrackAtY(e.clientY, drag.sourceFamily);

      drag.targetTrackNum = targetTrackNum;
      drag.ghostLeft = ghostLeft;
      drag.timelineOffset = timelineOffset;

      setDragVisual({
        mode: 'move',
        itemId: drag.itemId,
        ghostLeft,
        targetTrackNum,
        sourceTrackNum: drag.sourceTrackNum,
        sourceFamily: drag.sourceFamily,
      });
    } else if (drag.mode === 'retrim-left' || drag.mode === 'retrim-right') {
      const deltaSec = dx / pxPerSec;

      setDragVisual({
        mode: drag.mode,
        itemId: drag.itemId,
        ghostLeft: 0,
        targetTrackNum: drag.sourceTrackNum,
        sourceTrackNum: drag.sourceTrackNum,
        sourceFamily: drag.sourceFamily,
      });

      // Apply retrim per kind. Route through the PREVIEW channel so we don't
      // hit the WebSocket + disk on every pointer-move (a drag fires 40-50
      // moves). The final value commits on pointer-up.
      const next = applyRetrim(editDecision, drag, deltaSec);
      if (next) livePreview?.(next);
    }
  }, [editDecision, onEditDecisionChange, livePreview, tracks, pxPerSec, onPanPointerMove, resolveTrackAtY]);

  // Global pointer up — commit drag
  const handleDragPointerUp = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || !editDecision || !onEditDecisionChange) {
      onPanPointerUp();
      dragRef.current = null;
      setDragVisual(null);
      return;
    }

    // If pointer didn't move beyond deadzone, treat as a click (select clip)
    const dx = Math.abs(e.clientX - drag.startX);
    const dy = Math.abs(e.clientY - drag.startY);
    if (dx < DRAG_DEADZONE && dy < DRAG_DEADZONE) {
      dragRef.current = null;
      setDragVisual(null);
      if (onClipSelect) {
        onClipSelect(drag.itemId);
      }
      return;
    }

    if (drag.mode === 'move') {
      const toTrack = drag.targetTrackNum;
      const timelineOffset = drag.timelineOffset;

      // Branch on the item kind so each schema field gets updated correctly.
      // Cross-family drops were already blocked in resolveTrackAtY, so by
      // here sourceFamily === destination family.
      switch (drag.kind) {
        case 'v1':
        case 'v-overlay': {
          const clipIdx = drag.sourceIdx;
          const clip = editDecision.clips[clipIdx];
          if (!clip) break;
          const newClips = [...editDecision.clips];

          if (toTrack === 1) {
            // Landing on V1 — insert into the sequential spine.
            // Compute insert position from the hover timelineOffset.
            const [moved] = newClips.splice(clipIdx, 1);
            const v1Only = newClips.filter(c => (c.track ?? 1) === 1);
            let offset = 0;
            let insertAt = v1Only.length;
            for (let j = 0; j < v1Only.length; j++) {
              const vc = v1Only[j];
              const dur = Math.max(0.5, (vc.source_end - vc.source_start) / (vc.speed?.rate ?? 1));
              const mid = offset + dur / 2;
              if (timelineOffset < mid) { insertAt = j; break; }
              offset += dur;
            }
            const targetArrayIdx = insertAt < v1Only.length
              ? newClips.indexOf(v1Only[insertAt])
              : newClips.length;
            newClips.splice(targetArrayIdx, 0, { ...moved, track: 1, timeline_offset: undefined });
          } else {
            // Landing on V2+ — free placement via timeline_offset.
            newClips[clipIdx] = { ...clip, track: toTrack, timeline_offset: timelineOffset };
          }
          onEditDecisionChange({ ...editDecision, clips: newClips });
          break;
        }
        case 'title': {
          const titles = [...(editDecision.titles ?? [])];
          const t = titles[drag.sourceIdx];
          if (!t) break;
          // V-track N → FCPXML lane (N-1); V1 spine is not a valid title lane,
          // so clamp to lane≥1 (title dropped on V1 snaps to the first overlay).
          const lane = Math.max(1, toTrack - 1);
          titles[drag.sourceIdx] = { ...t, lane, timeline_offset: timelineOffset };
          onEditDecisionChange({ ...editDecision, titles });
          break;
        }
        case 'narration': {
          const narr = [...(editDecision.narration ?? [])];
          const n = narr[drag.sourceIdx];
          if (!n) break;
          narr[drag.sourceIdx] = { ...n, track: toTrack, timeline_offset: timelineOffset };
          onEditDecisionChange({ ...editDecision, narration: narr });
          break;
        }
        case 'music': {
          if (!editDecision.music) break;
          // Music uses `start` for its timeline position (no timeline_offset field).
          const m = { ...editDecision.music, track: toTrack, start: timelineOffset };
          onEditDecisionChange({ ...editDecision, music: m });
          break;
        }
      }
    } else if (drag.mode === 'retrim-left' || drag.mode === 'retrim-right') {
      // Retrim has been live-previewing via livePreview. Commit the final
      // state once so the server persists it and the undo stack records it.
      // editDecision already reflects all the preview mutations.
      onEditDecisionChange(editDecision);
    }

    dragRef.current = null;
    setDragVisual(null);
  }, [editDecision, onEditDecisionChange, onPanPointerUp, onClipSelect]);

  // ─── Empty state ────────────────────────────────────────────────────────

  if (!editDecision || editDecision.clips.length === 0) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(8,6,5,0.4)',
          borderRadius: '4px',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.25 }}>
            <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" />
            <line x1="2" y1="8" x2="22" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.5" />
            <line x1="2" y1="16" x2="22" y2="16" stroke="currentColor" strokeWidth="1" opacity="0.5" />
            <rect x="4" y="5" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="8" y="5" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="14" y="5" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="18" y="5" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="4" y="17" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="8" y="17" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="14" y="17" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="18" y="17" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.3" />
          </svg>
          <span
            style={{
              color: 'var(--text-muted)',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
            }}
          >
            Timeline will appear after FCPXML generation
          </span>
        </div>
      </div>
    );
  }

  // ─── Populated timeline ─────────────────────────────────────────────────

  const timelineName = editDecision.timeline?.name || 'Untitled Timeline';
  const fps = editDecision.timeline?.fps || 24;
  const clipCount = editDecision.clips.length;
  const hasNarration = (editDecision.narration?.length ?? 0) > 0;
  const hasMusic = !!editDecision.music;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(8,6,5,0.4)',
        borderRadius: '4px',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* ── Header bar ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '4px 10px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
          background: 'rgba(255,255,255,0.015)',
        }}
      >
        <span
          style={{
            color: 'var(--accent-blue)',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {timelineName}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: '9px', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
          {fps}fps
        </span>
        <div style={{ flex: 1 }} />

        {/* Stats pills */}
        <span style={statStyle}>
          {clipCount} clips
        </span>
        {hasNarration && <span style={{ ...statStyle, color: A1.color }}>VO</span>}
        {hasMusic && <span style={{ ...statStyle, color: A2.color }}>♫</span>}
        <span style={{ ...statStyle, color: 'var(--text-secondary)', fontWeight: 600 }}>
          {fmtShort(totalDuration)}
        </span>

        {/* Zoom slider */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            flexShrink: 0,
            marginLeft: '4px',
            borderLeft: '1px solid rgba(255,255,255,0.06)',
            paddingLeft: '8px',
          }}
        >
          <span style={zoomBtnStyle} onClick={() => setZoom((z) => Math.max(0.25, z / 1.3))}>
            −
          </span>
          <input
            type="range"
            min="0.25"
            max="4"
            step="0.05"
            value={zoom}
            onChange={(e) => setZoom(parseFloat(e.target.value))}
            style={{
              width: '60px',
              height: '3px',
              appearance: 'none',
              WebkitAppearance: 'none',
              background: `linear-gradient(90deg, var(--accent-blue) ${((zoom - 0.25) / 3.75) * 100}%, rgba(255,255,255,0.1) ${((zoom - 0.25) / 3.75) * 100}%)`,
              borderRadius: '2px',
              outline: 'none',
              cursor: 'pointer',
            }}
          />
          <span style={zoomBtnStyle} onClick={() => setZoom((z) => Math.min(4, z * 1.3))}>
            +
          </span>
          <span
            style={{
              fontSize: '8px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              minWidth: '28px',
              textAlign: 'right',
            }}
          >
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      {/* ── Track area ── */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
        {/* Track headers (fixed) */}
        <div
          style={{
            width: TRACK_HEADER_W,
            flexShrink: 0,
            borderRight: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            flexDirection: 'column',
            background: 'rgba(0,0,0,0.2)',
          }}
        >
          {/* Ruler header */}
          <div
            style={{
              height: RULER_H,
              borderBottom: '1px solid rgba(255,255,255,0.06)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <span
              style={{
                fontSize: '8px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)',
                letterSpacing: '0.06em',
                opacity: 0.6,
              }}
            >
              TC
            </span>
          </div>

          {/* Per-track headers */}
          {tracks.map((track) => (
            <div
              key={track.id}
              style={{
                height: track.height,
                borderBottom: `${TRACK_BORDER}px solid rgba(0,0,0,0.5)`,
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '0 6px',
              }}
            >
              <div
                style={{
                  width: '3px',
                  height: '14px',
                  borderRadius: '1px',
                  background: track.color,
                  opacity: 0.7,
                  flexShrink: 0,
                }}
              />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', minWidth: 0 }}>
                <span
                  style={{
                    fontSize: '9px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    color: 'var(--text-secondary)',
                    letterSpacing: '0.04em',
                    lineHeight: 1,
                  }}
                >
                  {track.name}
                </span>
                <span
                  style={{
                    fontSize: '7px',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                    letterSpacing: '0.02em',
                    lineHeight: 1,
                    opacity: 0.7,
                  }}
                >
                  {track.type === 'video' ? 'Video' : track.type === 'title' ? 'Title' : 'Audio'}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Scrollable timeline */}
        <div
          ref={scrollRef}
          onPointerDown={onPanPointerDown}
          onPointerMove={handleDragPointerMove}
          onPointerUp={(e) => { handleDragPointerUp(e); onPanPointerUp(); }}
          onPointerCancel={(e) => { handleDragPointerUp(e); onPanPointerUp(); }}
          style={{
            flex: 1,
            overflowX: 'auto',
            overflowY: 'hidden',
            position: 'relative',
            cursor: dragVisual ? 'grabbing' : 'grab',
          }}
        >
          <div style={{ width: timelineWidth, minWidth: '100%', position: 'relative' }}>
            {/* Timecode ruler */}
            <div
              onPointerDown={onRulerPointerDown}
              onPointerMove={onRulerPointerMove}
              onPointerUp={onRulerPointerUp}
              onPointerCancel={onRulerPointerUp}
              style={{
                height: RULER_H,
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                position: 'relative',
                background: 'rgba(0,0,0,0.15)',
                cursor: onSeek ? 'crosshair' : undefined,
              }}
            >
              {rulerTicks.map((tick, i) => (
                <React.Fragment key={i}>
                  <div
                    style={{
                      position: 'absolute',
                      left: tick.x,
                      top: tick.major ? 0 : RULER_H * 0.45,
                      width: '1px',
                      height: tick.major ? RULER_H : RULER_H * 0.55,
                      background: tick.major ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.06)',
                    }}
                  />
                  {tick.major && (
                    <span
                      style={{
                        position: 'absolute',
                        left: tick.x + 4,
                        top: '2px',
                        fontSize: '8px',
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--text-muted)',
                        letterSpacing: '0.02em',
                        whiteSpace: 'nowrap',
                        lineHeight: 1,
                        pointerEvents: 'none',
                        userSelect: 'none',
                      }}
                    >
                      {tick.label}
                    </span>
                  )}
                </React.Fragment>
              ))}
            </div>

            {/* Track lanes */}
            {tracks.map((track, trackIdx) => {
              return (
                <div
                  key={track.id}
                  data-track-id={track.id}
                  onClick={(e) => {
                    if (e.target === e.currentTarget && onClipSelect) onClipSelect(null);
                  }}
                  style={{
                    height: track.height,
                    borderBottom: `${TRACK_BORDER}px solid rgba(0,0,0,0.5)`,
                    position: 'relative',
                    background: trackIdx % 2 === 0 ? 'rgba(0,0,0,0.08)' : 'rgba(0,0,0,0.12)',
                  }}
                >
                  {track.items.map((item, itemIdx) => {
                    const left = item.tlStart * pxPerSec;
                    const width = Math.max(MIN_CLIP_W, (item.tlEnd - item.tlStart) * pxPerSec);
                    const isHovered =
                      hoveredItem?.trackIdx === trackIdx && hoveredItem?.itemIdx === itemIdx;
                    const isBeingDragged = dragVisual?.mode === 'move' && dragVisual.itemId === item.id;

                    return (
                      <ClipItem
                        key={item.id}
                        item={item}
                        track={track}
                        trackIdx={trackIdx}
                        itemIdx={itemIdx}
                        left={left}
                        width={width}
                        isHovered={isHovered}
                        isSelected={selectedClipId === item.id}
                        isCropping={cropStatuses?.[item.id]?.status === 'running'}
                        isDragging={isBeingDragged}
                        onHover={handleHover}
                        onMove={handleMove}
                        onLeave={handleLeave}
                        onDragStart={handleClipDragStart}
                        onSelect={onClipSelect ? (id) => onClipSelect(id) : undefined}
                      />
                    );
                  })}

                  {/* Drop indicator — highlight target track, any family */}
                  {dragVisual?.mode === 'move' && (() => {
                    // Only render the indicator on tracks in the same family
                    // as the item being dragged. The drag state carries its
                    // source family; we compare track.family to that.
                    if (track.family !== dragVisual.sourceFamily) return null;
                    if (track.trackNum !== dragVisual.targetTrackNum) return null;
                    const indicatorX = dragVisual.ghostLeft;
                    return (
                      <div
                        style={{
                          position: 'absolute',
                          left: indicatorX - 1,
                          top: 0,
                          width: 2,
                          height: track.height,
                          background: 'var(--accent-blue)',
                          borderRadius: 1,
                          zIndex: 20,
                          pointerEvents: 'none',
                          boxShadow: '0 0 6px var(--accent-blue)',
                        }}
                      />
                    );
                  })()}

                  {/* Transition diamonds between V1 clips */}
                  {track.type === 'video' &&
                    track.items.slice(0, -1).map((item, i) => {
                      const nextItem = track.items[i + 1];
                      if (!nextItem) return null;
                      const midX = item.tlEnd * pxPerSec;
                      return (
                        <div
                          key={`tr-${i}`}
                          style={{
                            position: 'absolute',
                            left: midX - 4,
                            top: track.height / 2 - 4,
                            width: '8px',
                            height: '8px',
                            transform: 'rotate(45deg)',
                            background: 'rgba(255,255,255,0.12)',
                            border: '1px solid rgba(255,255,255,0.18)',
                            borderRadius: '1px',
                            zIndex: 5,
                            pointerEvents: 'none',
                          }}
                        />
                      );
                    })}
                </div>
              );
            })}

            {/* Ghost clip for move drag */}
            {dragVisual?.mode === 'move' && (() => {
              // Find the dragged item across all tracks
              let dragItem: TrackItem | undefined;
              for (const t of tracks) {
                dragItem = t.items.find(it => it.id === dragVisual.itemId);
                if (dragItem) break;
              }
              if (!dragItem) return null;
              const ghostWidth = Math.max(MIN_CLIP_W, (dragItem.tlEnd - dragItem.tlStart) * pxPerSec);

              // Position ghost on the target track
              const targetTrackId = `V${dragVisual.targetTrackNum}`;
              const targetTrack = tracks.find(t => t.id === targetTrackId);
              let ghostTop = RULER_H + 2; // default
              if (targetTrack) {
                // Calculate cumulative Y offset to this track
                let yOffset = RULER_H;
                for (const t of tracks) {
                  if (t.id === targetTrackId) break;
                  yOffset += t.height + TRACK_BORDER;
                }
                ghostTop = yOffset + 2;
              } else {
                // New track — position below the last video track
                // Find end of all video tracks
                let lastVideoEnd = RULER_H;
                for (const t of tracks) {
                  if (t.type === 'video') {
                    lastVideoEnd += t.height + TRACK_BORDER;
                  } else break; // video tracks come first
                }
                ghostTop = lastVideoEnd + 4;
              }

              const isNewTrack = !targetTrack;
              const isChangingTrack = dragVisual.targetTrackNum !== dragVisual.sourceTrackNum;

              return (
                <div
                  style={{
                    position: 'absolute',
                    left: dragVisual.ghostLeft,
                    top: ghostTop,
                    width: ghostWidth,
                    height: VIDEO_TRACK_H - 4,
                    borderRadius: '3px',
                    background: `${dragItem.color}55`,
                    border: `2px solid ${isChangingTrack ? 'var(--accent-blue)' : dragItem.color}99`,
                    zIndex: 30,
                    pointerEvents: 'none',
                    boxShadow: `0 4px 12px rgba(0,0,0,0.4), 0 0 8px ${dragItem.color}33`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <span style={{
                    fontSize: '8px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    color: isNewTrack ? 'var(--accent-blue)' : dragItem.color,
                    opacity: 0.9,
                    pointerEvents: 'none',
                  }}>
                    {isNewTrack ? `+ V${dragVisual.targetTrackNum}` : dragItem.label}
                  </span>
                </div>
              );
            })()}

            {/* Playhead */}
            <div
              style={{
                position: 'absolute',
                left: playheadTime * pxPerSec,
                top: 0,
                width: '1px',
                height: '100%',
                background: 'var(--accent-red)',
                zIndex: 10,
                pointerEvents: 'none',
                boxShadow: '0 0 4px rgba(255,125,111,0.4)',
                transition: isSeeking ? 'none' : 'left 0.1s linear',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: '-4px',
                  width: 0,
                  height: 0,
                  borderLeft: '4px solid transparent',
                  borderRight: '4px solid transparent',
                  borderTop: '6px solid var(--accent-red)',
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Tooltip (portal to body for highest z-index) ── */}
      {hoveredItem && tooltipPos && !dragVisual && tracks[hoveredItem.trackIdx]?.items[hoveredItem.itemIdx] &&
        createPortal(
          <div
            style={{
              position: 'fixed',
              left: tooltipPos.x + 8,
              top: tooltipPos.y + 8,
              zIndex: 99999,
              background: 'rgba(18,15,13,0.96)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '6px',
              padding: '8px 10px',
              minWidth: '160px',
              maxWidth: '280px',
              pointerEvents: 'none',
              boxShadow: '0 6px 24px rgba(0,0,0,0.6)',
              backdropFilter: 'blur(12px)',
            }}
          >
            {tracks[hoveredItem.trackIdx].items[hoveredItem.itemIdx].tooltip.map((line, i) => (
              <div
                key={i}
                style={{
                  fontSize: i === 0 ? '10px' : '9px',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: i === 0 ? 700 : 400,
                  color: i === 0
                    ? tracks[hoveredItem.trackIdx].items[hoveredItem.itemIdx].color
                    : 'var(--text-muted)',
                  lineHeight: 1.6,
                }}
              >
                {line}
              </div>
            ))}
          </div>,
          document.body,
        )}
    </div>
  );
}

// ─── Shared styles ──────────────────────────────────────────────────────────

const statStyle: React.CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: '9px',
  fontFamily: 'var(--font-mono)',
  letterSpacing: '0.04em',
  flexShrink: 0,
};

const zoomBtnStyle: React.CSSProperties = {
  fontSize: '9px',
  color: 'var(--text-muted)',
  fontFamily: 'var(--font-mono)',
  cursor: 'pointer',
  userSelect: 'none',
};
