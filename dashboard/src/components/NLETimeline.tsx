import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type {
  EditDecision,
  EditDecisionClip,
  NarrationSegment,
  MusicTrack,
  TextOverlay,
} from '../hooks/useAgentChat';

// ─── Types ───────────────────────────────────────────────────────────────────

interface NLETimelineProps {
  editDecision: EditDecision | null;
  playheadTime?: number;
  onSeek?: (time: number) => void;
  onEditDecisionChange?: (updated: EditDecision) => void;
}

interface TrackItem {
  id: string;
  clipIndex: number;   // index in editDecision.clips (-1 for non-clip items)
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
  id: string;
  name: string;
  type: 'video' | 'audio' | 'title';
  height: number;
  color: string;
  items: TrackItem[];
}

// Drag modes
type DragMode = 'reorder' | 'retrim-left' | 'retrim-right' | null;

interface DragState {
  mode: DragMode;
  trackId: string;
  itemId: string;
  clipIndex: number;
  startX: number;
  startY: number;
  // For reorder
  originalOrder: number;  // index in clips array
  ghostOffset: number;    // px offset from clip left edge to pointer
  dropIndex: number;      // where to insert
  draggedDown: boolean;   // dragged below track → new track
  // For retrim
  originalStart: number;
  originalEnd: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const TRACK_HEADER_W = 52;
const RULER_H = 22;
const VIDEO_TRACK_H = 48;
const AUDIO_TRACK_H = 34;
const TITLE_TRACK_H = 30;
const TRACK_BORDER = 1;
const MIN_CLIP_W = 22;
const BASE_PX_PER_SEC = 18;
const RETRIM_HANDLE_W = 5;
const DRAG_DEADZONE = 4;
const DRAG_DOWN_THRESHOLD = 30; // px below track to trigger new track creation

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

function fmtTC(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * 24);
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(f)}`;
}

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

// ─── Track builder ──────────────────────────────────────────────────────────

function buildTracks(ed: EditDecision): { tracks: Track[]; totalDuration: number } {
  const tracks: Track[] = [];
  let totalDur = 0;

  // Video tracks — group clips by track number (V1, V2, etc.)
  if (ed.clips.length > 0) {
    // Group clips by track number, preserving original index
    const trackGroups = new Map<number, { clip: EditDecisionClip; origIndex: number }[]>();
    ed.clips.forEach((c, i) => {
      const trackNum = c.track ?? 1;
      if (!trackGroups.has(trackNum)) trackGroups.set(trackNum, []);
      trackGroups.get(trackNum)!.push({ clip: c, origIndex: i });
    });

    // Sort track numbers so V1 comes first
    const sortedTrackNums = [...trackGroups.keys()].sort((a, b) => a - b);

    for (const trackNum of sortedTrackNums) {
      const group = trackGroups.get(trackNum)!;
      let offset = 0;
      const items: TrackItem[] = group.map(({ clip: c, origIndex }) => {
        const dur = Math.max(0.5, (c.source_end - c.source_start) / (c.speed?.rate ?? 1));
        const start = offset;
        offset += dur;
        const pal = CLIP_PALETTE[origIndex % CLIP_PALETTE.length];
        return {
          id: c.id,
          clipIndex: origIndex,
          tlStart: start,
          tlEnd: offset,
          label: c.label || c.id,
          sublabel: baseName(c.source_file),
          color: pal.color,
          borderColor: pal.border,
          gainDb: c.gain_db,
          tooltip: [
            c.label || c.id,
            c.description || '',
            `Source: ${baseName(c.source_file)}`,
            `In: ${fmtShort(c.source_start)} → Out: ${fmtShort(c.source_end)}`,
            c.gain_db != null ? `Gain: ${c.gain_db}dB` : '',
            c.speed && c.speed.rate !== 1 ? `Speed: ${c.speed.rate}×` : '',
          ].filter(Boolean),
          type: 'video',
        };
      });
      totalDur = Math.max(totalDur, offset);
      const trackId = `V${trackNum}`;
      tracks.push({ id: trackId, name: trackId, type: 'video', height: VIDEO_TRACK_H, color: V1.color, items });
    }
  }

  // T1 — Titles
  const titles = ed.titles ?? [];
  if (titles.length > 0) {
    const items: TrackItem[] = titles.map((t, i) => ({
      id: `title-${i}`,
      clipIndex: -1,
      tlStart: t.timeline_offset,
      tlEnd: t.timeline_offset + t.duration,
      label: t.text.length > 20 ? t.text.slice(0, 19) + '…' : t.text,
      color: T1.color,
      borderColor: T1.border,
      tooltip: [`"${t.text}"`, `Duration: ${t.duration}s`],
      type: 'title',
    }));
    for (const it of items) totalDur = Math.max(totalDur, it.tlEnd);
    tracks.push({ id: 'T1', name: 'T1', type: 'title', height: TITLE_TRACK_H, color: T1.color, items });
  }

  // A1 — Narration
  const narr = ed.narration ?? [];
  if (narr.length > 0) {
    const items: TrackItem[] = narr.map((n, i) => ({
      id: `narr-${i}`,
      clipIndex: -1,
      tlStart: n.timeline_offset,
      tlEnd: n.timeline_offset + n.duration,
      label: 'Narration',
      sublabel: baseName(n.file),
      color: A1.color,
      borderColor: A1.border,
      gainDb: n.gain_db,
      tooltip: [`Narration`, `File: ${baseName(n.file)}`, `Duration: ${n.duration.toFixed(1)}s`, `Gain: ${n.gain_db}dB`],
      type: 'audio',
    }));
    for (const it of items) totalDur = Math.max(totalDur, it.tlEnd);
    tracks.push({ id: 'A1', name: 'A1', type: 'audio', height: AUDIO_TRACK_H, color: A1.color, items });
  }

  // A2 — Music
  if (ed.music) {
    const m = ed.music;
    const dur = m.duration > 0 ? m.duration : Math.max(totalDur, 10);
    const items: TrackItem[] = [{
      id: 'music-0',
      clipIndex: -1,
      tlStart: 0,
      tlEnd: dur,
      label: 'Music',
      sublabel: baseName(m.file),
      color: A2.color,
      borderColor: A2.border,
      gainDb: m.gain_db,
      tooltip: [`Music`, `File: ${baseName(m.file)}`, `Gain: ${m.gain_db}dB`, `Duration: ${dur.toFixed(1)}s`],
      type: 'audio',
    }];
    totalDur = Math.max(totalDur, dur);
    tracks.push({ id: 'A2', name: 'A2', type: 'audio', height: AUDIO_TRACK_H, color: A2.color, items });
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
  pxPerSec: number;
  isHovered: boolean;
  isDragging: boolean;
  isDropTarget: boolean;
  isVideoTrack: boolean;
  onHover: (trackIdx: number, itemIdx: number, e: React.MouseEvent) => void;
  onMove: (e: React.MouseEvent) => void;
  onLeave: () => void;
  onDragStart: (e: React.PointerEvent, item: TrackItem, edge: 'left' | 'right' | 'body') => void;
}

function ClipItem({
  item, track, trackIdx, itemIdx, left, width, pxPerSec,
  isHovered, isDragging, isDropTarget, isVideoTrack,
  onHover, onMove, onLeave, onDragStart,
}: ClipItemProps) {
  const isAudioTrack = track.type === 'audio';
  const isTitleTrack = track.type === 'title';
  const canDrag = isVideoTrack && item.clipIndex >= 0;

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
    if (!canDrag) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const relX = e.clientX - rect.left;
    if (relX <= RETRIM_HANDLE_W || relX >= rect.width - RETRIM_HANDLE_W) {
      setCursor('ew-resize');
    } else {
      setCursor('grab');
    }
  }, [canDrag, onMove]);

  return (
    <div
      onMouseEnter={(e) => onHover(trackIdx, itemIdx, e)}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => { onLeave(); setCursor('default'); }}
      onPointerDown={handlePointerDown}
      style={{
        position: 'absolute',
        left,
        top: 2,
        width,
        height: track.height - 4,
        borderRadius: '3px',
        overflow: 'hidden',
        cursor: canDrag ? cursor : 'default',
        background: isTitleTrack
          ? `linear-gradient(180deg, ${item.color}44, ${item.color}22)`
          : `linear-gradient(180deg, ${item.color}38, ${item.color}18, ${item.color}08)`,
        border: `1px solid ${isHovered ? `${item.borderColor}88` : `${item.color}44`}`,
        transition: isDragging ? 'none' : 'border-color 0.12s',
        boxShadow: isHovered
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
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export function NLETimeline({ editDecision, playheadTime = 0, onSeek, onEditDecisionChange }: NLETimelineProps) {
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
    dropIndex: number;
    draggedDown: boolean;
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
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [resolveSeekTime, onSeek]);

  const onRulerPointerMove = useCallback((e: React.PointerEvent) => {
    if (!seekingRef.current) return;
    const t = resolveSeekTime(e.clientX);
    if (t != null && onSeek) { onSeek(t); }
  }, [resolveSeekTime, onSeek]);

  const onRulerPointerUp = useCallback(() => {
    seekingRef.current = false;
  }, []);

  // ─── Clip drag (reorder / retrim) ─────────────────────────────────────

  const handleClipDragStart = useCallback((e: React.PointerEvent, item: TrackItem, edge: 'left' | 'right' | 'body') => {
    if (!editDecision || !onEditDecisionChange) return;
    e.stopPropagation();
    e.preventDefault();

    const clip = editDecision.clips[item.clipIndex];
    if (!clip) return;

    const mode: DragMode = edge === 'body' ? 'reorder' : edge === 'left' ? 'retrim-left' : 'retrim-right';

    // Calculate ghost offset for reorder
    const el = scrollRef.current;
    let ghostOffset = 0;
    if (el && mode === 'reorder') {
      const rect = el.getBoundingClientRect();
      const scrollX = el.scrollLeft;
      ghostOffset = (e.clientX - rect.left + scrollX) - item.tlStart * pxPerSec;
    }

    dragRef.current = {
      mode,
      trackId: 'V1',
      itemId: item.id,
      clipIndex: item.clipIndex,
      startX: e.clientX,
      startY: e.clientY,
      originalOrder: item.clipIndex,
      ghostOffset,
      dropIndex: item.clipIndex,
      draggedDown: false,
      originalStart: clip.source_start,
      originalEnd: clip.source_end,
    };

    // Capture pointer on the scroll container for global tracking
    const target = scrollRef.current || (e.target as HTMLElement);
    target.setPointerCapture(e.pointerId);

    setHoveredItem(null);
  }, [editDecision, onEditDecisionChange, pxPerSec]);

  // Global pointer move for drag operations
  const handleDragPointerMove = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || !editDecision) {
      // Fall through to pan
      onPanPointerMove(e);
      return;
    }

    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;

    // Deadzone
    if (Math.abs(dx) < DRAG_DEADZONE && Math.abs(dy) < DRAG_DEADZONE) return;

    if (drag.mode === 'reorder') {
      const el = scrollRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const scrollX = el.scrollLeft;
      const pointerXInTimeline = e.clientX - rect.left + scrollX;
      const ghostLeft = pointerXInTimeline - drag.ghostOffset;

      // Find V1 track to determine drop position
      const v1Track = tracks.find(t => t.id === 'V1');
      if (!v1Track) return;

      // Determine drop index based on ghost center position
      const ghostCenter = ghostLeft + ((v1Track.items[drag.clipIndex]?.tlEnd ?? 0) - (v1Track.items[drag.clipIndex]?.tlStart ?? 0)) * pxPerSec / 2;
      let dropIdx = 0;
      for (let i = 0; i < v1Track.items.length; i++) {
        const itemCenter = ((v1Track.items[i].tlStart + v1Track.items[i].tlEnd) / 2) * pxPerSec;
        if (ghostCenter > itemCenter) dropIdx = i + 1;
      }

      // Check if dragged below the V1 track
      const v1TrackEl = el.querySelector('[data-track-id="V1"]');
      const draggedDown = v1TrackEl
        ? e.clientY > (v1TrackEl.getBoundingClientRect().bottom + DRAG_DOWN_THRESHOLD)
        : false;

      drag.dropIndex = dropIdx;
      drag.draggedDown = draggedDown;

      setDragVisual({
        mode: 'reorder',
        itemId: drag.itemId,
        ghostLeft,
        dropIndex: dropIdx,
        draggedDown,
      });
    } else if (drag.mode === 'retrim-left' || drag.mode === 'retrim-right') {
      const deltaSec = dx / pxPerSec;
      const clip = editDecision.clips[drag.clipIndex];
      if (!clip) return;

      // Preview retrim visually — we'll commit on pointer up
      setDragVisual({
        mode: drag.mode,
        itemId: drag.itemId,
        ghostLeft: 0,
        dropIndex: 0,
        draggedDown: false,
      });

      // Live retrim preview: update edit decision in real-time
      const newClips = [...editDecision.clips];
      const c = { ...newClips[drag.clipIndex] };
      if (drag.mode === 'retrim-left') {
        c.source_start = Math.max(0, Math.min(drag.originalStart + deltaSec, c.source_end - 0.1));
      } else {
        c.source_end = Math.max(c.source_start + 0.1, drag.originalEnd + deltaSec);
      }
      newClips[drag.clipIndex] = c;
      onEditDecisionChange?.({ ...editDecision, clips: newClips });
    }
  }, [editDecision, onEditDecisionChange, tracks, pxPerSec, onPanPointerMove]);

  // Global pointer up — commit drag
  const handleDragPointerUp = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag || !editDecision || !onEditDecisionChange) {
      onPanPointerUp();
      dragRef.current = null;
      setDragVisual(null);
      return;
    }

    if (drag.mode === 'reorder') {
      const fromIdx = drag.originalOrder;
      let toIdx = drag.dropIndex;

      if (drag.draggedDown) {
        // Move clip to the next available video track
        const newClips = [...editDecision.clips];
        const clip = newClips[fromIdx];
        const currentTrack = clip.track ?? 1;
        // Find the highest track number currently in use
        const maxTrack = newClips.reduce((max, c) => Math.max(max, c.track ?? 1), 1);
        const nextTrack = currentTrack >= maxTrack ? maxTrack + 1 : currentTrack + 1;
        newClips[fromIdx] = { ...clip, track: nextTrack };
        onEditDecisionChange({ ...editDecision, clips: newClips });
      } else if (fromIdx !== toIdx) {
        // Reorder clips array
        const newClips = [...editDecision.clips];
        const [moved] = newClips.splice(fromIdx, 1);
        // Adjust toIdx if moving forward (since we removed an item before it)
        if (toIdx > fromIdx) toIdx--;
        newClips.splice(toIdx, 0, moved);
        onEditDecisionChange({ ...editDecision, clips: newClips });
      }
    }
    // Retrim is already committed via live updates in pointer move

    dragRef.current = null;
    setDragVisual(null);
  }, [editDecision, onEditDecisionChange, onPanPointerUp]);

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
  const trackCount = tracks.length;
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
            cursor: dragRef.current ? 'grabbing' : 'grab',
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
              const isVideoTrack = track.type === 'video';
              return (
                <div
                  key={track.id}
                  data-track-id={track.id}
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
                    const isBeingDragged = dragVisual?.mode === 'reorder' && dragVisual.itemId === item.id;

                    return (
                      <ClipItem
                        key={item.id}
                        item={item}
                        track={track}
                        trackIdx={trackIdx}
                        itemIdx={itemIdx}
                        left={left}
                        width={width}
                        pxPerSec={pxPerSec}
                        isHovered={isHovered}
                        isDragging={isBeingDragged}
                        isDropTarget={false}
                        isVideoTrack={isVideoTrack}
                        onHover={handleHover}
                        onMove={handleMove}
                        onLeave={handleLeave}
                        onDragStart={handleClipDragStart}
                      />
                    );
                  })}

                  {/* Drop indicator for reorder */}
                  {dragVisual?.mode === 'reorder' && isVideoTrack && !dragVisual.draggedDown && (() => {
                    const dropIdx = dragVisual.dropIndex;
                    const items = track.items;
                    let indicatorX = 0;
                    if (dropIdx >= items.length) {
                      indicatorX = (items[items.length - 1]?.tlEnd ?? 0) * pxPerSec;
                    } else {
                      indicatorX = (items[dropIdx]?.tlStart ?? 0) * pxPerSec;
                    }
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

            {/* Ghost clip for reorder drag */}
            {dragVisual?.mode === 'reorder' && (() => {
              const v1Track = tracks.find(t => t.id === 'V1');
              const dragItem = v1Track?.items.find(it => it.id === dragVisual.itemId);
              if (!v1Track || !dragItem) return null;
              const ghostWidth = Math.max(MIN_CLIP_W, (dragItem.tlEnd - dragItem.tlStart) * pxPerSec);
              const ghostTop = RULER_H + (dragVisual.draggedDown ? v1Track.height + 8 : 2);
              return (
                <div
                  style={{
                    position: 'absolute',
                    left: dragVisual.ghostLeft,
                    top: ghostTop,
                    width: ghostWidth,
                    height: v1Track.height - 4,
                    borderRadius: '3px',
                    background: `${dragItem.color}55`,
                    border: `2px solid ${dragItem.color}99`,
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
                    color: dragItem.color,
                    opacity: 0.9,
                    pointerEvents: 'none',
                  }}>
                    {dragVisual.draggedDown ? '+ New Track' : dragItem.label}
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
                transition: seekingRef.current ? 'none' : 'left 0.1s linear',
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
