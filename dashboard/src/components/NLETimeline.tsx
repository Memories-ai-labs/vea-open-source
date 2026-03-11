import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
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
}

interface TrackItem {
  id: string;
  tlStart: number;  // timeline seconds
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

// ─── Constants ───────────────────────────────────────────────────────────────

const TRACK_HEADER_W = 52;
const RULER_H = 22;
const VIDEO_TRACK_H = 48;
const AUDIO_TRACK_H = 34;
const TITLE_TRACK_H = 30;
const TRACK_BORDER = 1;
const MIN_CLIP_W = 22;
const BASE_PX_PER_SEC = 18;

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

// ─── Track builder ──────────────────────────────────────────────────────────

function buildTracks(ed: EditDecision): { tracks: Track[]; totalDuration: number } {
  const tracks: Track[] = [];
  let totalDur = 0;

  // V1 — Primary video spine (clips are sequential)
  if (ed.clips.length > 0) {
    let offset = 0;
    const items: TrackItem[] = ed.clips.map((c, i) => {
      const dur = Math.max(0.5, (c.source_end - c.source_start) / (c.speed?.rate ?? 1));
      const start = offset;
      offset += dur;
      const pal = CLIP_PALETTE[i % CLIP_PALETTE.length];
      return {
        id: c.id,
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
    tracks.push({ id: 'V1', name: 'V1', type: 'video', height: VIDEO_TRACK_H, color: V1.color, items });
  }

  // T1 — Titles
  const titles = ed.titles ?? [];
  if (titles.length > 0) {
    const items: TrackItem[] = titles.map((t, i) => ({
      id: `title-${i}`,
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

// ─── Component ───────────────────────────────────────────────────────────────

export function NLETimeline({ editDecision }: NLETimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [hoveredItem, setHoveredItem] = useState<{ trackIdx: number; itemIdx: number } | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const isDragging = useRef(false);
  const dragStartX = useRef(0);
  const dragScrollLeft = useRef(0);

  // Drag-to-pan
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const el = scrollRef.current;
    if (!el) return;
    isDragging.current = true;
    dragStartX.current = e.clientX;
    dragScrollLeft.current = el.scrollLeft;
    el.style.cursor = 'grabbing';
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging.current || !scrollRef.current) return;
    scrollRef.current.scrollLeft = dragScrollLeft.current - (e.clientX - dragStartX.current);
  }, []);

  const onPointerUp = useCallback(() => {
    isDragging.current = false;
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
      setHoveredItem({ trackIdx, itemIdx });
      setTooltipPos({ x: e.clientX, y: e.clientY });
    },
    [],
  );

  const handleMove = useCallback((e: React.MouseEvent) => {
    setTooltipPos({ x: e.clientX, y: e.clientY });
  }, []);

  const handleLeave = useCallback(() => {
    setHoveredItem(null);
  }, []);

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
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          style={{
            flex: 1,
            overflowX: 'auto',
            overflowY: 'hidden',
            position: 'relative',
            cursor: 'grab',
          }}
        >
          <div style={{ width: timelineWidth, minWidth: '100%', position: 'relative' }}>
            {/* Timecode ruler */}
            <div
              style={{
                height: RULER_H,
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                position: 'relative',
                background: 'rgba(0,0,0,0.15)',
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
            {tracks.map((track, trackIdx) => (
              <div
                key={track.id}
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
                  const isAudioTrack = track.type === 'audio';
                  const isTitleTrack = track.type === 'title';

                  return (
                    <div
                      key={item.id}
                      onMouseEnter={(e) => handleHover(trackIdx, itemIdx, e)}
                      onMouseMove={handleMove}
                      onMouseLeave={handleLeave}
                      style={{
                        position: 'absolute',
                        left,
                        top: 2,
                        width,
                        height: track.height - 4,
                        borderRadius: '3px',
                        overflow: 'hidden',
                        cursor: 'default',
                        background: isTitleTrack
                          ? `linear-gradient(180deg, ${item.color}44, ${item.color}22)`
                          : `linear-gradient(180deg, ${item.color}38, ${item.color}18, ${item.color}08)`,
                        border: `1px solid ${isHovered ? `${item.borderColor}88` : `${item.color}44`}`,
                        transition: 'border-color 0.12s',
                        boxShadow: isHovered
                          ? `inset 0 1px 0 ${item.color}55, 0 0 8px ${item.color}18`
                          : `inset 0 1px 0 ${item.color}33`,
                      }}
                    >
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
                          }}
                        >
                          {item.gainDb > 0 ? '+' : ''}
                          {item.gainDb}dB
                        </div>
                      )}
                    </div>
                  );
                })}

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
            ))}

            {/* Playhead */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: '1px',
                height: '100%',
                background: 'var(--accent-red)',
                zIndex: 10,
                pointerEvents: 'none',
                boxShadow: '0 0 4px rgba(255,125,111,0.4)',
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

      {/* ── Tooltip ── */}
      {hoveredItem && tooltipPos && tracks[hoveredItem.trackIdx]?.items[hoveredItem.itemIdx] && (
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x + 14,
            top: tooltipPos.y - 10,
            zIndex: 1000,
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
        </div>
      )}
    </div>
  );
}

// ─── Utility ────────────────────────────────────────────────────────────────

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
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
