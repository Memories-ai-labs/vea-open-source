import React, { useMemo, useRef, useState, useCallback } from 'react';
import type { EditDecision, EditDecisionClip } from '../hooks/useAgentChat';

// ─── Types ───────────────────────────────────────────────────────────────────

interface NLETimelineProps {
  editDecision: EditDecision | null;
}

interface ClipLayout {
  clip: EditDecisionClip;
  x: number;      // px offset from timeline start
  width: number;   // px width
  color: string;
  colorRaw: string; // hex without var()
  index: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const TRACK_HEADER_W = 52;
const RULER_H = 22;
const VIDEO_TRACK_H = 48;
const AUDIO_TRACK_H = 32;
const TRACK_GAP = 1;
const MIN_CLIP_W = 28;
const PX_PER_SEC = 18; // base zoom: 18px per second of timeline

const CLIP_COLORS = [
  { css: 'var(--accent-blue)',   hex: '#60d5c8' },
  { css: 'var(--accent-purple)', hex: '#d79bb5' },
  { css: 'var(--accent-green)',  hex: '#8ad27e' },
  { css: 'var(--accent-orange)', hex: '#f39a55' },
  { css: 'var(--accent-yellow)', hex: '#f1bf63' },
  { css: 'var(--accent-red)',    hex: '#ff7d6f' },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtTC(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * 24);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
}

function fmtShort(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** Generate a deterministic pseudo-waveform pattern for a clip */
function waveformPath(width: number, height: number, seed: number): string {
  const segments = Math.max(4, Math.floor(width / 3));
  const mid = height / 2;
  let d = `M 0 ${mid}`;
  for (let i = 0; i <= segments; i++) {
    const x = (i / segments) * width;
    // Pseudo-random amplitude from seed
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    const amp = (hash - Math.floor(hash)) * 0.85 + 0.15;
    const y = mid - amp * (mid - 2);
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  // Mirror bottom half
  for (let i = segments; i >= 0; i--) {
    const x = (i / segments) * width;
    const hash = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
    const amp = (hash - Math.floor(hash)) * 0.85 + 0.15;
    const y = mid + amp * (mid - 2);
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  d += ' Z';
  return d;
}

// ─── Component ───────────────────────────────────────────────────────────────

export function NLETimeline({ editDecision }: NLETimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [hoveredClip, setHoveredClip] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  // Compute clip layouts
  const { clips, totalDuration, timelineWidth } = useMemo(() => {
    if (!editDecision || editDecision.clips.length === 0) {
      return { clips: [] as ClipLayout[], totalDuration: 0, timelineWidth: 0 };
    }

    const totalDur = editDecision.clips.reduce(
      (sum, c) => sum + Math.max(0.5, c.source_end - c.source_start), 0
    );
    const tlWidth = Math.max(totalDur * PX_PER_SEC, 400);

    let x = 0;
    const layouts: ClipLayout[] = editDecision.clips.map((clip, i) => {
      const dur = Math.max(0.5, clip.source_end - clip.source_start);
      const w = Math.max(MIN_CLIP_W, (dur / totalDur) * tlWidth);
      const layout: ClipLayout = {
        clip,
        x,
        width: w,
        color: CLIP_COLORS[i % CLIP_COLORS.length].css,
        colorRaw: CLIP_COLORS[i % CLIP_COLORS.length].hex,
        index: i,
      };
      x += w;
      return layout;
    });

    return { clips: layouts, totalDuration: totalDur, timelineWidth: x };
  }, [editDecision]);

  // Ruler tick generation
  const rulerTicks = useMemo(() => {
    if (totalDuration <= 0) return [];
    const ticks: { x: number; label: string; major: boolean }[] = [];
    // Choose tick interval based on duration
    let interval = 1;
    if (totalDuration > 120) interval = 10;
    else if (totalDuration > 30) interval = 5;
    else if (totalDuration > 10) interval = 2;

    let cumDur = 0;
    for (let t = 0; t <= totalDuration; t += interval) {
      const x = (t / totalDuration) * timelineWidth;
      const major = t % (interval * 2) === 0 || t === 0;
      ticks.push({ x, label: fmtShort(t), major });
    }
    return ticks;
  }, [totalDuration, timelineWidth]);

  const handleClipHover = useCallback((index: number | null, e?: React.MouseEvent) => {
    setHoveredClip(index);
    if (e && index !== null) {
      setTooltipPos({ x: e.clientX, y: e.clientY });
    }
  }, []);

  // ─── Empty state ────────────────────────────────────────────────────────

  if (!editDecision || editDecision.clips.length === 0) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(8,6,5,0.4)',
        borderRadius: '4px',
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
        }}>
          {/* Film strip icon */}
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.25 }}>
            <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none" />
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
          <span style={{
            color: 'var(--text-muted)',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>
            Timeline will appear after FCPXML generation
          </span>
        </div>
      </div>
    );
  }

  // ─── Populated timeline ─────────────────────────────────────────────────

  const totalH = RULER_H + VIDEO_TRACK_H + TRACK_GAP + AUDIO_TRACK_H;
  const timelineName = editDecision.timeline?.name || 'Untitled Timeline';
  const fps = editDecision.timeline?.fps || 24;

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'rgba(8,6,5,0.4)',
      borderRadius: '4px',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* ── Timeline header bar ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '4px 10px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        flexShrink: 0,
        background: 'rgba(255,255,255,0.015)',
      }}>
        {/* Timeline name + info */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          minWidth: 0,
        }}>
          <span style={{
            color: 'var(--accent-blue)',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {timelineName}
          </span>
          <span style={{
            color: 'var(--text-muted)',
            fontSize: '9px',
            fontFamily: 'var(--font-mono)',
            flexShrink: 0,
          }}>
            {fps}fps
          </span>
        </div>

        <div style={{ flex: 1 }} />

        {/* Clip count + duration */}
        <span style={{
          color: 'var(--text-muted)',
          fontSize: '9px',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.04em',
          flexShrink: 0,
        }}>
          {editDecision.clips.length} clips
        </span>
        <span style={{
          color: 'var(--text-secondary)',
          fontSize: '9px',
          fontFamily: 'var(--font-mono)',
          fontWeight: 600,
          letterSpacing: '0.04em',
          flexShrink: 0,
        }}>
          {fmtShort(totalDuration)}
        </span>
      </div>

      {/* ── Track area: header + scrollable tracks ── */}
      <div style={{
        flex: 1,
        display: 'flex',
        minHeight: 0,
        overflow: 'hidden',
      }}>
        {/* Track headers (fixed left column) */}
        <div style={{
          width: TRACK_HEADER_W,
          flexShrink: 0,
          borderRight: '1px solid rgba(255,255,255,0.08)',
          display: 'flex',
          flexDirection: 'column',
          background: 'rgba(0,0,0,0.2)',
        }}>
          {/* Ruler header */}
          <div style={{
            height: RULER_H,
            borderBottom: '1px solid rgba(255,255,255,0.06)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <span style={{
              fontSize: '8px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              letterSpacing: '0.06em',
              opacity: 0.6,
            }}>
              TC
            </span>
          </div>

          {/* V1 header */}
          <div style={{
            height: VIDEO_TRACK_H,
            borderBottom: `${TRACK_GAP}px solid rgba(0,0,0,0.5)`,
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '0 6px',
          }}>
            <div style={{
              width: '3px',
              height: '14px',
              borderRadius: '1px',
              background: 'var(--accent-blue)',
              opacity: 0.7,
              flexShrink: 0,
            }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', minWidth: 0 }}>
              <span style={{
                fontSize: '9px',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                letterSpacing: '0.04em',
                lineHeight: 1,
              }}>
                V1
              </span>
              <span style={{
                fontSize: '7px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)',
                letterSpacing: '0.02em',
                lineHeight: 1,
                opacity: 0.7,
              }}>
                Video
              </span>
            </div>
          </div>

          {/* A1 header */}
          <div style={{
            height: AUDIO_TRACK_H,
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '0 6px',
          }}>
            <div style={{
              width: '3px',
              height: '14px',
              borderRadius: '1px',
              background: 'var(--accent-green)',
              opacity: 0.7,
              flexShrink: 0,
            }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', minWidth: 0 }}>
              <span style={{
                fontSize: '9px',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                letterSpacing: '0.04em',
                lineHeight: 1,
              }}>
                A1
              </span>
              <span style={{
                fontSize: '7px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)',
                letterSpacing: '0.02em',
                lineHeight: 1,
                opacity: 0.7,
              }}>
                Audio
              </span>
            </div>
          </div>
        </div>

        {/* Scrollable timeline area */}
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowX: 'auto',
            overflowY: 'hidden',
            position: 'relative',
          }}
        >
          <div style={{
            width: timelineWidth,
            minWidth: '100%',
            position: 'relative',
          }}>
            {/* ── Timecode ruler ── */}
            <div style={{
              height: RULER_H,
              borderBottom: '1px solid rgba(255,255,255,0.08)',
              position: 'relative',
              background: 'rgba(0,0,0,0.15)',
            }}>
              {rulerTicks.map((tick, i) => (
                <React.Fragment key={i}>
                  {/* Tick mark */}
                  <div style={{
                    position: 'absolute',
                    left: tick.x,
                    top: tick.major ? 0 : RULER_H * 0.45,
                    width: '1px',
                    height: tick.major ? RULER_H : RULER_H * 0.55,
                    background: tick.major
                      ? 'rgba(255,255,255,0.15)'
                      : 'rgba(255,255,255,0.06)',
                  }} />
                  {/* Label */}
                  {tick.major && (
                    <span style={{
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
                    }}>
                      {tick.label}
                    </span>
                  )}
                </React.Fragment>
              ))}
            </div>

            {/* ── Video track (V1) ── */}
            <div style={{
              height: VIDEO_TRACK_H,
              borderBottom: `${TRACK_GAP}px solid rgba(0,0,0,0.5)`,
              position: 'relative',
              background: 'rgba(0,0,0,0.08)',
            }}>
              {clips.map((cl) => {
                const isHovered = hoveredClip === cl.index;
                return (
                  <div
                    key={cl.clip.id}
                    onMouseEnter={(e) => handleClipHover(cl.index, e)}
                    onMouseMove={(e) => setTooltipPos({ x: e.clientX, y: e.clientY })}
                    onMouseLeave={() => handleClipHover(null)}
                    style={{
                      position: 'absolute',
                      left: cl.x,
                      top: 2,
                      width: cl.width,
                      height: VIDEO_TRACK_H - 4,
                      borderRadius: '3px',
                      overflow: 'hidden',
                      cursor: 'default',
                      // Layered background: color bar at top, dark fill, subtle noise
                      background: `
                        linear-gradient(180deg,
                          ${cl.colorRaw}38 0%,
                          ${cl.colorRaw}18 20%,
                          ${cl.colorRaw}08 100%
                        )
                      `,
                      border: `1px solid ${isHovered ? `${cl.colorRaw}88` : `${cl.colorRaw}44`}`,
                      transition: 'border-color 0.12s',
                      // Top highlight edge like DaVinci
                      boxShadow: isHovered
                        ? `inset 0 1px 0 ${cl.colorRaw}55, 0 0 8px ${cl.colorRaw}18`
                        : `inset 0 1px 0 ${cl.colorRaw}33`,
                    }}
                  >
                    {/* Top color bar (like DaVinci's clip header) */}
                    <div style={{
                      height: '3px',
                      background: `linear-gradient(90deg, ${cl.colorRaw}aa, ${cl.colorRaw}66)`,
                      flexShrink: 0,
                    }} />

                    {/* Clip content area */}
                    <div style={{
                      padding: '2px 5px 0',
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0px',
                      height: VIDEO_TRACK_H - 7, // minus top bar and borders
                    }}>
                      {/* Clip label */}
                      <span style={{
                        fontSize: '9px',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 600,
                        color: isHovered ? 'var(--text-primary)' : 'var(--text-secondary)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        lineHeight: 1.3,
                        transition: 'color 0.12s',
                      }}>
                        {cl.clip.label || cl.clip.id}
                      </span>

                      {/* Source info */}
                      <span style={{
                        fontSize: '7px',
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--text-muted)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        lineHeight: 1.2,
                        opacity: 0.7,
                      }}>
                        {fmtShort(cl.clip.source_start)} - {fmtShort(cl.clip.source_end)}
                      </span>

                      {/* Clip ID badge */}
                      <div style={{
                        marginTop: 'auto',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '3px',
                        paddingBottom: '1px',
                      }}>
                        <span style={{
                          fontSize: '7px',
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 700,
                          color: cl.colorRaw,
                          opacity: 0.8,
                          letterSpacing: '0.04em',
                          lineHeight: 1,
                        }}>
                          {cl.clip.id}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Transition diamonds between clips */}
              {clips.slice(0, -1).map((cl, i) => {
                const nextCl = clips[i + 1];
                if (!nextCl) return null;
                const midX = cl.x + cl.width;
                return (
                  <div
                    key={`tr-${i}`}
                    style={{
                      position: 'absolute',
                      left: midX - 4,
                      top: VIDEO_TRACK_H / 2 - 4,
                      width: '8px',
                      height: '8px',
                      transform: 'rotate(45deg)',
                      background: 'rgba(255,255,255,0.12)',
                      border: '1px solid rgba(255,255,255,0.18)',
                      borderRadius: '1px',
                      zIndex: 2,
                      pointerEvents: 'none',
                    }}
                  />
                );
              })}
            </div>

            {/* ── Audio track (A1) ── */}
            <div style={{
              height: AUDIO_TRACK_H,
              position: 'relative',
              background: 'rgba(0,0,0,0.12)',
            }}>
              {clips.map((cl) => {
                const dur = cl.clip.source_end - cl.clip.source_start;
                const isHovered = hoveredClip === cl.index;
                return (
                  <div
                    key={`a-${cl.clip.id}`}
                    onMouseEnter={(e) => handleClipHover(cl.index, e)}
                    onMouseMove={(e) => setTooltipPos({ x: e.clientX, y: e.clientY })}
                    onMouseLeave={() => handleClipHover(null)}
                    style={{
                      position: 'absolute',
                      left: cl.x,
                      top: 2,
                      width: cl.width,
                      height: AUDIO_TRACK_H - 4,
                      borderRadius: '2px',
                      overflow: 'hidden',
                      cursor: 'default',
                      background: `rgba(138,210,126,${isHovered ? '0.08' : '0.04'})`,
                      border: `1px solid rgba(138,210,126,${isHovered ? '0.28' : '0.15'})`,
                      transition: 'background 0.12s, border-color 0.12s',
                    }}
                  >
                    {/* Simulated waveform */}
                    <svg
                      width={cl.width}
                      height={AUDIO_TRACK_H - 4}
                      style={{ display: 'block' }}
                      viewBox={`0 0 ${cl.width} ${AUDIO_TRACK_H - 4}`}
                      preserveAspectRatio="none"
                    >
                      <path
                        d={waveformPath(cl.width, AUDIO_TRACK_H - 4, cl.index * 7 + 3)}
                        fill="rgba(138,210,126,0.25)"
                        stroke="rgba(138,210,126,0.4)"
                        strokeWidth="0.5"
                      />
                    </svg>

                    {/* Clip name overlay */}
                    <span style={{
                      position: 'absolute',
                      left: '4px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      fontSize: '7px',
                      fontFamily: 'var(--font-mono)',
                      color: 'rgba(138,210,126,0.6)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      maxWidth: cl.width - 8,
                      pointerEvents: 'none',
                      letterSpacing: '0.02em',
                    }}>
                      {cl.clip.id}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* ── Playhead (at position 0) ── */}
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: '1px',
              height: '100%',
              background: 'var(--accent-red)',
              zIndex: 10,
              pointerEvents: 'none',
              boxShadow: '0 0 4px rgba(255,125,111,0.4)',
            }}>
              {/* Playhead triangle */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: '-4px',
                width: 0,
                height: 0,
                borderLeft: '4px solid transparent',
                borderRight: '4px solid transparent',
                borderTop: '6px solid var(--accent-red)',
              }} />
            </div>
          </div>
        </div>
      </div>

      {/* ── Tooltip ── */}
      {hoveredClip !== null && tooltipPos && clips[hoveredClip] && (
        <div style={{
          position: 'fixed',
          left: tooltipPos.x + 14,
          top: tooltipPos.y - 10,
          zIndex: 1000,
          background: 'rgba(18,15,13,0.96)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: '6px',
          padding: '8px 10px',
          minWidth: '180px',
          maxWidth: '280px',
          pointerEvents: 'none',
          boxShadow: '0 6px 24px rgba(0,0,0,0.6)',
          backdropFilter: 'blur(12px)',
        }}>
          {(() => {
            const cl = clips[hoveredClip];
            const dur = cl.clip.source_end - cl.clip.source_start;
            const fileName = cl.clip.source_file.split('/').pop()?.replace(/^user_media_\w+_/, '') || cl.clip.source_file;
            return (
              <>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  marginBottom: '4px',
                }}>
                  <span style={{
                    fontSize: '10px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    color: cl.colorRaw,
                  }}>
                    {cl.clip.id}
                  </span>
                  {cl.clip.label && (
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                    }}>
                      {cl.clip.label}
                    </span>
                  )}
                </div>
                <div style={{
                  fontSize: '9px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  lineHeight: 1.6,
                }}>
                  <div>Source: {fileName.length > 30 ? fileName.slice(0, 30) + '...' : fileName}</div>
                  <div>In: {fmtTC(cl.clip.source_start)} &rarr; Out: {fmtTC(cl.clip.source_end)}</div>
                  <div>Duration: <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{dur.toFixed(1)}s</span></div>
                </div>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
