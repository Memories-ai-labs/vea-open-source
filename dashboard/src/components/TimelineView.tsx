import React, { useState } from 'react';
import type { Storyboard, Shot } from '../types';
import { Panel } from './Panel';
import {
  renderVideo,
  generateNarration,
  selectMusic,
  cropVideo,
  generateFcpxml,
} from '../api';

// ─── helpers ─────────────────────────────────────────────────────────────────

function scoreColor(score: number): string {
  if (score >= 0.75) return 'var(--accent-green)';
  if (score >= 0.45) return 'var(--accent-yellow)';
  return 'var(--accent-orange)';
}

function shotColor(index: number): string {
  const palette = [
    'var(--accent-blue)',
    'var(--accent-purple)',
    'var(--accent-green)',
    'var(--accent-orange)',
    'var(--accent-yellow)',
    'var(--accent-red)',
  ];
  return palette[index % palette.length];
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ─── Tooltip ─────────────────────────────────────────────────────────────────

interface TooltipData {
  shot: Shot;
  x: number;
  y: number;
}

function ShotTooltip({ data }: { data: TooltipData }) {
  const { shot, x, y } = data;
  const clip = shot.retrieved_clip;
  return (
    <div
      style={{
        position: 'fixed',
        left: x + 12,
        top: y - 8,
        zIndex: 100,
        background: 'var(--bg-card)',
        border: '1px solid var(--border-active)',
        borderRadius: '4px',
        padding: '8px 10px',
        minWidth: '220px',
        maxWidth: '300px',
        fontSize: '11px',
        pointerEvents: 'none',
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      }}
    >
      <div style={{ color: 'var(--text-primary)', fontWeight: 700, marginBottom: '4px' }}>
        {shot.id} · {shot.priority}
      </div>
      <div style={{ color: 'var(--text-secondary)', marginBottom: '4px', lineHeight: 1.4 }}>
        {shot.purpose}
      </div>
      <div style={{ color: 'var(--text-muted)', marginBottom: '2px' }}>
        Duration: {shot.duration_seconds.toFixed(1)}s
      </div>
      {clip && (
        <>
          <div style={{ color: 'var(--text-muted)', marginBottom: '2px' }}>
            Source: {clip.video_name}
          </div>
          <div style={{ color: 'var(--text-muted)', marginBottom: '2px' }}>
            Range: {fmtTime(clip.start_seconds)} → {fmtTime(clip.end_seconds)}
          </div>
          <div style={{ color: scoreColor(clip.score), fontWeight: 700 }}>
            Score: {(clip.score * 100).toFixed(0)}%
          </div>
          {clip.description && (
            <div
              style={{
                color: 'var(--text-muted)',
                marginTop: '4px',
                lineHeight: 1.4,
                fontStyle: 'italic',
              }}
            >
              {clip.description.slice(0, 120)}
            </div>
          )}
        </>
      )}
      {shot.narration && (
        <div
          style={{
            color: 'var(--accent-purple)',
            marginTop: '4px',
            lineHeight: 1.4,
            fontStyle: 'italic',
          }}
        >
          "{shot.narration.slice(0, 100)}"
        </div>
      )}
    </div>
  );
}

// ─── Timeline ────────────────────────────────────────────────────────────────

function Timeline({ storyboard }: { storyboard: Storyboard }) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const totalDur = storyboard.shots.reduce((s, sh) => s + sh.duration_seconds, 0) || 1;

  return (
    <div>
      {/* Track */}
      <div
        style={{
          overflowX: 'auto',
          paddingBottom: '4px',
        }}
      >
        <div
          style={{
            display: 'flex',
            gap: '2px',
            minWidth: '100%',
            height: '52px',
            alignItems: 'stretch',
          }}
        >
          {storyboard.shots.map((shot, i) => {
            const widthPct = (shot.duration_seconds / totalDur) * 100;
            const color = shotColor(i);
            const clip = shot.retrieved_clip;
            return (
              <div
                key={shot.id}
                onMouseEnter={(ev) =>
                  setTooltip({ shot, x: ev.clientX, y: ev.clientY })
                }
                onMouseMove={(ev) =>
                  setTooltip({ shot, x: ev.clientX, y: ev.clientY })
                }
                onMouseLeave={() => setTooltip(null)}
                style={{
                  flex: `0 0 ${widthPct}%`,
                  minWidth: '28px',
                  background: `${color}22`,
                  border: `1px solid ${color}88`,
                  borderRadius: '3px',
                  padding: '3px 4px',
                  overflow: 'hidden',
                  cursor: 'default',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                }}
              >
                <div
                  style={{
                    fontSize: '9px',
                    color,
                    fontWeight: 700,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {shot.id}
                </div>
                {clip && (
                  <div
                    style={{
                      fontSize: '9px',
                      color: 'var(--text-muted)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {clip.video_name.split('/').pop()?.slice(0, 12)}
                  </div>
                )}
                {clip && (
                  <div
                    style={{
                      fontSize: '9px',
                      color: scoreColor(clip.score),
                      fontWeight: 700,
                    }}
                  >
                    {(clip.score * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Time ruler */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            color: 'var(--text-muted)',
            fontSize: '9px',
            marginTop: '3px',
            paddingRight: '2px',
          }}
        >
          <span>0:00</span>
          <span>{fmtTime(totalDur / 2)}</span>
          <span>{fmtTime(totalDur)}</span>
        </div>
      </div>

      {tooltip && <ShotTooltip data={tooltip} />}

      {/* Shot list below timeline */}
      <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
        {storyboard.shots.map((shot, i) => {
          const clip = shot.retrieved_clip;
          const color = shotColor(i);
          return (
            <div
              key={shot.id}
              style={{
                display: 'flex',
                gap: '8px',
                alignItems: 'center',
                padding: '4px 7px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderLeft: `3px solid ${color}`,
                borderRadius: '0 3px 3px 0',
                fontSize: '11px',
              }}
            >
              <span style={{ color, fontWeight: 700, minWidth: '40px' }}>{shot.id}</span>
              <span
                style={{
                  color: 'var(--text-secondary)',
                  flex: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {shot.purpose}
              </span>
              {clip && (
                <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                  {clip.video_name.split('/').pop()?.slice(0, 18)}
                </span>
              )}
              <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                {shot.duration_seconds.toFixed(1)}s
              </span>
              {clip && (
                <span
                  style={{
                    color: scoreColor(clip.score),
                    fontWeight: 700,
                    flexShrink: 0,
                    minWidth: '28px',
                    textAlign: 'right',
                  }}
                >
                  {(clip.score * 100).toFixed(0)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Async Action Button ──────────────────────────────────────────────────────


function ActionBtn({
  label,
  onClick,
  result,
  loading,
  accent,
}: {
  label: string;
  onClick: () => void;
  result?: string;
  loading?: boolean;
  accent?: string;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        padding: '16px',
        minWidth: '200px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div className="eyebrow">{label}</div>
      <button
        onClick={onClick}
        disabled={loading}
        style={{
          padding: '12px 14px',
          background: loading ? 'rgba(255,255,255,0.03)' : (accent ? `${accent}16` : 'rgba(255,255,255,0.03)'),
          border: `1px solid ${accent ? `${accent}66` : 'var(--border)'}`,
          borderRadius: '999px',
          color: loading ? 'var(--text-muted)' : (accent ?? 'var(--text-primary)'),
          fontSize: '12px',
          fontWeight: 700,
          cursor: loading ? 'wait' : 'pointer',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          whiteSpace: 'nowrap',
        }}
      >
        {loading ? '…' : label}
      </button>
      {result && (
        <div
          style={{
            fontSize: '11px',
            color: 'var(--text-muted)',
            maxWidth: '240px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={result}
        >
          {result}
        </div>
      )}
    </div>
  );
}

// ─── TimelineView ─────────────────────────────────────────────────────────────

interface TimelineViewProps {
  projectName: string;
  storyboard: Storyboard | null;
}

export function TimelineView({ projectName, storyboard }: TimelineViewProps) {
  const [fcpxmlPath, setFcpxmlPath] = useState<string>('');
  const [previewPath, setPreviewPath] = useState<string>('');
  const [finalPath, setFinalPath] = useState<string>('');
  const [audioPath, setAudioPath] = useState<string>('');
  const [musicPath, setMusicPath] = useState<string>('');
  const [cropPath, setCropPath] = useState<string>('');

  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string>('');

  // Mood / aspect inputs
  const [mood, setMood] = useState('');
  const [showMoodInput, setShowMoodInput] = useState(false);
  const [aspectRatio, setAspectRatio] = useState('0.5625'); // 9:16
  const [showAspectInput, setShowAspectInput] = useState(false);

  function setLoading_(key: string, val: boolean) {
    setLoading((prev) => ({ ...prev, [key]: val }));
  }

  async function act<T>(key: string, fn: () => Promise<T>): Promise<T | null> {
    setError('');
    setLoading_(key, true);
    try {
      return await fn();
    } catch (e: any) {
      setError(String(e?.message ?? e));
      return null;
    } finally {
      setLoading_(key, false);
    }
  }

  async function handleFcpxml() {
    const r = await act('fcpxml', () => generateFcpxml(projectName));
    if (r) setFcpxmlPath(r.fcpxml_path);
  }

  async function handlePreview() {
    const r = await act('preview', () => renderVideo(projectName, 'preview'));
    if (r) setPreviewPath(r.output_path);
  }

  async function handleFinal() {
    const r = await act('final', () => renderVideo(projectName, 'final'));
    if (r) setFinalPath(r.output_path);
  }

  async function handleNarration() {
    const r = await act('narration', () => generateNarration(projectName));
    if (r) setAudioPath(r.audio_path);
  }

  async function handleMusic() {
    setShowMoodInput(false);
    const r = await act('music', () => selectMusic(projectName, mood || undefined));
    if (r) setMusicPath(r.music_path);
  }

  async function handleCrop() {
    setShowAspectInput(false);
    const ratio = parseFloat(aspectRatio);
    if (isNaN(ratio)) { setError('Invalid aspect ratio'); return; }
    const r = await act('crop', () => cropVideo(projectName, ratio));
    if (r) setCropPath(r.fcpxml_path);
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        flex: 1,
        minHeight: 0,
      }}
    >
      {/* Timeline Panel */}
      <Panel title="Timeline" badge={storyboard ? `${storyboard.shots.length} shots` : undefined}>
        {storyboard ? (
          <Timeline storyboard={storyboard} />
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            No storyboard available.
          </span>
        )}
      </Panel>

      {/* Controls */}
      <Panel title="Controls">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', alignItems: 'start' }}>
          {/* FCPXML */}
          <ActionBtn
            label="Regen FCPXML"
            onClick={handleFcpxml}
            loading={loading['fcpxml']}
            result={fcpxmlPath}
            accent="var(--accent-purple)"
          />

          {/* Preview */}
          <ActionBtn
            label="Preview Render"
            onClick={handlePreview}
            loading={loading['preview']}
            result={previewPath}
            accent="var(--accent-blue)"
          />

          {/* Final */}
          <ActionBtn
            label="Final Render"
            onClick={handleFinal}
            loading={loading['final']}
            result={finalPath}
            accent="var(--accent-green)"
          />

          {/* Narration */}
          <ActionBtn
            label="Generate Narration"
            onClick={handleNarration}
            loading={loading['narration']}
            result={audioPath}
            accent="var(--accent-yellow)"
          />

          {/* Music */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              padding: '16px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <div className="eyebrow">Select Music</div>
            <button
              onClick={() => setShowMoodInput((v) => !v)}
              disabled={loading['music']}
              style={{
                padding: '12px 14px',
                background: 'var(--accent-orange)16',
                border: '1px solid var(--accent-orange)',
                borderRadius: '999px',
                color: 'var(--accent-orange)',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              {loading['music'] ? '…' : 'Select Music'}
            </button>
            {showMoodInput && (
              <div style={{ display: 'flex', gap: '4px' }}>
                <input
                  value={mood}
                  onChange={(e) => setMood(e.target.value)}
                  placeholder="mood (optional)"
                  onKeyDown={(e) => e.key === 'Enter' && handleMusic()}
                  style={inputStyle}
                />
                <button onClick={handleMusic} style={smBtnStyle}>Go</button>
              </div>
            )}
            {musicPath && (
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', maxWidth: '220px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={musicPath}>
                {musicPath}
              </div>
            )}
          </div>

          {/* Dynamic Crop */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              padding: '16px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <div className="eyebrow">Dynamic Crop</div>
            <button
              onClick={() => setShowAspectInput((v) => !v)}
              disabled={loading['crop']}
              style={{
                padding: '12px 14px',
                background: 'var(--accent-red)16',
                border: '1px solid var(--accent-red)',
                borderRadius: '999px',
                color: 'var(--accent-red)',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              {loading['crop'] ? '…' : 'Dynamic Crop'}
            </button>
            {showAspectInput && (
              <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                <select
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                  style={{ ...inputStyle, width: '100px' }}
                >
                  <option value="0.5625">9:16 (0.5625)</option>
                  <option value="1">1:1</option>
                  <option value="1.7778">16:9</option>
                  <option value="0.75">4:3</option>
                </select>
                <button onClick={handleCrop} style={smBtnStyle}>Go</button>
              </div>
            )}
            {cropPath && (
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', maxWidth: '220px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={cropPath}>
                {cropPath}
              </div>
            )}
          </div>
        </div>

        {error && (
          <div
            style={{
              marginTop: '8px',
              padding: '5px 8px',
              background: 'var(--accent-red)11',
              border: '1px solid var(--accent-red)',
              borderRadius: '3px',
              color: 'var(--accent-red)',
              fontSize: '11px',
            }}
          >
            {error}
          </div>
        )}
      </Panel>

      {/* Status Panel */}
      {(fcpxmlPath || previewPath || finalPath) && (
        <Panel title="Output Paths">
          {fcpxmlPath && <OutputRow label="FCPXML" value={fcpxmlPath} />}
          {previewPath && <OutputRow label="Preview" value={previewPath} />}
          {finalPath && <OutputRow label="Final" value={finalPath} />}
        </Panel>
      )}
    </div>
  );
}

function OutputRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '4px',
        fontSize: '12px',
        alignItems: 'center',
      }}
    >
      <span style={{ color: 'var(--text-muted)', minWidth: '52px' }}>{label}:</span>
      <span
        style={{
          color: 'var(--accent-green)',
          flex: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={value}
      >
        {value}
      </span>
      <button
        onClick={() => navigator.clipboard.writeText(value)}
        style={{
          background: 'none',
          border: '1px solid var(--border)',
          borderRadius: '3px',
          color: 'var(--text-muted)',
          fontSize: '10px',
          padding: '1px 5px',
          cursor: 'pointer',
          fontFamily: 'inherit',
          flexShrink: 0,
        }}
      >
        copy
      </button>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid var(--border)',
  borderRadius: '999px',
  color: 'var(--text-primary)',
  fontSize: '12px',
  padding: '10px 12px',
  width: '140px',
};

const smBtnStyle: React.CSSProperties = {
  padding: '10px 12px',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid var(--border)',
  borderRadius: '999px',
  color: 'var(--text-secondary)',
  fontSize: '11px',
  fontWeight: 700,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  cursor: 'pointer',
};
