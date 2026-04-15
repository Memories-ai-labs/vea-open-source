import React, { useCallback } from 'react';

export type AudioItemKind = 'narration' | 'music';

interface AudioInspectorProps {
  kind: AudioItemKind;
  file: string;
  duration: number;
  start: number;            // in-point in source file
  timelineOffset: number;   // absolute placement on timeline
  gainDb: number;
  measuredLufs?: number | null;
  // Effective render target. Set when the renderer uses a non-default target
  // (e.g. music drops to -28 LUFS when narration is present for voice-over
  // separation). Falls back to kind defaults: music -18, narration -16.
  targetLufsOverride?: number | null;
  onGainChange: (gainDb: number) => void;
  onClose?: () => void;
}

export default function AudioInspector({
  kind,
  file,
  duration,
  start,
  timelineOffset,
  gainDb,
  measuredLufs,
  targetLufsOverride,
  onGainChange,
  onClose,
}: AudioInspectorProps) {
  const baseName = (path: string) => path.split('/').pop() || path;

  const handleGain = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = parseFloat(e.target.value);
      if (!Number.isNaN(v)) onGainChange(v);
    },
    [onGainChange]
  );

  const target = targetLufsOverride ?? (kind === 'music' ? -18 : -16);
  // Music/narration gain_db is an OFFSET from target — the renderer auto-normalizes
  // using (target + offset) - measured. To hit target, offset = 0. Suggesting
  // `target - measured` (which is the literal-dB formula for source clips) would
  // double-subtract the measured loudness and render the track inaudible.
  const suggestedGain = measuredLufs == null ? null : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div
          style={{
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-dim)',
          }}
        >
          {kind === 'music' ? 'Music' : 'Narration'}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-dim)',
              cursor: 'pointer',
              fontSize: '14px',
              padding: '2px 6px',
            }}
            aria-label="Close inspector"
          >
            ✕
          </button>
        )}
      </div>

      <div
        style={{
          fontSize: '13px',
          color: 'var(--text-bright)',
          fontWeight: 500,
          wordBreak: 'break-all',
        }}
      >
        {baseName(file)}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          gap: '6px 12px',
          fontSize: '11px',
          color: 'var(--text-dim)',
        }}
      >
        <span>Duration</span>
        <span style={{ color: 'var(--text-bright)' }}>{duration.toFixed(2)}s</span>
        <span>Timeline at</span>
        <span style={{ color: 'var(--text-bright)' }}>{timelineOffset.toFixed(2)}s</span>
        <span>In-point</span>
        <span style={{ color: 'var(--text-bright)' }}>{start.toFixed(2)}s</span>
        {measuredLufs != null && (
          <>
            <span>Measured</span>
            <span style={{ color: 'var(--text-bright)' }}>{measuredLufs} LUFS</span>
          </>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '11px',
            color: 'var(--text-dim)',
          }}
        >
          <span>Gain</span>
          <span style={{ color: 'var(--text-bright)' }}>
            {gainDb >= 0 ? '+' : ''}
            {gainDb}dB
          </span>
        </div>
        <input
          type="range"
          min={-40}
          max={20}
          step={0.5}
          value={gainDb}
          onChange={handleGain}
          style={{ width: '100%' }}
        />
        {suggestedGain != null && Math.abs(suggestedGain - gainDb) > 0.5 && (
          <button
            onClick={() => onGainChange(suggestedGain)}
            style={{
              background: 'transparent',
              border: '1px solid var(--accent-blue)',
              color: 'var(--accent-blue)',
              cursor: 'pointer',
              fontSize: '11px',
              padding: '4px 8px',
              borderRadius: '3px',
              marginTop: '2px',
            }}
          >
            Apply suggested {suggestedGain >= 0 ? '+' : ''}
            {suggestedGain}dB (target {target} LUFS)
          </button>
        )}
      </div>
    </div>
  );
}
