import React, { useState, useMemo, useCallback, useEffect } from 'react';

// ─── Types ───────────────────────────────────────────────────────────────────

interface TimelineSettingsBarProps {
  width: number;
  height: number;
  onChange: (width: number, height: number) => void;
}

interface Preset {
  label: string;
  w: number;
  h: number;
}

// ─── Presets ─────────────────────────────────────────────────────────────────

const PRESETS: Preset[] = [
  { label: '1920 \u00d7 1080', w: 1920, h: 1080 },
  { label: '1080 \u00d7 1920', w: 1080, h: 1920 },
  { label: '1080 \u00d7 1080', w: 1080, h: 1080 },
  { label: '1080 \u00d7 1350', w: 1080, h: 1350 },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}

function aspectLabel(w: number, h: number): string {
  if (w <= 0 || h <= 0) return '--';
  const d = gcd(w, h);
  return `${w / d}:${h / d}`;
}

function matchPreset(w: number, h: number): string {
  const match = PRESETS.find((p) => p.w === w && p.h === h);
  return match ? match.label : 'Custom';
}

// ─── Component ───────────────────────────────────────────────────────────────

const TimelineSettingsBar: React.FC<TimelineSettingsBarProps> = ({
  width,
  height,
  onChange,
}) => {
  const [localW, setLocalW] = useState(String(width));
  const [localH, setLocalH] = useState(String(height));

  // Sync local inputs when props change externally
  useEffect(() => {
    setLocalW(String(width));
  }, [width]);
  useEffect(() => {
    setLocalH(String(height));
  }, [height]);

  const selectedPreset = useMemo(() => matchPreset(width, height), [width, height]);
  const ratio = useMemo(() => aspectLabel(width, height), [width, height]);

  const handlePresetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      const preset = PRESETS.find((p) => p.label === val);
      if (preset) {
        onChange(preset.w, preset.h);
      }
    },
    [onChange],
  );

  const handleSwap = useCallback(() => {
    onChange(height, width);
  }, [width, height, onChange]);

  const commitCustom = useCallback(() => {
    const w = parseInt(localW, 10);
    const h = parseInt(localH, 10);
    if (w > 0 && h > 0 && (w !== width || h !== height)) {
      onChange(w, h);
    } else {
      // Reset to current if invalid
      setLocalW(String(width));
      setLocalH(String(height));
    }
  }, [localW, localH, width, height, onChange]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        commitCustom();
      }
    },
    [commitCustom],
  );

  // ─── Styles ──────────────────────────────────────────────────────────────

  const bar: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    height: 32,
    padding: '0 10px',
    background: 'var(--bg-surface, #1a1a1a)',
    borderBottom: '1px solid var(--border)',
    userSelect: 'none',
    fontSize: 12,
  };

  const eyebrow: React.CSSProperties = {
    fontSize: 10,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--text-muted)',
    marginRight: 2,
    flexShrink: 0,
  };

  const select: React.CSSProperties = {
    height: 22,
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-primary)',
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '0 6px',
    outline: 'none',
    cursor: 'pointer',
    appearance: 'auto' as React.CSSProperties['appearance'],
  };

  const swapBtn: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 24,
    height: 22,
    fontSize: 13,
    color: 'var(--text-secondary)',
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    padding: 0,
    lineHeight: 1,
    flexShrink: 0,
  };

  const numInput: React.CSSProperties = {
    width: 56,
    height: 22,
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-primary)',
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '0 5px',
    outline: 'none',
    textAlign: 'center',
  };

  const sep: React.CSSProperties = {
    color: 'var(--text-muted)',
    fontSize: 11,
    flexShrink: 0,
  };

  const badge: React.CSSProperties = {
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    fontWeight: 600,
    color: 'var(--accent-green)',
    background: 'rgba(74, 222, 128, 0.08)',
    border: '1px solid rgba(74, 222, 128, 0.18)',
    borderRadius: 'var(--radius-md)',
    padding: '2px 6px',
    lineHeight: 1,
    flexShrink: 0,
  };

  // ─── Render ──────────────────────────────────────────────────────────────

  return (
    <div style={bar}>
      <span style={eyebrow}>Resolution</span>

      <select
        style={select}
        value={selectedPreset}
        onChange={handlePresetChange}
      >
        {PRESETS.map((p) => (
          <option key={p.label} value={p.label}>
            {p.label}
          </option>
        ))}
        <option value="Custom">Custom</option>
      </select>

      <button
        style={swapBtn}
        title="Swap width and height"
        onClick={handleSwap}
      >
        &#x21C4;
      </button>

      <input
        style={numInput}
        type="number"
        min={1}
        value={localW}
        onChange={(e) => setLocalW(e.target.value)}
        onBlur={commitCustom}
        onKeyDown={handleKeyDown}
        aria-label="Width"
      />
      <span style={sep}>&times;</span>
      <input
        style={numInput}
        type="number"
        min={1}
        value={localH}
        onChange={(e) => setLocalH(e.target.value)}
        onBlur={commitCustom}
        onKeyDown={handleKeyDown}
        aria-label="Height"
      />

      <span style={badge}>{ratio}</span>
    </div>
  );
};

export default TimelineSettingsBar;
