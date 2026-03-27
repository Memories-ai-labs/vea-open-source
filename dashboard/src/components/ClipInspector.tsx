import React, { useCallback, useMemo } from 'react';
import type { EditDecisionClip, TransformSettings, CropStatus } from '../hooks/useAgentChat';

interface ClipInspectorProps {
  clip: EditDecisionClip;
  cropStatus?: CropStatus;
  timelineWidth: number;
  timelineHeight: number;
  onTransformChange: (clipId: string, transform: TransformSettings) => void;
  onResetTransform: (clipId: string) => void;
  onRequestCrop: (clipId: string) => void;
  onClose?: () => void;
}

const MODE_COLORS: Record<string, string> = {
  fit: 'var(--accent-blue)',
  custom: 'var(--accent-yellow)',
  saliency: 'var(--accent-green)',
};

export default function ClipInspector({
  clip,
  cropStatus,
  timelineWidth,
  timelineHeight,
  onTransformChange,
  onResetTransform,
  onRequestCrop,
  onClose,
}: ClipInspectorProps) {
  const mode = clip.transform_mode ?? 'fit';
  const defaultScale = timelineWidth / (clip.source_width || 1920);

  const currentTransform = useMemo<TransformSettings>(() => {
    if (clip.transform) return clip.transform;
    return {
      scale_x: defaultScale,
      scale_y: defaultScale,
      position_x: 0,
      position_y: 0,
      rotation: 0,
    };
  }, [clip.transform, defaultScale]);

  const handleScaleChange = useCallback(
    (value: number) => {
      onTransformChange(clip.id, {
        ...currentTransform,
        scale_x: value,
        scale_y: value,
      });
    },
    [clip.id, currentTransform, onTransformChange],
  );

  const handlePositionChange = useCallback(
    (axis: 'x' | 'y', value: number) => {
      const key = axis === 'x' ? 'position_x' : 'position_y';
      onTransformChange(clip.id, {
        ...currentTransform,
        [key]: value,
      });
    },
    [clip.id, currentTransform, onTransformChange],
  );

  const isCropping = cropStatus?.status === 'running';
  const clipLabel = clip.description
    ? clip.description.slice(0, 28) + (clip.description.length > 28 ? '...' : '')
    : clip.id;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.clipLabel} title={clip.description || clip.id}>
          {clipLabel}
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              ...styles.modeBadge,
              background: MODE_COLORS[mode] ?? 'var(--accent-blue)',
            }}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </span>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: 16,
                padding: '0 2px',
                lineHeight: 1,
              }}
              title="Close inspector (Esc)"
            >
              ✕
            </button>
          )}
        </span>
      </div>

      {/* Scale */}
      <div style={styles.row}>
        <label style={styles.label}>Scale</label>
        <input
          type="range"
          min={0.5}
          max={4.0}
          step={0.01}
          value={currentTransform.scale_x}
          onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
          style={styles.slider}
        />
        <span style={styles.numericValue}>{currentTransform.scale_x.toFixed(2)}</span>
      </div>

      {/* Position X */}
      <div style={styles.row}>
        <label style={styles.label}>Pos X</label>
        <input
          type="number"
          min={-100}
          max={100}
          step={0.1}
          value={currentTransform.position_x}
          onChange={(e) => handlePositionChange('x', parseFloat(e.target.value) || 0)}
          style={styles.numericInput}
        />
      </div>

      {/* Position Y */}
      <div style={styles.row}>
        <label style={styles.label}>Pos Y</label>
        <input
          type="number"
          min={-100}
          max={100}
          step={0.1}
          value={currentTransform.position_y}
          onChange={(e) => handlePositionChange('y', parseFloat(e.target.value) || 0)}
          style={styles.numericInput}
        />
      </div>

      {/* Action buttons */}
      <div style={styles.buttonRow}>
        <button
          style={styles.resetButton}
          onClick={() => onResetTransform(clip.id)}
          title="Reset transform to default"
        >
          Reset
        </button>
        <button
          style={{
            ...styles.cropButton,
            opacity: isCropping ? 0.7 : 1,
            cursor: isCropping ? 'not-allowed' : 'pointer',
          }}
          onClick={() => !isCropping && onRequestCrop(clip.id)}
          disabled={isCropping}
          title="Run saliency-based dynamic crop"
        >
          {isCropping && <span style={styles.spinner} />}
          {isCropping ? 'Cropping...' : 'Dynamic Crop'}
        </button>
      </div>

      {/* Crop status feedback */}
      {cropStatus && cropStatus.status !== 'idle' && (
        <div
          style={{
            ...styles.statusLine,
            color:
              cropStatus.status === 'error'
                ? 'var(--accent-red)'
                : cropStatus.status === 'complete'
                  ? 'var(--accent-green)'
                  : 'var(--text-muted)',
          }}
        >
          {cropStatus.step || cropStatus.status}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    padding: '8px 10px',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border)',
    fontSize: 11,
    color: 'var(--text-primary)',
    minWidth: 200,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 6,
    marginBottom: 2,
  },
  clipLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-primary)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: 1,
  },
  modeBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: '1px 6px',
    borderRadius: 3,
    color: '#000',
    whiteSpace: 'nowrap',
    flexShrink: 0,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  label: {
    fontSize: 11,
    color: 'var(--text-secondary)',
    width: 36,
    flexShrink: 0,
  },
  slider: {
    flex: 1,
    height: 4,
    accentColor: 'var(--accent-blue)',
    cursor: 'pointer',
  },
  numericValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: 'var(--text-primary)',
    width: 36,
    textAlign: 'right' as const,
    flexShrink: 0,
  },
  numericInput: {
    flex: 1,
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    padding: '2px 4px',
    border: '1px solid var(--border)',
    borderRadius: 3,
    background: 'transparent',
    color: 'var(--text-primary)',
    outline: 'none',
  },
  buttonRow: {
    display: 'flex',
    gap: 6,
    marginTop: 4,
  },
  resetButton: {
    flex: 1,
    fontSize: 11,
    fontWeight: 500,
    padding: '3px 0',
    borderRadius: 3,
    border: '1px solid var(--border)',
    background: 'transparent',
    color: 'var(--text-secondary)',
    cursor: 'pointer',
  },
  cropButton: {
    flex: 1,
    fontSize: 11,
    fontWeight: 600,
    padding: '3px 0',
    borderRadius: 3,
    border: 'none',
    background: 'var(--accent-green)',
    color: '#000',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  spinner: {
    display: 'inline-block',
    width: 10,
    height: 10,
    border: '2px solid rgba(0,0,0,0.2)',
    borderTopColor: '#000',
    borderRadius: '50%',
    animation: 'spin 0.6s linear infinite',
  },
  statusLine: {
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
};
