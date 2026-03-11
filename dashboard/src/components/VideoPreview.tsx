import React, { useRef, useEffect } from 'react';
import type { RenderState } from '../hooks/useAgentChat';

interface VideoPreviewProps {
  projectName: string;
  renderState: RenderState;
  hasEditDecision: boolean;
  onRequestRender?: () => void;
}

export function VideoPreview({ projectName, renderState, hasEditDecision, onRequestRender }: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Auto-play when render completes
  useEffect(() => {
    if (renderState.status === 'complete' && videoRef.current) {
      videoRef.current.load();
    }
  }, [renderState.status, renderState.filename]);

  const videoUrl = renderState.filename
    ? `/video-edit/v2/projects/${encodeURIComponent(projectName)}/renders/${encodeURIComponent(renderState.filename)}`
    : null;

  // ── Rendering state ──
  if (renderState.status === 'rendering') {
    return (
      <div style={containerStyle}>
        <div style={centerColumnStyle}>
          {/* Spinner */}
          <div style={{
            width: '32px',
            height: '32px',
            border: '2px solid rgba(255,255,255,0.08)',
            borderTop: '2px solid var(--accent-blue)',
            borderRadius: '50%',
            animation: 'spinner 1s linear infinite',
          }} />
          <span style={labelStyle}>Rendering preview...</span>
          {/* Progress bar */}
          <div style={{
            width: '80%',
            maxWidth: '180px',
            height: '3px',
            background: 'rgba(255,255,255,0.06)',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${renderState.progress}%`,
              height: '100%',
              background: 'var(--accent-blue)',
              borderRadius: '2px',
              transition: 'width 0.5s ease-out',
            }} />
          </div>
          <span style={{ ...labelStyle, fontSize: '9px' }}>
            {Math.round(renderState.progress)}%
          </span>
        </div>
      </div>
    );
  }

  // ── Complete — show video ──
  if (renderState.status === 'complete' && videoUrl) {
    return (
      <div style={{
        ...containerStyle,
        padding: 0,
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
          background: 'rgba(255,255,255,0.015)',
        }}>
          <span style={{
            fontSize: '9px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--accent-green)',
            textTransform: 'uppercase',
          }}>
            Preview
          </span>
          <span style={{
            fontSize: '8px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-muted)',
            flex: 1,
          }}>
            {renderState.filename}
          </span>
          {onRequestRender && (
            <button onClick={onRequestRender} style={renderBtnStyle} title="Re-render">
              &#8635;
            </button>
          )}
        </div>
        {/* Video player */}
        <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
          <video
            ref={videoRef}
            controls
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
            }}
          >
            <source src={videoUrl} type="video/mp4" />
          </video>
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (renderState.status === 'error') {
    return (
      <div style={containerStyle}>
        <div style={centerColumnStyle}>
          <span style={{ color: 'var(--accent-red)', fontSize: '18px' }}>!</span>
          <span style={{ ...labelStyle, color: 'var(--accent-red)' }}>Render failed</span>
          <span style={{
            ...labelStyle,
            fontSize: '9px',
            color: 'var(--text-muted)',
            maxWidth: '90%',
            textAlign: 'center',
            lineHeight: 1.5,
          }}>
            {renderState.error}
          </span>
          {onRequestRender && (
            <button onClick={onRequestRender} style={{ ...renderBtnStyle, marginTop: '4px', padding: '4px 12px' }}>
              Retry Render
            </button>
          )}
        </div>
      </div>
    );
  }

  // ── Idle / empty state ──
  return (
    <div style={containerStyle}>
      <div style={centerColumnStyle}>
        {/* Monitor icon */}
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.2 }}>
          <rect x="2" y="3" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" />
          <line x1="8" y1="21" x2="16" y2="21" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="12" y1="17" x2="12" y2="21" stroke="currentColor" strokeWidth="1.5" />
          {/* Play triangle */}
          <path d="M10 7.5L15 10.5L10 13.5V7.5Z" fill="currentColor" opacity="0.4" />
        </svg>
        <span style={labelStyle}>
          {hasEditDecision
            ? 'Waiting for Resolve render...'
            : 'Preview will appear after edit generation'}
        </span>
        {hasEditDecision && onRequestRender && (
          <button onClick={onRequestRender} style={{ ...renderBtnStyle, padding: '4px 12px' }}>
            Render Preview
          </button>
        )}
        {hasEditDecision && (
          <span style={{ ...labelStyle, fontSize: '8px', color: 'var(--text-muted)' }}>
            Ensure DaVinci Resolve Studio is running
          </span>
        )}
      </div>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  background: 'rgba(8,6,5,0.4)',
  borderRadius: '4px',
  overflow: 'hidden',
};

const centerColumnStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
};

const labelStyle: React.CSSProperties = {
  fontSize: '10px',
  fontFamily: 'var(--font-mono)',
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: 'var(--text-muted)',
};

const renderBtnStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '3px',
  color: 'var(--text-secondary)',
  fontSize: '10px',
  fontFamily: 'var(--font-mono)',
  cursor: 'pointer',
  padding: '2px 6px',
  letterSpacing: '0.04em',
};
