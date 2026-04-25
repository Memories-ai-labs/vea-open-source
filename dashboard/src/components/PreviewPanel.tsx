import React, { useState, useRef, useEffect, useCallback, useImperativeHandle, forwardRef } from 'react';
import type { RenderState, EditDecision, FfmpegRenderState, FfmpegQuality } from '../hooks/useAgentChat';

export interface PreviewPanelHandle {
  seekTo: (time: number) => void;
  togglePlayback: () => void;
}

interface PreviewPanelProps {
  projectName: string;
  ffmpegRenderState: FfmpegRenderState;
  resolveRenderState: RenderState;
  ffmpegQualityPref: FfmpegQuality;
  editDecision: EditDecision | null;
  onRequestFfmpegRender: (quality: FfmpegQuality) => void;
  onSetFfmpegQualityPref: (quality: FfmpegQuality) => void;
  onRequestResolveRender: () => void;
  onTimeUpdate?: (time: number) => void;
  playheadTime: number;
  resolveRunning: boolean | null;
}

type TabKey = 'ffmpeg' | 'resolve';

export const PreviewPanel = forwardRef<PreviewPanelHandle, PreviewPanelProps>(function PreviewPanel(
  {
    projectName,
    ffmpegRenderState,
    resolveRenderState,
    ffmpegQualityPref,
    editDecision,
    onRequestFfmpegRender,
    onSetFfmpegQualityPref,
    onRequestResolveRender,
    onTimeUpdate,
    playheadTime,
    resolveRunning,
  },
  ref,
) {
  const [activeTab, setActiveTab] = useState<TabKey>('ffmpeg');
  const ffmpegVideoRef = useRef<HTMLVideoElement>(null);
  const resolveVideoRef = useRef<HTMLVideoElement>(null);
  const [scrubFrameUrl, setScrubFrameUrl] = useState<string | null>(null);
  const prevFrameUrlRef = useRef<string | null>(null);
  const throttleRef = useRef<number>(0);
  const abortRef = useRef<AbortController | null>(null);

  const activeRender: RenderState =
    activeTab === 'ffmpeg' ? ffmpegRenderState : resolveRenderState;

  useImperativeHandle(ref, () => ({
    seekTo(time: number) {
      const v = activeTab === 'ffmpeg' ? ffmpegVideoRef.current : resolveVideoRef.current;
      if (v) v.currentTime = time;
    },
    togglePlayback() {
      const v = activeTab === 'ffmpeg' ? ffmpegVideoRef.current : resolveVideoRef.current;
      if (v) {
        if (v.paused) v.play();
        else v.pause();
      }
    },
  }), [activeTab, ffmpegRenderState.status, resolveRenderState.status]);

  const handleTimeUpdate = useCallback((e: React.SyntheticEvent<HTMLVideoElement>) => {
    if (onTimeUpdate) {
      onTimeUpdate(e.currentTarget.currentTime);
    }
  }, [onTimeUpdate]);

  // Auto-load when render completes
  useEffect(() => {
    if (ffmpegRenderState.status === 'complete' && ffmpegVideoRef.current) {
      ffmpegVideoRef.current.load();
    }
  }, [ffmpegRenderState.status, ffmpegRenderState.filename]);

  useEffect(() => {
    if (resolveRenderState.status === 'complete' && resolveVideoRef.current) {
      resolveVideoRef.current.load();
    }
  }, [resolveRenderState.status, resolveRenderState.filename]);

  // Scrub frame fetching — throttled to 200ms
  useEffect(() => {
    if (activeRender.status === 'complete') return; // Don't fetch frames when video is available
    if (!editDecision || !editDecision.clips || editDecision.clips.length === 0) return;

    const now = Date.now();
    if (now - throttleRef.current < 200) return;
    throttleRef.current = now;

    // Abort previous request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const url = `/video-edit/v2/projects/${encodeURIComponent(projectName)}/preview/frame?t=${playheadTime}`;
    fetch(url, { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error('Frame fetch failed');
        return res.blob();
      })
      .then(blob => {
        const objectUrl = URL.createObjectURL(blob);
        // Revoke old URL
        if (prevFrameUrlRef.current) {
          URL.revokeObjectURL(prevFrameUrlRef.current);
        }
        prevFrameUrlRef.current = objectUrl;
        setScrubFrameUrl(objectUrl);
      })
      .catch(() => {
        // Ignore abort errors and failures — keep showing previous frame
      });
  }, [playheadTime, activeRender.status, editDecision, projectName]);

  // Clean up object URLs on unmount
  useEffect(() => {
    return () => {
      if (prevFrameUrlRef.current) {
        URL.revokeObjectURL(prevFrameUrlRef.current);
      }
    };
  }, []);

  const hasEditDecision = editDecision !== null && editDecision.clips && editDecision.clips.length > 0;

  function dotColor(rs: RenderState): string {
    if (rs.status === 'complete') return 'var(--accent-green)';
    if (rs.status === 'rendering') return 'var(--accent-blue)';
    return 'var(--text-muted)';
  }

  function renderVideoUrl(rs: RenderState): string | null {
    return rs.filename
      ? `/video-edit/v2/projects/${encodeURIComponent(projectName)}/renders/${encodeURIComponent(rs.filename)}`
      : null;
  }

  const ffmpegUrl = renderVideoUrl(ffmpegRenderState);
  const resolveUrl = renderVideoUrl(resolveRenderState);

  // ── Render content for active tab ──
  function renderContent() {
    // Complete — show video
    if (activeRender.status === 'complete') {
      const videoUrl = activeTab === 'ffmpeg' ? ffmpegUrl : resolveUrl;
      const videoRef = activeTab === 'ffmpeg' ? ffmpegVideoRef : resolveVideoRef;
      if (!videoUrl) return renderEmptyState();
      return (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
          <video
            ref={videoRef}
            controls
            onTimeUpdate={handleTimeUpdate}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          >
            <source src={videoUrl} type="video/mp4" />
          </video>
        </div>
      );
    }

    // Rendering — show progress
    if (activeRender.status === 'rendering') {
      const label =
        activeTab === 'ffmpeg'
          ? `Rendering ffmpeg (${ffmpegRenderState.quality === 'full' ? 'full res' : '480p'})…`
          : 'Rendering resolve…';
      return (
        <div style={centerColumnStyle}>
          <div style={{
            width: '32px',
            height: '32px',
            border: '2px solid rgba(255,255,255,0.08)',
            borderTop: '2px solid var(--accent-blue)',
            borderRadius: '50%',
            animation: 'spinner 1s linear infinite',
          }} />
          <span style={labelStyle}>{label}</span>
          <div style={{
            width: '80%',
            maxWidth: '180px',
            height: '3px',
            background: 'rgba(255,255,255,0.06)',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${activeRender.progress}%`,
              height: '100%',
              background: 'var(--accent-blue)',
              borderRadius: '2px',
              transition: 'width 0.5s ease-out',
            }} />
          </div>
          <span style={{ ...labelStyle, fontSize: '9px' }}>
            {Math.round(activeRender.progress)}%
          </span>
        </div>
      );
    }

    // Error
    if (activeRender.status === 'error') {
      return (
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
            {activeRender.error}
          </span>
        </div>
      );
    }

    // Idle — scrub frame or empty
    if (hasEditDecision && scrubFrameUrl) {
      return (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
          <img
            src={scrubFrameUrl}
            alt="Scrub frame"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </div>
      );
    }

    return renderEmptyState();
  }

  function renderEmptyState() {
    if (activeTab === 'resolve' && resolveRunning === false) {
      return (
        <div style={centerColumnStyle}>
          <span style={labelStyle}>DaVinci Resolve not detected</span>
          <span style={{ ...labelStyle, fontSize: '9px', maxWidth: '80%', textAlign: 'center', lineHeight: 1.5 }}>
            Install Resolve and restart the backend, or stay on the ffmpeg tab.
          </span>
        </div>
      );
    }
    return (
      <div style={centerColumnStyle}>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.2 }}>
          <rect x="2" y="3" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" />
          <line x1="8" y1="21" x2="16" y2="21" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="12" y1="17" x2="12" y2="21" stroke="currentColor" strokeWidth="1.5" />
          <path d="M10 7.5L15 10.5L10 13.5V7.5Z" fill="currentColor" opacity="0.4" />
        </svg>
        <span style={labelStyle}>
          {hasEditDecision
            ? 'Ready to render preview'
            : 'Preview will appear after edit generation'}
        </span>
      </div>
    );
  }

  const ffmpegBusy = ffmpegRenderState.status === 'rendering';
  const resolveBusy = resolveRenderState.status === 'rendering';

  return (
    <div style={containerStyle}>
      {/* Tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        flexShrink: 0,
        background: 'rgba(255,255,255,0.015)',
      }}>
        {([
          { key: 'ffmpeg' as TabKey, label: `ffmpeg (${ffmpegRenderState.quality === 'full' ? 'full' : '480p'})`, rs: ffmpegRenderState as RenderState },
          { key: 'resolve' as TabKey, label: 'resolve', rs: resolveRenderState },
        ]).map(({ key, label, rs }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            style={{
              flex: 1,
              padding: '5px 8px',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === key ? '2px solid var(--accent-blue)' : '2px solid transparent',
              color: activeTab === key ? 'var(--text-secondary)' : 'var(--text-muted)',
              fontSize: '9px',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '5px',
            }}
          >
            <span style={{
              display: 'inline-block',
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: dotColor(rs),
              flexShrink: 0,
              ...(rs.status === 'rendering' ? { animation: 'spinner 1s linear infinite' } : {}),
            }} />
            {label}
          </button>
        ))}
      </div>

      {/* Content area */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        {renderContent()}
      </div>

      {/* Action bar */}
      {hasEditDecision && (
        <div style={{
          display: 'flex',
          gap: '4px',
          padding: '4px 8px',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
          background: 'rgba(255,255,255,0.015)',
          alignItems: 'center',
        }}>
          {activeTab === 'ffmpeg' ? (
            <>
              {/* Quality toggle (480p | full) */}
              <div
                role="group"
                aria-label="FFmpeg quality"
                style={{
                  display: 'inline-flex',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}
              >
                {(['draft', 'full'] as FfmpegQuality[]).map((q) => {
                  const selected = ffmpegQualityPref === q;
                  return (
                    <button
                      key={q}
                      onClick={() => onSetFfmpegQualityPref(q)}
                      style={{
                        ...toggleBtnStyle,
                        background: selected ? 'rgba(77,163,255,0.18)' : 'transparent',
                        color: selected ? 'var(--text-primary)' : 'var(--text-muted)',
                      }}
                      title={
                        q === 'draft'
                          ? 'Fast 480p preview (default)'
                          : 'Full timeline resolution — slower, higher quality'
                      }
                    >
                      {q === 'draft' ? '480p' : 'full'}
                    </button>
                  );
                })}
              </div>
              <button
                onClick={() => onRequestFfmpegRender(ffmpegQualityPref)}
                disabled={ffmpegBusy}
                style={{
                  ...actionBtnStyle,
                  opacity: ffmpegBusy ? 0.5 : 1,
                  cursor: ffmpegBusy ? 'not-allowed' : 'pointer',
                }}
                title={`Re-render ffmpeg at ${ffmpegQualityPref === 'full' ? 'full res' : '480p'}`}
              >
                {ffmpegBusy ? 'Rendering...' : 'Re-render'}
              </button>
            </>
          ) : (
            <button
              onClick={onRequestResolveRender}
              disabled={resolveBusy || resolveRunning !== true}
              style={{
                ...actionBtnStyle,
                opacity: (resolveBusy || resolveRunning !== true) ? 0.5 : 1,
                cursor: (resolveBusy || resolveRunning !== true) ? 'not-allowed' : 'pointer',
              }}
              title={resolveRunning !== true ? 'Requires DaVinci Resolve' : undefined}
            >
              {resolveBusy
                ? 'Rendering...'
                : resolveRunning !== true
                  ? 'Render via Resolve (unavailable)'
                  : 'Render via Resolve'}
            </button>
          )}
        </div>
      )}
    </div>
  );
});

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

const actionBtnStyle: React.CSSProperties = {
  flex: 1,
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '3px',
  color: 'var(--text-secondary)',
  fontSize: '9px',
  fontFamily: 'var(--font-mono)',
  cursor: 'pointer',
  padding: '4px 8px',
  letterSpacing: '0.04em',
  textTransform: 'uppercase',
};

const toggleBtnStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  fontFamily: 'var(--font-mono)',
  fontSize: '9px',
  letterSpacing: '0.04em',
  textTransform: 'uppercase',
  padding: '4px 8px',
  cursor: 'pointer',
};
