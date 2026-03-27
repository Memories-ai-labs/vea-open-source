import { useRef, useEffect, useState } from 'react';
import type { EditDecisionClip, TransformSettings, CropStatus } from '../hooks/useAgentChat';

interface TransformPreviewProps {
  clip: EditDecisionClip;
  projectName: string;
  timelineWidth: number;
  timelineHeight: number;
  cropStatus?: CropStatus;
}

export default function TransformPreview({
  clip,
  projectName,
  timelineWidth,
  timelineHeight,
  cropStatus,
}: TransformPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 300, h: 200 });
  const midpoint = clip.source_start + (clip.source_end - clip.source_start) / 2;

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerSize({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Seek to clip midpoint
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoaded = () => {
      video.currentTime = midpoint;
      video.pause();
    };

    if (video.readyState >= 1) {
      video.currentTime = midpoint;
      video.pause();
    } else {
      video.addEventListener('loadedmetadata', handleLoaded);
      return () => video.removeEventListener('loadedmetadata', handleLoaded);
    }
  }, [clip.source_file, clip.source_start, clip.source_end, midpoint]);

  const srcW = clip.source_width || 1920;
  const srcH = clip.source_height || 1080;
  const tlAspect = timelineWidth / timelineHeight;
  const srcAspect = srcW / srcH;

  const t = clip.transform as TransformSettings | undefined;

  // Compute the visual scale: how much the source is scaled to fill the timeline frame
  const fitScale = t ? t.scale_x : timelineWidth / srcW;

  // The timeline frame dimensions within the container
  // Use a padding-based approach for the frame: fill available width, compute height from aspect
  const availW = Math.max(containerSize.w - 16, 100); // 8px padding each side
  const availH = Math.max(containerSize.h - 16, 80);

  let frameW: number, frameH: number;
  if (availW / availH > tlAspect) {
    frameH = availH;
    frameW = frameH * tlAspect;
  } else {
    frameW = availW;
    frameH = frameW / tlAspect;
  }

  // The video fills the frame — at scale=1 the source matches the frame dimensions
  // For "fit" mode (no transform), the video is scaled to fit the timeline width
  const videoDisplayW = frameW * (srcW / timelineWidth) * fitScale;
  const videoDisplayH = frameH * (srcH / timelineHeight) * fitScale;

  // Position offset from transform (FCPXML units → pixels)
  const pxPerUnit = frameW / (timelineWidth / 19.2);
  const posX = t ? t.position_x * pxPerUnit : 0;
  const posY = t ? -t.position_y * pxPerUnit : 0;

  const src = `http://localhost:8000/video-edit/v2/projects/${projectName}/footage/${clip.source_file}`;
  const isCropping = cropStatus?.status === 'running';

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        background: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Timeline frame boundary */}
      <div
        style={{
          position: 'relative',
          width: frameW,
          height: frameH,
          overflow: 'hidden',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: 'var(--radius-md)',
        }}
      >
        <video
          ref={videoRef}
          src={src}
          muted
          preload="metadata"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: videoDisplayW,
            height: videoDisplayH,
            objectFit: 'cover',
            transform: `translate(calc(-50% + ${posX}px), calc(-50% + ${posY}px))`,
            opacity: isCropping ? 0.3 : 1,
            transition: 'opacity 0.3s',
          }}
        />
        {/* Crop loading overlay */}
        {isCropping && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}>
            <div style={{
              width: 28,
              height: 28,
              border: '3px solid rgba(255,255,255,0.2)',
              borderTopColor: 'var(--accent-green)',
              borderRadius: '50%',
              animation: 'spinner 0.8s linear infinite',
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--text-secondary)',
            }}>
              Analyzing saliency...
            </span>
          </div>
        )}
      </div>
      {/* Clip label */}
      {clip.label && (
        <div
          style={{
            position: 'absolute',
            bottom: 4,
            left: 4,
            padding: '2px 6px',
            background: 'rgba(0, 0, 0, 0.7)',
            borderRadius: 'var(--radius-md)',
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            maxWidth: 'calc(100% - 8px)',
          }}
        >
          {clip.label}
        </div>
      )}
    </div>
  );
}
