import React from 'react';

interface PanelProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  badge?: string | number;
  badgeColor?: string;
}

export function Panel({ title, children, className = '', badge, badgeColor }: PanelProps) {
  return (
    <div
      className={className}
      style={{
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-panel)',
        border: '1px solid var(--border)',
        borderRadius: '4px',
        overflow: 'hidden',
        minHeight: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 10px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-card)',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            color: 'var(--text-secondary)',
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          {title}
        </span>
        {badge !== undefined && (
          <span
            style={{
              marginLeft: 'auto',
              background: badgeColor ?? 'var(--bg-hover)',
              color: badgeColor ? '#fff' : 'var(--text-secondary)',
              fontSize: '10px',
              fontWeight: 700,
              padding: '1px 6px',
              borderRadius: '10px',
              letterSpacing: '0.04em',
            }}
          >
            {badge}
          </span>
        )}
      </div>
      {/* Scrollable body */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '8px 10px',
          minHeight: 0,
        }}
      >
        {children}
      </div>
    </div>
  );
}
