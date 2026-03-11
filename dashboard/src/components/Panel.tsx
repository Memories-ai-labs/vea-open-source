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
        background: 'linear-gradient(180deg, rgba(35, 27, 24, 0.94), rgba(22, 17, 16, 0.94))',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        overflow: 'hidden',
        minHeight: 0,
        boxShadow: 'var(--shadow-md)',
        backdropFilter: 'blur(18px)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '14px 16px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01))',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.16em',
            textTransform: 'uppercase',
          }}
        >
          {title}
        </span>
        {badge !== undefined && (
          <span
            style={{
              marginLeft: 'auto',
              background: badgeColor ? `${badgeColor}22` : 'rgba(255, 255, 255, 0.05)',
              border: `1px solid ${badgeColor ? `${badgeColor}66` : 'rgba(255, 255, 255, 0.08)'}`,
              color: badgeColor ?? 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              fontWeight: 700,
              padding: '4px 10px',
              borderRadius: '999px',
              letterSpacing: '0.08em',
            }}
          >
            {badge}
          </span>
        )}
      </div>
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '16px',
          minHeight: 0,
        }}
      >
        {children}
      </div>
    </div>
  );
}
