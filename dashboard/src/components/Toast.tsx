import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { createPortal } from 'react-dom';

type ToastLevel = 'success' | 'error' | 'info' | 'warning';

interface ToastItem {
  id: number;
  message: string;
  level: ToastLevel;
  detail?: string;
}

interface ToastContextValue {
  toast: (message: string, level?: ToastLevel, detail?: string) => void;
}

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

let _toastId = 0;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((message: string, level: ToastLevel = 'info', detail?: string) => {
    const id = ++_toastId;
    setToasts(prev => [...prev, { id, message, level, detail }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, level === 'error' ? 6000 : 3500);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {createPortal(
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          maxWidth: '400px',
          pointerEvents: 'none',
        }}>
          {toasts.map(t => (
            <ToastCard key={t.id} item={t} onDismiss={() => dismiss(t.id)} />
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}

const LEVEL_STYLES: Record<ToastLevel, { color: string; bg: string; border: string; icon: string }> = {
  success: { color: 'var(--accent-green)', bg: 'rgba(138,210,126,0.08)', border: 'rgba(138,210,126,0.25)', icon: '\u2713' },
  error:   { color: 'var(--accent-red)',   bg: 'rgba(255,125,111,0.08)', border: 'rgba(255,125,111,0.25)', icon: '!' },
  info:    { color: 'var(--accent-blue)',  bg: 'rgba(96,213,200,0.08)',  border: 'rgba(96,213,200,0.25)',  icon: 'i' },
  warning: { color: 'var(--accent-yellow)', bg: 'rgba(241,191,99,0.08)', border: 'rgba(241,191,99,0.25)', icon: '\u26A0' },
};

function ToastCard({ item, onDismiss }: { item: ToastItem; onDismiss: () => void }) {
  const s = LEVEL_STYLES[item.level];
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  return (
    <div
      onClick={onDismiss}
      style={{
        pointerEvents: 'auto',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '10px',
        padding: '10px 14px',
        borderRadius: '10px',
        background: s.bg,
        border: `1px solid ${s.border}`,
        backdropFilter: 'blur(16px)',
        boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        transform: visible ? 'translateX(0)' : 'translateX(100%)',
        opacity: visible ? 1 : 0,
        transition: 'transform 0.25s ease-out, opacity 0.25s ease-out',
      }}
    >
      <span style={{
        color: s.color,
        fontWeight: 800,
        fontSize: '12px',
        width: '16px',
        height: '16px',
        borderRadius: '50%',
        border: `1.5px solid ${s.color}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        marginTop: '1px',
      }}>
        {s.icon}
      </span>
      <div style={{ minWidth: 0 }}>
        <div style={{
          color: 'var(--text-primary)',
          fontSize: '12px',
          fontWeight: 600,
          lineHeight: 1.4,
        }}>
          {item.message}
        </div>
        {item.detail && (
          <div style={{
            color: 'var(--text-muted)',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            marginTop: '3px',
            lineHeight: 1.4,
            wordBreak: 'break-all',
          }}>
            {item.detail}
          </div>
        )}
      </div>
    </div>
  );
}
