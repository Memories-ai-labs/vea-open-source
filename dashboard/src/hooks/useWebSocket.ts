import { useEffect, useRef, useState, useCallback } from 'react';
import type { PlanningEvent } from '../types';

// Derive the WebSocket base from the page origin so deploys behind a proxy,
// ngrok, or any non-localhost origin work without hard-coded URLs. Falls back
// to localhost only in server-side render contexts.
const WS_PROTO = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_HOST = typeof window !== 'undefined' ? window.location.host : 'localhost:8000';
const WS_BASE = `${WS_PROTO}//${WS_HOST}/video-edit/v2/session`;

interface UseWebSocketResult {
  events: PlanningEvent[];
  connected: boolean;
  send: (msg: object) => void;
}

export function useWebSocket(projectName: string | null): UseWebSocketResult {
  const [events, setEvents] = useState<PlanningEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(500);
  const mountedRef = useRef(true);
  const projectRef = useRef(projectName);
  const connectRef = useRef<() => void>(() => {});

  useEffect(() => {
    projectRef.current = projectName;
  }, [projectName]);

  const connect = useCallback(() => {
    const name = projectRef.current;
    if (!name || !mountedRef.current) return;

    const url = `${WS_BASE}/${encodeURIComponent(name)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      setConnected(true);
      backoffRef.current = 500;
    };

    ws.onmessage = (ev) => {
      if (!mountedRef.current) return;
      try {
        const event: PlanningEvent = JSON.parse(ev.data as string);
        setEvents((prev) => [...prev, event]);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      wsRef.current = null;
      // exponential backoff, max 5s
      const delay = Math.min(backoffRef.current, 5000);
      backoffRef.current = Math.min(backoffRef.current * 2, 5000);
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current && projectRef.current === name) {
          connectRef.current();
        }
      }, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;

    if (!projectName) {
      setEvents([]);
      setConnected(false);
      return;
    }

    // Reset state when project changes
    setEvents([]);
    setConnected(false);
    backoffRef.current = 500;

    // Cancel any pending reconnect for the old project
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    // Close existing socket
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent reconnect from old socket
      wsRef.current.close();
      wsRef.current = null;
    }

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [projectName, connect]);

  const send = useCallback((msg: object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { events, connected, send };
}
