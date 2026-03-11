import { useEffect, useRef, useState, useCallback } from 'react';
import type { PlanningEvent } from '../types';

const WS_BASE = 'ws://localhost:8000/video-edit/v2/session';

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
          connect();
        }
      }, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

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
