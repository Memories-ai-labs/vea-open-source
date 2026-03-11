import { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE = 'ws://localhost:8000/video-edit/v2/agent';

export interface AgentEvent {
  type: string;
  data: Record<string, any>;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: string;
}

export interface ScratchpadState {
  comprehension: string;
  creative_direction: string;
  planning: string;
  fcpxml: string;
}

interface UseAgentChatResult {
  events: AgentEvent[];
  messages: ChatMessage[];
  scratchpads: ScratchpadState;
  connected: boolean;
  busy: boolean;
  send: (text: string) => void;
}

export function useAgentChat(projectName: string | null): UseAgentChatResult {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [scratchpads, setScratchpads] = useState<ScratchpadState>({
    comprehension: '',
    creative_direction: '',
    planning: '',
    fcpxml: '',
  });
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
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

    const url = `${WS_BASE}/${encodeURIComponent(name)}/chat`;
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
        const event: AgentEvent = JSON.parse(ev.data as string);

        // Handle specific event types
        switch (event.type) {
          case 'init':
            // New connection — reset events to just this init
            setEvents([event]);
            setBusy(false);
            if (event.data.scratchpads) {
              setScratchpads(event.data.scratchpads as ScratchpadState);
            }
            if (event.data.chat_history) {
              setMessages(event.data.chat_history as ChatMessage[]);
            }
            return; // skip the append below

          case 'agent_message':
            setMessages((prev) => [...prev, {
              role: 'model',
              text: event.data.text,
              timestamp: event.data.timestamp || new Date().toISOString(),
            }]);
            break;

          case 'user_message':
            // Server echo of user message — already added locally
            break;

          case 'scratchpad_update':
            if (event.data.name && event.data.content !== undefined) {
              setScratchpads((prev) => ({
                ...prev,
                [event.data.name]: event.data.content,
              }));
            }
            break;

          case 'done':
            setBusy(false);
            break;

          case 'error':
            setBusy(false);
            break;
        }

        // Append to events array (init is handled above with reset)
        setEvents((prev) => [...prev, event]);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      wsRef.current = null;
      const delay = Math.min(backoffRef.current, 5000);
      backoffRef.current = Math.min(backoffRef.current * 2, 5000);
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current && projectRef.current === name) {
          connect();
        }
      }, delay);
    };

    ws.onerror = () => { ws.close(); };
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    if (!projectName) {
      setEvents([]);
      setMessages([]);
      setScratchpads({ comprehension: '', creative_direction: '', planning: '', fcpxml: '' });
      setConnected(false);
      setBusy(false);
      return;
    }

    // Reset
    setEvents([]);
    setMessages([]);
    setConnected(false);
    setBusy(false);
    backoffRef.current = 500;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
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

  const send = useCallback((text: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Add message locally immediately
      setMessages((prev) => [...prev, {
        role: 'user',
        text,
        timestamp: new Date().toISOString(),
      }]);
      setBusy(true);
      wsRef.current.send(JSON.stringify({ type: 'user_message', text }));
    }
  }, []);

  return { events, messages, scratchpads, connected, busy, send };
}
