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

export interface ScratchpadTimestamps {
  comprehension: string | null;
  creative_direction: string | null;
  planning: string | null;
  fcpxml: string | null;
}

export interface EditDecisionClip {
  id: string;
  source_file: string;
  source_start: number;
  source_end: number;
  label?: string;
  description?: string;
  gain_db?: number | null;
  speed?: { rate: number } | null;
  transition_after?: { type: string; duration_seconds: number } | null;
  track?: number;
}

export interface NarrationSegment {
  file: string;
  timeline_offset: number;
  start: number;
  duration: number;
  gain_db: number;
}

export interface MusicTrack {
  file: string;
  start: number;
  duration: number;
  gain_db: number;
}

export interface TextOverlay {
  text: string;
  timeline_offset: number;
  duration: number;
  font_size?: number;
}

export interface EditDecision {
  timeline?: { name?: string; fps?: number; width?: number; height?: number };
  clips: EditDecisionClip[];
  narration?: NarrationSegment[];
  music?: MusicTrack | null;
  titles?: TextOverlay[];
  fcpxml_path?: string;
}

export interface RenderState {
  status: 'idle' | 'rendering' | 'complete' | 'error';
  progress: number; // 0-100
  filename: string | null;
  error: string | null;
}

interface UseAgentChatResult {
  events: AgentEvent[];
  messages: ChatMessage[];
  scratchpads: ScratchpadState;
  scratchpadTimestamps: ScratchpadTimestamps;
  editDecision: EditDecision | null;
  renderState: RenderState;
  connected: boolean;
  busy: boolean;
  send: (text: string) => void;
  clearAndReconnect: () => void;
  updateEditDecision: (updated: EditDecision) => void;
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
  const [scratchpadTimestamps, setScratchpadTimestamps] = useState<ScratchpadTimestamps>({
    comprehension: null,
    creative_direction: null,
    planning: null,
    fcpxml: null,
  });
  const [editDecision, setEditDecision] = useState<EditDecision | null>(null);
  const [renderState, setRenderState] = useState<RenderState>({
    status: 'idle', progress: 0, filename: null, error: null,
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
          case 'init': {
            // New connection — hydrate from persisted state
            setBusy(false);
            if (event.data.scratchpads) {
              setScratchpads(event.data.scratchpads as ScratchpadState);
            }
            if (event.data.scratchpad_timestamps) {
              setScratchpadTimestamps(event.data.scratchpad_timestamps as ScratchpadTimestamps);
            }
            if (event.data.chat_history) {
              setMessages(event.data.chat_history as ChatMessage[]);
            }
            if (event.data.edit_decision) {
              setEditDecision(event.data.edit_decision as EditDecision);
            }
            if (event.data.render_state) {
              const rs = event.data.render_state;
              setRenderState({
                status: rs.status || 'idle',
                progress: rs.progress ?? (rs.status === 'complete' ? 100 : 0),
                filename: rs.filename || null,
                error: rs.error || null,
              });
            }
            // Replay persisted events (tool calls, results, scratchpad updates)
            if (event.data.event_log && Array.isArray(event.data.event_log)) {
              const replayed: AgentEvent[] = (event.data.event_log as Array<{ type: string; data: Record<string, any> }>).map((e) => ({
                type: e.type,
                data: e.data,
              }));
              setEvents([event, ...replayed]);
            } else {
              setEvents([event]);
            }
            return; // skip the append below
          }

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
              if (event.data.timestamp) {
                setScratchpadTimestamps((prev) => ({
                  ...prev,
                  [event.data.name]: event.data.timestamp,
                }));
              }
            }
            break;

          case 'timeline_update':
            if (event.data.edit_decision) {
              setEditDecision({
                ...event.data.edit_decision,
                fcpxml_path: event.data.fcpxml_path,
              });
            }
            break;

          case 'render_start':
            setRenderState({ status: 'rendering', progress: 0, filename: null, error: null });
            break;

          case 'render_progress':
            setRenderState(prev => ({ ...prev, progress: event.data.percent || 0 }));
            break;

          case 'render_complete':
            setRenderState({
              status: 'complete',
              progress: 100,
              filename: event.data.filename || null,
              error: null,
            });
            break;

          case 'render_error':
            setRenderState(prev => ({
              ...prev,
              status: 'error',
              error: event.data.error || 'Render failed',
            }));
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
      setScratchpadTimestamps({ comprehension: null, creative_direction: null, planning: null, fcpxml: null });
      setEditDecision(null);
      setRenderState({ status: 'idle', progress: 0, filename: null, error: null });
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

  const requestRender = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setRenderState({ status: 'rendering', progress: 0, filename: null, error: null });
      wsRef.current.send(JSON.stringify({ type: 'render' }));
    }
  }, []);

  const updateEditDecision = useCallback((updated: EditDecision) => {
    setEditDecision(updated);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'edit_decision_update', edit_decision: updated }));
    }
  }, []);

  const clearAndReconnect = useCallback(() => {
    // Immediately wipe all local state
    setEvents([]);
    setMessages([]);
    setScratchpads({ comprehension: '', creative_direction: '', planning: '', fcpxml: '' });
    setScratchpadTimestamps({ comprehension: null, creative_direction: null, planning: null, fcpxml: null });
    setEditDecision(null);
    setRenderState({ status: 'idle', progress: 0, filename: null, error: null });
    setBusy(false);
    // Close and reconnect WebSocket so backend sends fresh init
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    backoffRef.current = 500;
    // Small delay so the backend clear endpoint finishes before reconnect
    setTimeout(() => { if (mountedRef.current) connect(); }, 300);
  }, [connect]);

  return { events, messages, scratchpads, scratchpadTimestamps, editDecision, renderState, connected, busy, send, requestRender, clearAndReconnect, updateEditDecision };
}
