import { useEffect, useRef, useState, useCallback } from 'react';

// Derive WS base from the page origin so non-localhost deploys work.
const WS_PROTO = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_HOST = typeof window !== 'undefined' ? window.location.host : 'localhost:8000';
const WS_BASE = `${WS_PROTO}//${WS_HOST}/video-edit/v2/agent`;

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

export interface TransformSettings {
  scale_x: number;
  scale_y: number;
  position_x: number;
  position_y: number;
  rotation: number;
}

export interface EditDecisionClip {
  id: string;
  source_file: string;
  source_start: number;
  source_end: number;
  label?: string;
  description?: string;
  gain_db?: number | null;
  measured_loudness_lufs?: number | null;
  speed?: { rate: number } | null;
  transform?: TransformSettings | null;
  transform_mode?: 'fit' | 'custom' | 'saliency';
  source_width?: number;
  source_height?: number;
  track?: number;
  timeline_offset?: number;
}

export interface NarrationSegment {
  file: string;
  timeline_offset: number;
  start: number;
  duration: number;
  gain_db: number;
  measured_loudness_lufs?: number | null;
  track?: number;                       // audio lane (default 1 = A1)
}

export interface MusicTrack {
  file: string;
  start: number;
  duration: number;
  gain_db: number;
  measured_loudness_lufs?: number | null;
  track?: number;                       // audio lane (default 2 = A2)
}

export interface TextOverlay {
  text: string;
  timeline_offset: number;
  duration: number;
  font_size?: number;
  style?: 'title' | 'subtitle';
  position?: 'center' | 'bottom' | 'top';
  lane?: number;                        // which V-track above the spine (default 1)
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
  outputPath: string | null;
  error: string | null;
}

export interface CropStatus {
  clip_id: string;
  status: 'idle' | 'running' | 'complete' | 'error';
  step?: string;
  error?: string;
}

export type FfmpegQuality = 'draft' | 'full';

export interface FfmpegRenderState extends RenderState {
  quality: FfmpegQuality;
}

interface UseAgentChatResult {
  events: AgentEvent[];
  messages: ChatMessage[];
  scratchpads: ScratchpadState;
  scratchpadTimestamps: ScratchpadTimestamps;
  editDecision: EditDecision | null;
  ffmpegRenderState: FfmpegRenderState;
  resolveRenderState: RenderState;
  ffmpegQualityPref: FfmpegQuality;
  cropStatuses: Record<string, CropStatus>;
  footageFiles: string[];
  indexedFiles: string[];
  videoMeta: Record<string, {
    video_no: string | null;
    indexed_at: string | null;
    duration_seconds: number | null;
  }>;
  reindexingFiles: Set<string>;
  needsIndexing: boolean;
  indexingState: {
    status: 'idle' | 'running' | 'complete' | 'error';
    percent: number;
    message: string;
    videos_total?: number;
    videos_done?: number;
  };
  connected: boolean;
  busy: boolean;
  send: (text: string) => void;
  requestResolveRender: () => void;
  requestFfmpegRender: (quality: FfmpegQuality) => void;
  setFfmpegQualityPref: (quality: FfmpegQuality) => void;
  triggerIndex: (files?: string[]) => void;
  clearAndReconnect: () => void;
  updateEditDecision: (updated: EditDecision) => void;
  /**
   * Update the local editDecision state WITHOUT sending to the server or
   * pushing history. Used for live drag previews so a retrim doesn't fire
   * 50 WebSocket messages per second. Pair with ``updateEditDecision``
   * at drag-end to commit the final state.
   */
  previewEditDecision: (updated: EditDecision) => void;
  requestCropClip: (clipId: string) => void;
  /** Undo last edit-decision change. No-op if ``canUndo`` is false. */
  undo: () => void;
  /** Redo the next edit-decision change. No-op if ``canRedo`` is false. */
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

const HISTORY_MAX = 50;

/** Structural deep-equal for plain JSON-compatible values.
 *
 * Used to avoid pushing a history entry when the server echoes back the
 * exact same EditDecision we just sent. JSON roundtrip is cheap compared to
 * the API call that produced the update, and covers every shape we need.
 */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
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
  const [editDecision, setEditDecisionRaw] = useState<EditDecision | null>(null);

  // ── Undo / redo history ────────────────────────────────────────────────
  // History for the current editDecision only. Each user edit (or incoming
  // agent edit via timeline_update) pushes the previous ``present`` onto
  // ``past`` and clears ``future``. Undo/redo apply a snapshot locally AND
  // send it back to the server via updateEditDecision so the change sticks.
  //
  // Use refs for the stacks (don't need to re-render when they change) and
  // useState only for the canUndo/canRedo booleans that the UI needs.
  const historyRef = useRef<{ past: EditDecision[]; future: EditDecision[] }>(
    { past: [], future: [] },
  );
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const applyingHistoryRef = useRef(false);
  const currentEditRef = useRef<EditDecision | null>(null);

  // Always keep currentEditRef aligned so undo/redo can read the "present"
  // synchronously without waiting for a re-render cycle.
  useEffect(() => { currentEditRef.current = editDecision; }, [editDecision]);

  // Every write path goes through this wrapper. Marks whether the change is
  // a "commit" (push onto past, clear future) or a "replace" (new baseline,
  // wipes history). Replace is used for fresh WS init / project switch.
  const setEditDecision = useCallback(
    (next: EditDecision | null, opts: { kind: 'commit' | 'replace' } = { kind: 'commit' }) => {
      const prev = currentEditRef.current;

      // Fast path — no-op if the next state is structurally equal to prev.
      // Avoids a history entry when a server echo arrives after our own
      // optimistic update.
      if (prev && next && deepEqual(prev, next)) {
        return;
      }

      if (opts.kind === 'replace' || next === null) {
        historyRef.current = { past: [], future: [] };
        setCanUndo(false);
        setCanRedo(false);
      } else if (!applyingHistoryRef.current && prev) {
        // Real user/agent edit — push the outgoing value onto past,
        // drop the redo stack.
        const { past } = historyRef.current;
        historyRef.current = {
          past: [...past, prev].slice(-HISTORY_MAX),
          future: [],
        };
        setCanUndo(true);
        setCanRedo(false);
      }
      // When applyingHistoryRef is true, the stacks were already arranged
      // inside undo()/redo() — we just set state here.
      applyingHistoryRef.current = false;
      currentEditRef.current = next;
      setEditDecisionRaw(next);
    },
    [],
  );
  const [resolveRenderState, setResolveRenderState] = useState<RenderState>({
    status: 'idle', progress: 0, filename: null, outputPath: null, error: null,
  });
  const [ffmpegRenderState, setFfmpegRenderState] = useState<FfmpegRenderState>({
    status: 'idle', progress: 0, filename: null, outputPath: null, error: null, quality: 'draft',
  });
  const [ffmpegQualityPref, setFfmpegQualityPrefState] = useState<FfmpegQuality>('draft');
  const [cropStatuses, setCropStatuses] = useState<Record<string, CropStatus>>({});
  const [footageFiles, setFootageFiles] = useState<string[]>([]);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);
  const [videoMeta, setVideoMeta] = useState<Record<string, {
    video_no: string | null;
    indexed_at: string | null;
    duration_seconds: number | null;
  }>>({});
  const [reindexingFiles, setReindexingFiles] = useState<Set<string>>(new Set());
  const [needsIndexing, setNeedsIndexing] = useState<boolean>(false);
  const [indexingState, setIndexingState] = useState<{
    status: 'idle' | 'running' | 'complete' | 'error';
    percent: number;
    message: string;
    videos_total?: number;
    videos_done?: number;
  }>({ status: 'idle', percent: 0, message: '' });
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
            if (event.data.footage_files) setFootageFiles(event.data.footage_files as string[]);
            if (event.data.indexed_files) setIndexedFiles(event.data.indexed_files as string[]);
            if (event.data.video_meta) setVideoMeta(event.data.video_meta as any);
            setReindexingFiles(new Set());
            setNeedsIndexing(Boolean(event.data.needs_indexing));
            if (event.data.indexing_state) {
              setIndexingState(event.data.indexing_state as any);
            } else {
              setIndexingState({ status: 'idle', percent: 0, message: '' });
            }
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
              // init is a fresh snapshot — wipe any prior history so the
              // reconnected session starts with a clean undo stack.
              setEditDecision(event.data.edit_decision as EditDecision, { kind: 'replace' });
            }
            if (event.data.ffmpeg_render_state) {
              const frs = event.data.ffmpeg_render_state;
              setFfmpegRenderState({
                status: frs.status || 'idle',
                progress: frs.progress ?? (frs.status === 'complete' ? 100 : 0),
                filename: frs.filename || null,
                outputPath: frs.output_path || null,
                error: frs.error || null,
                quality: (frs.quality === 'full' ? 'full' : 'draft') as FfmpegQuality,
              });
            }
            if (event.data.resolve_render_state) {
              const rrs = event.data.resolve_render_state;
              setResolveRenderState({
                status: rrs.status || 'idle',
                progress: rrs.progress ?? (rrs.status === 'complete' ? 100 : 0),
                filename: rrs.filename || null,
                outputPath: rrs.output_path || null,
                error: rrs.error || null,
              });
            }
            if (event.data.ffmpeg_quality_pref) {
              const q = event.data.ffmpeg_quality_pref;
              setFfmpegQualityPrefState(q === 'full' ? 'full' : 'draft');
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

          case 'index_progress':
            setIndexingState({
              status: event.data.status || 'running',
              percent: event.data.percent ?? 0,
              message: event.data.message || '',
              videos_total: event.data.videos_total,
              videos_done: event.data.videos_done,
            });
            // Once complete, refresh footage/indexed lists from the data
            if (event.data.status === 'complete') {
              setNeedsIndexing(false);
              if (event.data.indexed_files) setIndexedFiles(event.data.indexed_files as string[]);
              if (event.data.video_meta) setVideoMeta(event.data.video_meta as any);
              setReindexingFiles(new Set());
            } else if (event.data.status === 'error') {
              setReindexingFiles(new Set());
            }
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

          case 'render_start': {
            if (event.data.renderer === 'ffmpeg') {
              const q = (event.data.quality === 'full' ? 'full' : 'draft') as FfmpegQuality;
              setFfmpegRenderState({
                status: 'rendering', progress: 0, filename: null, outputPath: null, error: null, quality: q,
              });
            } else {
              setResolveRenderState({ status: 'rendering', progress: 0, filename: null, outputPath: null, error: null });
            }
            break;
          }

          case 'render_progress': {
            if (event.data.renderer === 'ffmpeg') {
              setFfmpegRenderState(prev => ({ ...prev, progress: event.data.percent || 0 }));
            } else {
              setResolveRenderState(prev => ({ ...prev, progress: event.data.percent || 0 }));
            }
            break;
          }

          case 'render_complete': {
            if (event.data.renderer === 'ffmpeg') {
              const q = (event.data.quality === 'full' ? 'full' : 'draft') as FfmpegQuality;
              setFfmpegRenderState({
                status: 'complete', progress: 100,
                filename: event.data.filename || null,
                outputPath: event.data.output_path || null,
                error: null, quality: q,
              });
            } else {
              setResolveRenderState({
                status: 'complete', progress: 100,
                filename: event.data.filename || null,
                outputPath: event.data.output_path || null,
                error: null,
              });
            }
            break;
          }

          case 'render_error': {
            if (event.data.renderer === 'ffmpeg') {
              setFfmpegRenderState(prev => ({
                ...prev, status: 'error',
                error: event.data.error || 'Render failed',
              }));
            } else {
              setResolveRenderState(prev => ({
                ...prev, status: 'error',
                error: event.data.error || 'Render failed',
              }));
            }
            break;
          }

          case 'ffmpeg_quality_pref': {
            const q = event.data.quality === 'full' ? 'full' : 'draft';
            setFfmpegQualityPrefState(q as FfmpegQuality);
            break;
          }

          case 'crop_progress':
            setCropStatuses(prev => ({
              ...prev,
              [event.data.clip_id]: {
                clip_id: event.data.clip_id,
                status: 'running',
                step: event.data.step,
              },
            }));
            break;

          case 'crop_complete':
            setCropStatuses(prev => ({
              ...prev,
              [event.data.clip_id]: {
                clip_id: event.data.clip_id,
                status: 'complete',
              },
            }));
            // Update clip transform in edit decision
            if (event.data.transform && event.data.edit_decision) {
              setEditDecision(event.data.edit_decision as EditDecision);
            }
            break;

          case 'crop_error':
            setCropStatuses(prev => ({
              ...prev,
              [event.data.clip_id]: {
                clip_id: event.data.clip_id,
                status: 'error',
                error: event.data.error,
              },
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
      } catch (err) {
        // Log malformed messages instead of silently dropping them —
        // silent drops made protocol bugs invisible during development.
        console.warn('[agentChat] Dropped malformed WS message:', err, ev.data);
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      wsRef.current = null;
      // Cap the reconnect delay at 5 s, but also cap the backoff state itself
      // so it cannot grow beyond the cap. Start at 500 ms (see BACKOFF_INITIAL).
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
      setResolveRenderState({ status: 'idle', progress: 0, filename: null, outputPath: null, error: null });
      setFfmpegRenderState({ status: 'idle', progress: 0, filename: null, outputPath: null, error: null, quality: 'draft' });
      setFfmpegQualityPrefState('draft');
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

  const requestResolveRender = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setResolveRenderState({ status: 'rendering', progress: 0, filename: null, outputPath: null, error: null });
      wsRef.current.send(JSON.stringify({ type: 'render_resolve' }));
    }
  }, []);

  const requestFfmpegRender = useCallback((quality: FfmpegQuality) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setFfmpegRenderState(prev => ({
        ...prev, status: 'rendering', progress: 0, filename: null, outputPath: null, error: null, quality,
      }));
      setFfmpegQualityPrefState(quality);
      wsRef.current.send(JSON.stringify({ type: 'render_ffmpeg', quality }));
    }
  }, []);

  const setFfmpegQualityPref = useCallback((quality: FfmpegQuality) => {
    setFfmpegQualityPrefState(quality);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_ffmpeg_quality', quality }));
    }
  }, []);

  const updateEditDecision = useCallback((updated: EditDecision) => {
    setEditDecision(updated, { kind: 'commit' });
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'edit_decision_update', edit_decision: updated }));
    }
  }, [setEditDecision]);

  // Preview channel — local state only, no server, no history. Used while a
  // drag is in flight to give responsive feedback without thrashing the WS.
  const previewEditDecision = useCallback((updated: EditDecision) => {
    setEditDecisionRaw(updated);
    currentEditRef.current = updated;
  }, []);

  const undo = useCallback(() => {
    const { past, future } = historyRef.current;
    const present = currentEditRef.current;
    if (past.length === 0 || !present) return;
    const prev = past[past.length - 1];
    historyRef.current = {
      past: past.slice(0, -1),
      future: [present, ...future].slice(0, HISTORY_MAX),
    };
    setCanUndo(historyRef.current.past.length > 0);
    setCanRedo(true);
    // Apply the restored snapshot locally AND send it to the server so the
    // change persists and downstream agents see the same state.
    applyingHistoryRef.current = true;
    setEditDecisionRaw(prev);
    currentEditRef.current = prev;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'edit_decision_update', edit_decision: prev }));
    }
  }, []);

  const redo = useCallback(() => {
    const { past, future } = historyRef.current;
    const present = currentEditRef.current;
    if (future.length === 0 || !present) return;
    const next = future[0];
    historyRef.current = {
      past: [...past, present].slice(-HISTORY_MAX),
      future: future.slice(1),
    };
    setCanUndo(true);
    setCanRedo(historyRef.current.future.length > 0);
    applyingHistoryRef.current = true;
    setEditDecisionRaw(next);
    currentEditRef.current = next;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'edit_decision_update', edit_decision: next }));
    }
  }, []);

  const requestCropClip = useCallback((clipId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setCropStatuses(prev => ({
        ...prev,
        [clipId]: { clip_id: clipId, status: 'running', step: 'starting' },
      }));
      wsRef.current.send(JSON.stringify({ type: 'crop_clip', clip_id: clipId }));
    }
  }, []);

  const triggerIndex = useCallback((files?: string[]) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIndexingState({
        status: 'running',
        percent: 1,
        message: files && files.length
          ? `Re-indexing ${files.length} file(s)...`
          : 'Requesting indexing...',
      });
      if (files && files.length) {
        setReindexingFiles(new Set(files));
      }
      wsRef.current.send(JSON.stringify({ type: 'index_now', files: files || null }));
    }
  }, []);

  const clearAndReconnect = useCallback(() => {
    // Immediately wipe all local state
    setEvents([]);
    setMessages([]);
    setScratchpads({ comprehension: '', creative_direction: '', planning: '', fcpxml: '' });
    setScratchpadTimestamps({ comprehension: null, creative_direction: null, planning: null, fcpxml: null });
    setEditDecision(null);
    setResolveRenderState({ status: 'idle', progress: 0, filename: null, outputPath: null, error: null });
    setFfmpegRenderState({ status: 'idle', progress: 0, filename: null, outputPath: null, error: null, quality: 'draft' });
    setFfmpegQualityPrefState('draft');
    setCropStatuses({});
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

  return { events, messages, scratchpads, scratchpadTimestamps, editDecision, ffmpegRenderState, resolveRenderState, ffmpegQualityPref, cropStatuses, footageFiles, indexedFiles, videoMeta, reindexingFiles, needsIndexing, indexingState, connected, busy, send, requestResolveRender, requestFfmpegRender, setFfmpegQualityPref, clearAndReconnect, updateEditDecision, previewEditDecision, requestCropClip, triggerIndex, undo, redo, canUndo, canRedo };
}
