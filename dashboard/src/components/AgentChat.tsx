import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import type { AgentEvent, ChatMessage, ScratchpadState, ScratchpadTimestamps, EditDecision, TransformSettings, RenderState, CropStatus } from '../hooks/useAgentChat';
import type { ProjectSummary } from '../types';
import { listProjects, clearGists, clearPlanning, clearMemories, indexProject, getResolveStatus, getSystemInfo, setMainModel, setVideoModel } from '../api';
import type { SystemInfo } from '../api';
import { useBreakpoint } from '../hooks/useBreakpoint';
import { SimpleMarkdown } from './SimpleMarkdown';
import { NLETimeline } from './NLETimeline';
import { PreviewPanel } from './PreviewPanel';
import type { PreviewPanelHandle } from './PreviewPanel';
import ClipInspector from './ClipInspector';
import AudioInspector from './AudioInspector';
import TransformPreview from './TransformPreview';
import TimelineSettingsBar from './TimelineSettingsBar';
import { useToast } from './Toast';

interface AgentChatProps {
  project: ProjectSummary;
  events: AgentEvent[];
  messages: ChatMessage[];
  scratchpads: ScratchpadState;
  scratchpadTimestamps: ScratchpadTimestamps;
  editDecision: EditDecision | null;
  renderState: RenderState;
  draftRenderState: RenderState;
  cropStatuses: Record<string, CropStatus>;
  connected: boolean;
  busy: boolean;
  needsIndexing?: boolean;
  indexingState?: {
    status: 'idle' | 'running' | 'complete' | 'error';
    percent: number;
    message: string;
    videos_total?: number;
    videos_done?: number;
  };
  videoMeta?: Record<string, {
    video_no: string | null;
    indexed_at: string | null;
    duration_seconds: number | null;
  }>;
  reindexingFiles?: Set<string>;
  onSend: (text: string) => void;
  onRequestRender: () => void;
  onRequestDraftRender: () => void;
  onTriggerIndex?: (files?: string[]) => void;
  onBack: () => void;
  onClearState: () => void;
  onEditDecisionChange: (updated: EditDecision) => void;
  onEditDecisionPreview: (updated: EditDecision) => void;
  onRequestCropClip: (clipId: string) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

function formatTime(iso: string | undefined): string {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  } catch { return ''; }
}

/** True when we should render Mac-style modifier glyphs (⌘, ⇧) in shortcuts. */
function isMacLike(): boolean {
  if (typeof navigator === 'undefined') return false;
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
}

/** Compact label for the header pill. "anthropic/claude-opus-4.6" → "Opus 4.6" */
function shortModelLabel(id: string | undefined): string {
  if (!id) return '…';
  const slug = id.includes('/') ? id.split('/')[1] : id;
  const m = slug.match(/(opus|sonnet|haiku|gpt|gemini)[-_ ]?([\d.]+(?:[-\s][a-z]+)?)/i);
  if (m) return `${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}`;
  return slug;
}

const PAD_LABELS: Record<string, string> = {
  comprehension: 'Comprehension',
  creative_direction: 'Creative Direction',
  planning: 'Planning',
  fcpxml: 'FCPXML',
};

const PAD_COLORS: Record<string, string> = {
  comprehension: 'var(--accent-blue)',
  creative_direction: 'var(--accent-yellow)',
  planning: 'var(--accent-green)',
  fcpxml: 'var(--accent-purple)',
};

const PAD_EMPTY_HINTS: Record<string, string> = {
  comprehension: 'Index your footage to generate a comprehension overview. Use the Manage menu to trigger indexing.',
  creative_direction: 'Start a conversation with a creative brief — the agent will fill this in as it explores your footage.',
  planning: 'The agent populates this during the planning loop with shot lists and edit structure.',
  fcpxml: 'Ask the agent to generate an FCPXML once planning is complete. The timeline XML will appear here.',
};

// ── Drag divider (horizontal, between upper and lower rows) ──
function useHorizontalDivider(
  upperPct: number,
  setUpperPct: React.Dispatch<React.SetStateAction<number>>,
  containerRef: React.RefObject<HTMLDivElement | null>,
) {
  const dragging = useRef(false);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((ev.clientY - rect.top) / rect.height) * 100;
      setUpperPct(Math.max(25, Math.min(75, pct)));
    };

    const onUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [upperPct, setUpperPct, containerRef]);

  return onMouseDown;
}

export function AgentChat({
  project: initialProject, events, messages, scratchpads, scratchpadTimestamps, editDecision, renderState, draftRenderState, cropStatuses, connected, busy, needsIndexing, indexingState, videoMeta, reindexingFiles, onSend, onRequestRender, onRequestDraftRender, onTriggerIndex, onBack, onClearState, onEditDecisionChange, onEditDecisionPreview, onRequestCropClip, onUndo, onRedo, canUndo, canRedo,
}: AgentChatProps) {
  const { toast } = useToast();
  const [project, setProject] = useState(initialProject);
  const [input, setInput] = useState('');
  const [activePad, setActivePad] = useState<string>('comprehension');
  const [resolveRunning, setResolveRunning] = useState<boolean | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [manageLoading, setManageLoading] = useState<string | null>(null);
  const [manageMsg, setManageMsg] = useState('');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [modelOpen, setModelOpen] = useState(false);
  const [modelLoading, setModelLoading] = useState<string | null>(null);
  const [modelDropdownPos, setModelDropdownPos] = useState<{ top: number; right: number } | null>(null);
  const [, setIndexing] = useState(false);
  const [expandedFile, setExpandedFile] = useState<string | null>(null);
  const [playheadTime, setPlayheadTime] = useState(0);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const previewPanelRef = useRef<PreviewPanelHandle>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const manageRef = useRef<HTMLDivElement>(null);
  const manageBtnRef = useRef<HTMLButtonElement>(null);
  const modelRef = useRef<HTMLDivElement>(null);
  const modelBtnRef = useRef<HTMLButtonElement>(null);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; right: number } | null>(null);
  const isTablet = useBreakpoint(1080);

  // Layout: upper row pct (of the main content area), lower gets the rest
  const [upperPct, setUpperPct] = useState(55);
  // Timeline/chat horizontal split in lower row
  const [timelinePct, setTimelinePct] = useState(65);

  const mainContentRef = useRef<HTMLDivElement>(null);
  const onRowDrag = useHorizontalDivider(upperPct, setUpperPct, mainContentRef);

  // Tick every 30s to keep "updated X ago" labels fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(iv);
  }, []);

  // Poll DaVinci Resolve status
  useEffect(() => {
    let mounted = true;
    async function check() {
      try {
        const r = await getResolveStatus();
        if (mounted) setResolveRunning(r.running);
      } catch {
        if (mounted) setResolveRunning(null);
      }
    }
    check();
    const iv = setInterval(check, 15000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  // Escape key deselects clip
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedClipId) setSelectedClipId(null);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedClipId]);

  // Cmd/Ctrl+Z → undo, Cmd+Shift+Z or Ctrl+Y → redo.
  // Skip when focus is in a text input/textarea/contenteditable so the browser's
  // native undo keeps working inside the chat input, inspector fields, etc.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const active = document.activeElement as HTMLElement | null;
      if (active) {
        const tag = active.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || active.isContentEditable) {
          return;
        }
      }
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      // Undo: (Cmd|Ctrl)+Z without Shift
      if (e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        onUndo();
        return;
      }
      // Redo: (Cmd|Ctrl)+Shift+Z  OR  Ctrl+Y
      if ((e.key.toLowerCase() === 'z' && e.shiftKey) || e.key.toLowerCase() === 'y') {
        e.preventDefault();
        onRedo();
        return;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onUndo, onRedo]);

  // Toast popups when renders finish or fail (transition-aware so we only fire once per render)
  const prevDraftStatusRef = useRef<RenderState['status']>('idle');
  const prevFinalStatusRef = useRef<RenderState['status']>('idle');
  useEffect(() => {
    const prev = prevDraftStatusRef.current;
    const curr = draftRenderState.status;
    if (prev !== curr) {
      if (curr === 'complete' && prev === 'rendering') {
        toast('Draft render complete', 'success');
      } else if (curr === 'error' && prev === 'rendering') {
        toast(`Draft render failed${draftRenderState.error ? `: ${draftRenderState.error.slice(0, 100)}` : ''}`, 'error');
      }
      prevDraftStatusRef.current = curr;
    }
  }, [draftRenderState.status, draftRenderState.error, toast]);
  useEffect(() => {
    const prev = prevFinalStatusRef.current;
    const curr = renderState.status;
    if (prev !== curr) {
      if (curr === 'complete' && prev === 'rendering') {
        toast('Final render complete', 'success');
      } else if (curr === 'error' && prev === 'rendering') {
        toast(`Final render failed${renderState.error ? `: ${renderState.error.slice(0, 100)}` : ''}`, 'error');
      }
      prevFinalStatusRef.current = curr;
    }
  }, [renderState.status, renderState.error, toast]);

  // Horizontal column drag divider for timeline/chat split in lower row
  const lowerRowRef = useRef<HTMLDivElement>(null);
  const colDragging = useRef(false);
  const onColDragDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    colDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev: MouseEvent) => {
      if (!colDragging.current || !lowerRowRef.current) return;
      const rect = lowerRowRef.current.getBoundingClientRect();
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setTimelinePct(Math.max(30, Math.min(80, pct)));
    };
    const onUp = () => {
      colDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const { projects } = await listProjects();
      const fresh = projects.find(p => p.project_name === initialProject.project_name);
      if (fresh) setProject(fresh);
    } catch { /* ignore */ }
  }, [initialProject.project_name]);

  useEffect(() => { refresh(); }, [refresh]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!manageOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (manageRef.current?.contains(target)) return;
      if (manageBtnRef.current?.contains(target)) return;
      setManageOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [manageOpen]);

  // Fetch current + available main_llm models once on mount.
  useEffect(() => {
    getSystemInfo().then(setSystemInfo).catch(() => {/* header pill just stays empty */});
  }, []);

  // Close model dropdown on outside click
  useEffect(() => {
    if (!modelOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (modelRef.current?.contains(target)) return;
      if (modelBtnRef.current?.contains(target)) return;
      setModelOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [modelOpen]);

  // Space bar to play/pause when not typing
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT' || (e.target as HTMLElement).isContentEditable) return;
      e.preventDefault();
      previewPanelRef.current?.togglePlayback();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, events]);

  // Current activity — only show spinner when agent is actively working
  const currentActivity = useMemo(() => {
    if (!busy) return null;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      switch (e.type) {
        case 'tool_call':
          return { label: toolLabel(e.data.tool, e.data.args), active: true };
        case 'tool_result':
          continue;
        case 'done':
        case 'error':
          if (e.type === 'error') return { label: `Error: ${e.data.message}`, active: false };
          return null;
      }
    }
    return { label: 'Thinking...', active: true };
  }, [events, busy]);

  // Timeline items — merge tool_call + tool_result into single entries
  const timeline = useMemo(() => {
    const items: TimelineItem[] = [];
    let msgIdx = 0;

    const lastToolCallIdx = events.reduce((acc, e, i) => e.type === 'tool_call' ? i : acc, -1);
    const hasDoneEvent = lastToolCallIdx >= 0 && events.slice(lastToolCallIdx).some(e => e.type === 'done' || e.type === 'error');
    const turnDone = hasDoneEvent || (!busy && lastToolCallIdx >= 0);

    const toolCallCompleted: boolean[] = [];
    const toolCalls: AgentEvent[] = [];
    const toolResults: AgentEvent[] = [];
    for (const e of events) {
      if (e.type === 'tool_call') toolCalls.push(e);
      if (e.type === 'tool_result') toolResults.push(e);
    }
    const resultUsed = new Array(toolResults.length).fill(false);
    for (let ci = 0; ci < toolCalls.length; ci++) {
      const tc = toolCalls[ci];
      let found = false;
      for (let ri = 0; ri < toolResults.length; ri++) {
        if (!resultUsed[ri] && toolResults[ri].data.tool === tc.data.tool) {
          toolCallCompleted.push(true);
          resultUsed[ri] = true;
          found = true;
          break;
        }
      }
      if (!found) toolCallCompleted.push(turnDone);
    }

    const refineEvents: AgentEvent[] = events.filter(e => e.type === 'refine_progress');

    let toolCallIdx = 0;
    for (const e of events) {
      if (e.type === 'init') continue;
      while (msgIdx < messages.length) {
        const m = messages[msgIdx];
        if (m.timestamp <= (e.data.timestamp || '')) {
          items.push({ kind: 'message', message: m });
          msgIdx++;
        } else break;
      }
      if (e.type === 'tool_call') {
        const completed = toolCallCompleted[toolCallIdx] || false;
        items.push({ kind: 'tool_call', event: e, completed });
        toolCallIdx++;
      }
      else if (e.type === 'tool_result') continue;
      else if (e.type === 'refine_progress') {
        const isLast = e === refineEvents[refineEvents.length - 1];
        const done = !isLast || turnDone;
        items.push({ kind: 'refine_progress', event: e, completed: done });
      }
      else if (e.type === 'scratchpad_update') items.push({ kind: 'scratchpad_update', event: e });
    }
    while (msgIdx < messages.length) {
      items.push({ kind: 'message', message: messages[msgIdx] });
      msgIdx++;
    }
    return items;
  }, [events, messages, busy]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    onSend(text);
    setInput('');
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  const handleTimelineSeek = useCallback((time: number) => {
    setPlayheadTime(time);
    previewPanelRef.current?.seekTo(time);
  }, []);

  const selectedClip = useMemo(() => {
    if (!selectedClipId || !editDecision) return null;
    return editDecision.clips.find(c => c.id === selectedClipId) || null;
  }, [selectedClipId, editDecision]);

  // Audio item selection (narration / music) — id format: "narr-N" or "music-0"
  const selectedAudio = useMemo(() => {
    if (!selectedClipId || !editDecision) return null;
    if (selectedClipId.startsWith('narr-')) {
      const idx = parseInt(selectedClipId.slice(5), 10);
      const seg = editDecision.narration?.[idx];
      if (!seg) return null;
      return {
        kind: 'narration' as const,
        index: idx,
        file: seg.file,
        duration: seg.duration,
        start: seg.start,
        timelineOffset: seg.timeline_offset,
        gainDb: seg.gain_db,
        measuredLufs: seg.measured_loudness_lufs ?? null,
      };
    }
    if (selectedClipId === 'music-0') {
      const m = editDecision.music;
      if (!m) return null;
      return {
        kind: 'music' as const,
        index: 0,
        file: m.file,
        duration: m.duration,
        start: m.start,
        timelineOffset: 0,
        gainDb: m.gain_db,
        measuredLufs: m.measured_loudness_lufs ?? null,
      };
    }
    return null;
  }, [selectedClipId, editDecision]);

  const handleAudioGainChange = useCallback((newGain: number) => {
    if (!editDecision || !selectedAudio) return;
    if (selectedAudio.kind === 'narration') {
      const newNarration = [...(editDecision.narration ?? [])];
      newNarration[selectedAudio.index] = {
        ...newNarration[selectedAudio.index],
        gain_db: newGain,
      };
      onEditDecisionChange({ ...editDecision, narration: newNarration });
    } else if (selectedAudio.kind === 'music' && editDecision.music) {
      onEditDecisionChange({
        ...editDecision,
        music: { ...editDecision.music, gain_db: newGain },
      });
    }
  }, [editDecision, selectedAudio, onEditDecisionChange]);

  const tlWidth = editDecision?.timeline?.width ?? 1920;
  const tlHeight = editDecision?.timeline?.height ?? 1080;

  const handleTransformChange = useCallback((clipId: string, transform: TransformSettings) => {
    if (!editDecision) return;
    const updated = {
      ...editDecision,
      clips: editDecision.clips.map(c =>
        c.id === clipId ? { ...c, transform, transform_mode: 'custom' as const } : c
      ),
    };
    onEditDecisionChange(updated);
  }, [editDecision, onEditDecisionChange]);

  const handleResetTransform = useCallback((clipId: string) => {
    if (!editDecision) return;
    const clip = editDecision.clips.find(c => c.id === clipId);
    if (!clip) return;
    const srcW = clip.source_width || 1920;
    const fitScale = tlWidth / srcW;
    const resetTransform: TransformSettings = {
      scale_x: fitScale, scale_y: fitScale,
      position_x: 0, position_y: 0, rotation: 0,
    };
    const updated = {
      ...editDecision,
      clips: editDecision.clips.map(c =>
        c.id === clipId ? { ...c, transform: resetTransform, transform_mode: 'fit' as const } : c
      ),
    };
    onEditDecisionChange(updated);
  }, [editDecision, onEditDecisionChange, tlWidth]);

  const handleResolutionChange = useCallback((w: number, h: number) => {
    if (!editDecision) return;
    const updated = {
      ...editDecision,
      timeline: { ...(editDecision.timeline || {}), width: w, height: h },
      clips: editDecision.clips.map(c => {
        if (c.transform_mode === 'fit' || !c.transform_mode) {
          const srcW = c.source_width || 1920;
          const fitScale = w / srcW;
          return {
            ...c,
            transform: { scale_x: fitScale, scale_y: fitScale, position_x: 0, position_y: 0, rotation: 0 },
          };
        }
        return c;
      }),
    };
    onEditDecisionChange(updated);
  }, [editDecision, onEditDecisionChange]);

  async function handleIndex(files?: string[]) {
    // Use the WebSocket path so progress streams to the indexing banner.
    // Falls back to REST if onTriggerIndex isn't wired (shouldn't happen).
    if (onTriggerIndex) {
      onTriggerIndex(files);
      setManageOpen(false);
      toast(
        files && files.length
          ? `Re-indexing ${files[0]}${files.length > 1 ? ` (+${files.length - 1} more)` : ''}...`
          : 'Indexing started — watch progress in the Media Pool',
        'success',
      );
      return;
    }
    setIndexing(true);
    try {
      await indexProject(project.project_name);
      toast('Indexing started', 'success');
      await refresh();
    } catch (e: any) {
      toast(`Indexing failed: ${e.message ?? String(e)}`, 'error');
    }
    finally { setIndexing(false); }
  }

  const padContent = scratchpads[activePad as keyof ScratchpadState] || '';
  const isIndexed = project.status !== 'new';

  // ── RENDER ──

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>

      {/* ════════════════════════════════════════════════════════════
          ROW 1: TOOLBAR (36px fixed)
          ════════════════════════════════════════════════════════════ */}
      <div style={{
        height: '36px',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '0 12px',
        background: 'rgba(22,17,16,0.95)',
        borderBottom: '1px solid var(--border)',
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
      }}>
        {/* Left cluster: back + project name */}
        <button
          onClick={onBack}
          style={{
            ...toolbarBtnStyle,
            display: 'flex', alignItems: 'center', gap: '4px',
          }}
        >
          <span style={{ fontSize: '12px' }}>&larr;</span> Back
        </button>

        <div style={{ width: '1px', height: '18px', background: 'var(--border)', flexShrink: 0 }} />

        <span style={{
          fontWeight: 700,
          fontSize: '12px',
          letterSpacing: '-0.02em',
          color: 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          maxWidth: '180px',
        }}>
          {project.project_name}
        </span>
        <span style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          workspace
        </span>

        <div style={{ flex: 1 }} />

        {/* Right cluster: status pills + manage */}
        <span className="pill" style={{ fontSize: '9px', padding: '2px 8px' }}>
          <span style={{ color: 'var(--accent-blue)', fontWeight: 800 }}>{project.video_count}</span> footage
        </span>

        <span className="pill" style={{ fontSize: '9px', padding: '2px 8px' }}>
          <span style={{ color: isIndexed ? 'var(--accent-green)' : 'var(--text-muted)', fontWeight: 800 }}>{project.indexed_files?.length || 0}</span> indexed
        </span>

        {/* Main-LLM switcher — affects the agent loop, not video tasks. */}
        <div style={{ position: 'relative' }}>
          <button
            ref={modelBtnRef}
            title={systemInfo ? `Main LLM: ${systemInfo.main_llm}\nVideo LLM: ${systemInfo.video_llm}` : 'Model'}
            onClick={() => {
              setModelOpen(o => {
                if (!o && modelBtnRef.current) {
                  const rect = modelBtnRef.current.getBoundingClientRect();
                  setModelDropdownPos({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
                }
                return !o;
              });
            }}
            className="pill"
            style={{
              fontSize: '9px',
              padding: '2px 8px',
              background: modelOpen ? 'rgba(255,255,255,0.08)' : undefined,
              cursor: systemInfo ? 'pointer' : 'default',
              border: '1px solid rgba(255,255,255,0.08)',
              display: 'inline-flex', alignItems: 'center', gap: '4px',
            }}
            disabled={!systemInfo}
          >
            <span style={{ color: 'var(--accent-blue)', fontWeight: 800 }}>
              {shortModelLabel(systemInfo?.main_llm)}
            </span>
            <span style={{ color: 'var(--text-muted)' }}>{'\u25BE'}</span>
          </button>

          {modelOpen && modelDropdownPos && systemInfo && createPortal(
            <div ref={modelRef} style={{
              position: 'fixed',
              top: modelDropdownPos.top,
              right: modelDropdownPos.right,
              zIndex: 1000,
              background: 'rgba(28,25,23,0.97)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              padding: '6px',
              minWidth: '240px',
              maxHeight: 'min(420px, calc(100vh - 80px))',
              overflowY: 'auto',
              display: 'grid',
              gap: '1px',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              backdropFilter: 'blur(12px)',
            }}>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '2px 8px 4px' }}>
                Main agent LLM
              </div>
              {systemInfo.available_main_models.map(m => {
                const active = m.id === systemInfo.main_llm;
                return (
                  <button
                    key={m.id}
                    disabled={modelLoading !== null || active}
                    onClick={async () => {
                      if (active) return;
                      setModelLoading(m.id);
                      try {
                        await setMainModel(m.id);
                        const fresh = await getSystemInfo();
                        setSystemInfo(fresh);
                        toast(`Switched to ${m.name}`, 'success');
                        setModelOpen(false);
                      } catch (e: any) {
                        toast(`Failed to switch model: ${e.message}`, 'error');
                      } finally {
                        setModelLoading(null);
                      }
                    }}
                    style={{
                      textAlign: 'left',
                      padding: '5px 8px',
                      borderRadius: 'var(--radius-sm)',
                      border: 'none',
                      background: active ? 'rgba(96,165,250,0.15)' : 'transparent',
                      color: 'var(--text-primary)',
                      cursor: active || modelLoading ? 'default' : 'pointer',
                      opacity: modelLoading && !active ? 0.4 : 1,
                      display: 'grid',
                      gap: '1px',
                    }}
                  >
                    <div style={{ fontSize: '11px', fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                      <span>{m.name}</span>
                      {active && <span style={{ fontSize: '9px', color: 'var(--accent-blue)' }}>active</span>}
                      {modelLoading === m.id && <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>switching…</span>}
                    </div>
                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', lineHeight: 1.3 }}>{m.hint}</div>
                  </button>
                );
              })}
              <div style={{ borderTop: '1px solid var(--border)', margin: '4px 0' }} />
              <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '2px 8px 4px' }}>
                Video LLM (refine / verify)
              </div>
              {systemInfo.available_video_models.map(m => {
                const active = m.id === systemInfo.video_llm;
                return (
                  <button
                    key={m.id}
                    disabled={modelLoading !== null || active}
                    onClick={async () => {
                      if (active) return;
                      setModelLoading(m.id);
                      try {
                        await setVideoModel(m.id);
                        const fresh = await getSystemInfo();
                        setSystemInfo(fresh);
                        toast(`Video LLM → ${m.name}`, 'success');
                        setModelOpen(false);
                      } catch (e: any) {
                        toast(`Failed to switch video LLM: ${e.message}`, 'error');
                      } finally {
                        setModelLoading(null);
                      }
                    }}
                    style={{
                      textAlign: 'left',
                      padding: '5px 8px',
                      borderRadius: 'var(--radius-sm)',
                      border: 'none',
                      background: active ? 'rgba(96,165,250,0.15)' : 'transparent',
                      color: 'var(--text-primary)',
                      cursor: active || modelLoading ? 'default' : 'pointer',
                      opacity: modelLoading && !active ? 0.4 : 1,
                      display: 'grid',
                      gap: '1px',
                    }}
                  >
                    <div style={{ fontSize: '11px', fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                      <span>{m.name}</span>
                      {active && <span style={{ fontSize: '9px', color: 'var(--accent-blue)' }}>active</span>}
                      {modelLoading === m.id && <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>switching…</span>}
                    </div>
                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', lineHeight: 1.3 }}>{m.hint}</div>
                  </button>
                );
              })}
            </div>,
            document.body
          )}
        </div>

        {/* Undo / redo — affects the edit_decision only. Keyboard: Cmd/Ctrl+Z, Cmd+Shift+Z (Ctrl+Y). */}
        <button
          title={`Undo (${isMacLike() ? '⌘Z' : 'Ctrl+Z'})`}
          onClick={onUndo}
          disabled={!canUndo}
          aria-label="Undo"
          style={{
            ...toolbarBtnStyle,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '28px', height: '28px', padding: 0, fontSize: '15px',
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'transparent',
            opacity: canUndo ? 1 : 0.35,
            cursor: canUndo ? 'pointer' : 'default',
          }}
        >
          {'\u21A9'}
        </button>
        <button
          title={`Redo (${isMacLike() ? '⌘⇧Z' : 'Ctrl+Y'})`}
          onClick={onRedo}
          disabled={!canRedo}
          aria-label="Redo"
          style={{
            ...toolbarBtnStyle,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '28px', height: '28px', padding: 0, fontSize: '15px',
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'transparent',
            opacity: canRedo ? 1 : 0.35,
            cursor: canRedo ? 'pointer' : 'default',
          }}
        >
          {'\u21AA'}
        </button>

        <div style={{ position: 'relative' }}>
          <button
            ref={manageBtnRef}
            title="Settings"
            onClick={() => {
              setManageOpen(o => {
                if (!o && manageBtnRef.current) {
                  const rect = manageBtnRef.current.getBoundingClientRect();
                  setDropdownPos({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
                }
                return !o;
              });
              setManageMsg('');
            }}
            style={{
              ...toolbarBtnStyle,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: '28px', height: '28px', padding: 0,
              fontSize: '14px',
              borderRadius: '6px',
              border: '1px solid rgba(255,255,255,0.08)',
              background: manageOpen ? 'rgba(255,255,255,0.08)' : 'transparent',
            }}
          >
            &#x2699;
          </button>

          {/* Dropdown rendered via portal */}
          {manageOpen && dropdownPos && createPortal(
            <div ref={manageRef} style={{
              position: 'fixed',
              top: dropdownPos.top,
              right: dropdownPos.right,
              zIndex: 1000,
              background: 'rgba(28,25,23,0.97)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              padding: '12px',
              minWidth: '220px',
              display: 'grid',
              gap: '6px',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              backdropFilter: 'blur(12px)',
            }}>
              {manageMsg && <div className="status-message info" style={{ fontSize: '11px', padding: '6px 10px' }}>{manageMsg}</div>}
              <DropdownBtn color="var(--text-secondary)" disabled={manageLoading === 'gists'} onClick={async () => { setManageLoading('gists'); try { await clearGists(project.project_name); setManageMsg('Gists cleared.'); toast('Gists cleared', 'success'); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); toast(`Failed to clear gists: ${e.message}`, 'error'); } finally { setManageLoading(null); } }}>
                {manageLoading === 'gists' ? 'Clearing...' : 'Clear gists'}
              </DropdownBtn>
              <DropdownBtn color="var(--text-secondary)" disabled={manageLoading === 'planning'} onClick={async () => { setManageLoading('planning'); onClearState(); try { await clearPlanning(project.project_name); setManageMsg('Planning + chat cleared.'); toast('Planning + chat cleared', 'success'); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); toast(`Failed to clear planning: ${e.message}`, 'error'); } finally { setManageLoading(null); } }}>
                {manageLoading === 'planning' ? 'Clearing...' : 'Clear planning + chat'}
              </DropdownBtn>
              <div style={{ borderTop: '1px solid var(--border)', margin: '4px 0' }} />
              <DropdownBtn color="var(--accent-red)" disabled={manageLoading === 'memories'} onClick={async () => { if (!confirm(`Delete Memories.ai uploads for "${project.project_name}"?`)) return; setManageLoading('memories'); try { const r = await clearMemories(project.project_name); setManageMsg(`Deleted ${r.deleted.length} video(s).`); toast(`Deleted ${r.deleted.length} video(s) from Memories.ai`, 'success'); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); toast(`Failed to delete from Memories.ai: ${e.message}`, 'error'); } finally { setManageLoading(null); } }}>
                {manageLoading === 'memories' ? 'Deleting...' : 'Delete from Memories.ai'}
              </DropdownBtn>
            </div>,
            document.body
          )}
        </div>

        <span className="pill" style={{ fontSize: '9px', padding: '2px 8px' }} title={resolveRunning === true ? 'DaVinci Resolve running' : resolveRunning === false ? 'DaVinci Resolve not detected' : 'Checking Resolve status...'}>
          <span
            style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: resolveRunning === true ? 'var(--accent-green)' : resolveRunning === false ? 'var(--accent-red)' : 'var(--text-muted)' }}
          />
          {resolveRunning === true ? 'Resolve' : resolveRunning === false ? 'No Resolve' : '?'}
        </span>

        <span className="pill" style={{ fontSize: '9px', padding: '2px 8px' }} title={connected ? 'WebSocket connected' : 'Disconnected'}>
          <span
            className={connected ? 'pulse-dot' : ''}
            style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: connected ? 'var(--accent-green)' : 'var(--accent-red)' }}
          />
          {connected ? 'OK' : 'Off'}
        </span>
      </div>

      {/* ════════════════════════════════════════════════════════════
          MAIN CONTENT (fills remaining space)
          ════════════════════════════════════════════════════════════ */}
      <div
        ref={mainContentRef}
        className="glass-card"
        style={{
          flex: 1,
          minHeight: 0,
          margin: '0',
          borderRadius: '0',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >

        {/* ── ROW 2: UPPER — Media Pool | Viewer | Inspector ── */}
        <div style={{
          height: isTablet ? '50%' : `${upperPct}%`,
          flexShrink: 0,
          display: 'flex',
          overflow: 'hidden',
        }}>

          {/* LEFT PANEL: Media Pool + Scratchpads */}
          {!isTablet && (
            <div style={{
              width: '22%',
              minWidth: '220px',
              maxWidth: '360px',
              borderRight: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}>
              {/* Media Pool header — with index button + progress */}
              <MediaPoolHeader
                hasFootage={project.footage_files.length > 0}
                allIndexed={project.footage_files.length > 0 && project.footage_files.every(f => project.indexed_files?.includes(f))}
                anyIndexed={(project.indexed_files?.length ?? 0) > 0}
                indexingState={indexingState}
                connected={connected}
                reindexingFiles={reindexingFiles}
                onIndex={() => handleIndex()}
              />
              {indexingState?.status === 'error' && !indexingState.message?.includes('progress') && (
                <div style={{
                  padding: '6px 10px',
                  fontSize: '10px',
                  color: 'var(--accent-red)',
                  background: 'rgba(255,80,80,0.06)',
                  borderBottom: '1px solid var(--border)',
                  lineHeight: 1.5,
                }}>
                  {indexingState.message}
                </div>
              )}

              {/* Footage files list */}
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                {project.footage_files.length === 0 ? (
                  <div style={{ padding: '12px 10px', color: 'var(--text-muted)', fontSize: '10px', fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
                    No footage yet. Drop files into <code>data/workspaces/{project.project_name}/footage/</code>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
                    {project.footage_files.map((f) => {
                      const indexed = project.indexed_files?.includes(f);
                      const meta = videoMeta?.[f];
                      const isReindexing = reindexingFiles?.has(f) ?? false;
                      const fileGist = project.video_gists?.[f] ?? '';
                      const thumbUrl = `/video-edit/v2/projects/${encodeURIComponent(project.project_name)}/footage/${encodeURIComponent(f)}/thumbnail`;
                      const isExpanded = expandedFile === f;
                      return (
                        <div key={f}>
                          <button
                            onClick={() => fileGist && setExpandedFile(isExpanded ? null : f)}
                            style={{
                              width: '100%',
                              padding: '4px 8px',
                              background: isExpanded ? 'rgba(255,255,255,0.04)' : 'transparent',
                              border: 'none',
                              color: 'var(--text-secondary)',
                              fontSize: '10px',
                              fontFamily: 'var(--font-mono)',
                              cursor: fileGist ? 'pointer' : 'default',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              textAlign: 'left',
                            }}
                          >
                            <img
                              src={thumbUrl}
                              alt=""
                              style={{
                                width: '32px', height: '20px', objectFit: 'cover',
                                borderRadius: '3px', background: 'rgba(0,0,0,0.3)', flexShrink: 0,
                              }}
                              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                            />
                            <FileIndexStatus
                              filename={f}
                              indexed={!!indexed}
                              isReindexing={isReindexing}
                              indexedAt={meta?.indexed_at ?? null}
                              connected={connected}
                              indexingRunning={indexingState?.status === 'running'}
                              onReindex={() => handleIndex([f])}
                            />
                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{f}</span>
                            {fileGist && <span style={{ fontSize: '8px', color: 'var(--text-muted)', flexShrink: 0 }}>{isExpanded ? '\u25BE' : '\u25B8'}</span>}
                          </button>
                          {isExpanded && fileGist && (
                            <div style={{
                              padding: '6px 10px 8px 46px',
                              color: 'var(--text-secondary)',
                              fontSize: '10px',
                              lineHeight: 1.7,
                              whiteSpace: 'pre-wrap',
                              background: 'rgba(0,0,0,0.1)',
                              borderBottom: '1px solid var(--border)',
                            }}>
                              {fileGist}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Divider between media pool and scratchpads */}
              <div style={{ borderTop: '1px solid var(--border)', flexShrink: 0 }} />

              {/* Scratchpad section header */}
              <div style={{
                padding: '6px 10px',
                borderBottom: '1px solid var(--border)',
                fontSize: '9px',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--text-muted)',
                flexShrink: 0,
              }}>
                Scratchpads
              </div>

              {/* Scratchpad tabs */}
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0, overflowX: 'auto' }}>
                {Object.keys(PAD_LABELS).map((key) => {
                  const ts = scratchpadTimestamps[key as keyof ScratchpadTimestamps];
                  const ago = timeAgo(ts);
                  return (
                    <button
                      key={key}
                      onClick={() => setActivePad(key)}
                      style={{
                        flex: 1,
                        padding: '5px 2px 4px',
                        background: activePad === key ? 'rgba(255,255,255,0.04)' : 'transparent',
                        border: 'none',
                        borderBottom: activePad === key ? `2px solid ${PAD_COLORS[key]}` : '2px solid transparent',
                        color: activePad === key ? PAD_COLORS[key] : 'var(--text-muted)',
                        fontSize: '8px',
                        fontFamily: 'var(--font-mono)',
                        letterSpacing: '0.04em',
                        textTransform: 'uppercase',
                        cursor: 'pointer',
                        whiteSpace: 'nowrap',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '1px',
                      }}
                    >
                      {PAD_LABELS[key]}
                      {ago && (
                        <span style={{ fontSize: '7px', opacity: 0.5, textTransform: 'none', letterSpacing: '0.02em' }}>
                          {ago}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Scratchpad content */}
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px 10px', fontSize: '11px' }}>
                {padContent ? (
                  <SimpleMarkdown text={padContent} />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '10px', padding: '4px 0', lineHeight: 1.8 }}>
                    {PAD_EMPTY_HINTS[activePad] || 'No content yet.'}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* CENTER PANEL: Program Viewer */}
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {selectedClip ? (
              <TransformPreview
                clip={selectedClip}
                projectName={project.project_name}
                timelineWidth={tlWidth}
                timelineHeight={tlHeight}
                cropStatus={cropStatuses[selectedClipId!]}
              />
            ) : (
              <PreviewPanel
                ref={previewPanelRef}
                projectName={project.project_name}
                draftRenderState={draftRenderState}
                finalRenderState={renderState}
                editDecision={editDecision}
                onRequestDraftRender={onRequestDraftRender}
                onRequestFinalRender={onRequestRender}
                onTimeUpdate={setPlayheadTime}
                playheadTime={playheadTime}
                resolveRunning={resolveRunning}
              />
            )}
            {/* Render buttons — always visible when a clip is selected (PreviewPanel has its own) */}
            {selectedClip && editDecision && editDecision.clips.length > 0 && (
              <div style={{
                display: 'flex',
                gap: '4px',
                padding: '4px 8px',
                borderTop: '1px solid rgba(255,255,255,0.06)',
                flexShrink: 0,
                background: 'rgba(255,255,255,0.015)',
              }}>
                <button
                  onClick={onRequestDraftRender}
                  disabled={draftRenderState.status === 'rendering'}
                  style={{
                    flex: 1, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '3px', color: 'var(--text-secondary)', fontSize: '9px', fontFamily: 'var(--font-mono)',
                    padding: '4px 8px', letterSpacing: '0.04em', textTransform: 'uppercase' as const,
                    opacity: draftRenderState.status === 'rendering' ? 0.5 : 1,
                    cursor: draftRenderState.status === 'rendering' ? 'not-allowed' : 'pointer',
                  }}
                >
                  {draftRenderState.status === 'rendering' ? 'Rendering...' : 'Render Draft'}
                </button>
                <button
                  onClick={onRequestRender}
                  disabled={renderState.status === 'rendering' || resolveRunning !== true}
                  style={{
                    flex: 1, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '3px', color: 'var(--text-secondary)', fontSize: '9px', fontFamily: 'var(--font-mono)',
                    padding: '4px 8px', letterSpacing: '0.04em', textTransform: 'uppercase' as const,
                    opacity: (renderState.status === 'rendering' || resolveRunning !== true) ? 0.5 : 1,
                    cursor: (renderState.status === 'rendering' || resolveRunning !== true) ? 'not-allowed' : 'pointer',
                  }}
                  title={resolveRunning !== true ? 'Requires DaVinci Resolve' : undefined}
                >
                  {renderState.status === 'rendering' ? 'Rendering...' : resolveRunning !== true ? 'Export Final (Resolve)' : 'Export Final'}
                </button>
              </div>
            )}
          </div>

          {/* RIGHT PANEL: Inspector */}
          {!isTablet && (
            <div style={{
              width: '22%',
              minWidth: '240px',
              maxWidth: '400px',
              borderLeft: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}>
              {/* Inspector header */}
              <div style={{
                padding: '6px 10px',
                borderBottom: '1px solid var(--border)',
                fontSize: '9px',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--text-muted)',
                flexShrink: 0,
              }}>
                Inspector
              </div>

              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                {selectedAudio ? (
                  <div style={{ padding: '10px' }}>
                    <AudioInspector
                      kind={selectedAudio.kind}
                      file={selectedAudio.file}
                      duration={selectedAudio.duration}
                      start={selectedAudio.start}
                      timelineOffset={selectedAudio.timelineOffset}
                      gainDb={selectedAudio.gainDb}
                      measuredLufs={selectedAudio.measuredLufs}
                      targetLufsOverride={
                        selectedAudio.kind === 'music' && (editDecision?.narration?.length ?? 0) > 0
                          ? -28
                          : null
                      }
                      onGainChange={handleAudioGainChange}
                      onClose={() => setSelectedClipId(null)}
                    />
                  </div>
                ) : selectedClip ? (
                  <>
                    <ClipInspector
                      clip={selectedClip}
                      cropStatus={cropStatuses[selectedClipId!]}
                      timelineWidth={tlWidth}
                      timelineHeight={tlHeight}
                      onTransformChange={handleTransformChange}
                      onResetTransform={handleResetTransform}
                      onRequestCrop={onRequestCropClip}
                      onClose={() => setSelectedClipId(null)}
                    />
                    {editDecision && editDecision.clips.length > 0 && (
                      <div style={{ padding: '8px 10px', borderTop: '1px solid var(--border)' }}>
                        <button
                          onClick={onRequestRender}
                          disabled={renderState.status === 'rendering'}
                          style={{
                            width: '100%',
                            padding: '5px 10px',
                            background: renderState.status === 'rendering' ? 'var(--surface-2)' : 'var(--accent-blue)',
                            color: '#fff',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: renderState.status === 'rendering' ? 'not-allowed' : 'pointer',
                            fontFamily: 'var(--font-mono)',
                            fontSize: 11,
                          }}
                        >
                          {renderState.status === 'rendering' ? 'Rendering...' : '\u21BB Render Preview'}
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  /* Edit-level settings when no clip selected */
                  <div style={{ padding: '10px', overflow: 'hidden', minWidth: 0 }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '12px', fontWeight: 600 }}>
                      Timeline Settings
                    </div>

                    {editDecision && editDecision.clips.length > 0 ? (
                      <>
                        <TimelineSettingsBar
                          width={tlWidth}
                          height={tlHeight}
                          onChange={handleResolutionChange}
                          compact
                        />

                        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                          <div style={inspectorRowStyle}>
                            <span style={inspectorLabelStyle}>Clips</span>
                            <span style={inspectorValueStyle}>{editDecision.clips.length}</span>
                          </div>
                          <div style={inspectorRowStyle}>
                            <span style={inspectorLabelStyle}>FPS</span>
                            <span style={inspectorValueStyle}>{editDecision.timeline?.fps ?? 24}</span>
                          </div>
                          {editDecision.timeline?.name && (
                            <div style={inspectorRowStyle}>
                              <span style={inspectorLabelStyle}>Name</span>
                              <span style={{ ...inspectorValueStyle, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {editDecision.timeline.name}
                              </span>
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div style={{ color: 'var(--text-muted)', fontSize: '10px', fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
                        No edit decision yet. Start a conversation with the agent to create one.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── Horizontal divider between upper and lower rows ── */}
        {!isTablet && (
          <div
            onMouseDown={onRowDrag}
            style={hDividerStyle}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
              const inner = e.currentTarget.firstChild as HTMLElement;
              if (inner) { inner.style.background = 'rgba(255,255,255,0.18)'; inner.style.width = '64px'; }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              const inner = e.currentTarget.firstChild as HTMLElement;
              if (inner) { inner.style.background = 'rgba(255,255,255,0.08)'; inner.style.width = '48px'; }
            }}
          >
            <div style={hDividerInnerStyle} />
          </div>
        )}

        {/* ── ROW 3: LOWER — Timeline | Chat ── */}
        <div
          ref={lowerRowRef}
          style={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            overflow: 'hidden',
          }}
        >

          {/* Timeline area */}
          <div style={{
            width: isTablet ? '100%' : `${timelinePct}%`,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
              <NLETimeline
                editDecision={editDecision}
                playheadTime={playheadTime}
                selectedClipId={selectedClipId}
                cropStatuses={cropStatuses}
                onSeek={handleTimelineSeek}
                onEditDecisionChange={onEditDecisionChange}
                onEditDecisionPreview={onEditDecisionPreview}
                onClipSelect={setSelectedClipId}
              />
            </div>
          </div>

          {/* Vertical divider between timeline and chat */}
          {!isTablet && (
            <div
              onMouseDown={onColDragDown}
              style={{
                width: '6px',
                cursor: 'col-resize',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                background: 'transparent',
                transition: 'background 0.15s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
                const inner = e.currentTarget.firstChild as HTMLElement;
                if (inner) { inner.style.background = 'rgba(255,255,255,0.2)'; inner.style.height = '40px'; }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                const inner = e.currentTarget.firstChild as HTMLElement;
                if (inner) { inner.style.background = 'rgba(255,255,255,0.08)'; inner.style.height = '28px'; }
              }}
            >
              <div style={{
                width: '2px', height: '28px', borderRadius: '999px',
                background: 'rgba(255,255,255,0.08)', transition: 'background 0.15s, height 0.15s',
              }} />
            </div>
          )}

          {/* Chat area */}
          {!isTablet && (
            <div style={{
              flex: 1,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              borderLeft: '1px solid var(--border)',
              overflow: 'hidden',
            }}>
              {/* Reconnection banner */}
              {!connected && (
                <div className="reconnect-banner">
                  <span className="activity-spinner" style={{ width: '10px', height: '10px', borderWidth: '1.5px' }} />
                  Reconnecting...
                </div>
              )}

              {/* Messages */}
              <div style={{
                flex: 1,
                minHeight: 0,
                overflowY: 'auto',
                padding: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
              }}>
                {timeline.length === 0 && !busy && !needsIndexing && (
                  <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '40px', fontSize: '12px', lineHeight: 1.8 }}>
                    Start a conversation to begin editing.<br />
                    The agent has access to your indexed footage.
                  </div>
                )}

                {timeline.map((item, i) => {
                  if (item.kind === 'message') {
                    const m = item.message!;
                    const isUser = m.role === 'user';
                    return (
                      <div
                        key={i}
                        className="slide-in"
                        style={{
                          alignSelf: isUser ? 'flex-end' : 'flex-start',
                          maxWidth: '90%',
                          padding: '8px 12px',
                          borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                          background: isUser
                            ? 'linear-gradient(135deg, rgba(96,213,200,0.18), rgba(241,191,99,0.12))'
                            : 'rgba(255,255,255,0.04)',
                          border: `1px solid ${isUser ? 'rgba(96,213,200,0.22)' : 'rgba(255,255,255,0.06)'}`,
                          color: 'var(--text-primary)',
                          fontSize: '12px',
                          lineHeight: 1.7,
                          wordBreak: 'break-word',
                        }}
                      >
                        {isUser ? m.text : <SimpleMarkdown text={m.text} />}
                        {m.timestamp && (
                          <div style={{ textAlign: 'right', marginTop: '3px', fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', opacity: 0.7, letterSpacing: '0.02em' }}>
                            {formatTime(m.timestamp)}
                          </div>
                        )}
                      </div>
                    );
                  }

                  if (item.kind === 'tool_call') {
                    const done = item.completed;
                    const ts = formatTime(item.event!.data.timestamp);
                    return (
                      <div key={i} style={toolCallStyle}>
                        {done
                          ? <span style={{ color: 'var(--accent-green)', fontSize: '10px', width: '10px', textAlign: 'center' }}>&#10003;</span>
                          : <span className="activity-spinner" style={{ color: 'var(--accent-blue)', width: '10px', height: '10px', borderWidth: '1.5px' }} />}
                        <span style={{ color: done ? 'var(--text-muted)' : undefined, flex: 1 }}>
                          {toolLabel(item.event!.data.tool, item.event!.data.args)}
                        </span>
                        {ts && <span style={{ fontSize: '9px', color: 'var(--text-muted)', opacity: 0.5, flexShrink: 0 }}>{ts}</span>}
                      </div>
                    );
                  }

                  if (item.kind === 'refine_progress') {
                    const done = item.completed;
                    return (
                      <div key={i} style={{ ...toolCallStyle, paddingLeft: '24px', fontSize: '10px', color: done ? 'var(--text-muted)' : 'var(--accent-yellow)' }}>
                        {done
                          ? <span style={{ color: 'var(--accent-green)', fontSize: '8px', width: '8px', textAlign: 'center' }}>&#10003;</span>
                          : <span className="activity-spinner" style={{ color: 'var(--accent-yellow)', width: '8px', height: '8px', borderWidth: '1.5px' }} />}
                        <span>{item.event!.data.message || item.event!.data.step}</span>
                      </div>
                    );
                  }

                  if (item.kind === 'scratchpad_update') {
                    const padName = item.event!.data.name || '';
                    return (
                      <div key={i} style={{ ...toolCallStyle, color: PAD_COLORS[padName] || 'var(--text-muted)' }}>
                        <span style={{ fontSize: '10px' }}>&#9998;</span>
                        <span>Updated {PAD_LABELS[padName] || padName} scratchpad</span>
                      </div>
                    );
                  }

                  return null;
                })}

                {currentActivity && currentActivity.active && (
                  <div className="activity-status slide-in" style={{ alignSelf: 'flex-start', marginTop: '4px' }}>
                    <span className="activity-spinner" />
                    <span style={{ color: 'var(--text-secondary)' }}>{currentActivity.label}</span>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <form onSubmit={handleSubmit} style={{
                padding: '8px 12px',
                borderTop: '1px solid var(--border)',
                display: 'flex',
                gap: '6px',
                flexShrink: 0,
              }}>
                <textarea
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    e.target.style.height = 'auto';
                    e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder={busy ? 'Agent is working...' : 'Type a message...'}
                  rows={1}
                  style={{
                    flex: 1,
                    background: 'rgba(18,15,14,0.55)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontSize: '12px',
                    padding: '8px 10px',
                    resize: 'none',
                    lineHeight: 1.5,
                    fontFamily: 'inherit',
                    minHeight: '34px',
                    maxHeight: '100px',
                    overflow: 'auto',
                  }}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || !connected}
                  style={{
                    ...btnStyle('var(--accent-blue)', !input.trim() || !connected),
                    padding: '8px 14px',
                    alignSelf: 'flex-end',
                  }}
                >
                  Send
                </button>
              </form>
            </div>
          )}
        </div>
      </div>

      {/* ── Tablet/mobile fallback: show chat + scratchpads below ── */}
      {isTablet && (
        <div
          className="glass-card"
          style={{
            margin: '0',
            borderRadius: '0',
            flex: 1,
            minHeight: 0,
            padding: 0,
            display: 'grid',
            gridTemplateColumns: '1fr',
            gridTemplateRows: 'minmax(0, 1fr) minmax(180px, 0.4fr)',
            overflow: 'hidden',
          }}
        >
          {/* Chat panel */}
          <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, borderBottom: '1px solid var(--border)' }}>
            {!connected && (
              <div className="reconnect-banner">
                <span className="activity-spinner" style={{ width: '10px', height: '10px', borderWidth: '1.5px' }} />
                Reconnecting...
              </div>
            )}
            <div style={{
              flex: 1, minHeight: 0, overflowY: 'auto', padding: '12px',
              display: 'flex', flexDirection: 'column', gap: '6px',
            }}>
              {timeline.length === 0 && !busy && (
                <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '40px', fontSize: '12px', lineHeight: 1.8 }}>
                  Start a conversation to begin editing.
                </div>
              )}
              {timeline.map((item, i) => {
                if (item.kind === 'message') {
                  const m = item.message!;
                  const isUser = m.role === 'user';
                  return (
                    <div key={i} className="slide-in" style={{
                      alignSelf: isUser ? 'flex-end' : 'flex-start',
                      maxWidth: '94%', padding: '8px 12px',
                      borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                      background: isUser ? 'linear-gradient(135deg, rgba(96,213,200,0.18), rgba(241,191,99,0.12))' : 'rgba(255,255,255,0.04)',
                      border: `1px solid ${isUser ? 'rgba(96,213,200,0.22)' : 'rgba(255,255,255,0.06)'}`,
                      color: 'var(--text-primary)', fontSize: '12px', lineHeight: 1.7, wordBreak: 'break-word',
                    }}>
                      {isUser ? m.text : <SimpleMarkdown text={m.text} />}
                    </div>
                  );
                }
                if (item.kind === 'tool_call') {
                  const done = item.completed;
                  return (
                    <div key={i} style={toolCallStyle}>
                      {done ? <span style={{ color: 'var(--accent-green)', fontSize: '10px' }}>&#10003;</span>
                        : <span className="activity-spinner" style={{ width: '10px', height: '10px', borderWidth: '1.5px' }} />}
                      <span style={{ color: done ? 'var(--text-muted)' : undefined, flex: 1 }}>{toolLabel(item.event!.data.tool, item.event!.data.args)}</span>
                    </div>
                  );
                }
                return null;
              })}
              {currentActivity && currentActivity.active && (
                <div className="activity-status slide-in" style={{ alignSelf: 'flex-start', marginTop: '4px' }}>
                  <span className="activity-spinner" />
                  <span style={{ color: 'var(--text-secondary)' }}>{currentActivity.label}</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} style={{
              padding: '8px 12px', borderTop: '1px solid var(--border)',
              display: 'flex', gap: '6px', flexShrink: 0,
            }}>
              <textarea
                value={input}
                onChange={(e) => { setInput(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px'; }}
                onKeyDown={handleKeyDown}
                placeholder={busy ? 'Agent is working...' : 'Type a message...'}
                rows={1}
                style={{
                  flex: 1, background: 'rgba(18,15,14,0.55)', border: '1px solid var(--border)',
                  borderRadius: '8px', color: 'var(--text-primary)', fontSize: '12px',
                  padding: '8px 10px', resize: 'none', lineHeight: 1.5, fontFamily: 'inherit',
                  minHeight: '34px', maxHeight: '100px', overflow: 'auto',
                }}
              />
              <button type="submit" disabled={!input.trim() || !connected} style={{ ...btnStyle('var(--accent-blue)', !input.trim() || !connected), padding: '8px 14px', alignSelf: 'flex-end' }}>Send</button>
            </form>
          </div>

          {/* Scratchpad panel for tablet */}
          <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, background: 'rgba(0,0,0,0.12)' }}>
            <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0, overflowX: 'auto' }}>
              {Object.keys(PAD_LABELS).map((key) => (
                <button key={key} onClick={() => setActivePad(key)} style={{
                  flex: '0 0 auto', padding: '6px 8px 4px', background: activePad === key ? 'rgba(255,255,255,0.04)' : 'transparent',
                  border: 'none', borderBottom: activePad === key ? `2px solid ${PAD_COLORS[key]}` : '2px solid transparent',
                  color: activePad === key ? PAD_COLORS[key] : 'var(--text-muted)', fontSize: '9px',
                  fontFamily: 'var(--font-mono)', letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', whiteSpace: 'nowrap',
                }}>{PAD_LABELS[key]}</button>
              ))}
            </div>
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '10px' }}>
              {padContent ? <SimpleMarkdown text={padContent} /> : (
                <div style={{ color: 'var(--text-muted)', fontSize: '10px', padding: '4px 0', lineHeight: 1.8 }}>
                  {PAD_EMPTY_HINTS[activePad] || 'No content yet.'}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Helpers ──

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 0) return 'just now';
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return 'just now';
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// ── Media Pool header (with index button + progress) ──

interface MediaPoolHeaderProps {
  hasFootage: boolean;
  allIndexed: boolean;
  anyIndexed: boolean;
  indexingState?: {
    status: 'idle' | 'running' | 'complete' | 'error';
    percent: number;
    message: string;
    videos_total?: number;
    videos_done?: number;
  };
  connected: boolean;
  reindexingFiles?: Set<string>;
  onIndex: () => void;
}

function MediaPoolHeader({ hasFootage, allIndexed, anyIndexed, indexingState, connected, reindexingFiles, onIndex }: MediaPoolHeaderProps) {
  const running = indexingState?.status === 'running';
  // A "global" run is when reindexingFiles is empty — i.e. we're indexing everything,
  // not just one file. Per-file re-indexes are visualized inline on the row instead.
  const globalRunning = running && (!reindexingFiles || reindexingFiles.size === 0);
  let buttonLabel = 'Index footage';
  if (allIndexed) buttonLabel = 'Re-index all';
  else if (anyIndexed) buttonLabel = 'Index missing';

  return (
    <div style={{
      borderBottom: '1px solid var(--border)',
      flexShrink: 0,
    }}>
      <div style={{
        padding: '6px 10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '8px',
      }}>
        <span style={{
          fontSize: '9px',
          fontFamily: 'var(--font-mono)',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          color: 'var(--text-muted)',
        }}>
          Media Pool
        </span>
        {hasFootage && (
          <button
            onClick={onIndex}
            disabled={!connected || running}
            title={running ? indexingState?.message || 'Indexing...' : buttonLabel}
            style={{
              border: '1px solid rgba(255,255,255,0.12)',
              background: globalRunning ? 'rgba(96,165,250,0.12)' : (anyIndexed ? 'transparent' : 'rgba(96,165,250,0.18)'),
              color: anyIndexed ? 'var(--text-secondary)' : 'var(--accent-blue)',
              fontSize: '9px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
              padding: '3px 8px',
              borderRadius: '4px',
              cursor: (!connected || running) ? 'not-allowed' : 'pointer',
              opacity: (!connected || running) ? 0.6 : 1,
              whiteSpace: 'nowrap',
            }}
          >
            {globalRunning
              ? `Indexing ${Math.round(indexingState?.percent ?? 0)}%`
              : running
                ? 'Working...'
                : buttonLabel}
          </button>
        )}
      </div>
      {globalRunning && (
        <div style={{ padding: '0 10px 6px' }}>
          <div style={{
            width: '100%',
            height: '3px',
            background: 'rgba(255,255,255,0.06)',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${Math.max(2, indexingState?.percent ?? 0)}%`,
              height: '100%',
              background: 'var(--accent-blue)',
              borderRadius: '2px',
              transition: 'width 0.3s ease-out',
            }} />
          </div>
          {indexingState?.videos_total != null && (
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '3px', textAlign: 'right' }}>
              {indexingState.videos_done ?? 0} / {indexingState.videos_total} videos
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Per-file index status icon with click popover ──

interface FileIndexStatusProps {
  filename: string;
  indexed: boolean;
  isReindexing: boolean;
  indexedAt: string | null;
  connected: boolean;
  indexingRunning: boolean;
  onReindex: () => void;
}

function FileIndexStatus({ filename, indexed, isReindexing, indexedAt, connected, indexingRunning, onReindex }: FileIndexStatusProps) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLSpanElement>(null);
  const [popPos, setPopPos] = useState<{ top: number; left: number } | null>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      const t = e.target as Node;
      if (btnRef.current && !btnRef.current.contains(t)) {
        // Check the popover element too via the data attribute
        const popEl = document.querySelector(`[data-fileidx-pop="${filename}"]`);
        if (popEl && popEl.contains(t)) return;
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open, filename]);

  const togglePopover = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPopPos({ top: rect.bottom + 4, left: rect.left });
    }
    setOpen(o => !o);
  };

  // Visual states
  let glyph: React.ReactNode = '\u25CB';      // ○ not indexed
  let color = 'var(--text-muted)';
  let title = 'Not indexed yet — click to index';
  if (isReindexing) {
    glyph = <span className="activity-spinner" style={{ width: '8px', height: '8px', borderWidth: '1.5px' }} />;
    color = 'var(--accent-blue)';
    title = 'Re-indexing...';
  } else if (indexed) {
    glyph = '\u2713';                          // ✓ indexed
    color = 'var(--accent-green)';
    title = indexedAt ? `Indexed ${timeAgo(indexedAt)} — click for options` : 'Indexed — click for options';
  }

  return (
    <>
      <span
        ref={btnRef}
        onClick={togglePopover}
        title={title}
        style={{
          fontSize: '9px',
          color,
          flexShrink: 0,
          width: '14px',
          height: '14px',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '3px',
          cursor: connected && !indexingRunning ? 'pointer' : 'default',
          background: open ? 'rgba(255,255,255,0.08)' : 'transparent',
        }}
      >
        {glyph}
      </span>
      {open && popPos && createPortal(
        <div
          data-fileidx-pop={filename}
          style={{
            position: 'fixed',
            top: popPos.top,
            left: popPos.left,
            zIndex: 1100,
            background: 'rgba(28,25,23,0.97)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '8px 10px',
            minWidth: '180px',
            display: 'grid',
            gap: '6px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            backdropFilter: 'blur(12px)',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <div style={{ color: 'var(--text-bright)', fontWeight: 600, wordBreak: 'break-all' }}>
            {filename}
          </div>
          <div style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            {indexed ? (
              <>
                Status: <span style={{ color: 'var(--accent-green)' }}>indexed</span>
                <br />
                {indexedAt ? `Indexed ${timeAgo(indexedAt)}` : 'No timestamp recorded'}
              </>
            ) : (
              <>Status: <span style={{ color: 'var(--text-muted)' }}>not indexed</span></>
            )}
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setOpen(false);
              onReindex();
            }}
            disabled={!connected || indexingRunning}
            style={{
              border: '1px solid rgba(255,255,255,0.12)',
              background: 'rgba(96,165,250,0.18)',
              color: 'var(--accent-blue)',
              fontSize: '9px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: (!connected || indexingRunning) ? 'not-allowed' : 'pointer',
              opacity: (!connected || indexingRunning) ? 0.6 : 1,
            }}
          >
            {indexed ? 'Re-index this file' : 'Index this file'}
          </button>
        </div>,
        document.body,
      )}
    </>
  );
}

type TimelineItem = {
  kind: 'message' | 'tool_call' | 'tool_result' | 'refine_progress' | 'scratchpad_update';
  message?: ChatMessage;
  event?: AgentEvent;
  completed?: boolean;
};

function toolLabel(toolName: string, args?: Record<string, any>): string {
  switch (toolName) {
    case 'ask_memories':
      return `Asking about footage: "${truncate(args?.question || '', 60)}"`;
    case 'search_footage':
      return `Searching clips: "${truncate(args?.query || '', 60)}"`;
    case 'update_scratchpad':
      return `Updating ${args?.name || 'scratchpad'}`;
    case 'generate_fcpxml': {
      let clipCount = 0;
      try {
        const ed = typeof args?.edit_decision_json === 'string' ? JSON.parse(args.edit_decision_json) : args?.edit_decision_json;
        clipCount = ed?.clips?.length || 0;
      } catch { /* ignore */ }
      return `Generating FCPXML (${clipCount} clips)`;
    }
    case 'refine_clip_timestamps': {
      const file = args?.source_file || '';
      const start = args?.source_start != null ? Number(args.source_start).toFixed(1) : '?';
      const end = args?.source_end != null ? Number(args.source_end).toFixed(1) : '?';
      const target = args?.target_duration != null ? Number(args.target_duration).toFixed(0) : '?';
      return `Refining ${truncate(file, 24)} [${start}s\u2013${end}s] \u2192 ${target}s`;
    }
    case 'generate_narration':
      return `Generating voiceover (${(args?.script || '').split(/\s+/).length} words)`;
    case 'select_music':
      return `Selecting music: "${truncate(args?.prompt || '', 50)}"`;
    case 'message_user':
      return 'Composing message...';
    default:
      return `Running ${toolName}`;
  }
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '...' : s;
}

// ── Styles ──

const hDividerStyle: React.CSSProperties = {
  height: '6px',
  cursor: 'row-resize',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
  position: 'relative',
  background: 'transparent',
  transition: 'background 0.15s',
};

const hDividerInnerStyle: React.CSSProperties = {
  width: '48px',
  height: '3px',
  borderRadius: '999px',
  background: 'rgba(255,255,255,0.08)',
  transition: 'background 0.15s, width 0.15s',
};

const toolCallStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '4px 8px',
  fontSize: '11px',
  fontFamily: 'var(--font-mono)',
  color: 'var(--text-muted)',
  letterSpacing: '0.02em',
};

const toolbarBtnStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  color: 'var(--text-secondary)',
  fontSize: '10px',
  fontFamily: 'var(--font-mono)',
  fontWeight: 600,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  cursor: 'pointer',
  padding: '4px 8px',
  borderRadius: '4px',
};


const inspectorRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '8px',
};

const inspectorLabelStyle: React.CSSProperties = {
  fontSize: '10px',
  color: 'var(--text-muted)',
  fontFamily: 'var(--font-mono)',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
};

const inspectorValueStyle: React.CSSProperties = {
  fontSize: '11px',
  color: 'var(--text-primary)',
  fontFamily: 'var(--font-mono)',
};

function DropdownBtn({ color, disabled = false, children, onClick }: { color: string; disabled?: boolean; children: React.ReactNode; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: '100%',
        padding: '8px 12px',
        background: 'transparent',
        border: 'none',
        borderRadius: '6px',
        color: disabled ? 'var(--text-muted)' : color,
        fontSize: '11px',
        fontWeight: 600,
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.04em',
        cursor: disabled ? 'not-allowed' : 'pointer',
        textAlign: 'left' as const,
        transition: 'background 0.15s',
      }}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
    >
      {children}
    </button>
  );
}

function btnStyle(color: string, disabled = false): React.CSSProperties {
  return {
    padding: '8px 14px',
    background: disabled ? 'rgba(255,255,255,0.03)' : `${color}18`,
    border: `1px solid ${disabled ? 'var(--border)' : `${color}77`}`,
    borderRadius: '999px',
    color: disabled ? 'var(--text-muted)' : color,
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    fontFamily: 'var(--font-mono)',
    cursor: disabled ? 'not-allowed' : 'pointer',
  };
}
