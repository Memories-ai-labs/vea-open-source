import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import type { AgentEvent, ChatMessage, ScratchpadState, ScratchpadTimestamps, EditDecision, RenderState } from '../hooks/useAgentChat';
import type { ProjectSummary } from '../types';
import { listProjects, clearGists, clearPlanning, clearMemories, indexProject } from '../api';
import { useBreakpoint } from '../hooks/useBreakpoint';
import { SimpleMarkdown } from './SimpleMarkdown';
import { NLETimeline } from './NLETimeline';
import { VideoPreview } from './VideoPreview';

interface AgentChatProps {
  project: ProjectSummary;
  events: AgentEvent[];
  messages: ChatMessage[];
  scratchpads: ScratchpadState;
  scratchpadTimestamps: ScratchpadTimestamps;
  editDecision: EditDecision | null;
  renderState: RenderState;
  connected: boolean;
  busy: boolean;
  onSend: (text: string) => void;
  onBack: () => void;
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

// ── Resizable row divider ──
const ROW_MINS = [48, 80, 200]; // min heights for top, middle, bottom
const ROW_MAXS = [400, 400, Infinity]; // max heights

function useDragDivider(
  rowHeights: number[],
  setRowHeights: React.Dispatch<React.SetStateAction<number[]>>,
  dividerIndex: number, // 0 = between row 0 and 1, 1 = between row 1 and 2
) {
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeights = useRef<number[]>([]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startY.current = e.clientY;
    startHeights.current = [...rowHeights];
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientY - startY.current;
      const h = [...startHeights.current];
      const above = dividerIndex;
      const below = dividerIndex + 1;
      let newAbove = h[above] + delta;
      let newBelow = h[below] - delta;

      // Clamp
      newAbove = Math.max(ROW_MINS[above], Math.min(ROW_MAXS[above], newAbove));
      newBelow = Math.max(ROW_MINS[below], Math.min(ROW_MAXS[below], newBelow));

      // Recalculate delta after clamping
      const actualDelta = newAbove - h[above];
      h[above] = h[above] + actualDelta;
      h[below] = h[below] - actualDelta;

      if (h[below] < ROW_MINS[below]) return;
      setRowHeights(h);
    };

    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [rowHeights, setRowHeights, dividerIndex]);

  return onMouseDown;
}

export function AgentChat({
  project: initialProject, events, messages, scratchpads, scratchpadTimestamps, editDecision, renderState, connected, busy, onSend, onBack,
}: AgentChatProps) {
  const [project, setProject] = useState(initialProject);
  const [input, setInput] = useState('');
  const [activePad, setActivePad] = useState<string>('comprehension');
  const [manageOpen, setManageOpen] = useState(false);
  const [manageLoading, setManageLoading] = useState<string | null>(null);
  const [manageMsg, setManageMsg] = useState('');
  const [indexing, setIndexing] = useState(false);
  const [expandedFile, setExpandedFile] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const manageRef = useRef<HTMLDivElement>(null);
  const manageBtnRef = useRef<HTMLButtonElement>(null);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; right: number } | null>(null);
  const isTablet = useBreakpoint(1080);
  const isMobile = useBreakpoint(720);

  // Row heights: [top, middle, bottom]. Bottom uses flex to fill remaining space.
  // We store top and middle as fixed px, bottom is flex: 1.
  const [rowHeights, setRowHeights] = useState<number[]>([120, 140, 500]);
  // Column split for middle row: percentage of width for the timeline (0-100)
  const [timelinePct, setTimelinePct] = useState(70);

  // Tick every 30s to keep "updated X ago" labels fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(iv);
  }, []);

  const onDrag0 = useDragDivider(rowHeights, setRowHeights, 0);
  const onDrag1 = useDragDivider(rowHeights, setRowHeights, 1);

  // Horizontal column drag divider for timeline/preview split
  const colDragging = useRef(false);
  const colContainerRef = useRef<HTMLDivElement>(null);
  const onColDragDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    colDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev: MouseEvent) => {
      if (!colDragging.current || !colContainerRef.current) return;
      const rect = colContainerRef.current.getBoundingClientRect();
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setTimelinePct(Math.max(30, Math.min(85, pct)));
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

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, events]);

  // Current activity
  const currentActivity = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      switch (e.type) {
        case 'tool_call':
          return { label: toolLabel(e.data.tool, e.data.args), active: true };
        case 'tool_result':
          continue;
        case 'done':
        case 'error':
          // Previous round finished — but a new round may be in progress
          if (busy) return { label: 'Thinking...', active: true };
          if (e.type === 'error') return { label: `Error: ${e.data.message}`, active: false };
          return null;
      }
    }
    if (busy) return { label: 'Thinking...', active: true };
    return null;
  }, [events, busy]);

  // Timeline items — merge tool_call + tool_result into single entries
  const timeline = useMemo(() => {
    const items: TimelineItem[] = [];
    let msgIdx = 0;

    // Build a set of tool_call timestamps that have a matching tool_result
    const completedTools = new Set<string>();
    for (const e of events) {
      if (e.type === 'tool_result' && e.data.tool) {
        // Match by tool name + find the latest unmatched tool_call
        completedTools.add(e.data.tool + '|' + e.data.timestamp);
      }
    }

    // Track which tool_calls have been completed (by index)
    let toolCallIdx = 0;
    const toolCallCompleted: boolean[] = [];
    const toolCalls: AgentEvent[] = [];
    const toolResults: AgentEvent[] = [];
    for (const e of events) {
      if (e.type === 'tool_call') toolCalls.push(e);
      if (e.type === 'tool_result') toolResults.push(e);
    }
    // Match tool_calls to tool_results by order (1st call → 1st result of same tool, etc.)
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
      if (!found) toolCallCompleted.push(false);
    }

    toolCallIdx = 0;
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
      // Skip tool_result as separate rows — they're merged into tool_call
      else if (e.type === 'tool_result') continue;
      else if (e.type === 'refine_progress') items.push({ kind: 'refine_progress', event: e });
      else if (e.type === 'scratchpad_update') items.push({ kind: 'scratchpad_update', event: e });
    }
    while (msgIdx < messages.length) {
      items.push({ kind: 'message', message: messages[msgIdx] });
      msgIdx++;
    }
    return items;
  }, [events, messages]);

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

  async function handleIndex() {
    setIndexing(true);
    try {
      await indexProject(project.project_name);
      await refresh();
    } catch { /* ignore */ }
    finally { setIndexing(false); }
  }

  const padContent = scratchpads[activePad as keyof ScratchpadState] || '';
  const isIndexed = project.status !== 'new';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: isMobile ? '10px' : '0' }}>
      {/* ── Top row: compact header bar ── */}
      <div
        className="glass-card"
        style={{
          margin: isMobile ? '0' : '16px 16px 0',
          padding: isMobile ? '10px 14px' : '0',
          flexShrink: 0,
          height: isTablet ? 'auto' : `${rowHeights[0]}px`,
          minHeight: isTablet ? undefined : `${ROW_MINS[0]}px`,
          maxHeight: isTablet ? undefined : `${ROW_MAXS[0]}px`,
          overflow: 'auto',
        }}
      >
        {/* Navigation bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: isMobile ? '0' : '12px 20px',
          flexWrap: isMobile ? 'wrap' : 'nowrap',
        }}>
          {/* Left: back + project name */}
          <button
            onClick={onBack}
            style={{
              ...btnStyle('var(--text-secondary)'),
              padding: '6px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              flexShrink: 0,
            }}
          >
            <span style={{ fontSize: '13px' }}>&larr;</span> Back
          </button>

          <div style={{ width: '1px', height: '20px', background: 'var(--border)', flexShrink: 0 }} />

          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', minWidth: 0, flex: '1 1 0' }}>
            <div
              title="Project name (rename not yet supported)"
              style={{
                fontSize: 'clamp(18px, 2.5vw, 26px)',
                fontWeight: 800,
                letterSpacing: '-0.04em',
                lineHeight: 1.2,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                borderBottom: '1px dashed transparent',
                transition: 'border-color 0.15s',
                cursor: 'default',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-strong)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'transparent'; }}
            >
              {project.project_name}
            </div>
            <span className="eyebrow" style={{ flexShrink: 0 }}>workspace</span>
          </div>

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Right: status pills + manage */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <span className="pill">
              <span style={{ color: 'var(--accent-blue)', fontSize: '13px', fontWeight: 800 }}>{project.video_count}</span>
              footage
            </span>

            <span className="pill">
              <span style={{ color: isIndexed ? 'var(--accent-green)' : 'var(--text-muted)', fontSize: '13px', fontWeight: 800 }}>{project.indexed_files?.length || 0}</span>
              indexed
            </span>

            <div style={{ width: '1px', height: '20px', background: 'var(--border)', flexShrink: 0 }} />

            <div style={{ position: 'relative' }}>
              <button
                ref={manageBtnRef}
                onClick={() => {
                  setManageOpen(o => {
                    if (!o && manageBtnRef.current) {
                      const rect = manageBtnRef.current.getBoundingClientRect();
                      setDropdownPos({ top: rect.bottom + 8, right: window.innerWidth - rect.right });
                    }
                    return !o;
                  });
                  setManageMsg('');
                }}
                style={{
                  ...btnStyle('var(--text-secondary)'),
                  padding: '6px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <span style={{ fontSize: '10px' }}>{manageOpen ? '▾' : '▸'}</span> Manage
              </button>

              {/* Dropdown rendered via portal to escape glass-card containing block */}
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
                  {project.footage_files.length > 0 && (
                    <DropdownBtn color={isIndexed ? 'var(--text-secondary)' : 'var(--accent-blue)'} disabled={indexing} onClick={handleIndex}>
                      {indexing ? 'Indexing...' : isIndexed ? 'Re-index footage' : 'Index footage'}
                    </DropdownBtn>
                  )}
                  <DropdownBtn color="var(--text-secondary)" disabled={manageLoading === 'gists'} onClick={async () => { setManageLoading('gists'); try { await clearGists(project.project_name); setManageMsg('Gists cleared.'); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); } finally { setManageLoading(null); } }}>
                    {manageLoading === 'gists' ? 'Clearing...' : 'Clear gists'}
                  </DropdownBtn>
                  <DropdownBtn color="var(--text-secondary)" disabled={manageLoading === 'planning'} onClick={async () => { setManageLoading('planning'); try { await clearPlanning(project.project_name); setManageMsg('Planning + chat cleared.'); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); } finally { setManageLoading(null); } }}>
                    {manageLoading === 'planning' ? 'Clearing...' : 'Clear planning + chat'}
                  </DropdownBtn>
                  <div style={{ borderTop: '1px solid var(--border)', margin: '4px 0' }} />
                  <DropdownBtn color="var(--accent-red)" disabled={manageLoading === 'memories'} onClick={async () => { if (!confirm(`Delete Memories.ai uploads for "${project.project_name}"?`)) return; setManageLoading('memories'); try { const r = await clearMemories(project.project_name); setManageMsg(`Deleted ${r.deleted.length} video(s).`); await refresh(); } catch (e: any) { setManageMsg(`Error: ${e.message}`); } finally { setManageLoading(null); } }}>
                    {manageLoading === 'memories' ? 'Deleting...' : 'Delete from Memories.ai'}
                  </DropdownBtn>
                </div>,
                document.body
              )}
            </div>

            <span className="pill" title={connected ? 'WebSocket connected' : 'Disconnected'}>
              <span
                className={connected ? 'pulse-dot' : ''}
                style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: connected ? 'var(--accent-green)' : 'var(--accent-red)' }}
              />
              {connected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Footage strip */}
        {project.footage_files.length > 0 && (
          <div style={{
            borderTop: '1px solid var(--border)',
            padding: isMobile ? '8px 14px' : '8px 20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            overflowX: 'auto',
          }}>
            <span className="eyebrow" style={{ flexShrink: 0 }}>footage</span>
            <div style={{ width: '1px', height: '16px', background: 'var(--border)', flexShrink: 0 }} />
            {project.footage_files.map((f) => {
              const indexed = project.indexed_files?.includes(f);
              const fileGist = project.video_gists?.[f] ?? '';
              return (
                <button
                  key={f}
                  onClick={() => fileGist && setExpandedFile(expandedFile === f ? null : f)}
                  style={{
                    flexShrink: 0,
                    padding: '5px 10px',
                    borderRadius: '999px',
                    border: `1px solid ${indexed ? 'rgba(138,210,126,0.22)' : 'rgba(255,255,255,0.06)'}`,
                    background: indexed ? 'rgba(138,210,126,0.06)' : 'rgba(255,255,255,0.03)',
                    color: 'var(--text-secondary)',
                    fontSize: '10px',
                    fontFamily: 'var(--font-mono)',
                    cursor: fileGist ? 'pointer' : 'default',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    maxWidth: '280px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  <span style={{ fontSize: '10px', color: indexed ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                    {indexed ? '✓' : '○'}
                  </span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{f}</span>
                  {fileGist && <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{expandedFile === f ? '▾' : '▸'}</span>}
                </button>
              );
            })}
          </div>
        )}

        {/* Expanded gist panel (if a file is selected) */}
        {expandedFile && project.video_gists?.[expandedFile] && (
          <div style={{
            borderTop: '1px solid var(--border)',
            padding: isMobile ? '10px 14px' : '10px 20px',
            maxHeight: '120px',
            overflowY: 'auto',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <span className="eyebrow" style={{ color: 'var(--accent-green)' }}>gist</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>{expandedFile}</span>
              <div style={{ flex: 1 }} />
              <button
                onClick={() => setExpandedFile(null)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px' }}
              >
                &times;
              </button>
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '11px', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {project.video_gists[expandedFile]}
            </div>
          </div>
        )}

        {/* No footage hint */}
        {project.footage_files.length === 0 && (
          <div style={{
            borderTop: '1px solid var(--border)',
            padding: isMobile ? '8px 14px' : '8px 20px',
            color: 'var(--text-muted)',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
          }}>
            No footage yet — drop files into <code>data/workspaces/{project.project_name}/footage/</code>
          </div>
        )}
      </div>

      {/* Divider between header and timeline */}
      {!isTablet && (
        <div
          onMouseDown={onDrag0}
          style={dividerStyle}
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
          <div style={dividerInnerStyle} />
        </div>
      )}

      {/* ── Middle row: NLE timeline + video preview ── */}
      <div
        ref={colContainerRef}
        className="glass-card"
        style={{
          margin: isMobile ? '10px 0 0' : '0 16px',
          padding: '0',
          flexShrink: 0,
          height: isTablet ? '120px' : `${rowHeights[1]}px`,
          minHeight: isTablet ? '80px' : `${ROW_MINS[1]}px`,
          maxHeight: isTablet ? '160px' : `${ROW_MAXS[1]}px`,
          overflow: 'hidden',
          display: 'flex',
        }}
      >
        {/* Timeline */}
        <div style={{ width: isTablet ? '100%' : `${timelinePct}%`, minWidth: 0, overflow: 'hidden' }}>
          <NLETimeline editDecision={editDecision} />
        </div>

        {/* Column divider */}
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
              width: '2px',
              height: '28px',
              borderRadius: '999px',
              background: 'rgba(255,255,255,0.08)',
              transition: 'background 0.15s, height 0.15s',
            }} />
          </div>
        )}

        {/* Video preview */}
        {!isTablet && (
          <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <VideoPreview
              projectName={project.project_name}
              renderState={renderState}
              hasEditDecision={editDecision !== null && editDecision.clips.length > 0}
            />
          </div>
        )}
      </div>

      {/* Divider */}
      {!isTablet && (
        <div
          onMouseDown={onDrag1}
          style={dividerStyle}
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
          <div style={dividerInnerStyle} />
        </div>
      )}

      {/* ── Bottom row: chat + scratchpads ── */}
      <div
        className="glass-card"
        style={{
          margin: isMobile ? '10px 0 0' : '0 16px 16px',
          flex: 1,
          minHeight: 0,
          padding: 0,
          display: 'grid',
          gridTemplateColumns: isTablet ? '1fr' : '1fr minmax(300px, 0.5fr)',
          gridTemplateRows: isTablet ? 'minmax(0, 1fr) minmax(220px, 0.5fr)' : undefined,
          overflow: 'hidden',
        }}
      >
        {/* Chat panel */}
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, borderRight: isTablet ? 'none' : '1px solid var(--border)', borderBottom: isTablet ? '1px solid var(--border)' : 'none' }}>
          {/* Messages */}
          <div style={{
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}>
            {timeline.length === 0 && !busy && (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '40px', fontSize: '13px', lineHeight: 1.8 }}>
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
                        maxWidth: isMobile ? '94%' : '85%',
                        padding: '10px 14px',
                      borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                      background: isUser
                        ? 'linear-gradient(135deg, rgba(96,213,200,0.18), rgba(241,191,99,0.12))'
                        : 'rgba(255,255,255,0.04)',
                      border: `1px solid ${isUser ? 'rgba(96,213,200,0.22)' : 'rgba(255,255,255,0.06)'}`,
                      color: 'var(--text-primary)',
                      fontSize: '13px',
                      lineHeight: 1.7,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {m.text}
                  </div>
                );
              }

              if (item.kind === 'tool_call') {
                const done = item.completed;
                return (
                  <div key={i} style={toolCallStyle}>
                    {done
                      ? <span style={{ color: 'var(--accent-green)', fontSize: '10px', width: '10px', textAlign: 'center' }}>&#10003;</span>
                      : <span className="activity-spinner" style={{ color: 'var(--accent-blue)', width: '10px', height: '10px', borderWidth: '1.5px' }} />}
                    <span style={{ color: done ? 'var(--text-muted)' : undefined }}>
                      {toolLabel(item.event!.data.tool, item.event!.data.args)}
                    </span>
                  </div>
                );
              }

              if (item.kind === 'refine_progress') {
                return (
                  <div key={i} style={{ ...toolCallStyle, paddingLeft: '24px', fontSize: '10px', color: 'var(--accent-yellow)' }}>
                    <span className="activity-spinner" style={{ color: 'var(--accent-yellow)', width: '8px', height: '8px', borderWidth: '1.5px' }} />
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
            padding: isMobile ? '10px 12px' : '12px 16px',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            flexDirection: isMobile ? 'column' : 'row',
            gap: '8px',
            flexShrink: 0,
          }}>
            <textarea
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                // Auto-grow
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
              }}
              onKeyDown={handleKeyDown}
              placeholder={busy ? 'Agent is working...' : 'Type a message...'}
              rows={1}
              style={{
                flex: 1,
                background: 'rgba(18,15,14,0.55)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                color: 'var(--text-primary)',
                fontSize: '13px',
                padding: '10px 14px',
                resize: 'none',
                lineHeight: 1.5,
                fontFamily: 'inherit',
                minHeight: '38px',
                maxHeight: '120px',
                overflow: 'auto',
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || !connected}
              style={{
                ...btnStyle('var(--accent-blue)', !input.trim() || !connected),
                padding: '10px 16px',
                alignSelf: isMobile ? 'stretch' : 'flex-end',
                width: isMobile ? '100%' : undefined,
              }}
            >
              Send
            </button>
          </form>
        </div>

        {/* Scratchpad panel */}
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, background: 'rgba(0,0,0,0.12)', maxHeight: isTablet ? '40vh' : undefined }}>
          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0, overflowX: 'auto' }}>
            {Object.keys(PAD_LABELS).map((key) => {
              const ts = scratchpadTimestamps[key as keyof ScratchpadTimestamps];
              const ago = timeAgo(ts);
              return (
                <button
                  key={key}
                  onClick={() => setActivePad(key)}
                  style={{
                    flex: isTablet ? '0 0 auto' : 1,
                    padding: '8px 4px 6px',
                    background: activePad === key ? 'rgba(255,255,255,0.04)' : 'transparent',
                    border: 'none',
                    borderBottom: activePad === key ? `2px solid ${PAD_COLORS[key]}` : '2px solid transparent',
                    color: activePad === key ? PAD_COLORS[key] : 'var(--text-muted)',
                    fontSize: '9px',
                    fontFamily: 'var(--font-mono)',
                    letterSpacing: '0.06em',
                    textTransform: 'uppercase',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    minWidth: isTablet ? '120px' : undefined,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                  }}
                >
                  {PAD_LABELS[key]}
                  {ago && (
                    <span style={{
                      fontSize: '7px',
                      opacity: 0.5,
                      textTransform: 'none',
                      letterSpacing: '0.02em',
                    }}>
                      {ago}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Content */}
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '14px' }}>
            {padContent ? (
              <SimpleMarkdown text={padContent} />
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', padding: '8px 0', lineHeight: 1.8 }}>
                {PAD_EMPTY_HINTS[activePad] || 'No content yet.'}
              </div>
            )}
          </div>
        </div>
      </div>
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

type TimelineItem = {
  kind: 'message' | 'tool_call' | 'tool_result' | 'refine_progress' | 'scratchpad_update';
  message?: ChatMessage;
  event?: AgentEvent;
  completed?: boolean;  // for tool_call: has a matching tool_result
};

function toolLabel(toolName: string, args?: Record<string, any>): string {
  switch (toolName) {
    case 'ask_memories':
      return `Asking about footage: "${truncate(args?.question || '', 60)}"`;
    case 'search_footage':
      return `Searching clips: "${truncate(args?.query || '', 60)}"`;
    case 'update_scratchpad':
      return `Updating ${args?.name || 'scratchpad'}`;
    case 'generate_fcpxml':
      return `Generating FCPXML (${args?.shots?.length || 0} shots)`;
    case 'refine_clip_timestamps': {
      const file = args?.source_file || '';
      const start = args?.source_start != null ? Number(args.source_start).toFixed(1) : '?';
      const end = args?.source_end != null ? Number(args.source_end).toFixed(1) : '?';
      const target = args?.target_duration != null ? Number(args.target_duration).toFixed(0) : '?';
      return `Refining ${truncate(file, 24)} [${start}s–${end}s] → ${target}s`;
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

const dividerStyle: React.CSSProperties = {
  height: '8px',
  margin: '0 16px',
  cursor: 'row-resize',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
  position: 'relative',
  background: 'transparent',
  transition: 'background 0.15s',
};

const dividerInnerStyle: React.CSSProperties = {
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
