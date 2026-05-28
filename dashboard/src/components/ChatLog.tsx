import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import type { PlanningEvent } from '../types';

// ── Types ────────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: 'user' | 'model' | 'system';
  kind: 'prompt' | 'inject' | 'reasoning' | 'storyboard_note' | 'tool_call' | 'tool_result';
  text: string;
  timestamp: string;
  iteration?: number;
  ingested?: boolean; // user messages that have been consumed by the planner
  toolType?: string; // 'chat' | 'search'
}

interface QueuedMessage {
  id: number;
  text: string;
  timestamp: string;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function ago(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  return `${Math.round(diff / 60)}m ago`;
}

let _queueId = 0;

// ── Component ────────────────────────────────────────────────────────────────

interface ChatLogProps {
  initialPrompt: string;
  events: PlanningEvent[];
  paused: boolean;
  planningDone: boolean;
  onInject: (prompt: string) => void;
  onPause: () => void;
  onResume: () => void;
}

export function ChatLog({
  initialPrompt,
  events,
  paused,
  planningDone,
  onInject,
  onPause,
  onResume,
}: ChatLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [input, setInput] = useState('');
  const [queue, setQueue] = useState<QueuedMessage[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Track which queued messages have been ingested (consumed by planner)
  const ingestedTexts = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) {
      if (e.event_type === 'prompt_injected' && e.data?.added) {
        for (const p of e.data.added as string[]) {
          set.add(p);
        }
      }
    }
    return set;
  }, [events]);

  // Auto-remove ingested messages from queue
  useEffect(() => {
    if (ingestedTexts.size === 0) return;
    setQueue((prev) => prev.filter((m) => !ingestedTexts.has(m.text)));
  }, [ingestedTexts]);

  // Build the conversation history from events
  const messages = useMemo(() => {
    const msgs: ChatMessage[] = [];

    if (initialPrompt) {
      const firstEvent = events[0];
      msgs.push({
        role: 'user',
        kind: 'prompt',
        text: initialPrompt,
        timestamp: firstEvent?.timestamp ?? new Date().toISOString(),
      });
    }

    for (const e of events) {
      if (e.event_type === 'tool_call_plan' && e.data?.reasoning) {
        msgs.push({
          role: 'model',
          kind: 'reasoning',
          text: e.data.reasoning,
          timestamp: e.timestamp,
          iteration: e.data?.iteration,
        });
      }
      if (e.event_type === 'tool_call') {
        const isChat = e.data?.type === 'chat';
        const label = isChat
          ? `Asking footage index: "${e.data?.question ?? ''}"`
          : `Searching footage: "${e.data?.query ?? ''}"`;
        msgs.push({
          role: 'system',
          kind: 'tool_call',
          text: label,
          timestamp: e.timestamp,
          iteration: e.data?.iteration,
          toolType: e.data?.type,
        });
      }
      if (e.event_type === 'tool_result') {
        const isChat = e.data?.type === 'chat';
        const summary = isChat
          ? (typeof e.data?.answer === 'string' ? e.data.answer.slice(0, 150) : '')
          : `Found ${(e.data?.clips as unknown[])?.length ?? 0} clips`;
        msgs.push({
          role: 'system',
          kind: 'tool_result',
          text: summary,
          timestamp: e.timestamp,
          iteration: e.data?.iteration,
          toolType: e.data?.type,
        });
      }
      if (e.event_type === 'prompt_injected' && e.data?.added) {
        for (const p of e.data.added as string[]) {
          msgs.push({
            role: 'user',
            kind: 'inject',
            text: p,
            timestamp: e.timestamp,
            ingested: true,
          });
        }
      }
      if (e.event_type === 'storyboard_update' && e.data?.notes) {
        msgs.push({
          role: 'model',
          kind: 'storyboard_note',
          text: e.data.notes,
          timestamp: e.timestamp,
          iteration: e.data?.iteration,
        });
      }
    }

    return msgs;
  }, [initialPrompt, events]);

  // Derive current activity from latest event
  const currentActivity = useMemo(() => {
    if (planningDone) return null;
    if (paused) return null;
    if (events.length === 0) return null;

    const last = events[events.length - 1];
    switch (last.event_type) {
      case 'iteration_start':
        return { label: 'Planning next moves…', color: 'var(--accent-yellow)' };
      case 'tool_call_plan':
        return { label: 'Deciding tool calls…', color: 'var(--accent-yellow)' };
      case 'tool_call':
        return last.data?.type === 'chat'
          ? { label: 'Querying footage index…', color: 'var(--accent-blue)' }
          : { label: 'Searching footage…', color: 'var(--accent-green)' };
      case 'tool_result':
        return { label: 'Processing results…', color: 'var(--text-muted)' };
      case 'storyboard_update':
        return { label: 'Updating storyboard…', color: 'var(--accent-purple)' };
      default:
        return null;
    }
  }, [events, planningDone, paused]);

  // Auto-scroll
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, queue.length]);

  const toggleExpand = useCallback((idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  function handleSubmit() {
    const trimmed = input.trim();
    if (!trimmed) return;

    // Queue the message locally and send to backend
    setQueue((prev) => [
      ...prev,
      { id: ++_queueId, text: trimmed, timestamp: new Date().toISOString() },
    ]);
    onInject(trimmed);
    setInput('');
    inputRef.current?.focus();
  }

  function handleDeleteQueued(id: number) {
    setQueue((prev) => prev.filter((m) => m.id !== id));
  }

  function handleStartEdit(msg: QueuedMessage) {
    setEditingId(msg.id);
    setEditText(msg.text);
  }

  function handleSaveEdit(id: number) {
    if (!editText.trim()) {
      handleDeleteQueued(id);
    } else {
      setQueue((prev) =>
        prev.map((m) => (m.id === id ? { ...m, text: editText.trim() } : m))
      );
    }
    setEditingId(null);
    setEditText('');
  }

  function handleCancelEdit() {
    setEditingId(null);
    setEditText('');
  }

  const userBorder = 'var(--accent-blue)';
  const modelBorder = 'var(--accent-yellow)';
  const queuedBorder = 'var(--accent-purple)';

  // Determine the status line
  const isRunning = !paused && !planningDone;
  const pausePending = paused && events.length > 0 && !events.some(
    (e) => e.event_type === 'paused'
  );

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      background: 'linear-gradient(180deg, rgba(35, 27, 24, 0.94), rgba(22, 17, 16, 0.94))',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
      minHeight: 0,
      flex: 1,
      boxShadow: 'var(--shadow-md)',
      backdropFilter: 'blur(18px)',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '14px 16px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01))',
        flexShrink: 0,
      }}>
        <span style={{
          color: 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          fontWeight: 600,
          letterSpacing: '0.16em',
          textTransform: 'uppercase',
        }}>
          Agent Chat
        </span>
        <span style={{
          marginLeft: 'auto',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          color: 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          fontWeight: 700,
          padding: '4px 10px',
          borderRadius: '999px',
          letterSpacing: '0.08em',
        }}>
          {messages.length + queue.length}
        </span>
      </div>

      {/* Scrollable messages area */}
      <div
        ref={scrollRef}
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          padding: '16px',
        }}
      >
        {messages.length === 0 && queue.length === 0 && (
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Planning session will appear here…
          </span>
        )}

        {/* Conversation history */}
        {messages.map((msg, i) => {
          // Tool calls / results: compact inline status
          if (msg.role === 'system') {
            const isResult = msg.kind === 'tool_result';
            const color = msg.toolType === 'chat' ? 'var(--accent-blue)' : 'var(--accent-green)';
            return (
              <div
                key={`msg-${i}`}
                className="slide-in"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '5px 10px',
                  fontSize: '10px',
                  color: isResult ? 'var(--text-muted)' : color,
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.02em',
                }}
              >
                {!isResult && <span className="activity-spinner" style={{ color, width: '10px', height: '10px', borderWidth: '1.5px' }} />}
                {isResult && <span style={{ color: 'var(--accent-green)', fontSize: '10px' }}>✓</span>}
                <span style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                }}>
                  {isResult ? `↳ ${msg.text}` : msg.text}
                </span>
                <span style={{ flexShrink: 0, color: 'var(--text-muted)', fontSize: '9px' }}>
                  {ago(msg.timestamp)}
                </span>
              </div>
            );
          }

          const isModel = msg.role === 'model';
          const border = isModel ? modelBorder : userBorder;
          const isExpanded = expanded.has(i);

          return (
            <div
              key={`msg-${i}`}
              className="slide-in"
              style={{
                padding: '8px 10px',
                borderLeft: `3px solid ${border}`,
                borderRadius: '0 6px 6px 0',
                background: isModel
                  ? 'rgba(241,191,99,0.04)'
                  : 'rgba(96,213,200,0.04)',
                cursor: isModel ? 'pointer' : 'default',
              }}
              onClick={isModel ? () => toggleExpand(i) : undefined}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{
                  fontSize: '9px',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: border,
                }}>
                  {msg.kind === 'prompt'
                    ? 'Brief'
                    : msg.kind === 'inject'
                      ? 'Steering'
                      : msg.kind === 'reasoning'
                        ? `Thinking · iter ${(msg.iteration ?? 0) + 1}`
                        : `Note · iter ${(msg.iteration ?? 0) + 1}`}
                </span>
                {msg.ingested && (
                  <span style={{
                    fontSize: '8px',
                    color: 'var(--accent-green)',
                    fontFamily: 'var(--font-mono)',
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                  }}>
                    ingested
                  </span>
                )}
                <span style={{ marginLeft: 'auto', fontSize: '9px', color: 'var(--text-muted)' }}>
                  {ago(msg.timestamp)}
                </span>
                {isModel && (
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    {isExpanded ? '▾' : '▸'}
                  </span>
                )}
              </div>

              {isModel && !isExpanded ? (
                <div style={{
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  lineHeight: 1.5,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {msg.text.slice(0, 120)}…
                </div>
              ) : (
                <div style={{
                  color: msg.role === 'user' ? 'var(--text-primary)' : 'var(--text-secondary)',
                  fontSize: msg.role === 'user' ? '12px' : '11px',
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap',
                  ...(isModel ? { maxHeight: '200px', overflowY: 'auto' as const } : {}),
                }}>
                  {msg.text}
                </div>
              )}
            </div>
          );
        })}

        {/* Live activity indicator */}
        {currentActivity && (
          <div className="activity-status slide-in" style={{ color: currentActivity.color }}>
            <span className="activity-spinner" style={{ color: currentActivity.color }} />
            {currentActivity.label}
          </div>
        )}

        {/* Queued messages (not yet ingested) */}
        {queue.length > 0 && (
          <>
            <div style={{
              padding: '6px 0',
              color: 'var(--text-muted)',
              fontSize: '9px',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              borderTop: '1px solid rgba(255,255,255,0.06)',
              marginTop: '4px',
            }}>
              Queued — will be ingested next iteration
            </div>
            {queue.map((qm) => (
              <div
                key={`q-${qm.id}`}
                className="slide-in"
                style={{
                  padding: '8px 10px',
                  borderLeft: `3px solid ${queuedBorder}`,
                  borderRadius: '0 6px 6px 0',
                  background: 'rgba(215,155,181,0.06)',
                  border: '1px dashed rgba(215,155,181,0.2)',
                  borderLeftStyle: 'solid',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{
                    fontSize: '9px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    color: queuedBorder,
                  }}>
                    Queued
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: '9px', color: 'var(--text-muted)' }}>
                    {ago(qm.timestamp)}
                  </span>
                </div>

                {editingId === qm.id ? (
                  <div style={{ display: 'grid', gap: '6px' }}>
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      autoFocus
                      rows={3}
                      style={{
                        background: 'rgba(255,255,255,0.04)',
                        border: '1px solid var(--border)',
                        borderRadius: '4px',
                        color: 'var(--text-primary)',
                        fontSize: '12px',
                        padding: '8px',
                        fontFamily: 'inherit',
                        resize: 'vertical',
                        lineHeight: 1.6,
                      }}
                    />
                    <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                      <button onClick={handleCancelEdit} style={tinyBtn('var(--text-muted)')}>
                        Cancel
                      </button>
                      <button onClick={() => handleSaveEdit(qm.id)} style={tinyBtn('var(--accent-green)')}>
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div style={{
                      color: 'var(--text-primary)',
                      fontSize: '12px',
                      lineHeight: 1.6,
                      whiteSpace: 'pre-wrap',
                    }}>
                      {qm.text}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', marginTop: '6px', justifyContent: 'flex-end' }}>
                      <button onClick={() => handleStartEdit(qm)} style={tinyBtn('var(--text-muted)')}>
                        Edit
                      </button>
                      <button onClick={() => handleDeleteQueued(qm.id)} style={tinyBtn('var(--accent-red)')}>
                        Delete
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </>
        )}
      </div>

      {/* Pinned footer: status + input */}
      <div style={{
        flexShrink: 0,
        padding: '12px 16px',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}>
        {/* Status + pause controls */}
        {!planningDone && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            {pausePending ? (
              <span style={{ fontSize: '11px', color: 'var(--accent-yellow)', flex: 1, lineHeight: 1.5 }}>
                Pausing after this iteration finishes…
              </span>
            ) : paused ? (
              <span style={{ fontSize: '11px', color: 'var(--accent-yellow)', flex: 1, lineHeight: 1.5 }}>
                Paused — take your time writing messages
              </span>
            ) : isRunning ? (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', flex: 1 }}>
                Running iteration…
              </span>
            ) : null}

            {!paused ? (
              <button onClick={onPause} style={tinyBtn('var(--accent-yellow)')}>
                Pause after iteration
              </button>
            ) : (
              <button onClick={onResume} style={tinyBtn('var(--accent-green)')}>
                Resume
              </button>
            )}
          </div>
        )}

        {planningDone && (
          <span style={{ fontSize: '11px', color: 'var(--accent-green)' }}>
            Planning complete
          </span>
        )}

        {/* Input area */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder={paused ? 'Write your steering notes…' : 'Queue a message for next iteration…'}
            rows={2}
            style={{
              flex: 1,
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              fontSize: '12px',
              padding: '10px 12px',
              fontFamily: 'inherit',
              resize: 'none',
              lineHeight: 1.5,
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim()}
            style={{
              flexShrink: 0,
              padding: '10px 14px',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: input.trim()
                ? 'rgba(96,213,200,0.12)'
                : 'rgba(255,255,255,0.03)',
              color: input.trim() ? 'var(--accent-blue)' : 'var(--text-muted)',
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              alignSelf: 'flex-end',
            }}
          >
            Queue
          </button>
        </div>
      </div>
    </div>
  );
}

function tinyBtn(color: string): React.CSSProperties {
  return {
    padding: '4px 10px',
    borderRadius: '4px',
    border: `1px solid ${color}44`,
    background: `${color}0a`,
    color,
    cursor: 'pointer',
    fontFamily: 'var(--font-mono)',
    fontSize: '9px',
    fontWeight: 700,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    whiteSpace: 'nowrap',
  };
}
