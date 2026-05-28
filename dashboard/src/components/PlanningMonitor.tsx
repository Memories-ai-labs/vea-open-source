import { useEffect, useRef, useMemo } from 'react';
import type { PlanningEvent, Storyboard, Shot, RetrievedClip } from '../types';
import { Panel } from './Panel';
import { ChatLog } from './ChatLog';

// ─── helpers ────────────────────────────────────────────────────────────────

function scoreColor(score: number): string {
  if (score >= 0.75) return 'var(--accent-green)';
  if (score >= 0.45) return 'var(--accent-yellow)';
  return 'var(--accent-orange)';
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function ago(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  return `${Math.round(diff / 60)}m ago`;
}

// ─── Knowledge Panel ─────────────────────────────────────────────────────────

function KnowledgePanel({ events }: { events: PlanningEvent[] }) {
  const chatResults = useMemo(
    () =>
      events.filter(
        (e) => e.event_type === 'tool_result' && e.data?.type === 'chat'
      ),
    [events]
  );

  const gist = useMemo(() => {
    const gistEvents = events.filter(
      (e) => e.event_type === 'storyboard_update' && e.data?.notes
    );
    return gistEvents.length > 0
      ? gistEvents[gistEvents.length - 1].data.notes
      : null;
  }, [events]);

  return (
    <Panel title="Knowledge" badge={chatResults.length} badgeColor="var(--accent-blue)">
      {gist && (
        <div
          style={{
            marginBottom: '10px',
            padding: '6px 8px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '3px',
            color: 'var(--text-primary)',
            fontSize: '12px',
            lineHeight: 1.5,
          }}
        >
          <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>GIST — </span>
          {gist}
        </div>
      )}
      {chatResults.length === 0 && !gist && (
        <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          Waiting for chat results…
        </span>
      )}
      {chatResults.map((e, i) => (
        <div
          key={i}
          className="slide-in"
          style={{
            marginBottom: '6px',
            padding: '5px 7px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '3px',
            fontSize: '12px',
          }}
        >
          <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginBottom: '2px' }}>
            iter {e.data?.iteration ?? '?'} · {ago(e.timestamp)}
          </div>
          {e.data?.question && (
            <div style={{ color: 'var(--accent-blue)', marginBottom: '3px' }}>
              Q: {e.data.question}
            </div>
          )}
          <div
            style={{
              color: 'var(--text-secondary)',
              lineHeight: 1.45,
              maxHeight: '60px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {typeof e.data?.answer === 'string'
              ? e.data.answer.slice(0, 300)
              : JSON.stringify(e.data?.answer ?? '').slice(0, 300)}
          </div>
        </div>
      ))}
    </Panel>
  );
}

// ─── Storyboard Panel ────────────────────────────────────────────────────────

function StoryboardPanel({ storyboard }: { storyboard: Storyboard | null }) {
  if (!storyboard) {
    return (
      <Panel title="Storyboard">
        <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          No storyboard yet…
        </span>
      </Panel>
    );
  }

  const totalDur = storyboard.shots.reduce((s, sh) => s + sh.duration_seconds, 0);

  return (
    <Panel
      title={`Storyboard · iter ${storyboard.iteration}`}
      badge={`${storyboard.shots.length} shots`}
    >
      {/* Meta row */}
      <div
        style={{
          display: 'flex',
          gap: '12px',
          marginBottom: '8px',
          color: 'var(--text-secondary)',
          fontSize: '11px',
        }}
      >
        <span>{storyboard.theme}</span>
        <span style={{ color: 'var(--text-muted)' }}>|</span>
        <span>{fmtTime(totalDur)} / {fmtTime(storyboard.target_duration_seconds)}</span>
      </div>

      {/* Shots */}
      {storyboard.shots.map((shot) => (
        <ShotRow key={shot.id} shot={shot} />
      ))}

      {/* Open questions */}
      {storyboard.open_questions.length > 0 && (
        <div style={{ marginTop: '8px' }}>
          <div
            style={{
              color: 'var(--text-muted)',
              fontSize: '10px',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              marginBottom: '4px',
            }}
          >
            Open Questions
          </div>
          {storyboard.open_questions.map((q, i) => (
            <div
              key={i}
              style={{
                color: 'var(--accent-yellow)',
                fontSize: '11px',
                marginBottom: '2px',
                paddingLeft: '8px',
                borderLeft: '2px solid var(--accent-yellow)',
              }}
            >
              {q}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function ShotRow({ shot }: { shot: Shot }) {
  const clip = shot.retrieved_clip;
  return (
    <div
      className="slide-in"
      style={{
        display: 'flex',
        gap: '7px',
        marginBottom: '5px',
        padding: '5px 7px',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '3px',
        fontSize: '12px',
        alignItems: 'flex-start',
      }}
    >
      {/* ID badge */}
      <span
        style={{
          background: 'var(--bg-hover)',
          color: 'var(--text-secondary)',
          fontSize: '10px',
          padding: '1px 5px',
          borderRadius: '3px',
          flexShrink: 0,
          fontWeight: 700,
        }}
      >
        {shot.id}
      </span>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            color: 'var(--text-primary)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {shot.purpose}
        </div>
        {clip && (
          <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '1px' }}>
            {clip.video_name} · {fmtTime(clip.start_seconds)}→{fmtTime(clip.end_seconds)}
          </div>
        )}
      </div>

      <div style={{ flexShrink: 0, textAlign: 'right' }}>
        <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
          {shot.duration_seconds.toFixed(1)}s
        </div>
        {clip && (
          <div
            style={{
              fontSize: '10px',
              color: scoreColor(clip.score),
              fontWeight: 700,
            }}
          >
            {(clip.score * 100).toFixed(0)}%
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Tool Call Feed ──────────────────────────────────────────────────────────

function ToolFeedPanel({ events }: { events: PlanningEvent[] }) {
  const feedRef = useRef<HTMLDivElement>(null);
  const feedEvents = useMemo(
    () =>
      events.filter((e) =>
        ['tool_call', 'tool_result', 'tool_error', 'iteration_start', 'tool_call_plan'].includes(
          e.event_type
        )
      ),
    [events]
  );

  useEffect(() => {
    const el = feedRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [feedEvents.length]);

  return (
    <Panel title="Tool Feed" badge={feedEvents.length}>
      <div ref={feedRef} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {feedEvents.length === 0 && (
          <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Waiting for tool calls…
          </span>
        )}
        {feedEvents.map((e, i) => (
          <FeedRow key={i} event={e} />
        ))}
      </div>
    </Panel>
  );
}

function FeedRow({ event }: { event: PlanningEvent }) {
  const { event_type, data, timestamp } = event;

  if (event_type === 'iteration_start') {
    return (
      <div
        className="slide-in"
        style={{
          padding: '3px 0',
          color: 'var(--text-muted)',
          fontSize: '10px',
          borderTop: '1px solid var(--border)',
          marginTop: '2px',
          letterSpacing: '0.04em',
        }}
      >
        ── ITERATION {data?.iteration ?? '?'} ──────────────────────
      </div>
    );
  }

  if (event_type === 'tool_call' || event_type === 'tool_call_plan') {
    const isChat = data?.type === 'chat';
    const color = isChat ? 'var(--accent-blue)' : 'var(--accent-green)';
    const icon = isChat ? '💬' : '🔍';
    const label = isChat ? data?.question : data?.query;
    return (
      <div
        className="slide-in"
        style={{
          padding: '4px 7px',
          background: 'var(--bg-card)',
          borderLeft: `2px solid ${color}`,
          borderRadius: '0 3px 3px 0',
          fontSize: '12px',
        }}
      >
        <span style={{ marginRight: '5px' }}>{icon}</span>
        <span style={{ color }}>
          {event_type === 'tool_call_plan' ? '[plan] ' : ''}
        </span>
        <span style={{ color: 'var(--text-secondary)' }}>{data?.purpose ?? ''}</span>
        {label && (
          <div
            style={{
              color: 'var(--text-muted)',
              fontSize: '11px',
              marginTop: '1px',
              fontStyle: 'italic',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {String(label).slice(0, 120)}
          </div>
        )}
      </div>
    );
  }

  if (event_type === 'tool_result') {
    const isChat = data?.type === 'chat';
    const color = isChat ? 'var(--accent-blue)' : 'var(--accent-green)';
    const resultSummary = isChat
      ? (typeof data?.answer === 'string' ? data.answer.slice(0, 100) : '')
      : `${data?.clips?.length ?? data?.count ?? 0} clips`;
    return (
      <div
        className="slide-in"
        style={{
          padding: '4px 7px',
          background: 'var(--bg-hover)',
          borderLeft: `2px solid ${color}`,
          borderRadius: '0 3px 3px 0',
          fontSize: '11px',
          color: 'var(--text-muted)',
          display: 'flex',
          justifyContent: 'space-between',
          gap: '8px',
        }}
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          ↳ {resultSummary}
        </span>
        <span style={{ flexShrink: 0, color: 'var(--text-muted)', fontSize: '10px' }}>
          {ago(timestamp)}
        </span>
      </div>
    );
  }

  if (event_type === 'tool_error') {
    return (
      <div
        className="slide-in"
        style={{
          padding: '4px 7px',
          background: 'var(--bg-card)',
          borderLeft: '2px solid var(--accent-red)',
          borderRadius: '0 3px 3px 0',
          fontSize: '11px',
          color: 'var(--accent-red)',
        }}
      >
        ✕ {data?.error ?? 'Tool error'}
      </div>
    );
  }

  return null;
}

// ─── Footage Map ─────────────────────────────────────────────────────────────

interface VideoSegment {
  start: number;
  end: number;
  score: number;
  query: string;
}

interface VideoEntry {
  name: string;
  segments: VideoSegment[];
  maxEnd: number;
}

function FootageMap({ events }: { events: PlanningEvent[] }) {
  const videoMap = useMemo(() => {
    const map = new Map<string, VideoEntry>();

    for (const e of events) {
      if (e.event_type !== 'tool_result' || e.data?.type !== 'search') continue;
      const clips: RetrievedClip[] = e.data?.clips ?? [];
      for (const clip of clips) {
        const key = clip.video_name;
        if (!map.has(key)) {
          map.set(key, { name: key, segments: [], maxEnd: 0 });
        }
        const entry = map.get(key)!;
        entry.segments.push({
          start: clip.start_seconds,
          end: clip.end_seconds,
          score: clip.score,
          query: clip.shot_query,
        });
        entry.maxEnd = Math.max(entry.maxEnd, clip.end_seconds);
      }
    }

    return Array.from(map.values());
  }, [events]);

  const totalClips = videoMap.reduce((s, v) => s + v.segments.length, 0);

  return (
    <Panel title="Footage Map" badge={totalClips} badgeColor="var(--accent-green)">
      {videoMap.length === 0 && (
        <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          No clips retrieved yet…
        </span>
      )}
      {videoMap.map((video) => (
        <VideoTrack key={video.name} video={video} />
      ))}
    </Panel>
  );
}

function VideoTrack({ video }: { video: VideoEntry }) {
  const duration = video.maxEnd > 0 ? video.maxEnd * 1.05 : 1;

  return (
    <div style={{ marginBottom: '12px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '3px',
          fontSize: '11px',
        }}
      >
        <span
          style={{
            color: 'var(--text-secondary)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            maxWidth: '75%',
          }}
        >
          {video.name}
        </span>
        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
          {video.segments.length} clips
        </span>
      </div>

      {/* Timeline bar */}
      <div
        style={{
          position: 'relative',
          height: '18px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: '3px',
          overflow: 'hidden',
        }}
      >
        {video.segments.map((seg, i) => {
          const left = (seg.start / duration) * 100;
          const width = Math.max(((seg.end - seg.start) / duration) * 100, 0.5);
          return (
            <div
              key={i}
              title={`${fmtTime(seg.start)}–${fmtTime(seg.end)} · score ${(seg.score * 100).toFixed(0)}%\n${seg.query}`}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${width}%`,
                top: '2px',
                bottom: '2px',
                background: scoreColor(seg.score),
                opacity: 0.75,
                borderRadius: '2px',
              }}
            />
          );
        })}
      </div>

      {/* Time axis labels */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          color: 'var(--text-muted)',
          fontSize: '9px',
          marginTop: '1px',
        }}
      >
        <span>0:00</span>
        <span>{fmtTime(duration)}</span>
      </div>
    </div>
  );
}

// ─── PlanningMonitor ─────────────────────────────────────────────────────────

interface PlanningMonitorProps {
  events: PlanningEvent[];
  projectName: string;
  initialPrompt: string;
  paused: boolean;
  planningDone: boolean;
  onPause: () => void;
  onResume: () => void;
  onInject: (prompt: string) => void;
}

export function PlanningMonitor(props: PlanningMonitorProps) {
  const {
    events,
    initialPrompt,
    paused,
    planningDone,
    onPause,
    onResume,
    onInject,
  } = props;
  const latestStoryboard = useMemo<Storyboard | null>(() => {
    const sbs = events.filter((e) => e.event_type === 'storyboard_update');
    return sbs.length > 0 ? (sbs[sbs.length - 1].data as Storyboard) : null;
  }, [events]);

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr minmax(280px, 0.55fr)',
        gridTemplateRows: '1fr 1fr',
        gap: '6px',
        flex: 1,
        minHeight: 0,
      }}
    >
      <KnowledgePanel events={events} />
      <StoryboardPanel storyboard={latestStoryboard} />
      <div style={{ gridColumn: '3', gridRow: '1 / -1', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <ChatLog
          initialPrompt={initialPrompt}
          events={events}
          paused={paused}
          planningDone={planningDone}
          onInject={onInject}
          onPause={onPause}
          onResume={onResume}
        />
      </div>
      <ToolFeedPanel events={events} />
      <FootageMap events={events} />
    </div>
  );
}
