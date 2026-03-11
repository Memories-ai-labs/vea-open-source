import React, { useState, useEffect, useMemo } from 'react';
import type { ProjectSummary, Storyboard } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import { PlanningMonitor } from './components/PlanningMonitor';
import { TimelineView } from './components/TimelineView';
import { ProjectBrowser } from './components/ProjectBrowser';
import {
  indexProject,
  startPlan,
  pausePlan,
  resumePlan,
  injectPrompt,
  generateFcpxml,
  getResolveStatus,
} from './api';

// ─── Shared button style helper ───────────────────────────────────────────────

function btn(color: string, disabled = false): React.CSSProperties {
  return {
    padding: '4px 10px',
    background: disabled ? 'var(--bg-hover)' : `${color}22`,
    border: `1px solid ${disabled ? 'var(--border)' : color}`,
    borderRadius: '3px',
    color: disabled ? 'var(--text-muted)' : color,
    fontSize: '12px',
    fontFamily: 'inherit',
    cursor: disabled ? 'not-allowed' : 'pointer',
  };
}

// ─── Status badge ─────────────────────────────────────────────────────────────

function StatusBadge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      padding: '2px 8px',
      background: `${color}22`,
      border: `1px solid ${color}`,
      borderRadius: '10px',
      color,
      fontSize: '11px',
      fontWeight: 700,
    }}>
      {label}
    </span>
  );
}

// ─── Project Detail (index + plan setup) ─────────────────────────────────────

interface ProjectDetailProps {
  project: ProjectSummary;
  onBack: () => void;
  onStartPlanning: (
    projectName: string,
    prompt: string,
    targetDuration: number,
    maxIterations: number
  ) => Promise<void>;
}

function ProjectDetail({ project, onBack, onStartPlanning }: ProjectDetailProps) {
  const [prompt, setPrompt] = useState('');
  const [targetDuration, setTargetDuration] = useState(60);
  const [maxIterations, setMaxIterations] = useState(5);
  const [indexing, setIndexing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState('');
  const [info, setInfo] = useState('');

  const isIndexed = project.status !== 'new';
  const hasFootage = project.footage_files.length > 0;

  const fieldStyle: React.CSSProperties = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '3px',
    color: 'var(--text-primary)',
    fontSize: '12px',
    padding: '6px 9px',
    fontFamily: 'inherit',
    width: '100%',
  };

  const labelStyle: React.CSSProperties = {
    color: 'var(--text-muted)',
    fontSize: '10px',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    marginBottom: '4px',
  };

  async function handleIndex() {
    setIndexing(true);
    setErr('');
    setInfo('Indexing footage — this may take several minutes…');
    try {
      const result = await indexProject(project.project_name);
      setInfo(`Indexed! Status: ${result.status}`);
    } catch (e: any) {
      setErr(e.message ?? String(e));
      setInfo('');
    } finally {
      setIndexing(false);
    }
  }

  async function handleStart() {
    if (!prompt.trim()) return;
    setStarting(true);
    setErr('');
    try {
      await onStartPlanning(project.project_name, prompt.trim(), targetDuration, maxIterations);
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Mini top bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '6px 14px',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <button onClick={onBack} style={{ ...btn('var(--text-muted)'), border: 'none', background: 'none', padding: '2px 0', cursor: 'pointer', fontSize: '12px' }}>
          ← Projects
        </button>
        <span style={{ color: 'var(--border)', fontSize: '12px' }}>|</span>
        <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{project.project_name}</span>
        <StatusBadge
          label={project.status}
          color={
            project.status === 'planning' ? 'var(--accent-yellow)' :
            project.status === 'indexed' ? 'var(--accent-blue)' :
            project.status === 'done' ? 'var(--accent-green)' :
            project.status === 'error' ? 'var(--accent-red)' :
            'var(--text-muted)'
          }
        />
      </div>

      {/* Body */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        padding: '32px 24px',
        gap: '24px',
        flexWrap: 'wrap',
      }}>

        {/* Footage card */}
        <div style={{
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: '6px',
          padding: '18px',
          width: '280px',
        }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '10px' }}>
            Footage
          </div>
          {project.footage_files.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px', lineHeight: 1.7 }}>
              No footage found.<br />
              Drop video files into:<br />
              <code style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>
                data/workspaces/{project.project_name}/footage/
              </code>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {project.footage_files.map((f) => (
                <div key={f} style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <span style={{ color: 'var(--accent-blue)', fontSize: '10px' }}>▶</span>
                  {f}
                </div>
              ))}
            </div>
          )}
          {hasFootage && !isIndexed && (
            <button
              onClick={handleIndex}
              disabled={indexing}
              style={{
                ...btn('var(--accent-blue)', indexing),
                marginTop: '14px',
                width: '100%',
                padding: '7px',
                fontWeight: 700,
              }}
            >
              {indexing ? 'Indexing…' : '→ Index Footage'}
            </button>
          )}
          {isIndexed && (
            <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--accent-green)' }}>
              ✓ Indexed ({project.video_count} video{project.video_count !== 1 ? 's' : ''})
            </div>
          )}
        </div>

        {/* Plan card */}
        <div style={{
          background: 'var(--bg-panel)',
          border: `1px solid ${isIndexed ? 'var(--border)' : 'var(--border)'}`,
          borderRadius: '6px',
          padding: '18px',
          width: '340px',
          opacity: isIndexed ? 1 : 0.5,
        }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '10px' }}>
            Plan
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <div style={labelStyle}>Prompt</div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={!isIndexed}
                placeholder="Create a 60-second highlight reel…"
                rows={4}
                style={{ ...fieldStyle, resize: 'vertical', lineHeight: 1.5 }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <div style={labelStyle}>Target Duration (s)</div>
                <input
                  type="number"
                  value={targetDuration}
                  onChange={(e) => setTargetDuration(Number(e.target.value))}
                  disabled={!isIndexed}
                  min={10} max={600}
                  style={fieldStyle}
                />
              </div>
              <div>
                <div style={labelStyle}>Max Iterations</div>
                <input
                  type="number"
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                  disabled={!isIndexed}
                  min={1} max={20}
                  style={fieldStyle}
                />
              </div>
            </div>

            <button
              onClick={handleStart}
              disabled={!isIndexed || !prompt.trim() || starting}
              style={{
                ...btn('var(--accent-blue)', !isIndexed || !prompt.trim() || starting),
                padding: '8px',
                fontWeight: 700,
                fontSize: '13px',
              }}
            >
              {starting ? 'Starting…' : 'Start Planning →'}
            </button>
          </div>
        </div>

        {/* Stats card */}
        {(project.clip_count > 0 || project.iteration_count > 0 || project.has_fcpxml || project.has_renders) && (
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '18px',
            width: '200px',
          }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '10px' }}>
              Progress
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
              {project.iteration_count > 0 && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Iterations</span>
                  <span style={{ color: 'var(--text-primary)' }}>{project.iteration_count}</span>
                </div>
              )}
              {project.clip_count > 0 && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Clips</span>
                  <span style={{ color: 'var(--text-primary)' }}>{project.clip_count}</span>
                </div>
              )}
              {project.has_storyboard && (
                <div style={{ color: 'var(--accent-blue)', fontSize: '11px' }}>✓ Storyboard</div>
              )}
              {project.has_fcpxml && (
                <div style={{ color: 'var(--accent-purple)', fontSize: '11px' }}>✓ FCPXML</div>
              )}
              {project.has_renders && (
                <div style={{ color: 'var(--accent-green)', fontSize: '11px' }}>✓ Rendered</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Status messages */}
      {(info || err) && (
        <div style={{ padding: '0 24px 16px', display: 'flex', justifyContent: 'center' }}>
          {info && <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>{info}</div>}
          {err && <div style={{ color: 'var(--accent-red)', fontSize: '12px' }}>{err}</div>}
        </div>
      )}
    </div>
  );
}

// ─── Top Bar (shown during planning / fcpxml_ready) ──────────────────────────

interface TopBarProps {
  projectName: string;
  phase: 'planning' | 'fcpxml_ready';
  paused: boolean;
  connected: boolean;
  iteration: number;
  clipCount: number;
  resolveRunning: boolean | null;
  onBack: () => void;
  onPause: () => void;
  onResume: () => void;
  injectValue: string;
  onInjectChange: (v: string) => void;
  onInjectSubmit: () => void;
  onGenerateFcpxml: () => void;
  generatingFcpxml: boolean;
  planningDone: boolean;
}

function TopBar({
  projectName, phase, paused, connected, iteration, clipCount, resolveRunning,
  onBack, onPause, onResume, injectValue, onInjectChange, onInjectSubmit,
  onGenerateFcpxml, generatingFcpxml, planningDone,
}: TopBarProps) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '6px 12px',
      background: 'var(--bg-panel)',
      borderBottom: '1px solid var(--border)',
      flexShrink: 0,
      flexWrap: 'wrap',
    }}>
      <button onClick={onBack} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', padding: 0 }}>
        ← Projects
      </button>
      <span style={{ color: 'var(--border)' }}>|</span>
      <span style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '13px' }}>{projectName}</span>

      {phase === 'planning' && (
        paused
          ? <StatusBadge label="Paused" color="var(--accent-yellow)" />
          : <StatusBadge label="Running" color="var(--accent-green)" />
      )}
      {phase === 'fcpxml_ready' && <StatusBadge label="Done" color="var(--accent-purple)" />}

      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
        iter <span style={{ color: 'var(--text-secondary)' }}>{iteration}</span>
      </span>
      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
        clips <span style={{ color: 'var(--text-secondary)' }}>{clipCount}</span>
      </span>

      {/* WebSocket indicator */}
      <span
        title={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
        className={connected ? 'pulse-dot' : ''}
        style={{
          display: 'inline-block',
          width: '7px', height: '7px',
          borderRadius: '50%',
          background: connected ? 'var(--accent-green)' : 'var(--accent-red)',
        }}
      />

      {/* Resolve indicator */}
      {resolveRunning !== null && (
        <span style={{ color: 'var(--text-muted)', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{
            display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%',
            background: resolveRunning ? 'var(--accent-green)' : 'var(--text-muted)',
          }} />
          Resolve
        </span>
      )}

      <div style={{ flex: 1 }} />

      {/* Inject prompt */}
      {phase === 'planning' && (
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          <input
            value={injectValue}
            onChange={(e) => onInjectChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onInjectSubmit()}
            placeholder="Inject prompt…"
            style={{
              background: 'var(--bg-base)', border: '1px solid var(--border)', borderRadius: '3px',
              color: 'var(--text-primary)', fontSize: '12px', padding: '4px 8px',
              fontFamily: 'inherit', width: '200px',
            }}
          />
          <button onClick={onInjectSubmit} disabled={!injectValue.trim()} style={btn('var(--text-secondary)', !injectValue.trim())}>
            Inject
          </button>
        </div>
      )}

      {/* Pause / Resume */}
      {phase === 'planning' && (
        paused
          ? <button onClick={onResume} style={btn('var(--accent-green)')}>Resume</button>
          : <button onClick={onPause} style={btn('var(--accent-yellow)')}>Pause</button>
      )}

      {/* Generate FCPXML */}
      {planningDone && phase === 'planning' && (
        <button
          onClick={onGenerateFcpxml}
          disabled={generatingFcpxml}
          style={{ ...btn('var(--accent-purple)', generatingFcpxml), fontWeight: 700 }}
        >
          {generatingFcpxml ? 'Generating…' : 'Generate FCPXML →'}
        </button>
      )}
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

type Phase = 'browser' | 'detail' | 'planning' | 'fcpxml_ready';

export default function App() {
  const [phase, setPhase] = useState<Phase>('browser');
  const [selectedProject, setSelectedProject] = useState<ProjectSummary | null>(null);
  const [projectName, setProjectName] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [currentStoryboard, setCurrentStoryboard] = useState<Storyboard | null>(null);
  const [resolveRunning, setResolveRunning] = useState<boolean | null>(null);
  const [injectText, setInjectText] = useState('');
  const [generatingFcpxml, setGeneratingFcpxml] = useState(false);

  const { events, connected } = useWebSocket(projectName);

  const iteration = useMemo(() => {
    const iters = events.filter((e) => e.event_type === 'iteration_start');
    if (iters.length === 0) return 0;
    return (iters[iters.length - 1].data?.iteration as number) ?? iters.length;
  }, [events]);

  const clipCount = useMemo(() => {
    let count = 0;
    for (const e of events) {
      if (e.event_type === 'tool_result' && e.data?.type === 'search') {
        count += (e.data?.clips as unknown[])?.length ?? 0;
      }
    }
    return count;
  }, [events]);

  useEffect(() => {
    const sbs = events.filter((e) => e.event_type === 'storyboard_update');
    if (sbs.length > 0) setCurrentStoryboard(sbs[sbs.length - 1].data as Storyboard);
  }, [events]);

  useEffect(() => {
    const last = [...events].reverse().find(
      (e) => e.event_type === 'paused' || e.event_type === 'resumed'
    );
    if (last) setPaused(last.event_type === 'paused');
  }, [events]);

  const planningDone = useMemo(
    () => events.some((e) => e.event_type === 'done' || e.event_type === 'stopped_early'),
    [events]
  );

  // Poll Resolve status
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try { const r = await getResolveStatus(); if (!cancelled) setResolveRunning(r.running); }
      catch { /* ignore */ }
    }
    poll();
    const id = setInterval(poll, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  function handleSelectProject(project: ProjectSummary) {
    setSelectedProject(project);
    setPhase('detail');
  }

  function handleSelectProjectByName(name: string) {
    // Called from ProjectBrowser when project name is known but we don't have full summary yet
    setSelectedProject({ project_name: name, status: 'new', video_count: 0, clip_count: 0, iteration_count: 0, footage_files: [], has_storyboard: false, has_fcpxml: false, has_renders: false, last_updated: null });
    setPhase('detail');
  }

  async function handleStartPlanning(
    name: string,
    prompt: string,
    targetDuration: number,
    maxIterations: number
  ) {
    await startPlan(name, prompt, targetDuration, maxIterations);
    setProjectName(name);
    setPhase('planning');
    setCurrentStoryboard(null);
    setPaused(false);
  }

  async function handlePause() {
    if (!projectName) return;
    try { await pausePlan(projectName); } catch { /* ignore */ }
  }

  async function handleResume() {
    if (!projectName) return;
    try { await resumePlan(projectName); } catch { /* ignore */ }
  }

  async function handleInject() {
    if (!projectName || !injectText.trim()) return;
    try { await injectPrompt(projectName, injectText.trim()); setInjectText(''); }
    catch { /* ignore */ }
  }

  async function handleGenerateFcpxml() {
    if (!projectName) return;
    setGeneratingFcpxml(true);
    try { await generateFcpxml(projectName); setPhase('fcpxml_ready'); }
    catch { /* stay on planning */ }
    finally { setGeneratingFcpxml(false); }
  }

  // ── Browser ──
  if (phase === 'browser') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <ProjectBrowser onSelectProject={handleSelectProjectByName} />
      </div>
    );
  }

  // ── Project Detail (index + plan setup) ──
  if (phase === 'detail' && selectedProject) {
    return (
      <ProjectDetail
        project={selectedProject}
        onBack={() => setPhase('browser')}
        onStartPlanning={handleStartPlanning}
      />
    );
  }

  // ── Planning Monitor / Timeline ──
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <TopBar
        projectName={projectName!}
        phase={phase === 'fcpxml_ready' ? 'fcpxml_ready' : 'planning'}
        paused={paused}
        connected={connected}
        iteration={iteration}
        clipCount={clipCount}
        resolveRunning={resolveRunning}
        onBack={() => setPhase('browser')}
        onPause={handlePause}
        onResume={handleResume}
        injectValue={injectText}
        onInjectChange={setInjectText}
        onInjectSubmit={handleInject}
        onGenerateFcpxml={handleGenerateFcpxml}
        generatingFcpxml={generatingFcpxml}
        planningDone={planningDone}
      />
      <div style={{ flex: 1, minHeight: 0, padding: '6px', display: 'flex', flexDirection: 'column' }}>
        {phase === 'planning' && projectName && (
          <PlanningMonitor
            events={events}
            projectName={projectName}
            onPause={handlePause}
            onResume={handleResume}
            onInject={async (p) => { if (projectName) await injectPrompt(projectName, p); }}
            paused={paused}
          />
        )}
        {phase === 'fcpxml_ready' && projectName && (
          <TimelineView projectName={projectName} storyboard={currentStoryboard} />
        )}
      </div>
    </div>
  );
}
