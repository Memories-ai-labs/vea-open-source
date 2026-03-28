import { useEffect, useCallback, useMemo, useState, useRef } from 'react';
import type { ProjectSummary } from '../types';
import { listProjects, createProject } from '../api';
import { useBreakpoint } from '../hooks/useBreakpoint';
import { useToast } from './Toast';

function statusColor(status: string): string {
  switch (status) {
    case 'ready': return 'var(--accent-cyan, #60d5c8)';
    case 'planning': return 'var(--accent-yellow)';
    case 'indexed': return 'var(--accent-blue)';
    case 'done': return 'var(--accent-green)';
    case 'error': return 'var(--accent-red)';
    default: return 'var(--text-muted)';
  }
}

function humanizeStatus(status: string): string {
  const map: Record<string, string> = {
    new: 'New',
    ready: 'Ready',
    planning: 'Planning',
    plan_ready: 'Plan Ready',
    indexed: 'Indexed',
    rendered: 'Rendered',
    done: 'Done',
    error: 'Error',
  };
  return map[status] || status.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function cleanFileName(name: string): string {
  let cleaned = name;
  if (cleaned.startsWith('user_media_')) cleaned = cleaned.slice('user_media_'.length);
  return cleaned.replace(/_/g, ' ');
}

function relativeTime(iso: string | null): string {
  if (!iso) return 'awaiting activity';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'updated just now';
  if (mins < 60) return `updated ${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `updated ${hrs}h ago`;
  return `updated ${Math.floor(hrs / 24)}d ago`;
}

interface ProjectCardProps {
  project: ProjectSummary;
  onSelect: (name: string) => void;
}

function ProjectCard({ project, onSelect }: ProjectCardProps) {
  const accent = statusColor(project.status);

  return (
    <button
      onClick={() => onSelect(project.project_name)}
      style={{
        textAlign: 'left',
        background: 'linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: '18px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        boxShadow: 'var(--shadow-md)',
        transition: 'transform 0.18s ease, border-color 0.18s ease, background 0.18s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.borderColor = 'var(--border-strong)';
        e.currentTarget.style.background = 'linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03))';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.borderColor = 'var(--border)';
        e.currentTarget.style.background = 'linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))';
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div className="eyebrow" style={{ marginBottom: '8px' }}>project</div>
          <div
            style={{
              color: 'var(--text-primary)',
              fontSize: '20px',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {project.project_name}
          </div>
        </div>
        <span
          style={{
            flexShrink: 0,
            padding: '7px 10px',
            borderRadius: '999px',
            border: `1px solid ${accent}66`,
            background: `${accent}18`,
            color: accent,
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
          }}
        >
          {humanizeStatus(project.status)}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '10px' }}>
        <StatBlock label="videos" value={project.video_count} />
        <StatBlock label="clips" value={project.clip_count} />
        <StatBlock label="iterations" value={project.iteration_count} />
        <StatBlock
          label="outputs"
          value={[
            project.has_storyboard ? 'story' : null,
            project.has_fcpxml ? 'xml' : null,
            project.has_renders ? 'render' : null,
          ].filter(Boolean).length || 'none'}
        />
      </div>

      {project.footage_files.length > 0 && (
        <div
          style={{
            padding: '12px 14px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            color: 'var(--text-secondary)',
            fontSize: '12px',
            lineHeight: 1.6,
          }}
        >
          {project.footage_files.slice(0, 2).map(cleanFileName).join(' \u00B7 ')}
          {project.footage_files.length > 2 && ` +${project.footage_files.length - 2} more`}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', color: 'var(--text-muted)', fontSize: '11px' }}>
        <span>{relativeTime(project.last_updated)}</span>
        <span style={{ color: accent }}>Open project</span>
      </div>
    </button>
  );
}

function StatBlock({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        padding: '10px 12px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div style={{ color: 'var(--text-primary)', fontSize: '18px', fontWeight: 800 }}>{value}</div>
      <div className="eyebrow" style={{ marginTop: '4px' }}>{label}</div>
    </div>
  );
}

interface CreateFormProps {
  onCreated: (name: string) => void;
  disabled?: boolean;
}

function CreateForm({ onCreated, disabled }: CreateFormProps) {
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const { toast } = useToast();

  async function handleCreate() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setLoading(true);
    setErr('');
    try {
      await createProject(trimmed);
      toast(`Project "${trimmed}" created`, 'success');
      onCreated(trimmed);
      setName('');
    } catch (e: any) {
      const msg = e.message ?? String(e);
      setErr(msg);
      toast(`Failed to create project: ${msg}`, 'error');
    } finally {
      setLoading(false);
    }
  }

  const valid = /^[a-zA-Z0-9_-]+$/.test(name.trim());

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <div className="eyebrow" style={{ marginBottom: '8px' }}>project name</div>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && valid && handleCreate()}
          placeholder="festival-cut"
          disabled={disabled || loading}
          style={{
            width: '100%',
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--text-primary)',
            padding: '14px 16px',
            fontSize: '13px',
          }}
        />
      </div>

      <button
        onClick={handleCreate}
        disabled={!valid || loading || disabled}
        style={{
          padding: '14px 16px',
          borderRadius: '999px',
          border: '1px solid var(--accent-blue)',
          background: valid ? 'linear-gradient(90deg, rgba(96,213,200,0.24), rgba(241,191,99,0.18))' : 'rgba(255,255,255,0.03)',
          color: valid ? 'var(--text-primary)' : 'var(--text-muted)',
          cursor: valid ? 'pointer' : 'not-allowed',
          fontSize: '12px',
          fontWeight: 700,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}
      >
        {loading ? 'Creating project...' : 'Create workspace'}
      </button>

      {err && <div style={{ color: 'var(--accent-red)', fontSize: '12px' }}>{err}</div>}

      <div style={{ color: 'var(--text-muted)', fontSize: '12px', lineHeight: 1.7 }}>
        Creates <code>data/workspaces/&lt;name&gt;/footage/</code> so footage can be dropped in immediately.
      </div>
    </div>
  );
}

interface ProjectBrowserProps {
  onSelectProject: (projectName: string) => void;
}

export function ProjectBrowser({ onSelectProject }: ProjectBrowserProps) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [loadTimeout, setLoadTimeout] = useState(false);
  const isTablet = useBreakpoint(1100);
  const isMobile = useBreakpoint(720);
  const loadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async (isPolling = false) => {
    if (!isPolling) {
      setLoading(true);
      setErr('');
      setLoadTimeout(false);
      if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
      loadTimerRef.current = setTimeout(() => setLoadTimeout(true), 8000);
    }
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const res = await fetch(`/video-edit/v2/projects`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
      setLoadTimeout(false);
      setProjects(data.projects);
      setErr('');
    } catch (e: any) {
      if (e.name === 'AbortError') {
        if (!isPolling) {
          setLoadTimeout(true);
          setErr('Backend not responding — is the server running?');
        }
      } else {
        if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
        setLoadTimeout(false);
        setErr(e.message ?? String(e));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => { if (loadTimerRef.current) clearTimeout(loadTimerRef.current); };
  }, [refresh]);

  useEffect(() => {
    const id = setInterval(() => refresh(true), 5000);
    return () => clearInterval(id);
  }, [refresh]);

  const summary = useMemo(() => {
    const planning = projects.filter((p) => p.status === 'planning').length;
    const indexed = projects.filter((p) => p.status !== 'new').length;
    const rendered = projects.filter((p) => p.has_renders).length;
    return { total: projects.length, planning, indexed, rendered };
  }, [projects]);

  return (
    <div className="dashboard-shell">
      <div
        className="glass-card"
        style={{
          maxWidth: '1320px',
          margin: '0 auto',
          padding: isMobile ? '18px' : isTablet ? '22px' : '28px',
          display: 'grid',
          gap: isMobile ? '20px' : '28px',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: isTablet ? '1fr' : 'minmax(0, 1.5fr) minmax(280px, 0.9fr)',
            gap: isMobile ? '18px' : '24px',
            alignItems: 'start',
          }}
        >
          <div style={{ display: 'grid', gap: isMobile ? '10px' : '14px' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{
                fontSize: 'clamp(24px, 4vw, 38px)',
                fontWeight: 800,
                lineHeight: 1,
                letterSpacing: '-0.04em',
                fontFamily: 'var(--font-sans)',
              }}>
                VEA
              </div>
              <span className="eyebrow">video editing automation</span>
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '13px', lineHeight: 1.7, maxWidth: '52ch' }}>
              Create a workspace, drop footage, index, plan, and render — all from here.
            </div>
            <div className="metric-strip" style={{ maxWidth: '520px' }}>
              <div className="metric-card">
                <span className="metric-value">{summary.total}</span>
                <span className="metric-label">projects</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{summary.indexed}</span>
                <span className="metric-label">indexed</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{summary.planning}</span>
                <span className="metric-label">active loops</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{summary.rendered}</span>
                <span className="metric-label">renders</span>
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gap: '16px' }}>
            <div
              style={{
                padding: isMobile ? '18px' : '22px',
                borderRadius: 'var(--radius-lg)',
                background: 'linear-gradient(180deg, rgba(96,213,200,0.12), rgba(241,191,99,0.06))',
                border: '1px solid rgba(96,213,200,0.16)',
              }}
            >
              <div className="eyebrow" style={{ marginBottom: '10px', color: 'var(--accent-blue)' }}>
                New project
              </div>
              <CreateForm
                onCreated={(name) => {
                  refresh();
                  onSelectProject(name);
                }}
              />
            </div>

          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: '8px' }}>project library</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
              Pick up a workspace in progress or open a fresh one.
            </div>
          </div>
          <button
            onClick={refresh}
            style={{
              padding: '12px 16px',
              borderRadius: '999px',
              border: '1px solid var(--border)',
              background: 'rgba(255,255,255,0.03)',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              fontSize: '11px',
            }}
          >
            Refresh library
          </button>
        </div>

        {err && <div className="status-message error">{err}</div>}

        {loading && projects.length === 0 && !loadTimeout && (
          <div className="status-message info">Loading workspaces...</div>
        )}

        {loadTimeout && projects.length === 0 && (
          <div className="status-message error" style={{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'space-between' }}>
            <span>Could not load projects — the backend may be unavailable.</span>
            <button
              onClick={refresh}
              style={{
                padding: '6px 14px',
                borderRadius: '999px',
                border: '1px solid rgba(255,125,111,0.4)',
                background: 'rgba(255,125,111,0.12)',
                color: 'var(--accent-red)',
                cursor: 'pointer',
                fontSize: '11px',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                flexShrink: 0,
              }}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && projects.length === 0 && !err ? (
          <div
            style={{
              borderRadius: 'var(--radius-lg)',
              border: '1px dashed var(--border-strong)',
              padding: '48px 28px',
              textAlign: 'center',
              color: 'var(--text-secondary)',
              background: 'rgba(255,255,255,0.02)',
            }}
          >
            <div className="eyebrow" style={{ marginBottom: '12px' }}>No projects yet</div>
            <div style={{ fontSize: '24px', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '8px' }}>
              Start with a workspace, not a blank page.
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
              Create a project on the right, then drop footage into its `footage/` folder.
            </div>
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: isMobile ? '1fr' : 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '16px',
            }}
          >
            {projects.map((p) => (
              <ProjectCard key={p.project_name} project={p} onSelect={onSelectProject} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
