import React, { useState, useEffect, useCallback } from 'react';
import type { ProjectSummary } from '../types';
import { listProjects, createProject } from '../api';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function statusColor(status: string): string {
  switch (status) {
    case 'planning': return 'var(--accent-yellow)';
    case 'indexed':  return 'var(--accent-blue)';
    case 'done':     return 'var(--accent-green)';
    case 'error':    return 'var(--accent-red)';
    default:         return 'var(--text-muted)';
  }
}

function relativeTime(iso: string | null): string {
  if (!iso) return '—';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

// ─── Project Card ─────────────────────────────────────────────────────────────

interface ProjectCardProps {
  project: ProjectSummary;
  onSelect: (name: string) => void;
}

function ProjectCard({ project, onSelect }: ProjectCardProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={() => onSelect(project.project_name)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? 'var(--bg-hover)' : 'var(--bg-card)',
        border: `1px solid ${hovered ? 'var(--border-active)' : 'var(--border)'}`,
        borderRadius: '6px',
        padding: '14px 16px',
        cursor: 'pointer',
        transition: 'background 0.1s, border-color 0.1s',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px' }}>
          {project.project_name}
        </span>
        <span style={{
          fontSize: '10px',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          color: statusColor(project.status),
          background: `${statusColor(project.status)}18`,
          border: `1px solid ${statusColor(project.status)}40`,
          borderRadius: '3px',
          padding: '1px 6px',
        }}>
          {project.status}
        </span>
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: '14px', fontSize: '11px', color: 'var(--text-secondary)' }}>
        <span title="footage files">{project.video_count} video{project.video_count !== 1 ? 's' : ''}</span>
        {project.clip_count > 0 && <span title="retrieved clips">{project.clip_count} clips</span>}
        {project.iteration_count > 0 && <span>{project.iteration_count} iter</span>}
        {project.has_fcpxml && <span style={{ color: 'var(--accent-purple)' }}>FCPXML ✓</span>}
        {project.has_renders && <span style={{ color: 'var(--accent-green)' }}>Rendered ✓</span>}
      </div>

      {/* Footage list (if any and not indexed yet) */}
      {project.status === 'new' && project.footage_files.length > 0 && (
        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
          {project.footage_files.slice(0, 3).join(', ')}
          {project.footage_files.length > 3 && ` +${project.footage_files.length - 3} more`}
        </div>
      )}

      {/* Footer */}
      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
        {relativeTime(project.last_updated)}
      </div>
    </div>
  );
}

// ─── Create Project Form ──────────────────────────────────────────────────────

interface CreateFormProps {
  onCreated: (name: string) => void;
  disabled?: boolean;
}

function CreateForm({ onCreated, disabled }: CreateFormProps) {
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  async function handleCreate() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setLoading(true);
    setErr('');
    try {
      await createProject(trimmed);
      onCreated(trimmed);
      setName('');
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  const valid = /^[a-zA-Z0-9_-]+$/.test(name.trim());

  return (
    <div style={{ display: 'flex', gap: '6px', flexDirection: 'column' }}>
      <div style={{ display: 'flex', gap: '6px' }}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && valid && handleCreate()}
          placeholder="new-project-name"
          disabled={disabled || loading}
          style={{
            flex: 1,
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '3px',
            color: 'var(--text-primary)',
            fontSize: '12px',
            padding: '6px 9px',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleCreate}
          disabled={!valid || loading || disabled}
          style={{
            background: valid ? 'var(--accent-blue)' : 'var(--bg-hover)',
            border: 'none',
            borderRadius: '3px',
            color: valid ? '#fff' : 'var(--text-muted)',
            cursor: valid ? 'pointer' : 'not-allowed',
            fontSize: '12px',
            fontFamily: 'inherit',
            padding: '6px 14px',
            whiteSpace: 'nowrap',
          }}
        >
          {loading ? '…' : '+ Create'}
        </button>
      </div>
      {err && <div style={{ fontSize: '11px', color: 'var(--accent-red)' }}>{err}</div>}
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Creates <code style={{ color: 'var(--text-secondary)' }}>data/workspaces/&lt;name&gt;/footage/</code> — drop your video files there, then index.
      </div>
    </div>
  );
}

// ─── ProjectBrowser ───────────────────────────────────────────────────────────

interface ProjectBrowserProps {
  onSelectProject: (projectName: string) => void;
}

export function ProjectBrowser({ onSelectProject }: ProjectBrowserProps) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr('');
    try {
      const { projects: ps } = await listProjects();
      setProjects(ps);
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Auto-refresh every 5s while page is visible
  useEffect(() => {
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      padding: '24px',
      gap: '20px',
      maxWidth: '900px',
      margin: '0 auto',
      width: '100%',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
            VEA Dashboard
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Video Editing Automation — Playground
          </div>
        </div>
        <button
          onClick={refresh}
          style={{
            background: 'none',
            border: '1px solid var(--border)',
            borderRadius: '3px',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            fontSize: '11px',
            fontFamily: 'inherit',
            padding: '4px 10px',
          }}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Two-column layout: projects + create */}
      <div style={{ display: 'flex', gap: '20px', flex: 1, minHeight: 0 }}>

        {/* Left: project list */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
            Projects ({projects.length})
          </div>

          {loading && projects.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Loading…</div>
          )}
          {err && (
            <div style={{ color: 'var(--accent-red)', fontSize: '12px' }}>{err}</div>
          )}
          {!loading && projects.length === 0 && !err && (
            <div style={{
              color: 'var(--text-muted)',
              fontSize: '12px',
              background: 'var(--bg-card)',
              border: '1px dashed var(--border)',
              borderRadius: '6px',
              padding: '20px',
              textAlign: 'center',
              lineHeight: 2,
            }}>
              No projects yet.<br />
              Create one on the right →
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
            {projects.map((p) => (
              <ProjectCard key={p.project_name} project={p} onSelect={onSelectProject} />
            ))}
          </div>
        </div>

        {/* Right: create new project */}
        <div style={{
          width: '260px',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
        }}>
          <div style={{ fontSize: '10px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
            New Project
          </div>
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '16px',
          }}>
            <CreateForm
              onCreated={(name) => {
                refresh();
                onSelectProject(name);
              }}
            />
          </div>

          {/* Quick guide */}
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '14px',
            fontSize: '11px',
            color: 'var(--text-muted)',
            lineHeight: 1.7,
          }}>
            <div style={{ color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>Quick start</div>
            <ol style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <li>Create a project</li>
              <li>Drop footage into <code>footage/</code></li>
              <li>Click the project → Index</li>
              <li>Set a prompt → Plan</li>
              <li>Generate FCPXML → Render</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
