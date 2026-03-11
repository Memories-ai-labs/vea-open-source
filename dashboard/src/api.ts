import type { ProjectSummary, SessionStatus } from './types';

const BASE = '/video-edit/v2';

async function req<T>(
  path: string,
  method: string = 'GET',
  body?: object
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${method} ${path} → ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function listProjects(): Promise<{ projects: ProjectSummary[] }> {
  return req('/projects');
}

export function createProject(projectName: string): Promise<{ status: string; path: string }> {
  return req(`/projects/create?project_name=${encodeURIComponent(projectName)}`, 'POST');
}

export function indexProject(
  projectName: string,
  sourceDir?: string,
  startFresh?: boolean
): Promise<{ project_name: string; status: string; gist: string }> {
  return req('/index', 'POST', {
    project_name: projectName,
    ...(sourceDir !== undefined ? { source_dir: sourceDir } : {}),
    start_fresh: startFresh ?? false,
  });
}

export function startPlan(
  projectName: string,
  prompt: string,
  targetDuration: number,
  maxIterations: number
): Promise<{ status: string }> {
  return req('/plan/start', 'POST', {
    project_name: projectName,
    prompt,
    target_duration: targetDuration,
    max_iterations: maxIterations,
  });
}

export function getStatus(projectName: string): Promise<SessionStatus> {
  return req(`/session/${encodeURIComponent(projectName)}/status`);
}

export function pausePlan(projectName: string): Promise<void> {
  return req(`/session/${encodeURIComponent(projectName)}/pause`, 'POST');
}

export function resumePlan(projectName: string): Promise<void> {
  return req(`/session/${encodeURIComponent(projectName)}/resume`, 'POST');
}

export function injectPrompt(projectName: string, prompt: string): Promise<void> {
  return req(`/session/${encodeURIComponent(projectName)}/inject`, 'POST', { prompt });
}

export function generateFcpxml(projectName: string): Promise<{ fcpxml_path: string }> {
  return req(`/session/${encodeURIComponent(projectName)}/fcpxml`, 'POST');
}

export function generateNarration(
  projectName: string,
  overrideScript?: string
): Promise<{ audio_path: string }> {
  return req(`/session/${encodeURIComponent(projectName)}/narration`, 'POST', {
    ...(overrideScript !== undefined ? { override_script: overrideScript } : {}),
  });
}

export function selectMusic(
  projectName: string,
  mood?: string
): Promise<{ music_path: string }> {
  return req(`/session/${encodeURIComponent(projectName)}/music`, 'POST', {
    ...(mood !== undefined ? { mood } : {}),
  });
}

export function cropVideo(
  projectName: string,
  aspectRatio: number
): Promise<{ fcpxml_path: string }> {
  return req(`/session/${encodeURIComponent(projectName)}/crop`, 'POST', {
    aspect_ratio: aspectRatio,
  });
}

export function renderVideo(
  projectName: string,
  quality: 'preview' | 'final'
): Promise<{ output_path: string }> {
  return req(`/session/${encodeURIComponent(projectName)}/render`, 'POST', { quality });
}

export function getResolveStatus(): Promise<{
  running: boolean;
  version?: string;
  studio: boolean;
  error?: string;
}> {
  return req('/resolve/status');
}
