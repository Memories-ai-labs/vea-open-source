import { useState, useEffect, useCallback } from 'react';
import type { ProjectSummary } from './types';
import { useAgentChat } from './hooks/useAgentChat';
import { ProjectBrowser } from './components/ProjectBrowser';
import { AgentChat } from './components/AgentChat';

// ─── App ─────────────────────────────────────────────────────────────────────

function makeStub(name: string): ProjectSummary {
  return { project_name: name, status: 'new', video_count: 0, clip_count: 0, iteration_count: 0, footage_files: [], indexed_files: [], video_gists: {}, gist: '', has_storyboard: false, has_fcpxml: false, has_renders: false, last_updated: null };
}

/** Read project name from URL hash (#/project/<name>). Returns null if on home. */
function readHash(): string | null {
  const hash = window.location.hash;
  const match = hash.match(/^#\/project\/(.+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

export default function App() {
  const initialProject = readHash();
  const [selectedProject, setSelectedProject] = useState<ProjectSummary | null>(
    initialProject ? makeStub(initialProject) : null,
  );
  const [agentProjectName, setAgentProjectName] = useState<string | null>(initialProject);

  const agent = useAgentChat(agentProjectName);

  // Sync hash → state on popstate (browser back/forward)
  useEffect(() => {
    function onHashChange() {
      const name = readHash();
      if (name) {
        setSelectedProject(makeStub(name));
        setAgentProjectName(name);
      } else {
        setSelectedProject(null);
        setAgentProjectName(null);
      }
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const handleSelectProject = useCallback((name: string) => {
    setSelectedProject(makeStub(name));
    setAgentProjectName(name);
    window.location.hash = `#/project/${encodeURIComponent(name)}`;
  }, []);

  const handleBack = useCallback(() => {
    setAgentProjectName(null);
    setSelectedProject(null);
    window.location.hash = '';
  }, []);

  // ── Browser ──
  if (!selectedProject || !agentProjectName) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <ProjectBrowser onSelectProject={handleSelectProject} />
      </div>
    );
  }

  // ── Workspace (agent chat + project info) ──
  return (
    <AgentChat
      project={selectedProject}
      events={agent.events}
      messages={agent.messages}
      scratchpads={agent.scratchpads}
      scratchpadTimestamps={agent.scratchpadTimestamps}
      editDecision={agent.editDecision}
      renderState={agent.renderState}
      connected={agent.connected}
      busy={agent.busy}
      onSend={agent.send}
      onRequestRender={agent.requestRender}
      onBack={handleBack}
    />
  );
}
