import { useState, useEffect, useCallback } from 'react';
import type { ProjectSummary } from './types';
import { useAgentChat } from './hooks/useAgentChat';
import { ProjectBrowser } from './components/ProjectBrowser';
import { AgentChat } from './components/AgentChat';
import { ToastProvider } from './components/Toast';

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

  // Update project metadata when agent connects and sends init data
  useEffect(() => {
    if (selectedProject && (agent.footageFiles.length > 0 || agent.indexedFiles.length > 0)) {
      setSelectedProject(prev => prev ? {
        ...prev,
        video_count: agent.footageFiles.length,
        footage_files: agent.footageFiles,
        indexed_files: agent.indexedFiles,
        status: agent.indexedFiles.length > 0 ? 'indexed' : prev.status,
      } : prev);
    }
  }, [agent.footageFiles, agent.indexedFiles, selectedProject?.project_name]);

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
      <ToastProvider>
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
          <ProjectBrowser onSelectProject={handleSelectProject} />
        </div>
      </ToastProvider>
    );
  }

  // ── Workspace (agent chat + project info) ──
  return (
    <ToastProvider>
      <AgentChat
        project={selectedProject}
        events={agent.events}
        messages={agent.messages}
        scratchpads={agent.scratchpads}
        scratchpadTimestamps={agent.scratchpadTimestamps}
        editDecision={agent.editDecision}
        renderState={agent.renderState}
        draftRenderState={agent.draftRenderState}
        cropStatuses={agent.cropStatuses}
        connected={agent.connected}
        busy={agent.busy}
        onSend={agent.send}
        onRequestRender={agent.requestRender}
        onRequestDraftRender={agent.requestDraftRender}
        onBack={handleBack}
        onClearState={agent.clearAndReconnect}
        onEditDecisionChange={agent.updateEditDecision}
        onRequestCropClip={agent.requestCropClip}
      />
    </ToastProvider>
  );
}
