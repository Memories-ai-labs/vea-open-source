# VEA Dashboard

React + Vite + TypeScript frontend for the VEA agentic video editor. This is **not** a standalone app — it talks to the VEA FastAPI backend (`src/app.py`) over REST and WebSocket.

For the full project README, setup instructions, and architecture docs, see the repo root:

- [`../README.md`](../README.md) — quick start and product overview
- [`../docs/onboarding.md`](../docs/onboarding.md) — step-by-step developer setup
- [`../docs/architecture.md`](../docs/architecture.md) — system architecture, including the dashboard section

## Running

The dashboard is normally built once and served by FastAPI at **http://localhost:8000/app**:

```bash
# From the repo root
cd dashboard && npm install && npm run build && cd ..
./dev.sh up
```

For UI development with hot reload, run the Vite dev server alongside the backend:

```bash
./dev.sh up --frontend-dev
# Backend on http://localhost:8000
# Vite dev server on http://localhost:5173 (proxies API + WebSocket to :8000)
```

The proxy is configured in `vite.config.ts`; both REST (`/video-edit/*`) and WebSocket upgrades are forwarded to the backend.

## Key files

- `src/App.tsx` — root component, project routing
- `src/hooks/useAgentChat.ts` — single WebSocket hook that owns all real-time state (chat, scratchpads, edit decision, render status, indexing progress)
- `src/components/AgentChat.tsx` — main editing workspace UI
- `src/components/NLETimeline.tsx` — multi-track timeline visualization (V1 video, T1 titles, A1 narration, A2 music)
- `src/components/AudioInspector.tsx` — gain / LUFS panel for selected audio clips
- `src/components/PreviewPanel.tsx` — draft / final render preview tabs
- `src/api.ts` — REST helpers
