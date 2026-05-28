import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Asset base path differs between dev server and production build:
//   - `npm run dev`   → served from / (Vite dev server, port 5173)
//   - `npm run build` → mounted by FastAPI under /app (see src/app.py)
// Without the build-time /app/ base, the static index.html references
// /assets/... which 404s because StaticFiles only serves under /app/.
// Override with VITE_BASE_PATH if a deploy mounts the dashboard somewhere
// else (e.g. behind a different reverse proxy prefix).
export default defineConfig(({ command }) => ({
  base: process.env.VITE_BASE_PATH ?? (command === 'build' ? '/app/' : '/'),
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // ws: true so the WebSocket upgrade for /video-edit/v2/agent/{project}/chat
      // is forwarded correctly (otherwise vite only proxies HTTP).
      '/video-edit': { target: 'http://127.0.0.1:8000', ws: true, changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
}))
