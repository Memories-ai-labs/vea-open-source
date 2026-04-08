import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      // ws: true so the WebSocket upgrade for /video-edit/v2/agent/{project}/chat
      // is forwarded correctly (otherwise vite only proxies HTTP).
      '/video-edit': { target: 'http://127.0.0.1:8000', ws: true, changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
