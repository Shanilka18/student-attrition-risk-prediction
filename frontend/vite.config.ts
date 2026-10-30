import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const PROXY_CONFIG = {
  '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/meta': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/leaderboard': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/models': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/research-answer': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/data': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/explain': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/predict': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/train': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/retrain': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/jobs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/static': { target: 'http://127.0.0.1:8000', changeOrigin: true },
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: PROXY_CONFIG,
  },
  preview: {
    port: 5173,
    proxy: PROXY_CONFIG,
  },
})
