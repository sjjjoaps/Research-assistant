import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/agent': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Disable proxy buffering so SSE events stream through immediately
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['x-accel-buffering'] = 'no'
            proxyRes.headers['cache-control'] = 'no-cache'
          })
        },
      },
      '/documents': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/research': 'http://localhost:8000',
      '/community': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
    host: true,
    allowedHosts: ['.lhr.life'],
  },
})
