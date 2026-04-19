import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/agent': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/research': 'http://localhost:8000',
      '/community': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
