import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Ensure this matches your frontend port
    proxy: {
      // Proxy API requests to the backend server
      '/api': {
        target: 'http://localhost:3001', // Your backend server address
        changeOrigin: true, // Recommended for virtual hosted sites
        // secure: false,      // Uncomment if your backend is not HTTPS (usually not needed for localhost)
        // rewrite: (path) => path.replace(/^\/api/, '') // Use if your backend doesn't expect /api prefix
      }
    }
  }
})
