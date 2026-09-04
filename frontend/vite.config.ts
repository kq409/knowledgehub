import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Increase timeout for long audio file transcriptions
        // Large audio files can take several minutes to process
        timeout: 600000, // 10 minutes
        proxyTimeout: 600000, // 10 minutes
        configure: (proxy: any) => {
          proxy.on('error', (err: Error, _req: unknown, res: any) => {
            console.error('[vite] http proxy error:', err.message);
            if (res?.writeHead && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(
                JSON.stringify({
                  detail:
                    'Backend is not reachable. Start the API server on port 8000.',
                })
              );
            }
          });
          proxy.on('proxyRes', (proxyRes: { headers: Record<string, unknown> }) => {
            // Prevent proxy/nginx from buffering streamed LLM tokens
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['x-accel-buffering'] = 'no';
            // Keep FastAPI slash-redirects on the frontend origin instead of
            // leaking http://localhost:8000 into the browser.
            const location = proxyRes.headers.location;
            if (typeof location === 'string' && location.includes('://')) {
              try {
                const url = new URL(location);
                if (url.pathname.startsWith('/api')) {
                  proxyRes.headers.location = `${url.pathname}${url.search}`;
                }
              } catch {
                // Leave the original header if it is not a valid URL.
              }
            }
          });
        },
      },
    },
  },
});
