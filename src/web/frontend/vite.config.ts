import { defineConfig } from "vite";

// Dev server proxies the API to the FastAPI backend (python -m web.backend, port 8000).
export default defineConfig({
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
