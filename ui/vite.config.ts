import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend only talks to the local API: the proxy keeps the calls on the
// dev server origin, so neither CORS nor a base URL scattered through the
// code is needed.
//
// The backend port comes from the environment because `run.py --api-port`
// moves it. With the port hard-coded here that flag started a backend the
// dashboard could not reach, and the failure looked like an empty API rather
// than a misrouted proxy.
const apiPort = process.env.BACKTEST_API_PORT ?? "8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: `http://127.0.0.1:${apiPort}`, changeOrigin: false },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
