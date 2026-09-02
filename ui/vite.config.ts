import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend only talks to the local API: the proxy keeps the calls on the
// dev server origin, so neither CORS nor a base URL scattered through the
// code is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: false },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
