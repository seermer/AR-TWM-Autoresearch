import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", chunkSizeWarningLimit: 8000 },
  server: { proxy: { "/api": "http://127.0.0.1:7860", "/raw": "http://127.0.0.1:7860" } },
});
