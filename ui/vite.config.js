import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";

export default defineConfig({
  plugins: [react()],
  envPrefix: "CTRON_",
  test: {
    environment: "jsdom",
    setupFiles: "vitest.setup.ts",
  },
  server: {
    port: 3000,
    host: true, // needed for the Docker Container port mapping to work
  },
});
