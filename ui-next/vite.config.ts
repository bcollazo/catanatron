import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  envPrefix: ["VITE_", "CTRON_"],
  server: { host: true, port: 3000 },
  test: { environment: "jsdom" },
});
