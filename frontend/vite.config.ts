import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Python package serves the build; `proteus serve` injects the session token into index.html.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/proteus_music/api/static", emptyOutDir: true },
});
