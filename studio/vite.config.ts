import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    strictPort: true,
    port: 1420,
  },
  envPrefix: "VITE_",
  build: {
    target: "es2024",
    minify: "oxc",
    sourcemap: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "react-vendor",
              test: /node_modules[\\/]react(?:-dom)?[\\/]/,
              priority: 30,
            },
            {
              name: "xyflow-vendor",
              test: /node_modules[\\/]@xyflow[\\/]/,
              priority: 20,
            },
            {
              name: "validation-vendor",
              test: /node_modules[\\/]zod[\\/]/,
              priority: 10,
            },
          ],
        },
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
