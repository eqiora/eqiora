import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  testMatch: 'editor.spec.ts',
  use: { baseURL: 'http://127.0.0.1:18926', headless: true },
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:18926',
    reuseExistingServer: false,
  },
});
