import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  testMatch: 'host.spec.ts',
  use: { baseURL: process.env.EQIORA_JUPYTER_URL ?? 'http://127.0.0.1:18927', headless: true },
  workers: 1,
  timeout: 60000,
});
