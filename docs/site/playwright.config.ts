import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { defineConfig } from '@playwright/test';

const baseURL = process.env.EQIORA_SITE_BASE_URL;
if (baseURL !== 'http://127.0.0.1:4173') {
  throw new Error('EQIORA_SITE_BASE_URL must bind the isolated loopback site at 127.0.0.1:4173');
}
const browserRoot = process.env.PLAYWRIGHT_BROWSERS_PATH;
if (!browserRoot?.endsWith('/eqiora-pw-1.64.1-r1248')) {
  throw new Error('PLAYWRIGHT_BROWSERS_PATH does not name the pinned r1248 supply');
}

const require = createRequire(import.meta.url);
const corePackage = require.resolve('playwright-core/package.json');
const browsers = JSON.parse(
  readFileSync(resolve(dirname(corePackage), 'browsers.json'), 'utf8'),
) as {
  browsers: Array<{ name: string; revision: string; browserVersion: string }>;
};
for (const name of ['chromium', 'chromium-headless-shell']) {
  const browser = browsers.browsers.find((entry) => entry.name === name);
  if (browser?.revision !== '1248' || browser.browserVersion !== '156.0.8078.4') {
    throw new Error(`unexpected ${name} identity: ${JSON.stringify(browser)}`);
  }
}

const scratch = process.env.EQIORA_API_SCRATCH;
if (!scratch) throw new Error('EQIORA_API_SCRATCH is required');

export default defineConfig({
  testDir: './tests',
  outputDir: resolve(scratch, 'playwright-results'),
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 2,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: [['line']],
  use: {
    baseURL,
    browserName: 'chromium',
    colorScheme: 'light',
    locale: 'en-GB',
    serviceWorkers: 'block',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium-156-r1248' }],
});
