import { readFileSync } from 'node:fs';
import { expect, test, type Browser } from '@playwright/test';
import { BASE_URL, launchOfficialBrowser } from './support';

const model = readFileSync(new URL('../../../examples/decay.eqi', import.meta.url), 'utf8');
const program = readFileSync(new URL('../../../examples/python/textbook_decay.py', import.meta.url), 'utf8');
let browser: Browser;
test.beforeAll(async () => { browser = await launchOfficialBrowser(); });
test.afterAll(async () => { await browser?.close(); });

for (const width of [1280, 320]) {
  test(`Get started run/edit and downloads remain usable without JavaScript at ${width}px`, async () => {
    const context = await browser.newContext({
      baseURL: BASE_URL, javaScriptEnabled: false, viewport: { width, height: 900 },
    });
    const page = await context.newPage();
    await page.goto('/get-started/');
    await expect(page.getByRole('heading', { level: 1, name: 'Get started' })).toBeVisible();
    const main = page.getByRole('main');
    await expect(main).toContainText("uv pip install --python .venv/bin/python --reinstall-package eqiora ./eqiora-source");
    const revisionLink = main.getByRole('link', { name: 'The checked-out revision', exact: true });
    const sourceHref = await revisionLink.getAttribute('href');
    const sourceSha = sourceHref?.match(/\/blob\/([0-9a-f]{40})\//)?.[1];
    expect(sourceSha).toBeTruthy();
    await expect(main).toContainText(`git -C eqiora-source fetch --depth 1 origin ${sourceSha}`);
    await expect(main).toContainText('git -C eqiora-source checkout --detach FETCH_HEAD');
    await expect(main).toContainText('uv run --no-project --python .venv/bin/python python run.py');
    await expect(main).toContainText('0.3678794412');
    await expect(main).toContainText('0.1353352833');
    await expect(main).toContainText('EQ0603');
    for (const [name, bytes] of [['decay.eqi', model], ['run.py', program]]) {
      const link = main.getByRole('link', { name: `Download ${name}`, exact: true });
      await expect(link).toHaveAttribute('download', name);
      const href = await link.getAttribute('href');
      expect(href?.startsWith('/')).toBe(true);
      const response = await context.request.get(href!);
      expect(response.ok()).toBe(true);
      expect(await response.text()).toBe(bytes);
      await expect(main.getByRole('region', { name, exact: true })).toContainText(bytes.trim(), { useInnerText: true });
    }
    await page.keyboard.press('Tab');
    let reachedDownload = false;
    for (let step = 0; step < 70; step++) {
      reachedDownload = await page.evaluate(() => document.activeElement?.getAttribute('download') === 'decay.eqi');
      if (reachedDownload) break;
      await page.keyboard.press('Tab');
    }
    expect(reachedDownload).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    for (const href of ['/guides/run-and-inspect/', '/gallery/', '/reference/']) {
      await expect(main.locator(`a[href="${href}"]`)).toHaveCount(1);
    }
    await page.emulateMedia({ media: 'print' });
    await expect(main.locator('.katex-display').first()).toBeVisible();
    await page.emulateMedia({ media: 'screen' });
    await main.getByRole('link', { name: 'derive this ODE', exact: true }).click();
    await expect(page.getByRole('heading', { level: 1, name: '4. Ordinary differential equations' })).toBeVisible();
    await expect(page.getByRole('region', { name: 'decay.eqi', exact: true })).toContainText(model.trim(), { useInnerText: true });
    await expect(page.getByRole('region', { name: 'run.py', exact: true })).toContainText(program.trim(), { useInnerText: true });
    await context.close();
  });
}
