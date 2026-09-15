import { expect, test } from '@playwright/test';

import { assertNoSeriousAxeViolations } from './support';

const pages = [
  '/reference/language/',
  '/reference/language/declarations/',
  '/reference/language/units/',
  '/reference/language/equations/',
  '/reference/language/composition/',
  '/reference/standard-packages/',
  '/reference/standard-packages/electrical/',
  '/reference/standard-packages/controls/',
  '/reference/standard-packages/continuum/',
];

test('Reference navigation reaches language and physical sources', async ({ page }) => {
  await page.goto('/reference/');
  await page.locator('main').getByRole('link', { name: /Language syntax Eqiora Language/ }).click();
  await expect(page).toHaveURL(/\/reference\/language\/$/);
  await page.locator('main').getByRole('link', { name: 'Declarations', exact: true }).click();
  await expect(page.locator('main')).toContainText('variable current: A;');
  await page.goto('/reference/standard-packages/');
  await page.locator('main').getByRole('link', { name: 'Electrical components', exact: true }).click();
  await expect(page.locator('main')).toContainText('IdealVoltageSource');
  await expect(page.locator('main a[href*="/blob/"]').first()).toBeVisible();
});

test('Current Reference links, source and edit destinations are present', async ({ page, baseURL }) => {
  expect(baseURL).toBeTruthy();
  for (const route of pages) {
    const response = await page.goto(route);
    expect(response?.ok(), route).toBe(true);
    await expect(page.locator('main')).not.toContainText('0.1.0a7');
    const sourceLinks = page.locator(
      'main a[href^="https://github.com/nkiyohara/eqiora/blob/"], '
      + 'main a[href^="https://github.com/nkiyohara/eqiora/tree/"]',
    );
    await expect(sourceLinks.first(), `${route} links to its source`).toBeVisible();
    for (const source of await sourceLinks.all()) {
      await expect(source).toHaveAttribute('href', /\/(?:blob|tree)\/[0-9a-f]{40}\//);
    }
    await assertNoSeriousAxeViolations(page);
    await expect(page.getByRole('link', { name: /Edit page/i })).toHaveAttribute(
      'href', /github\.com\/nkiyohara\/eqiora\/edit\/main\/docs\/site\/src\/content\/docs\/reference\//,
    );
    const links = await page.locator('main a[href]').evaluateAll((elements) =>
      elements.map((element) => (element as HTMLAnchorElement).href),
    );
    for (const href of new Set(links)) {
      const url = new URL(href);
      if (url.origin !== new URL(baseURL!).origin) continue;
      const target = await page.request.get(url.pathname);
      expect(target.ok(), `${route} -> ${href}`).toBe(true);
      if (url.pathname === route && url.hash) {
        expect(await page.locator(`[id=${JSON.stringify(decodeURIComponent(url.hash.slice(1)))}]`).count(), href).toBe(1);
      }
    }
  }
});

for (const width of [390, 1440]) {
  for (const theme of ['light', 'dark']) {
    test(`Reference examples remain readable at ${width}px in ${theme}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      for (const route of ['/reference/language/declarations/', '/reference/standard-packages/continuum/']) {
        await page.goto(route);
        await page.locator('html').evaluate((element, value) => element.setAttribute('data-theme', value), theme);
        await expect(page.locator('h1')).toBeVisible();
        const code = page.locator('pre[data-language="eqiora"]').first();
        await expect(code).toBeVisible();
        await expect(code).toHaveAttribute('aria-label', /(?:declarations|elastic-body)\.eqi/);
        expect(await code.innerText()).toContain('model ');
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        expect(overflow).toBeLessThanOrEqual(1);
      }
      await page.screenshot({ path: testInfo.outputPath('continuum.png'), fullPage: true });
    });
  }
}

test('Current Reference is discoverable through the site search', async ({ page }) => {
  await page.goto('/reference/');
  await page.getByRole('button', { name: /search/i }).click();
  const dialog = page.getByRole('dialog', { name: 'Search' });
  await dialog.getByRole('textbox', { name: 'Search', exact: true }).fill('IdealVoltageSource');
  const result = dialog.getByRole('link', { name: /Electrical components/i }).first();
  await expect(result).toBeVisible();
  await result.click();
  await expect(page).toHaveURL(/\/reference\/standard-packages\/electrical\//);
});
