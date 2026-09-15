import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test, type Browser } from '@playwright/test';
import { BASE_URL, launchOfficialBrowser, assertNoPageOverflow, assertNoSeriousAxeViolations, assertTableInventory, installTableObserver, TABLE_ROUTES, SITE_ROUTES } from './support';

let browser: Browser;
test.beforeAll(async () => { browser = await launchOfficialBrowser(); });
test.afterAll(async () => { await browser?.close(); });

test('Learn table math retains an accessible alternative and all painted text checks', async () => {
  const context = await browser.newContext({ baseURL: BASE_URL });
  const page = await context.newPage();
  await installTableObserver(page);
  const expected = TABLE_ROUTES.find(({ route }) => route.endsWith('/models-not-simulations/'))!;
  await page.goto(expected.route);
  const baseline = await assertTableInventory(page, expected);
  expect(Object.values(baseline.failures).every((failures) => failures === 0)).toBe(true);
  for (const mutation of ['missing-mathml', 'hidden-mathml', 'hidden-presentation', 'hidden-prose']) {
    await page.goto(expected.route);
    await page.locator('main table').evaluate((table, change) => {
      if (change === 'missing-mathml') table.querySelector('.katex-mathml')!.remove();
      if (change === 'hidden-mathml') table.querySelector('.katex-mathml')!.setAttribute('aria-hidden', 'true');
      if (change === 'hidden-presentation') (table.querySelector('.katex-html') as HTMLElement).style.display = 'none';
      if (change === 'hidden-prose') (table.querySelector('tbody tr td:last-child') as HTMLElement).style.visibility = 'hidden';
    }, mutation);
    const observation = await assertTableInventory(page, expected);
    expect(mutation.endsWith('mathml') ? observation.failures.text : observation.failures.concealment, mutation).toBeGreaterThan(0);
  }
  await context.close();
});

const representatives = [
  '/', '/get-started/', '/learn/',
  '/learn/mathematical-modeling/fields-spatial-domains/',
  '/guides/modeling/', '/gallery/karman-vortex-street/',
  '/reference/language/', '/contributing/architecture/',
];

for (const width of [1440, 320]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`reader destinations retain section navigation and readable content at ${width}px ${colorScheme}`, async () => {
      test.setTimeout(120_000);
      const context = await browser.newContext({ baseURL: BASE_URL, viewport: { width, height: 900 }, colorScheme });
      const page = await context.newPage();
      for (const route of representatives) {
        expect((await page.goto(route))?.status(), route).toBe(200);
        const primary = page.getByRole('navigation', { name: 'Primary', exact: true });
        await expect(primary.getByRole('link')).toHaveText(['Learn', 'Guides', 'Gallery', 'Reference']);
        for (const link of await primary.getByRole('link').all()) {
          await expect(link).toBeVisible();
          const bounds = await link.boundingBox();
          expect(bounds!.x).toBeGreaterThanOrEqual(0);
          expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
        }
        const brand = page.getByRole('banner').getByRole('link', { name: 'Eqiora', exact: true });
        expect(await brand.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
        await expect(page.getByRole('banner').getByRole('link', { name: 'Get started', exact: true })).toBeVisible();
        await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
        await assertNoPageOverflow(page);
        await assertNoSeriousAxeViolations(page);
        if (route !== '/') await expect(page.getByRole('navigation', { name: 'Breadcrumb' })).toContainText('Home');
        if (width === 1440 && route.startsWith('/guides/')) {
          await expect(page.locator('#starlight__sidebar')).toContainText('Choose a task');
          await expect(page.locator('#starlight__sidebar')).not.toContainText('Component composition');
        }
        if (width === 320 && route === '/guides/modeling/') {
          await page.getByRole('button', { name: 'Search', exact: true }).click();
          await expect(page.getByRole('dialog')).toBeVisible();
          await page.keyboard.press('Escape');
          await page.getByRole('button', { name: 'Menu', exact: true }).click();
          await expect(page.locator('#starlight__sidebar')).toBeVisible();
          await page.keyboard.press('Escape');
          await expect(page.locator('#starlight__sidebar')).toBeHidden();
        }
        if (['/', '/learn/', '/guides/modeling/'].includes(route)) {
          await page.screenshot({ path: resolve(process.env.EQIORA_API_SCRATCH!, `reader-${route.replaceAll('/', '-') || 'home'}-${width}-${colorScheme}.png`), fullPage: route !== '/guides/modeling/' });
        }
      }
      await context.close();
    });
  }
}

test('canonical Python guide bodies, heading targets and local crosslinks render on-site', async () => {
  const context = await browser.newContext({ baseURL: BASE_URL });
  const page = await context.newPage();
  for (const name of ['modeling', 'execution-and-arrays', 'differentiation']) {
    await page.goto(`/guides/${name}/`);
    const source = readFileSync(new URL(`../../python/${name}.md`, import.meta.url), 'utf8');
    const headings = source.split('\n').filter((line) => line.startsWith('## ')).map((line) => line.slice(3));
    const main = page.getByRole('main');
    for (const heading of headings) await expect(main.getByRole('heading', { level: 2, name: heading, exact: true })).toHaveCount(1);
    for (const link of await page.locator('.right-sidebar a[href^="#"]').all()) {
      const target = (await link.getAttribute('href'))!.slice(1);
      expect(await page.locator('[id]').evaluateAll((elements, id) => elements.filter((element) => element.id === id).length, target)).toBe(1);
    }
    await expect(main.locator('.eq-guide-source')).toHaveText('View guide source');
    await expect(main.getByRole('link', { name: 'View guide source' })).toHaveAttribute('href', `https://github.com/nkiyohara/eqiora/blob/${process.env.EQIORA_SITE_SOURCE_SHA}/docs/python/${name}.md`);
    expect(await main.locator('pre').count()).toBeGreaterThan(0);
  }
  await expect(page.getByRole('main').getByRole('link', { name: 'Execution, diagnostics, and arrays' })).toHaveAttribute('href', '/guides/execution-and-arrays/');
  await context.close();
});

test('Home to released run to Guide and Learn works with keyboard and no JavaScript', async () => {
  const context = await browser.newContext({ baseURL: BASE_URL, javaScriptEnabled: false, viewport: { width: 320, height: 900 } });
  const page = await context.newPage();
  await page.goto('/');
  const start = page.getByRole('banner').getByRole('link', { name: 'Get started', exact: true });
  await start.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/get-started\/$/);
  await expect(page.getByRole('main')).toContainText('0.3678794412');
  const guide = page.getByRole('main').getByRole('link', { name: 'run a model and inspect its result', exact: true });
  await guide.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/guides\/run-and-inspect\/$/);
  await expect(page.getByRole('main')).toContainText('same source revision');
  await page.getByRole('main').getByRole('link', { name: 'ODE lesson and exercises' }).click();
  await expect(page).toHaveURL(/\/learn\/mathematical-modeling\/ordinary-differential-equations\/$/);
  await page.emulateMedia({ media: 'print' });
  await expect(page.locator('.katex-display').first()).toBeVisible();
  await page.emulateMedia({ media: 'screen' });
  await page.getByRole('main').getByRole('link', { name: 'Next: Fields and spatial domains' }).click();
  await expect(page.getByRole('heading', { name: '5. Fields and spatial domains', exact: true })).toBeVisible();
  await assertNoPageOverflow(page);
  await page.goto('/gallery/karman-vortex-street/');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(page.locator('img.eq-gallery-motion__still')).toBeVisible();
  await expect(page.locator('video.eq-gallery-motion__video')).toBeHidden();
  await context.close();
});

test('search classifies migrated destinations and displaced routes are absent', async () => {
  const context = await browser.newContext({ baseURL: BASE_URL });
  const page = await context.newPage();
  await page.goto('/');
  const results = await page.evaluate(async () => {
    const dynamicImport = new Function('url', 'return import(url)');
    const index = await dynamicImport('/pagefind/pagefind.js');
    return Promise.all(['constitutive laws', 'structured failures'].map(async (query) => {
      const matches = await index.search(query);
      return Promise.all(matches.results.map(async (item) => (await item.data()).url));
    }));
  });
  expect(results[0].some((url) => url.includes('/learn/mathematical-modeling/constitutive-laws/'))).toBe(true);
  expect(results[1].some((url) => url.includes('/guides/execution-and-arrays/'))).toBe(true);
  await page.getByRole('button', { name: /Search/ }).first().click();
  await page.locator('dialog input.pagefind-ui__search-input').fill('structured failures');
  await expect(page.locator('[data-eq-content-type]').filter({ hasText: 'Content type: Guide' }).first()).toBeVisible();
  for (const route of ['/textbooks/', '/python/', '/api/', '/examples/', '/concepts/', '/architecture/', '/gallery/transient-cylinder-startup/', '/capabilities/', '/reference/mcp/', '/reference/control-v2/']) {
    expect((await context.request.get(route)).status(), route).toBe(404);
  }
  await context.close();
});


test('Learn opens every subject directly without JavaScript', async () => {
  const context = await browser.newContext({ baseURL: BASE_URL, javaScriptEnabled: false });
  const page = await context.newPage();
  const subjects = SITE_ROUTES.filter((route) => /^\/learn\/[^/]+\/$/.test(route));
  expect(subjects).toHaveLength(7);
  for (const route of subjects) {
    await page.goto('/learn/');
    const link = page.getByRole('main').locator(`a[href="${route}"]`).first();
    await expect(link).toBeVisible();
    await link.focus();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(new URL(route, BASE_URL).href);
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
    expect(await page.getByRole('main').locator('a[href^="/learn/"]').count()).toBeGreaterThan(0);
  }
  await context.close();
});
