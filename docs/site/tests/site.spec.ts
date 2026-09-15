import { expect, test } from '@playwright/test';

import {
  assertAccessibleTooltip,
  assertCoreVisible,
  assertNoFakeExecutionControls,
  assertNoSeriousAxeViolations,
  assertSemanticStages,
  assertVisibleSourceFallback,
  rejectExternalRequests,
  ROUTES,
} from './support';

test('homepage headline stays readable on desktop and mobile', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 375, height: 812 }]) {
    await page.setViewportSize(viewport);
    await page.goto('/');
    const headline = page.getByRole('heading', { level: 1, name: 'Any physics. One language.', exact: true });
    await expect(headline).toBeVisible();
    await expect(headline).toHaveText('Any physics. One language.');
    const bounds = await headline.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width);
    expect(await headline.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  }
});

test('homepage wake moves, pauses, and links to its walkthrough', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('/');
  const video = page.locator('.eq-preview__video');
  await expect(video).toBeVisible();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.currentTime)).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Pause animation', exact: true }).click();
  expect(await video.evaluate((node: HTMLVideoElement) => node.paused)).toBe(true);
  await page.getByRole('button', { name: 'Play animation', exact: true }).click();
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.paused)).toBe(false);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(video).toBeHidden();
  await expect(page.locator('.eq-preview__still')).toBeVisible();
  expect(await video.evaluate((node: HTMLVideoElement) => node.paused)).toBe(true);
  await page.getByRole('link', { name: 'Explore the Kármán vortex street', exact: true }).click();
  await expect(page).toHaveURL(/\/gallery\/karman-vortex-street\/$/);
});

test('required routes, semantic stages, controls, and 404 are real static surfaces', async ({ page }) => {
  const external = await rejectExternalRequests(page);
  for (const route of ROUTES) {
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    await assertCoreVisible(page);
  }

  await page.goto('/');
  await expect(page.getByRole('banner').getByRole('link', { name: 'Eqiora', exact: true })).toHaveAttribute('href', '/');
  await expect(page.locator('.eq-actions').getByRole('link', { name: 'Get started', exact: true })).toHaveAttribute('href', '/get-started/');
  await expect(page.getByRole('link', { name: 'Explore simulations', exact: true })).toHaveAttribute('href', '/gallery/');
  await expect(page.locator('.eq-preview__video')).toBeVisible();
  await expect(page.locator('.eq-preview__label')).toContainText('Flow simulation');
  await assertAccessibleTooltip(
    page,
    page.getByRole('button', { name: /search/i }).filter({ visible: true }).first(),
    /search/i,
  );
  await assertAccessibleTooltip(
    page,
    page.getByRole('combobox', { name: /theme/i }).filter({ visible: true }).first(),
    /theme/i,
  );

  await page.goto('/gallery/');
  const card = page.getByRole('link', { name: /Exact-cylinder steady Stokes/i }).first();
  await expect(card).toHaveAttribute('href', '/gallery/exact-cylinder-steady-stokes/');
  await card.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/gallery\/exact-cylinder-steady-stokes\/$/);

  await assertSemanticStages(page);
  await assertNoFakeExecutionControls(page);
  const missing = await page.goto('/this-route-does-not-exist');
  expect(missing?.status()).toBe(404);
  await expect(page.getByRole('heading', { level: 1, name: '404', exact: true })).toBeVisible();
  await expect(page.getByText(/Page not found/i)).toBeVisible();

  const oldSocial = await page.goto('/assets/social-card.svg');
  expect(oldSocial?.status()).toBe(404);
  expect(oldSocial?.headers().location).toBeUndefined();
  const social = await page.goto('/social-card.svg');
  expect(social?.status()).toBe(200);
  expect(social?.headers()['content-type']).toContain('image/svg+xml');
  expect(external).toEqual([]);
});

test('mixed-boundary elasticity is a static source-traced second gallery surface', async ({ page }) => {
  const external = await rejectExternalRequests(page);
  await page.goto('/gallery/');
  const card = page.getByRole('link', { name: /Mixed-boundary linear elasticity/i });
  await expect(card).toHaveAttribute('href', '/gallery/mixed-boundary-elasticity/');
  await card.click();
  await expect(page).toHaveURL(/\/gallery\/mixed-boundary-elasticity\/$/);
  await expect(
    page.getByRole('img', {
      name: /Reference and deformed meshes for the 2D mixed-boundary/i,
    }),
  ).toBeVisible();
  await assertNoFakeExecutionControls(page);
  expect(external).toEqual([]);
});

test('Kármán vortex street publishes accessible caller-owned motion', async ({ page }) => {
  const external = await rejectExternalRequests(page);
  await page.goto('/gallery/');
  const card = page.getByRole('link', { name: 'Kármán vortex street', exact: true });
  await expect(card).toHaveAttribute('href', '/gallery/karman-vortex-street/');
  await card.click();
  await expect(page).toHaveURL(/\/gallery\/karman-vortex-street\/$/);
  await expect(page.getByRole('heading', { name: 'Numerical method', exact: true })).toBeVisible();

  const video = page.locator('video.eq-gallery-motion__video');
  await expect(video).toHaveAttribute('controls', '');
  await expect(video.locator('source[type="video/webm"]')).toHaveCount(1);
  await expect(video.locator('source[type="video/mp4"]')).toHaveCount(1);
  await expect(video).not.toHaveAttribute('autoplay', /.*/u);
  await expect(page.locator('#wake-motion-description')).toContainText(
    'alternating signed vorticity',
  );
  await assertNoSeriousAxeViolations(page);

  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(video).toBeHidden();
  await expect(page.locator('img.eq-gallery-motion__still')).toBeVisible();
  await assertNoFakeExecutionControls(page);
  expect(external).toEqual([]);
});

test('Kármán vortex-street links to its current Python source', async ({ page }) => {
  await page.goto('/gallery/karman-vortex-street/');
  const sourceSha = process.env.EQIORA_SITE_SOURCE_SHA;
  await expect(page.locator(`a[href="https://github.com/nkiyohara/eqiora/blob/${sourceSha}/examples/python/karman_vortex_street.py"]`).first()).toBeVisible();
  await expect(page.getByRole('link', { name: 'Get started', exact: true }).first()).toBeVisible();
});

test('Pagefind returns one representative from every public reference family', async ({ page }) => {
  const external = await rejectExternalRequests(page);
  await page.goto('/');
  const expectations = [
    ['eqiora Diagnostic', '/reference/python/eqiora/'],
    ['eqiora::Diagnostic', '/reference/rust/'],
    ['eqiora::api::CadBoxIntentV1', '/reference/rust/'],
    ['eqiora::api module', '/reference/rust/'],
    ['eqiora check', '/reference/cli/'],
  ] as const;
  for (const [query, expectedRoute] of expectations) {
    const urls = await page.evaluate(async (searchQuery) => {
      const dynamicImport = new Function('specifier', 'return import(specifier)') as (
        specifier: string,
      ) => Promise<{ search: (query: string) => Promise<{ results: Array<{ data: () => Promise<{ url: string }> }> }> }>;
      const pagefind = await dynamicImport('/pagefind/pagefind.js');
      const result = await pagefind.search(searchQuery);
      return Promise.all(result.results.slice(0, 12).map(async (entry) => (await entry.data()).url));
    }, query);
    const paths = urls.map((url) => new URL(url, 'http://127.0.0.1:4173').pathname);
    expect(paths, query).toContain(expectedRoute);
  }
  expect(external).toEqual([]);
});

test('JavaScript-disabled core remains navigable and mathematically complete', async ({ browser }) => {
  test.setTimeout(120_000);
  const context = await browser.newContext({
    baseURL: 'http://127.0.0.1:4173',
    javaScriptEnabled: false,
    serviceWorkers: 'block',
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();
  const external = await rejectExternalRequests(page);
  for (const route of ROUTES) {
    await page.goto(route);
    await assertCoreVisible(page);
  }
  await page.goto('/gallery/exact-cylinder-steady-stokes/');
  await assertSemanticStages(page);
  await expect(page.getByRole('img', { name: /Steady Stokes pressure around a cylinder/i })).toBeVisible();
  expect(await page.locator('math').count()).toBeGreaterThanOrEqual(2);
  expect(await page.locator('.katex-html').count()).toBeGreaterThanOrEqual(2);
  await assertVisibleSourceFallback(page);
  await page.getByRole('link', { name: 'Gallery', exact: true }).filter({ visible: true }).first().click();
  await expect(page).toHaveURL(/\/gallery\/$/);
  expect(external).toEqual([]);
  await context.close();
});
