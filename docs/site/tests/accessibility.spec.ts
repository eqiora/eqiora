import { expect, test, type Browser, type Page } from '@playwright/test';

import {
  assertCoreVisible,
  assertHonest320Reflow,
  assertKeyboardFocusVisible,
  assertMinimumTargetSizes,
  assertNoFakeExecutionControls,
  assertNoPageOverflow,
  assertNoSeriousAxeViolations,
  assertOrdinaryRoutePage,
  assertOrdinaryRoutePlan,
  assertProductTableRouteInvariant,
  assertProductTableRouteGreen,
  assertReducedMotion,
  assertSemanticStages,
  assertVisibleSourceFallback,
  attachGrossScreenshot,
  BASE_URL,
  conditionDependentAxeRuleIds,
  createOrdinaryRoutePlan,
  DIAGNOSTIC_ROUTE,
  installTableObserver,
  launchOfficialBrowser,
  measureSitePhase,
  navigateSitePage,
  rejectExternalRequests,
  reportSitePhaseMetrics,
  resetSitePhaseMetrics,
  ROUTES,
  SITE_ROUTES,
  TABLE_ROUTES,
  type ConditionAxeProjection,
} from './support';

async function assertCylinderContent(page: Page): Promise<void> {
  const sourceSha = process.env.EQIORA_SITE_SOURCE_SHA;
  expect(sourceSha).toMatch(/^[0-9a-f]{40}$/u);
  await assertSemanticStages(page);
  await assertVisibleSourceFallback(page);
  const figures = page.getByRole('figure').getByRole('img');
  await expect(figures).toHaveCount(2); // Geometry and the single current pressure plot.
  for (let offset = 0; offset < 2; offset += 1) await expect(figures.nth(offset)).toBeVisible();
  expect(await page.locator('math').count()).toBeGreaterThanOrEqual(2);
  expect(await page.locator('math[display="block"]').count()).toBeGreaterThan(0);
  expect(await page.locator('math:not([display="block"])').count()).toBeGreaterThan(0);
  expect(await page.locator('.katex-html').count()).toBeGreaterThanOrEqual(2);
  await expect(
    page.getByRole('link', { name: 'Stage 4 Submit and result', exact: true }),
  ).toHaveAttribute('href', '#submit-and-result');
  await expect(
    page.getByRole('link', {
      name: 'Eqiora source form: Python resolve/run path',
      exact: true,
    }),
  ).toHaveAttribute(
    'href',
    `https://github.com/nkiyohara/eqiora/blob/${sourceSha}/examples/python/exact_cylinder_stokes.py#L51-L67`,
  );
  await assertNoFakeExecutionControls(page);
}

async function assertDiagnosticIdentity(page: Page): Promise<void> {
  await expect(page).toHaveTitle('Diagnostic in eqiora - Rust');
  await expect(page.locator('body')).toHaveClass(/\brustdoc\b.*\bstruct\b/);
  await expect(page.locator('main')).toHaveCount(1);
  await expect(page.locator('main h1')).toContainText('Struct Diagnostic');
  await expect(page.locator('main h1')).toContainText('Copy item path');
  await assertNoSeriousAxeViolations(page);
}

async function runOrdinaryChunk(id: 'A' | 'B' | 'C'): Promise<void> {
  const plan = createOrdinaryRoutePlan();
  expect(assertOrdinaryRoutePlan(plan)).toEqual([...SITE_ROUTES]);
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  const external = await rejectExternalRequests(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  for (const route of plan[id]) {
    await navigateSitePage(page, route);
    await assertOrdinaryRoutePage(page, route);
    if (route !== '/gallery/exact-cylinder-steady-stokes/') {
      await assertNoSeriousAxeViolations(page);
    }
  }
  expect(external).toEqual([]);
  await context.close();
  expect(context.pages()).toEqual([]);
}

type ProductTableProjection = 'invariant' | 'dynamic' | 'forced-color';

type ProductTableMatrixCell = Readonly<{
  forcedColors: 'none' | 'active';
  width: 1280 | 390 | 320;
  projections: readonly ProductTableProjection[];
}>;

const PRODUCT_TABLE_MATRIX = [
  {
    forcedColors: 'none',
    width: 1280,
    projections: ['invariant', 'dynamic', 'forced-color'],
  },
  { forcedColors: 'none', width: 390, projections: ['dynamic'] },
  { forcedColors: 'none', width: 320, projections: ['dynamic'] },
  {
    forcedColors: 'active',
    width: 1280,
    projections: ['dynamic', 'forced-color'],
  },
  { forcedColors: 'active', width: 390, projections: ['dynamic'] },
  { forcedColors: 'active', width: 320, projections: ['dynamic'] },
] as const satisfies readonly ProductTableMatrixCell[];

const EXPECTED_PRODUCT_TABLE_MATRIX_KEYS = [
  'none/1280',
  'none/390',
  'none/320',
  'active/1280',
  'active/390',
  'active/320',
] as const;

function tableMatrixKey(cell: ProductTableMatrixCell): string {
  return `${cell.forcedColors}/${cell.width}`;
}

function hasTableProjection(
  cell: ProductTableMatrixCell,
  projection: ProductTableProjection,
): boolean {
  return cell.projections.includes(projection);
}

function axeProjectionForCell(
  cell: ProductTableMatrixCell,
): ConditionAxeProjection {
  return hasTableProjection(cell, 'forced-color') ? 'forced-color' : 'scroll-only';
}

function validateProductTableMatrix(plan: readonly ProductTableMatrixCell[]): void {
  const expectedKeys: readonly string[] = EXPECTED_PRODUCT_TABLE_MATRIX_KEYS;
  const expected = new Set(expectedKeys);
  const actualKeys = plan.map(tableMatrixKey);
  const unknown = actualKeys.find((key) => !expected.has(key));
  if (unknown) throw new Error(`TABLE-MATRIX-UNKNOWN: ${unknown}`);
  const seen = new Set<string>();
  for (const key of actualKeys) {
    if (seen.has(key)) throw new Error(`TABLE-MATRIX-DUPLICATE: ${key}`);
    seen.add(key);
  }
  const missing = expectedKeys.find((key) => !seen.has(key));
  if (missing) throw new Error(`TABLE-MATRIX-MISSING: ${missing}`);
  if (actualKeys.length !== expectedKeys.length) {
    throw new Error(`TABLE-MATRIX-CARDINALITY: ${actualKeys.length}`);
  }
  const outOfOrder = actualKeys.findIndex((key, index) => key !== expectedKeys[index]);
  if (outOfOrder !== -1) {
    throw new Error(
      `TABLE-MATRIX-ORDER: ${actualKeys[outOfOrder]} at ${outOfOrder}, expected ${expectedKeys[outOfOrder]}`,
    );
  }
  const invariantCells = plan.filter((cell) => hasTableProjection(cell, 'invariant'));
  if (invariantCells.length !== 1 || tableMatrixKey(invariantCells[0]) !== 'none/1280') {
    throw new Error(
      `TABLE-MATRIX-INVARIANT: ${invariantCells.map(tableMatrixKey).join(',') || 'missing'}`,
    );
  }
  const missingDynamic = plan.find((cell) => !hasTableProjection(cell, 'dynamic'));
  if (missingDynamic) {
    throw new Error(`TABLE-MATRIX-DYNAMIC: ${tableMatrixKey(missingDynamic)}`);
  }
  const forcedColorKeys = plan
    .filter((cell) => hasTableProjection(cell, 'forced-color'))
    .map(tableMatrixKey);
  if (forcedColorKeys.join(',') !== 'none/1280,active/1280') {
    throw new Error(`TABLE-MATRIX-FORCED-COLOR: ${forcedColorKeys.join(',') || 'missing'}`);
  }
}

function assertProductTableMatrixPlan(): void {
  const positive = PRODUCT_TABLE_MATRIX.map((cell) => ({ ...cell }));
  expect(() => validateProductTableMatrix(positive)).not.toThrow();

  const missing = positive.slice(0, -1);
  expect(() => validateProductTableMatrix(missing)).toThrow(
    'TABLE-MATRIX-MISSING: active/320',
  );

  const duplicated = [...positive.slice(0, -1), { ...positive[0] }];
  expect(() => validateProductTableMatrix(duplicated)).toThrow(
    'TABLE-MATRIX-DUPLICATE: none/1280',
  );

  const forcedBeforeOrdinary = [...positive.slice(3), ...positive.slice(0, 3)];
  expect(() => validateProductTableMatrix(forcedBeforeOrdinary)).toThrow(
    'TABLE-MATRIX-ORDER: active/1280 at 0, expected none/1280',
  );

  const omittedInvariant = positive.map((cell) => ({
    ...cell,
    projections: cell.projections.filter((projection) => projection !== 'invariant'),
  }));
  expect(() => validateProductTableMatrix(omittedInvariant)).toThrow(
    'TABLE-MATRIX-INVARIANT: missing',
  );

  const duplicatedInvariant = positive.map((cell, index) => ({
    ...cell,
    projections:
      index === 1 ? [...cell.projections, 'invariant' as const] : cell.projections,
  }));
  expect(() => validateProductTableMatrix(duplicatedInvariant)).toThrow(
    'TABLE-MATRIX-INVARIANT: none/1280,none/390',
  );

  const omittedDynamic = positive.map((cell, index) => ({
    ...cell,
    projections:
      index === 5
        ? cell.projections.filter((projection) => projection !== 'dynamic')
        : cell.projections,
  }));
  expect(() => validateProductTableMatrix(omittedDynamic)).toThrow(
    'TABLE-MATRIX-DYNAMIC: active/320',
  );

  const omittedForcedColor = positive.map((cell) => ({
    ...cell,
    projections: cell.projections.filter((projection) => projection !== 'forced-color'),
  }));
  expect(() => validateProductTableMatrix(omittedForcedColor)).toThrow(
    'TABLE-MATRIX-FORCED-COLOR: missing',
  );
}

test.describe.configure({ mode: 'serial' });

let browser: Browser;

test.beforeAll(async () => {
  resetSitePhaseMetrics();
  browser = await launchOfficialBrowser(true);
});

test.afterAll(async () => {
  if (browser) await measureSitePhase('browserLifecycle', () => browser.close());
  reportSitePhaseMetrics('accessibility.spec.ts');
});

for (const id of ['A', 'B', 'C'] as const) {
  test(`00${id} exact deterministic ordinary site chunk is complete and green`, async () => {
    test.setTimeout(300_000);
    await runOrdinaryChunk(id);
  });
}

test('00D exact real Rustdoc Diagnostic ordinary chunk is complete and green', async () => {
  test.setTimeout(300_000);
  const plan = createOrdinaryRoutePlan();
  expect(assertOrdinaryRoutePlan(plan)).toEqual([...SITE_ROUTES]);
  expect(ROUTES).toHaveLength(61);
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  const external = await rejectExternalRequests(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  await navigateSitePage(page, DIAGNOSTIC_ROUTE);
  await assertOrdinaryRoutePage(page, DIAGNOSTIC_ROUTE);
  await assertDiagnosticIdentity(page);
  expect(external).toEqual([]);
  await context.close();
  expect(context.pages()).toEqual([]);
});

test('01 honest 320px O-1 through O-4 composition and retained interaction controls pass', async ({}, testInfo) => {
  test.setTimeout(180_000);
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  const external = await rejectExternalRequests(page);
  await page.setViewportSize({ width: 320, height: 844 });
  await navigateSitePage(page, '/');
  await assertHonest320Reflow(page);
  await assertNoPageOverflow(page);
  await assertMinimumTargetSizes(page.getByRole('banner').getByRole('link'));
  await assertMinimumTargetSizes(page.getByRole('banner').getByRole('button'));
  await assertMinimumTargetSizes(page.locator('.eq-actions').getByRole('link', { name: 'Get started', exact: true }));
  await assertMinimumTargetSizes(page.getByRole('link', { name: 'Explore simulations', exact: true }));
  await assertKeyboardFocusVisible(page, page.locator('.eq-actions').getByRole('link', { name: 'Get started', exact: true }));

  await navigateSitePage(page, '/gallery/exact-cylinder-steady-stokes/');
  await assertHonest320Reflow(page);
  await assertCoreVisible(page);
  await assertNoPageOverflow(page);
  await assertCylinderContent(page);

  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme, forcedColors: 'none' });
    await page.setViewportSize({ width: 1440, height: 900 });
    await navigateSitePage(page, '/');
    await attachGrossScreenshot(page, testInfo, `home-desktop-${colorScheme}`);
    await navigateSitePage(page, '/gallery/exact-cylinder-steady-stokes/');
    await attachGrossScreenshot(page, testInfo, `cylinder-desktop-${colorScheme}`);
    await page.setViewportSize({ width: 390, height: 844 });
    await navigateSitePage(page, '/');
    await assertNoPageOverflow(page);
    await attachGrossScreenshot(page, testInfo, `home-mobile-${colorScheme}`);
    await navigateSitePage(page, '/gallery/exact-cylinder-steady-stokes/');
    await assertNoPageOverflow(page);
    await attachGrossScreenshot(page, testInfo, `cylinder-mobile-${colorScheme}`);
  }

  await page.emulateMedia({ colorScheme: 'light', forcedColors: 'none', reducedMotion: 'reduce' });
  await navigateSitePage(page, '/gallery/exact-cylinder-steady-stokes/');
  await assertReducedMotion(page);
  await navigateSitePage(page, '/gallery/mixed-boundary-elasticity/');
  await assertReducedMotion(page);
  expect(external).toEqual([]);
  await context.close();
});

test('table structure is complete across current site routes', async () => {
  test.setTimeout(600_000);
  assertProductTableMatrixPlan();
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  await installTableObserver(page);
  const external = await rejectExternalRequests(page);
  const invariantCell = PRODUCT_TABLE_MATRIX.find((cell) =>
    hasTableProjection(cell, 'invariant'),
  );
  if (!invariantCell) throw new Error('TABLE-MATRIX-INVARIANT: missing');
  await page.emulateMedia({ forcedColors: invariantCell.forcedColors });
  await page.setViewportSize({ width: invariantCell.width, height: 900 });

  let invariantRoutes = 0;
  for (const expected of TABLE_ROUTES) {
    await assertProductTableRouteInvariant(page, expected);
    invariantRoutes += 1;
  }
  expect(invariantRoutes).toBe(40);
  await navigateSitePage(page, '/reference/python/eqiora/');
  await expect(page.locator('main table')).toHaveCount(0);

  expect(external).toEqual([]);
  await context.close();
  expect(context.pages()).toEqual([]);
});

test('tables reflow and remain accessible across widths and forced colours', async () => {
  test.setTimeout(1_200_000);
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  await installTableObserver(page);
  const external = await rejectExternalRequests(page);
  const work = {
    cells: 0,
    routes: 0,
    dynamicProjections: 0,
    conditionAxeCalls: 0,
    conditionAxeRuleApplications: 0,
    contexts: 1,
    pages: 1,
  };

  try {
    for (const cell of PRODUCT_TABLE_MATRIX) {
      const key = tableMatrixKey(cell);
      await test.step(`matrix ${key}`, async () => {
        await page.emulateMedia({ forcedColors: cell.forcedColors });
        await page.setViewportSize({ width: cell.width, height: 900 });
        for (const expected of TABLE_ROUTES) {
          const externalBefore = external.length;
          await test.step(`${key} ${expected.route}`, async () => {
            await assertProductTableRouteGreen(
              page,
              expected,
              axeProjectionForCell(cell),
            );
            expect(
              external.slice(externalBefore),
              `external requests at ${key} ${expected.route}`,
            ).toEqual([]);
          });
          work.routes += 1;
          work.dynamicProjections += 1;
          work.conditionAxeCalls += 1;
          work.conditionAxeRuleApplications += conditionDependentAxeRuleIds(
            axeProjectionForCell(cell),
          ).length;
        }
      });
      work.cells += 1;
    }
    expect(external).toEqual([]);
    expect(work).toEqual({
      cells: 6,
      // Each declared route runs in six cells; two cells apply three Axe
      // rules and the remaining four apply one. Derive from the route owner,
      // not observed work, so newly documented tables retain full coverage.
      routes: TABLE_ROUTES.length * 6,
      dynamicProjections: TABLE_ROUTES.length * 6,
      conditionAxeCalls: TABLE_ROUTES.length * 6,
      conditionAxeRuleApplications: TABLE_ROUTES.length * (2 * 3 + 4),
      contexts: 1,
      pages: 1,
    });
  } finally {
    await context.close();
  }
  expect(context.pages()).toEqual([]);
});

test('03 forced colours retain core content and exact non-table accessibility boundaries', async () => {
  test.setTimeout(300_000);
  const context = await browser.newContext({
    baseURL: BASE_URL,
    locale: 'en-GB',
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  const external = await rejectExternalRequests(page);
  await page.emulateMedia({ forcedColors: 'active' });
  await page.setViewportSize({ width: 320, height: 844 });
  for (const route of ROUTES) {
    await navigateSitePage(page, route);
    await assertCoreVisible(page);
    await assertNoPageOverflow(page);
    if (route === '/gallery/exact-cylinder-steady-stokes/') {
      await assertCylinderContent(page);
    }
    if (!TABLE_ROUTES.some((expected) => expected.route === route)) {
      await assertNoSeriousAxeViolations(page);
    }
  }
  await navigateSitePage(page, '/');
  await assertKeyboardFocusVisible(page, page.locator('.eq-actions').getByRole('link', { name: 'Get started', exact: true }));
  expect(external).toEqual([]);
  await context.close();
});
