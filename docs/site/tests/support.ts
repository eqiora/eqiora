import AxeBuilder from '@axe-core/playwright';
import axe from 'axe-core';
import {
  chromium,
  expect,
  type Browser,
  type CDPSession,
  type Locator,
  type Page,
  type TestInfo,
} from '@playwright/test';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { lstat, readFile, realpath } from 'node:fs/promises';

export const BASE_URL = 'http://127.0.0.1:4173';
export const DIAGNOSTIC_ROUTE = '/reference/rust/api/eqiora/struct.Diagnostic.html';
export const SITE_ROUTES = [
  '/',
  '/contributing/architecture/',
  '/capabilities/',
  '/guides/how-eqiora-fits-together/',
  '/contributing/',
  '/evidence/',
  '/gallery/',
  '/gallery/exact-cylinder-steady-stokes/',
  '/gallery/mixed-boundary-elasticity/',
  '/gallery/karman-vortex-street/',
  '/get-started/',
  '/guides/',
  '/guides/run-and-inspect/',
  '/guides/differentiation/',
  '/guides/execution-and-arrays/',
  '/guides/modeling/',
  '/reference/',
  '/reference/language/',
  '/reference/language/composition/',
  '/reference/language/declarations/',
  '/reference/language/equations/',
  '/reference/language/units/',
  '/reference/standard-packages/',
  '/reference/standard-packages/continuum/',
  '/reference/standard-packages/electrical/',
  '/reference/cli/',
  '/reference/control-v2/',
  '/reference/mcp/',
  '/reference/python/',
  '/reference/python/diff/',
  '/reference/python/eqiora/',
  '/reference/python/fem/',
  '/reference/python/formulation/',
  '/reference/python/fvm/',
  '/reference/python/lang/',
  '/reference/python/solve/',
  '/reference/python/time/',
  '/reference/python/viewer/',
  '/reference/python/fluid/',
  '/reference/python/fsi/',
  '/reference/python/geometry/',
  '/reference/python/jax/',
  '/reference/python/matplotlib/',
  '/reference/python/meshing/',
  '/reference/python/solid/',
  '/reference/python/torch/',
  '/reference/python/trajectory/',
  '/reference/rust/',
  '/release-notes/',
  '/learn/',
  '/learn/mathematical-modeling/',
  '/learn/mathematical-modeling/algebraic-relations-networks/',
  '/learn/mathematical-modeling/boundary-interface-conditions/',
  '/learn/mathematical-modeling/conservation-laws/',
  '/learn/mathematical-modeling/constitutive-laws/',
  '/learn/mathematical-modeling/fields-spatial-domains/',
  '/learn/mathematical-modeling/models-not-simulations/',
  '/learn/mathematical-modeling/ordinary-differential-equations/',
  '/learn/mathematical-modeling/quantities-dimensions-units/',
  '/reference/standard-packages/controls/',
  '/reference/python/units/',
  '/reference/python/colab/',
  '/learn/circuits-dynamics/component-laws/',
  '/learn/circuits-dynamics/conserving-networks/',
  '/learn/circuits-dynamics/decay-and-storage/',
  '/learn/circuits-dynamics/',
  '/learn/circuits-dynamics/resistor-divider/',
  '/learn/circuits-dynamics/voltage-current-power/',
  '/learn/fluid-mechanics/computing-incompressible-flow/',
  '/learn/fluid-mechanics/continuum-and-statics/',
  '/learn/fluid-mechanics/cylinder-wakes/',
  '/learn/fluid-mechanics/exact-parallel-flows/',
  '/learn/fluid-mechanics/',
  '/learn/fluid-mechanics/mass-and-momentum/',
  '/learn/fluid-mechanics/references/',
  '/learn/fluid-mechanics/stokes-and-boundaries/',
  '/learn/heat-transfer/assessing-a-thermal-calculation/',
  '/learn/heat-transfer/boundaries-and-interfaces/',
  '/learn/heat-transfer/conservation-and-fourier-law/',
  '/learn/heat-transfer/heated-body-in-eqiora/',
  '/learn/heat-transfer/',
  '/learn/heat-transfer/steady-conduction/',
  '/learn/heat-transfer/transient-storage/',
  '/learn/inverse-problems/differentiating-a-solved-model/',
  '/learn/inverse-problems/heat-source-recovery/',
  '/learn/inverse-problems/',
  '/learn/inverse-problems/measurements-to-parameters/',
  '/learn/inverse-problems/noise-scaling-and-design/',
  '/learn/inverse-problems/sensitivity-and-identifiability/',
  '/learn/numerical-simulation/errors-and-residuals/',
  '/learn/numerical-simulation/finite-volume-balance/',
  '/learn/numerical-simulation/',
  '/learn/numerical-simulation/refinement-and-reproducibility/',
  '/learn/numerical-simulation/time-integration/',
  '/learn/numerical-simulation/weak-forms-and-finite-elements/',
  '/learn/structural-mechanics/',
  '/learn/structural-mechanics/interpreting-results/',
  '/learn/structural-mechanics/loads-energy/',
  '/learn/structural-mechanics/mixed-boundary-square/',
  '/learn/structural-mechanics/strain-stress/',
  '/learn/structural-mechanics/virtual-work/',
  '/404.html',
] as const;

export const ROUTES = [...SITE_ROUTES, DIAGNOSTIC_ROUTE] as const;

export const STAGES = [
  { id: 'problem-setup', step: 1, title: 'Problem setup' },
  { id: 'model-definition', step: 2, title: 'Eqiora model definition' },
  { id: 'mesh-and-boundaries', step: 3, title: 'Mesh and boundaries' },
  { id: 'submit-and-result', step: 4, title: 'Submit and result' },
  { id: 'pressure-visualization', step: 5, title: 'Pressure visualization' },
  { id: 'reading-pressure', step: 6, title: 'Reading the pressure plot' },
] as const;

export const TABLE_SELECTORS = {
  generic: '.sl-markdown-content table:not(:where(.not-content *))',
  direct: '.sl-markdown-content > table',
  component: '.sl-markdown-content .eq-stage__body > table',
} as const;

type TableRouteShape = Readonly<{
  route: string;
  tables: number;
  direct: number;
  component: number;
}>;

export const TABLE_ROUTES = [
  { route: '/learn/mathematical-modeling/models-not-simulations/', tables: 1, direct: 1, component: 0 },
  { route: '/learn/mathematical-modeling/algebraic-relations-networks/', tables: 1, direct: 1, component: 0 },
  { route: '/learn/mathematical-modeling/fields-spatial-domains/', tables: 1, direct: 1, component: 0 },
  { route: '/capabilities/', tables: 1, direct: 1, component: 0 },
  { route: '/evidence/', tables: 0, direct: 0, component: 0 },
  { route: '/gallery/exact-cylinder-steady-stokes/', tables: 0, direct: 0, component: 0 },
  { route: '/reference/control-v2/', tables: 1, direct: 1, component: 0 },
  { route: '/reference/language/', tables: 1, direct: 1, component: 0 },
  { route: '/reference/language/declarations/', tables: 1, direct: 1, component: 0 },
  { route: '/reference/language/units/', tables: 1, direct: 1, component: 0 },
  { route: '/reference/python/', tables: 1, direct: 1, component: 0 },
  { route: '/reference/rust/', tables: 2, direct: 2, component: 0 },
  { route: '/reference/standard-packages/', tables: 2, direct: 2, component: 0 },
  { route: '/reference/standard-packages/electrical/', tables: 1, direct: 1, component: 0 },
] as const satisfies readonly TableRouteShape[];

export type TableRoute = (typeof TABLE_ROUTES)[number];
export type OrdinaryRoutePlan = Readonly<{
  A: readonly string[];
  B: readonly string[];
  C: readonly string[];
}>;

const REFERENCE_START = SITE_ROUTES.indexOf('/reference/');

export function createOrdinaryRoutePlan(): OrdinaryRoutePlan {
  const plan = {
    A: SITE_ROUTES.filter((route) => route === '/evidence/'),
    B: SITE_ROUTES.slice(0, REFERENCE_START).filter((route) => route !== '/evidence/'),
    C: SITE_ROUTES.slice(REFERENCE_START),
  };
  return Object.freeze({
    A: Object.freeze(plan.A),
    B: Object.freeze(plan.B),
    C: Object.freeze(plan.C),
  });
}

export function assertOrdinaryRoutePlan(plan: OrdinaryRoutePlan): readonly string[] {
  if (REFERENCE_START < 1) throw new Error('route authority missing /reference/');
  if (SITE_ROUTES.length !== 102 || new Set(SITE_ROUTES).size !== 102) {
    throw new Error('route authority is not 102 unique entries');
  }
  const entries = (['A', 'B', 'C'] as const).flatMap((chunk) =>
    plan[chunk].map((route) => ({ chunk, route })),
  );
  const authority = new Set<string>(SITE_ROUTES);
  const unknown = entries.find(({ route }) => !authority.has(route));
  if (unknown) throw new Error(`ORDER-UNKNOWN ${unknown.chunk}: ${unknown.route}`);

  const seen = new Set<string>();
  for (const { route } of entries) {
    if (seen.has(route)) throw new Error(`ORDER-DUPLICATE: ${route}`);
    seen.add(route);
  }
  const missing = SITE_ROUTES.find((route) => !seen.has(route));
  if (missing) throw new Error(`ORDER-MISSING: ${missing}`);

  const expected = createOrdinaryRoutePlan();
  const cardinalities = { A: 1, B: 15, C: 86 } as const;
  for (const chunk of ['A', 'B', 'C'] as const) {
    if (plan[chunk].length !== cardinalities[chunk]) {
      throw new Error(`ORDER-CARDINALITY ${chunk}: ${plan[chunk].length}`);
    }
    const expectedMembers = new Set(expected[chunk]);
    const wrongMember = plan[chunk].find((route) => !expectedMembers.has(route));
    if (wrongMember) throw new Error(`ORDER-MEMBERSHIP ${chunk}: ${wrongMember}`);
    if (plan[chunk].some((route, index) => route !== expected[chunk][index])) {
      throw new Error(`ORDER-REORDER ${chunk}`);
    }
  }
  if (entries.length !== 102 || seen.size !== 102) {
    throw new Error('ORDER-UNION is not exactly 102 entries');
  }

  const byRoute = new Map(entries.map((entry) => [entry.route, entry]));
  if (byRoute.size !== 102) throw new Error('ORDER-CANONICAL duplicate identity');
  const canonical = SITE_ROUTES.map((route) => {
    const entry = byRoute.get(route);
    if (!entry) throw new Error(`ORDER-CANONICAL missing: ${route}`);
    return entry.route;
  });
  if (canonical.some((route, index) => route !== SITE_ROUTES[index])) {
    throw new Error('ORDER-CANONICAL authority mismatch');
  }
  return Object.freeze(canonical);
}

const CHROME_VERSION = '151.0.7922.34';
const CHROME_BYTES = 290_614_600;
const CHROME_SHA256 = '0b20b130e7edd9dd51873be867761295fe0cfad490c2b9a64f95bd3cfc08fa71';
const BROWSERS_JSON_SHA256 = 'f306eed529599b1eaf2f8a85db9de2b23e1a3fe36c2b66434b7c9434fb627a99';

type AxeRuleDefinition = Readonly<{
  id: string;
  impact: string | null;
  enabled: boolean;
  tags: string[];
}>;

const axeInternals = axe as typeof axe & {
  _audit: { rules: AxeRuleDefinition[]; tagExclude: string[] };
};

if (axe.version !== '4.13.0') {
  throw new Error(`unexpected axe identity: ${axe.version}`);
}

const axeExcludedTags = Object.freeze([...axeInternals._audit.tagExclude].sort());
if (axeExcludedTags.join(',') !== 'deprecated,experimental') {
  throw new Error(`axe default excluded tags changed: ${axeExcludedTags.join(',')}`);
}

export const SERIOUS_CRITICAL_AXE_RULE_IDS = Object.freeze(
  axeInternals._audit.rules
    .filter(
      (rule) =>
        rule.enabled &&
        (rule.impact === 'serious' || rule.impact === 'critical') &&
        !rule.tags.some((tag) => axeExcludedTags.includes(tag)),
    )
    .map((rule) => rule.id),
);

export const CONDITION_DEPENDENT_AXE_RULE_IDS = Object.freeze([
  'color-contrast',
  'link-in-text-block',
  'scrollable-region-focusable',
]);

export type ConditionAxeProjection = 'forced-color' | 'scroll-only';

export function conditionDependentAxeRuleIds(
  projection: ConditionAxeProjection,
): readonly string[] {
  return projection === 'forced-color'
    ? CONDITION_DEPENDENT_AXE_RULE_IDS
    : ['scrollable-region-focusable'];
}

if (
  SERIOUS_CRITICAL_AXE_RULE_IDS.length !== 63 ||
  new Set(SERIOUS_CRITICAL_AXE_RULE_IDS).size !== 63
) {
  throw new Error(
    `axe serious/critical projection changed: ${SERIOUS_CRITICAL_AXE_RULE_IDS.length}`,
  );
}

export type SiteVerificationPhase =
  | 'browserLifecycle'
  | 'navigation'
  | 'tableInvariant'
  | 'tableDynamic'
  | 'axeAnalysis'
  | 'screenshot';

type SitePhaseMetric = { calls: number; work: number; elapsedMs: number };

const SITE_PHASES: readonly SiteVerificationPhase[] = [
  'browserLifecycle',
  'navigation',
  'tableInvariant',
  'tableDynamic',
  'axeAnalysis',
  'screenshot',
];

let sitePhaseMetrics = new Map<SiteVerificationPhase, SitePhaseMetric>();

export function resetSitePhaseMetrics(): void {
  sitePhaseMetrics = new Map(
    SITE_PHASES.map((phase) => [phase, { calls: 0, work: 0, elapsedMs: 0 }]),
  );
}

function recordSitePhase(
  phase: SiteVerificationPhase,
  elapsedMs: number,
  work = 1,
): void {
  const metric = sitePhaseMetrics.get(phase) ?? { calls: 0, work: 0, elapsedMs: 0 };
  metric.calls += 1;
  metric.work += work;
  metric.elapsedMs += elapsedMs;
  sitePhaseMetrics.set(phase, metric);
}

export async function measureSitePhase<T>(
  phase: SiteVerificationPhase,
  task: () => Promise<T>,
  work = 1,
): Promise<T> {
  const started = performance.now();
  try {
    return await task();
  } finally {
    recordSitePhase(phase, performance.now() - started, work);
  }
}

export function reportSitePhaseMetrics(scope: string): void {
  const metrics = Object.fromEntries(
    SITE_PHASES.map((phase) => {
      const metric = sitePhaseMetrics.get(phase) ?? { calls: 0, work: 0, elapsedMs: 0 };
      return [phase, { ...metric, elapsedMs: Math.round(metric.elapsedMs) }];
    }),
  );
  console.log(`EQIORA-SITE-WORK ${JSON.stringify({ scope, metrics })}`);
}

resetSitePhaseMetrics();

export async function launchOfficialBrowser(headless = true): Promise<Browser> {
  const started = performance.now();
  try {
    const browserRoot = process.env.PLAYWRIGHT_BROWSERS_PATH;
    expect(browserRoot?.endsWith('/eqiora-pw-1.62.1-r1234')).toBe(true);
    const root = await realpath(browserRoot!);
    expect(root.endsWith('/eqiora-pw-1.62.1-r1234')).toBe(true);
    const executablePath = resolve(root, 'chromium-1234/chrome-linux64/chrome');
    const executableLink = await lstat(executablePath);
    expect(executableLink.isSymbolicLink()).toBe(false);
    expect(executableLink.isFile()).toBe(true);
    expect(executableLink.mode & 0o777).toBe(0o755);
    expect(executableLink.size).toBe(CHROME_BYTES);
    expect(createHash('sha256').update(await readFile(executablePath)).digest('hex')).toBe(
      CHROME_SHA256,
    );

    const require = createRequire(import.meta.url);
    const corePackage = require.resolve('playwright-core/package.json');
    const browsersJson = resolve(dirname(corePackage), 'browsers.json');
    expect(createHash('sha256').update(await readFile(browsersJson)).digest('hex')).toBe(
      BROWSERS_JSON_SHA256,
    );
    const manifest = JSON.parse(await readFile(browsersJson, 'utf8')) as {
      browsers: Array<{ name: string; revision: string; browserVersion: string }>;
    };
    for (const name of ['chromium', 'chromium-headless-shell']) {
      expect(manifest.browsers.find((entry) => entry.name === name)).toMatchObject({
        revision: '1234',
        browserVersion: CHROME_VERSION,
      });
    }

    const browser = await chromium.launch({ executablePath, headless });
    expect(browser.version()).toBe(CHROME_VERSION);
    return browser;
  } finally {
    recordSitePhase('browserLifecycle', performance.now() - started);
  }
}

export async function navigateSitePage(page: Page, route: string): Promise<void> {
  const response = await measureSitePhase('navigation', () => page.goto(route));
  expect(response?.ok(), route).toBe(true);
}

export async function rejectExternalRequests(page: Page): Promise<string[]> {
  const attempts: string[] = [];
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (
      url.protocol === 'data:' ||
      url.protocol === 'blob:' ||
      url.origin === BASE_URL
    ) {
      await route.continue();
      return;
    }
    attempts.push(url.href);
    await route.abort('blockedbyclient');
  });
  return attempts;
}

export async function assertOrdinaryRoutePage(page: Page, route: string): Promise<void> {
  const observation = await page.evaluate((expectedRoute) => {
    const main = document.querySelector('main');
    const heading = main?.querySelector('h1');
    if (!(main instanceof HTMLElement) || !(heading instanceof HTMLElement)) {
      throw new Error(`missing ordinary core at ${expectedRoute}`);
    }
    const mainStyle = getComputedStyle(main);
    const headingStyle = getComputedStyle(heading);
    const mainBox = main.getBoundingClientRect();
    const headingBox = heading.getBoundingClientRect();
    return {
      pathname: location.pathname,
      mainTextLength: main.innerText.trim().length,
      mainVisible:
        mainStyle.display !== 'none' &&
        mainStyle.visibility === 'visible' &&
        Number.parseFloat(mainStyle.opacity) > 0 &&
        mainBox.width > 0 &&
        mainBox.height > 0,
      headingVisible:
        headingStyle.display !== 'none' &&
        headingStyle.visibility === 'visible' &&
        Number.parseFloat(headingStyle.opacity) > 0 &&
        headingBox.width > 0 &&
        headingBox.height > 0,
      documentClient: document.documentElement.clientWidth,
      documentScroll: document.documentElement.scrollWidth,
      bodyScroll: document.body.scrollWidth,
    };
  }, route);
  expect(observation.pathname).toBe(route);
  expect(observation.mainTextLength).toBeGreaterThan(20);
  expect(observation.mainVisible).toBe(true);
  expect(observation.headingVisible).toBe(true);
  expect(observation.documentClient).toBeGreaterThan(0);
  expect(observation.documentScroll).toBeLessThanOrEqual(observation.documentClient + 1);
  expect(observation.bodyScroll).toBeLessThanOrEqual(observation.documentClient + 1);
}

const tableObserverSessions = new WeakMap<Page, CDPSession>();

export async function installTableObserver(page: Page): Promise<void> {
  expect(page.url()).toBe('about:blank');
  await expect(page.locator('main, table')).toHaveCount(0);
  expect(tableObserverSessions.has(page)).toBe(false);
  const session = await page.context().newCDPSession(page);
  await session.send('Runtime.enable');
  tableObserverSessions.set(page, session);
}

export async function assertCoreVisible(page: Page): Promise<void> {
  const main = page.getByRole('main');
  await expect(main).toBeVisible();
  await expect(main.getByRole('heading', { level: 1 })).toBeVisible();
  expect((await main.innerText()).trim().length).toBeGreaterThan(20);
}

export async function assertSemanticStages(page: Page): Promise<void> {
  const sections = page.locator('main section.eq-stage');
  await expect(sections).toHaveCount(STAGES.length);
  for (const [offset, stage] of STAGES.entries()) {
    const name = `Stage ${stage.step} ${stage.title}`;
    const section = sections.nth(offset);
    await expect(section).toHaveAttribute('id', stage.id);
    await expect(section).toHaveAttribute('data-step', String(stage.step));
    const heading = section.getByRole('heading', { level: 2, name, exact: true });
    await expect(heading).toHaveCount(1);
    await expect(heading).toBeVisible();
    const region = page.getByRole('region', { name, exact: true });
    await expect(region).toHaveCount(1);
    expect(await region.evaluate((element, expected) => element === expected, await section.elementHandle())).toBe(
      true,
    );
    const emoji = heading.locator('.eq-stage-marker__emoji');
    await expect(emoji).toHaveCount(1);
    await expect(emoji).toHaveAttribute('aria-hidden', 'true');
  }
}

export async function assertVisibleSourceFallback(page: Page): Promise<void> {
  const fallbacks = page.getByText('Eqiora source form', { exact: true });
  expect(await fallbacks.count()).toBeGreaterThan(0);
  for (let offset = 0; offset < (await fallbacks.count()); offset += 1) {
    await expect(fallbacks.nth(offset)).toBeVisible();
  }
}

export async function seriousAxeViolations(page: Page) {
  const result = await measureSitePhase(
    'axeAnalysis',
    () =>
      new AxeBuilder({ page })
        // Keep every violation node; callers do not consume passing-node selectors.
        .options({ resultTypes: ['violations'] })
        .withRules([...SERIOUS_CRITICAL_AXE_RULE_IDS])
        .analyze(),
    SERIOUS_CRITICAL_AXE_RULE_IDS.length,
  );
  expect(
    result.violations.every(
      (violation) => violation.impact === 'serious' || violation.impact === 'critical',
    ),
  ).toBe(true);
  return result.violations;
}

export async function conditionDependentAxeViolations(
  page: Page,
  projection: ConditionAxeProjection = 'forced-color',
) {
  const rules = conditionDependentAxeRuleIds(projection);
  expect(
    rules.every((rule) => SERIOUS_CRITICAL_AXE_RULE_IDS.includes(rule)),
  ).toBe(true);
  const result = await measureSitePhase(
    'axeAnalysis',
    () =>
      new AxeBuilder({ page })
        .options({ resultTypes: ['violations'] })
        .withRules([...rules])
        .analyze(),
    rules.length,
  );
  return result.violations;
}

export async function assertNoSeriousAxeViolations(page: Page): Promise<void> {
  expect(await seriousAxeViolations(page)).toEqual([]);
}

export async function assertNoPageOverflow(page: Page): Promise<void> {
  const dimensions = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    documentScroll: document.documentElement.scrollWidth,
    bodyScroll: document.body.scrollWidth,
  }));
  expect(
    dimensions.documentScroll,
    `document overflowed: ${JSON.stringify(dimensions)}`,
  ).toBeLessThanOrEqual(dimensions.client + 1);
  expect(dimensions.bodyScroll, `body overflowed: ${JSON.stringify(dimensions)}`).toBeLessThanOrEqual(
    dimensions.client + 1,
  );
}

export async function assertHonest320Reflow(page: Page): Promise<void> {
  expect(page.viewportSize()).toEqual({ width: 320, height: 844 });
  const observation = await page.evaluate(() => ({
    innerWidth,
    documentClient: document.documentElement.clientWidth,
    visualScale: visualViewport?.scale ?? 1,
    viewportMeta: document.querySelector('meta[name="viewport"]')?.getAttribute('content') ?? null,
    roots: ['html', 'body', 'main'].map((selector) => {
      const node = document.querySelector(selector);
      if (!node) throw new Error(`missing ${selector}`);
      const style = getComputedStyle(node);
      return { selector, zoom: style.zoom, transform: style.transform };
    }),
  }));
  expect(observation.innerWidth).toBe(320);
  expect(observation.documentClient).toBeGreaterThan(0);
  expect(observation.documentClient).toBeLessThanOrEqual(320);
  expect(observation.visualScale).toBe(1);
  if (observation.viewportMeta !== null) {
    const directives = observation.viewportMeta
      .toLowerCase()
      .split(',')
      .map((directive) => directive.trim());
    expect(directives).toContain('width=device-width');
    const initialScale = directives.find((directive) => directive.startsWith('initial-scale='));
    if (initialScale !== undefined) expect(initialScale).toBe('initial-scale=1');
    expect(directives.some((directive) => directive.startsWith('minimum-scale='))).toBe(false);
    expect(directives.some((directive) => directive.startsWith('maximum-scale='))).toBe(false);
    expect(directives.some((directive) => directive.startsWith('user-scalable='))).toBe(false);
  }
  for (const root of observation.roots) {
    expect(root.zoom, `${root.selector} scales the CSS viewport`).toBe('1');
    expect(root.transform, `${root.selector} transforms the CSS viewport`).toBe('none');
  }
}

export async function assertMinimumTargetSizes(locator: Locator): Promise<void> {
  const count = await locator.count();
  expect(count).toBeGreaterThan(0);
  for (let offset = 0; offset < count; offset += 1) {
    const target = locator.nth(offset);
    if (!(await target.isVisible())) continue;
    const box = await target.boundingBox();
    expect(box, `target ${offset} has no box`).not.toBeNull();
    expect(box!.width, `target ${offset} is narrower than 44px`).toBeGreaterThanOrEqual(44);
    expect(box!.height, `target ${offset} is shorter than 44px`).toBeGreaterThanOrEqual(44);
  }
}

export async function assertKeyboardFocusVisible(page: Page, target: Locator): Promise<void> {
  await page.locator('body').click({ position: { x: 1, y: 1 } });
  const handle = await target.elementHandle();
  expect(handle).not.toBeNull();
  let reached = false;
  for (let offset = 0; offset < 100; offset += 1) {
    await page.keyboard.press('Tab');
    reached = await page.evaluate((element) => document.activeElement === element, handle);
    if (reached) break;
  }
  expect(reached, 'target is absent from the keyboard focus order').toBe(true);
  const focus = await target.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      visiblePseudo: element.matches(':focus-visible'),
      outlineStyle: style.outlineStyle,
      outlineWidth: Number.parseFloat(style.outlineWidth),
      boxShadow: style.boxShadow,
    };
  });
  expect(focus.visiblePseudo).toBe(true);
  expect(
    (focus.outlineStyle !== 'none' && focus.outlineWidth >= 1) || focus.boxShadow !== 'none',
    `focus is not visibly drawn: ${JSON.stringify(focus)}`,
  ).toBe(true);
}

export async function assertAccessibleTooltip(
  page: Page,
  control: Locator,
  name: RegExp,
): Promise<void> {
  await expect(control).toBeVisible();
  const nativeTitle = (await control.getAttribute('title'))?.trim();
  if (nativeTitle) {
    expect(nativeTitle).toMatch(name);
    return;
  }
  await control.hover();
  await expect(page.getByRole('tooltip', { name })).toBeVisible();
}

export async function assertNoFakeExecutionControls(page: Page): Promise<void> {
  const executionLabel = /\b(run|submit|reset|start|begin|try|solv\w*|execut\w*|simulat\w*|comput\w*|calculat\w*|launch\w*|evaluat\w*|process\w*|generat\w*|analy[sz]\w*|predict\w*)\b/i;
  const actionLeading = /^\s*(run|submit|reset|start|begin|try|solv\w*|execut\w*|simulat\w*|comput\w*|calculat\w*|launch\w*|evaluat\w*|process\w*|generat\w*|analy[sz]\w*|predict\w*)\b/i;
  const normalize = (value: string) => value.trim().split(/\s+/u).filter(Boolean).join(' ');
  expect(await page.getByRole('button', { name: executionLabel }).count()).toBe(0);
  expect(await page.getByRole('link', { name: actionLeading }).count()).toBe(0);
  const controls = page.locator('button, [role="button"], input[type="button" i], input[type="submit" i], input[type="reset" i], input[type="image" i]');
  for (let offset = 0; offset < (await controls.count()); offset += 1) {
    const control = controls.nth(offset);
    expect(normalize(await control.innerText())).not.toMatch(executionLabel);
    const alternatives = await control.locator('img').evaluateAll((images) =>
      images
        .filter((image) => !image.closest('[hidden], [aria-hidden="true"]'))
        .map((image) => (image as HTMLImageElement).alt),
    );
    for (const alternative of alternatives) {
      expect(normalize(alternative)).not.toMatch(executionLabel);
    }
  }
  const inputs = page.locator('input[type="button" i], input[type="submit" i], input[type="reset" i], input[type="image" i]');
  for (let offset = 0; offset < (await inputs.count()); offset += 1) {
    expect(normalize(await inputs.nth(offset).inputValue())).not.toMatch(executionLabel);
  }
  const links = page.locator('a, [role="link"]');
  for (let offset = 0; offset < (await links.count()); offset += 1) {
    const link = links.nth(offset);
    expect(normalize(await link.innerText())).not.toMatch(actionLeading);
    const href = (await link.getAttribute('href'))?.trim().toLowerCase();
    expect(['', '#', 'javascript:void(0)']).not.toContain(href ?? '');
  }
}

type TableObservation = {
  document: { inner: number; client: number; scroll: number; bodyScroll: number };
  counts: { main: number; all: number; generic: number; direct: number; component: number };
  failures: {
    tag: number;
    relation: number;
    wrapper: number;
    role: number;
    focus: number;
    handler: number;
    rows: number;
    headers: number;
    cells: number;
    links: number;
    text: number;
    selection: number;
    size: number;
    bounds: number;
    overlap: number;
    concealment: number;
    localOverflow: number;
    parentLocalOverflow: number;
  };
  boundaries: Array<{ table: number; failures: string[] }>;
  tables: Array<{
    client: number;
    scroll: number;
    left: number;
    right: number;
    display: string;
    overflowX: string;
    tableLayout: string;
    direct: boolean;
    component: boolean;
  }>;
};

type TableProjection = 'complete' | 'invariant' | 'dynamic';

async function observeTables(
  page: Page,
  projection: TableProjection = 'complete',
): Promise<TableObservation> {
  const session = tableObserverSessions.get(page);
  if (!session) throw new Error('table observer was not installed before fixture or navigation');
  const started = performance.now();
  const observation = await page.evaluate(({ selectors, projection }) => {
    const runInvariant = projection !== 'dynamic';
    const runDynamic = projection !== 'invariant';
    const normalize = (value: string) => value.trim().split(/\s+/u).filter(Boolean).join(' ');
    const tables = Array.from(document.querySelectorAll<HTMLTableElement>('main table'));
    const cssLength = (value: string, extent: number): number | null => {
      const match = value.trim().match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(px|%)?$/iu);
      if (!match) return null;
      const magnitude = Number.parseFloat(match[1]);
      return match[2]?.toLowerCase() === '%' ? (magnitude * extent) / 100 : magnitude;
    };
    const zeroLength = (value: string) => {
      const parsed = cssLength(value, 1);
      return parsed !== null && Math.abs(parsed) < Number.EPSILON;
    };
    const splitTopLevelComma = (value: string) => {
      const parts: string[] = [];
      let start = 0;
      let depth = 0;
      let quote = '';
      for (let offset = 0; offset < value.length; offset += 1) {
        const character = value[offset];
        if (quote) {
          if (character === '\\') offset += 1;
          else if (character === quote) quote = '';
        } else if (character === '"' || character === "'") quote = character;
        else if (character === '\\') offset += 1;
        else if (character === '(') depth += 1;
        else if (character === ')') depth -= 1;
        else if (character === ',' && depth === 0) {
          parts.push(value.slice(start, offset).trim());
          start = offset + 1;
        }
      }
      parts.push(value.slice(start).trim());
      return parts;
    };
    const fullyClipped = (element: Element, style: CSSStyleDeclaration) => {
      const legacy = style.clip.trim().match(/^rect\((.*)\)$/iu);
      if (legacy) {
        const coordinates = splitTopLevelComma(legacy[1]);
        if (coordinates.length === 4) {
          const [top, right, bottom, left] = coordinates.map((value) => cssLength(value, 1));
          if (
            top !== null &&
            right !== null &&
            bottom !== null &&
            left !== null &&
            (bottom <= top || right <= left)
          ) return true;
        }
      }

      const clipPath = style.clipPath.trim();
      if (!clipPath || clipPath === 'none') return false;
      const basicShape = clipPath.match(
        /^([a-z-]+)\(([\s\S]*)\)(?:\s+(?:margin-box|border-box|padding-box|content-box|fill-box|stroke-box|view-box))?$/iu,
      );
      if (!basicShape) return false;
      const name = basicShape[1].toLowerCase();
      const body = basicShape[2].trim();
      if (name === 'circle') {
        const radius = body.split(/\s+at\s+/iu, 1)[0].trim();
        return zeroLength(radius);
      }
      if (name === 'ellipse') {
        const radii = body.split(/\s+at\s+/iu, 1)[0].trim().split(/\s+/u);
        return radii.length === 2 && radii.some(zeroLength);
      }
      if (name === 'inset') {
        const offsets = body.split(/\s+round\s+/iu, 1)[0].trim().split(/\s+/u);
        if (offsets.length < 1 || offsets.length > 4) return false;
        const expanded =
          offsets.length === 1
            ? [offsets[0], offsets[0], offsets[0], offsets[0]]
            : offsets.length === 2
              ? [offsets[0], offsets[1], offsets[0], offsets[1]]
              : offsets.length === 3
                ? [offsets[0], offsets[1], offsets[2], offsets[1]]
                : offsets;
        const box = element.getBoundingClientRect();
        const top = cssLength(expanded[0], box.height);
        const right = cssLength(expanded[1], box.width);
        const bottom = cssLength(expanded[2], box.height);
        const left = cssLength(expanded[3], box.width);
        return (
          top !== null &&
          right !== null &&
          bottom !== null &&
          left !== null &&
          (top + bottom >= box.height || left + right >= box.width)
        );
      }
      if (name === 'polygon') {
        const points = splitTopLevelComma(body);
        const fillRule = /^(evenodd|nonzero)$/iu.exec(points[0] ?? '')?.[1]?.toLowerCase();
        if (fillRule) points.shift();
        if (points.length < 3) return false;
        const box = element.getBoundingClientRect();
        const coordinates = points.map((point) => {
          const pair = point.trim().split(/\s+/u);
          if (pair.length !== 2) return null;
          const x = cssLength(pair[0], box.width);
          const y = cssLength(pair[1], box.height);
          return x === null || y === null ? null : { x, y };
        });
        if (coordinates.some((point) => point === null)) return false;
        const concrete = coordinates as Array<{ x: number; y: number }>;
        const segments = concrete.map((start, index) => ({
          start,
          end: concrete[(index + 1) % concrete.length],
        }));
        const epsilon = 1e-7;
        const cross = (left: { x: number; y: number }, right: { x: number; y: number }) =>
          left.x * right.y - left.y * right.x;
        const yEvents = concrete.map(({ y }) => y);
        for (let left = 0; left < segments.length; left += 1) {
          const first = segments[left];
          const firstVector = {
            x: first.end.x - first.start.x,
            y: first.end.y - first.start.y,
          };
          for (let right = left + 1; right < segments.length; right += 1) {
            const second = segments[right];
            const secondVector = {
              x: second.end.x - second.start.x,
              y: second.end.y - second.start.y,
            };
            const denominator = cross(firstVector, secondVector);
            if (Math.abs(denominator) <= epsilon) continue;
            const displacement = {
              x: second.start.x - first.start.x,
              y: second.start.y - first.start.y,
            };
            const firstDistance = cross(displacement, secondVector) / denominator;
            const secondDistance = cross(displacement, firstVector) / denominator;
            if (
              firstDistance > epsilon &&
              firstDistance < 1 - epsilon &&
              secondDistance > epsilon &&
              secondDistance < 1 - epsilon
            ) {
              yEvents.push(first.start.y + firstDistance * firstVector.y);
            }
          }
        }
        const levels = yEvents
          .sort((left, right) => left - right)
          .filter((value, index, all) => index === 0 || Math.abs(value - all[index - 1]) > epsilon);
        for (let level = 0; level + 1 < levels.length; level += 1) {
          if (levels[level + 1] - levels[level] <= epsilon) continue;
          const y = (levels[level] + levels[level + 1]) / 2;
          const crossings = segments
            .filter(({ start, end }) => (start.y <= y && end.y > y) || (end.y <= y && start.y > y))
            .map(({ start, end }) => ({
              x: start.x + ((y - start.y) * (end.x - start.x)) / (end.y - start.y),
              winding: end.y > start.y ? 1 : -1,
            }))
            .sort((left, right) => left.x - right.x);
          let winding = 0;
          for (let crossing = 0; crossing < crossings.length; ) {
            const x = crossings[crossing].x;
            let delta = 0;
            let next = crossing;
            while (next < crossings.length && Math.abs(crossings[next].x - x) <= epsilon) {
              delta += crossings[next].winding;
              next += 1;
            }
            winding += delta;
            if (
              next < crossings.length &&
              crossings[next].x - x > epsilon &&
              (fillRule === 'evenodd' ? Math.abs(winding) % 2 === 1 : winding !== 0)
            ) return false;
            crossing = next;
          }
        }
        return true;
      }
      return false;
    };
    const relevantChain = (table: HTMLTableElement) => {
      const chain: Element[] = [table];
      let ancestor = table.parentElement;
      while (ancestor) {
        chain.push(ancestor);
        if (ancestor.matches('.sl-markdown-content')) break;
        ancestor = ancestor.parentElement;
      }
      return chain;
    };
    const failures = {
      tag: 0,
      relation: 0,
      wrapper: 0,
      role: 0,
      focus: 0,
      handler: 0,
      rows: 0,
      headers: 0,
      cells: 0,
      links: 0,
      text: 0,
      selection: 0,
      size: 0,
      bounds: 0,
      overlap: 0,
      concealment: 0,
      localOverflow: 0,
      parentLocalOverflow: 0,
    };
    const boundaries: Array<{ table: number; failures: string[] }> = [];
    const observations = tables.map((table, tableIndex) => {
      const before = { ...failures };
      const style = getComputedStyle(table);
      const box = table.getBoundingClientRect();
      const direct = table.matches(selectors.direct);
      const component = table.matches(selectors.component);
      const generic = table.matches(selectors.generic);
      const chain = relevantChain(table);
      const parents = chain.slice(1);
      const structural = runInvariant
        ? [table, ...table.querySelectorAll('table, thead, tbody, tfoot, tr, th, td')]
        : [];
      const cells = Array.from(table.querySelectorAll<HTMLTableCellElement>('th, td'));
      const guarded = runInvariant ? [...new Set([...structural, ...parents])] : [];
      if (runInvariant) {
        failures.tag += table.tagName === 'TABLE' ? 0 : 1;
        failures.relation += generic && direct !== component ? 0 : 1;
        failures.wrapper +=
          table.parentElement?.matches('.sl-markdown-content, .eq-stage__body') === true ? 0 : 1;
        failures.role += guarded.some((node) => node.hasAttribute('role')) ? 1 : 0;
        failures.focus +=
          chain.some(
            (node) =>
              node.hasAttribute('tabindex') ||
              (node instanceof HTMLElement && node.tabIndex >= 0) ||
              (node.getAttribute('contenteditable') ?? '').toLowerCase() === 'true',
          ) || table.querySelector('[tabindex], [contenteditable="true" i]')
            ? 1
            : 0;
        failures.handler += guarded.some((node) =>
          ['keydown', 'keypress', 'keyup'].some(
            (event) =>
              node.hasAttribute(`on${event}`) ||
              typeof (node as unknown as Record<string, unknown>)[`on${event}`] === 'function',
          ),
        ) ? 1 : 0;
        failures.rows +=
          table.rows.length > 0 &&
          Array.from(table.rows).every((row) => row.cells.length > 0)
            ? 0
            : 1;
        failures.headers += table.querySelectorAll('th').length > 0 ? 0 : 1;
        failures.cells += cells.length > 0 ? 0 : 1;
      }
      const links = runInvariant
        ? Array.from(table.querySelectorAll<HTMLAnchorElement>('a'))
        : [];
      if (runInvariant) {
        failures.links += links.every(
          (link) =>
            normalize(link.innerText).length > 0 &&
            Boolean(link.getAttribute('href')?.trim()) &&
            !['#', 'javascript:void(0)'].includes(link.getAttribute('href')!.trim().toLowerCase()),
        ) ? 0 : 1;
      }
      // KaTeX exposes the equation once to assistive technology through MathML
      // and paints its aria-hidden HTML sibling. The clipped MathML/TeX source
      // is not a second visible table label; every painted glyph still receives
      // the ordinary text checks below. Admit only this complete dual form.
      const mathAlternatives = new Set<Element>();
      for (const math of table.querySelectorAll('.katex')) {
        const alternative = math.querySelector(':scope > .katex-mathml');
        const presentation = math.querySelector(':scope > .katex-html[aria-hidden="true"]');
        const semantics = alternative?.querySelector('math > semantics');
        const alternativeStyle = alternative && getComputedStyle(alternative);
        const complete = alternative && presentation && semantics &&
          !alternative.querySelector('[aria-hidden="true"], [hidden]') &&
          !alternative.hasAttribute('hidden') &&
          alternative.getAttribute('aria-hidden') !== 'true' &&
          alternativeStyle?.display !== 'none' &&
          alternativeStyle?.visibility === 'visible' &&
          normalize(semantics.querySelector('annotation[encoding="application/x-tex"]')?.textContent ?? '').length > 0 &&
          Array.from(semantics.children).some((child) => child.tagName.toLowerCase() !== 'annotation' && normalize(child.textContent ?? '').length > 0) &&
          normalize(presentation.textContent ?? '').length > 0;
        if (complete) mathAlternatives.add(alternative);
        else if (runInvariant) failures.text += 1;
      }
      const textNodes = cells.flatMap((cell) => {
        const nodes: Text[] = [];
        const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
          const node = walker.currentNode as Text;
          const alternative = node.parentElement?.closest('.katex-mathml');
          if (normalize(node.data).length > 0 && !(alternative && mathAlternatives.has(alternative))) nodes.push(node);
        }
        return nodes;
      });
      const textFacts = runDynamic
        ? textNodes.map((node) => {
            const owner = node.parentElement;
            const range = document.createRange();
            range.selectNodeContents(node);
            const boxes = Array.from(range.getClientRects());
            const ancestors: Element[] = [];
            let current: Element | null = owner;
            while (current) {
              ancestors.push(current);
              if (current === table) break;
              current = current.parentElement;
            }
            const styles = ancestors.map((element) => ({
              element,
              style: getComputedStyle(element),
            }));
            return {
              bounded:
                owner !== null &&
                boxes.length > 0 &&
                boxes.every((textBox) => textBox.width > 0 && textBox.height > 0) &&
                Number.parseFloat(getComputedStyle(owner).fontSize) >= 12,
              selectable: styles.every(
                ({ style: textStyle }) => textStyle.userSelect !== 'none',
              ),
              visible: styles.every(({ element, style: textStyle }) => {
                const horizontalTruncation =
                  element instanceof HTMLElement &&
                  /hidden|clip/u.test(textStyle.overflowX) &&
                  element.scrollWidth > element.clientWidth + 1;
                const verticalTruncation =
                  element instanceof HTMLElement &&
                  /hidden|clip/u.test(textStyle.overflowY) &&
                  element.scrollHeight > element.clientHeight + 1;
                return (
                  textStyle.display !== 'none' &&
                  !/hidden|collapse/u.test(textStyle.visibility) &&
                  Number.parseFloat(textStyle.opacity) > 0 &&
                  !fullyClipped(element, textStyle) &&
                  !horizontalTruncation &&
                  !verticalTruncation &&
                  !(textStyle.textOverflow === 'ellipsis' && horizontalTruncation)
                );
              }),
            };
          })
        : [];
      if (runInvariant) {
        failures.text +=
          textNodes.length > 0 &&
          cells.every(
            (cell) =>
              normalize(cell.innerText).length > 0 &&
              normalize(cell.textContent ?? '').length > 0,
          )
          ? 0
          : 1;
      }
      if (runDynamic) {
        failures.selection +=
          textFacts.length > 0 && textFacts.every(({ selectable }) => selectable) ? 0 : 1;
        failures.size +=
          box.width > 0 &&
          box.height > 0 &&
          cells.every((cell) => {
            const cellBox = cell.getBoundingClientRect();
            return (
              cellBox.width > 0 &&
              cellBox.height > 0 &&
              Number.parseFloat(getComputedStyle(cell).fontSize) >= 12
            );
          }) &&
          textFacts.every(({ bounded }) => bounded)
            ? 0
            : 1;
        failures.bounds +=
          box.left >= -1 && box.right <= document.documentElement.clientWidth + 1 ? 0 : 1;
        const cellBoxes = cells.map((cell) => cell.getBoundingClientRect());
        failures.overlap += cellBoxes.some((first, firstIndex) =>
          cellBoxes.slice(firstIndex + 1).some(
            (second) =>
              Math.min(first.right, second.right) - Math.max(first.left, second.left) > 1 &&
              Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top) > 1,
          ),
        )
          ? 1
          : 0;
        failures.concealment +=
          style.visibility === 'visible' &&
          Number.parseFloat(style.opacity) > 0 &&
          !fullyClipped(table, style) &&
          !/hidden|clip/.test(style.overflowX) &&
          cells.every((cell) => {
            const cellStyle = getComputedStyle(cell);
            return (
              cellStyle.visibility === 'visible' &&
              Number.parseFloat(cellStyle.opacity) > 0 &&
              !fullyClipped(cell, cellStyle) &&
              !/hidden|clip/.test(cellStyle.overflowX) &&
              cellStyle.textOverflow !== 'ellipsis'
            );
          }) &&
          textFacts.every(({ visible }) => visible)
            ? 0
            : 1;
        failures.localOverflow += table.scrollWidth > table.clientWidth + 1 ? 1 : 0;
        failures.parentLocalOverflow += parents.some((parent) => {
          const parentStyle = getComputedStyle(parent);
          return (
            /auto|scroll/u.test(parentStyle.overflowX) &&
            parent instanceof HTMLElement &&
            parent.scrollWidth > parent.clientWidth + 1
          );
        }) ? 1 : 0;
      }
      const tableFailures = Object.keys(failures).filter(
        (failure) =>
          failures[failure as keyof typeof failures] >
          before[failure as keyof typeof before],
      );
      if (tableFailures.length > 0) {
        boundaries.push({ table: tableIndex, failures: tableFailures });
      }
      return {
        client: table.clientWidth,
        scroll: table.scrollWidth,
        left: box.left,
        right: box.right,
        display: style.display,
        overflowX: style.overflowX,
        tableLayout: style.tableLayout,
        direct,
        component,
      };
    });
    return {
      document: {
        inner: innerWidth,
        client: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth,
        bodyScroll: document.body.scrollWidth,
      },
      counts: {
        main: tables.length,
        all: document.querySelectorAll('table').length,
        generic: document.querySelectorAll(selectors.generic).length,
        direct: document.querySelectorAll(selectors.direct).length,
        component: document.querySelectorAll(selectors.component).length,
      },
      failures,
      boundaries,
      tables: observations,
    };
  }, { selectors: TABLE_SELECTORS, projection });
  if (projection !== 'dynamic') {
    const registered = await session.send('Runtime.evaluate', {
      expression: `(() => {
      const unique = new Set();
      for (const table of document.querySelectorAll('main table')) {
        unique.add(table);
        let ancestor = table.parentElement;
        while (ancestor) {
          unique.add(ancestor);
          if (ancestor.matches('.sl-markdown-content')) break;
          ancestor = ancestor.parentElement;
        }
      }
      let count = 0;
      for (const node of unique) {
        const listeners = getEventListeners(node);
        for (const type of ['keydown', 'keyup', 'keypress']) count += (listeners[type] || []).length;
      }
      return count;
    })()`,
      includeCommandLineAPI: true,
      returnByValue: true,
    });
    if (registered.exceptionDetails || typeof registered.result.value !== 'number') {
      throw new Error('registered keyboard-handler observation failed closed');
    }
    observation.failures.handler += registered.result.value > 0 ? 1 : 0;
    if (registered.result.value > 0) {
      observation.boundaries.push({ table: -1, failures: ['handler'] });
    }
  }
  const elapsedMs = performance.now() - started;
  if (projection !== 'dynamic') {
    recordSitePhase('tableInvariant', elapsedMs, observation.counts.main);
  }
  if (projection !== 'invariant') {
    recordSitePhase('tableDynamic', elapsedMs, observation.counts.main);
  }
  return observation;
}

export async function assertTableInventory(
  page: Page,
  expected: TableRoute,
  projection: TableProjection = 'complete',
): Promise<TableObservation> {
  expect(new URL(page.url()).pathname).toBe(expected.route);
  const observation = await observeTables(page, projection);
  expect(observation.counts, `table inventory ${expected.route}`).toEqual({
    main: expected.tables,
    all: expected.tables,
    generic: expected.tables,
    direct: expected.direct,
    component: expected.component,
  });
  if (projection !== 'dynamic') {
    expect(
      observation.failures.relation,
      `table relation boundaries ${expected.route}: ${JSON.stringify(observation.boundaries)}`,
    ).toBe(0);
  }
  expect(observation.tables.filter((table) => table.direct)).toHaveLength(
    expected.direct,
  );
  expect(observation.tables.filter((table) => table.component)).toHaveLength(expected.component);
  return observation;
}

function assertTableInvariantFacts(observation: TableObservation): void {
  expect({
    tag: observation.failures.tag,
    relation: observation.failures.relation,
    wrapper: observation.failures.wrapper,
    role: observation.failures.role,
    focus: observation.failures.focus,
    handler: observation.failures.handler,
    rows: observation.failures.rows,
    headers: observation.failures.headers,
    cells: observation.failures.cells,
    links: observation.failures.links,
    text: observation.failures.text,
  }, `invariant table boundaries: ${JSON.stringify(observation.boundaries)}`).toEqual({
    tag: 0,
    relation: 0,
    wrapper: 0,
    role: 0,
    focus: 0,
    handler: 0,
    rows: 0,
    headers: 0,
    cells: 0,
    links: 0,
    text: 0,
  });
}

function assertTableContentAndSemantics(observation: TableObservation): void {
  assertTableInvariantFacts(observation);
  expect(observation.document.inner).toBeGreaterThanOrEqual(observation.document.client);
  expect(observation.document.scroll).toBeLessThanOrEqual(observation.document.client + 1);
  expect(observation.document.bodyScroll).toBeLessThanOrEqual(observation.document.client + 1);
  const { localOverflow: _, ...failures } = observation.failures;
  expect(failures).toEqual({
    tag: 0,
    relation: 0,
    wrapper: 0,
    role: 0,
    focus: 0,
    handler: 0,
    rows: 0,
    headers: 0,
    cells: 0,
    links: 0,
    text: 0,
    selection: 0,
    size: 0,
    bounds: 0,
    overlap: 0,
    concealment: 0,
    parentLocalOverflow: 0,
  });
}

export async function assertProductTableRouteInvariant(
  page: Page,
  expected: TableRoute,
): Promise<TableObservation> {
  await navigateSitePage(page, expected.route);
  await assertCoreVisible(page);
  const observation = await assertTableInventory(page, expected, 'invariant');
  assertTableInvariantFacts(observation);
  return observation;
}

function assertTableDynamicFacts(observation: TableObservation): void {
  expect(observation.document.inner).toBeGreaterThanOrEqual(observation.document.client);
  expect(observation.document.scroll).toBeLessThanOrEqual(observation.document.client + 1);
  expect(observation.document.bodyScroll).toBeLessThanOrEqual(observation.document.client + 1);
  expect({
    selection: observation.failures.selection,
    size: observation.failures.size,
    bounds: observation.failures.bounds,
    overlap: observation.failures.overlap,
    concealment: observation.failures.concealment,
    parentLocalOverflow: observation.failures.parentLocalOverflow,
  }, `dynamic table boundaries: ${JSON.stringify(observation.boundaries)}`).toEqual({
    selection: 0,
    size: 0,
    bounds: 0,
    overlap: 0,
    concealment: 0,
    parentLocalOverflow: 0,
  });
}

export async function assertProductTableRouteGreen(
  page: Page,
  expected: TableRoute,
  axeProjection: ConditionAxeProjection,
): Promise<TableObservation> {
  await navigateSitePage(page, expected.route);
  await assertCoreVisible(page);
  const observation = await assertTableInventory(page, expected, 'dynamic');
  assertTableDynamicFacts(observation);
  expect(
    observation.failures.localOverflow,
    `overflow table boundaries ${expected.route}: ${JSON.stringify(observation.boundaries)}`,
  ).toBe(0);
  expect(
    await conditionDependentAxeViolations(page, axeProjection),
  ).toEqual([]);
  return observation;
}

export async function assertReducedMotion(page: Page): Promise<void> {
  const moving = await page.evaluate(() => {
    const duration = (value: string) =>
      value
        .split(',')
        .map((part) => part.trim())
        .some((part) => (part.endsWith('ms') ? Number.parseFloat(part) : Number.parseFloat(part) * 1000) > 1);
    return [...document.querySelectorAll('header *, main *')]
      .filter((element) => {
        const style = getComputedStyle(element);
        return duration(style.animationDuration) || duration(style.transitionDuration);
      })
      .map((element) => `${element.tagName.toLowerCase()} ${element.getAttribute('aria-label') ?? element.textContent?.trim().slice(0, 40) ?? ''}`);
  });
  expect(moving).toEqual([]);
}

export async function attachGrossScreenshot(
  page: Page,
  testInfo: TestInfo,
  name: string,
): Promise<void> {
  await assertCoreVisible(page);
  const main = await page.getByRole('main').boundingBox();
  expect(main).not.toBeNull();
  expect(main!.width).toBeGreaterThan(100);
  expect(main!.height).toBeGreaterThan(100);
  const body = await measureSitePhase('screenshot', () => page.screenshot({ fullPage: true }));
  await testInfo.attach(name, {
    body,
    contentType: 'image/png',
  });
}
