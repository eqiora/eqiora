import { expect, test } from '@playwright/test';

const source = '%%eqiora model\n// A source cell\nmodel Decay {\n  parameter rate: 1 / s = 2;\n  relation evolution continuous { derivative(x) + rate * x = 0; }\n}';

test('canonical grammar highlights a never-executed source cell and preserves Python', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => 'editor' in window);
  expect(await page.evaluate(() => (window as any).editor.language())).toBe('python');
  await page.evaluate((source) => (window as any).editor.open(source), source);
  await expect.poll(() => page.evaluate(() => (window as any).editor.language())).toBe('eqiora');
  const tokens = await page.evaluate(() => (window as any).editor.tokens());
  for (const [text, name] of [
    ['%%eqiora model', 'meta'], ['// A source cell', 'comment'], ['model', 'keyword'],
    ['Decay', 'typeName'], ['s', 'typeName'], ['2', 'number'], ['derivative', 'variableName.standard'],
  ]) expect(tokens).toContainEqual({ text, name });
  await expect(page.locator('.cm-content')).toContainText('model Decay');
  expect(await page.locator('.cm-line span').count()).toBeGreaterThan(8);
});

test('typing/removing the magic switches language and reopening restores it from source', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => 'editor' in window);
  for (const [text, expected] of [
    [source, 'eqiora'], ['%%eqiorax model\nx = 1', 'python'], [source, 'eqiora'],
    ['x = 1\nprint(x)', 'python'], ['# %%eqiora model\nx = 1', 'python'],
    ['text = "%%eqiora model"', 'python'], ['%%eqiora\nmodel M {}', 'eqiora'],
  ]) {
    await page.evaluate((text) => (window as any).editor.replace(text), text);
    await expect.poll(() => page.evaluate(() => (window as any).editor.language())).toBe(expected);
  }
  await page.evaluate((source) => (window as any).editor.open(source), source);
  expect(await page.evaluate(() => (window as any).editor.language())).toBe('eqiora');
  await page.evaluate(() => (window as any).editor.replace('if True:\n    print(42)'));
  const tokens = await page.evaluate(() => (window as any).editor.tokens());
  expect(tokens).toContainEqual({ name: 'if', text: 'if' });
  expect(tokens).toContainEqual({ name: 'Number', text: '42' });
});

const hoverSource = '%%eqiora model\n// 🦀 日本語\nmodel M(){parameter speed:1=2;}';
const hoverOffset = hoverSource.indexOf('speed');
const hoverStart = Array.from(hoverSource.slice(0, hoverOffset)).length;

test('hover sends current code-point source position and renders only plain text', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => 'hover' in window);
  await page.evaluate(source => (window as any).hover.open(source), hoverSource);
  await page.evaluate(pos => (window as any).hover.request(pos), hoverOffset + 2);
  await expect.poll(() => page.evaluate(() => (window as any).hover.requests().length)).toBe(1);
  const requests = await page.evaluate(() => (window as any).hover.requests());
  expect(requests[0].data).toEqual({ source: hoverSource, cursor: hoverStart + 2 });
  const text = 'parameter speed\n<img src=x onerror=alert(1)>\nNotation: v';
  await page.evaluate(result => (window as any).hover.reply(result), [hoverStart, hoverStart + 5, text]);
  await expect(page.locator('.eqiora-source-hover')).toHaveText(text);
  expect(await page.locator('.eqiora-source-hover img').count()).toBe(0);
  await page.evaluate(() => (window as any).hover.remoteClose());
  await expect(page.locator('.eqiora-source-hover')).toHaveText(text);
  await page.evaluate(() => (window as any).hover.restart());
  await expect(page.locator('.eqiora-source-hover')).toHaveCount(0);
});

for (const invalidation of ['edit', 'restart', 'switchKernel', 'disconnect', 'detach', 'destroy', 'disposeKernel']) {
  test(`hover rejects a late response after ${invalidation}`, async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => 'hover' in window);
    await page.evaluate(source => (window as any).hover.open(source), hoverSource);
    await page.evaluate(pos => (window as any).hover.request(pos), hoverOffset + 2);
    await expect.poll(() => page.evaluate(() => (window as any).hover.requests().length)).toBe(1);
    await page.evaluate(({ invalidation, source }) => {
      const h = (window as any).hover;
      if (invalidation === 'edit') { h.edit(source + ' '); h.edit(source); }
      else h[invalidation]();
      h.reply([Array.from(source.slice(0, source.indexOf('speed'))).length,
        Array.from(source.slice(0, source.indexOf('speed'))).length + 5, 'stale']);
    }, { invalidation, source: hoverSource });
    await expect(page.locator('.eqiora-source-hover')).toHaveCount(0);
    await expect.poll(() => page.evaluate(() => (window as any).hover.requests()[0].isDisposed)).toBe(true);
  });
}

test('hover omits Python, unowned and ambiguous cells and times out unloaded kernels', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => 'hover' in window);
  for (const mode of ['python', 'detach', 'ambiguous', 'busy']) {
    await page.evaluate(({ mode, source, position }) => {
      const h = (window as any).hover;
      h.open(mode === 'python' ? 'speed = 2' : source);
      if (mode !== 'python') h[mode]();
      h.request(mode === 'python' ? 2 : position);
    }, { mode, source: hoverSource, position: hoverOffset + 2 });
    expect(await page.evaluate(() => (window as any).hover.requests())).toEqual([]);
  }
  await page.evaluate(source => (window as any).hover.open(source), hoverSource);
  await page.evaluate(pos => (window as any).hover.request(pos), hoverOffset + 2);
  await expect.poll(() => page.evaluate(() => (window as any).hover.requests()[0]?.closed), { timeout: 5000 }).toBe(true);
  await expect(page.locator('.eqiora-source-hover')).toHaveCount(0);
});
