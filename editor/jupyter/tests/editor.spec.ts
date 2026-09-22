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
