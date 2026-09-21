import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';

for (const route of ['lab/tree', 'notebooks']) {
  test(`${route}: source highlighting before execution, dynamic switching and saved reopen`, async ({ page, request }) => {
    const notebook = JSON.parse(await readFile(new URL('../decay.ipynb', import.meta.url), 'utf8'));
    const name = `${route.replace('/', '-')}-${Date.now()}-test.ipynb`;
    const token = process.env.EQIORA_JUPYTER_TOKEN ?? 'eqiora-test';
    const query = `?token=${encodeURIComponent(token)}`;
    const created = await request.put(`/api/contents/${name}${query}`, { data: { type: 'notebook', content: notebook } });
    expect(created.ok()).toBeTruthy();
    try {
      await page.goto(`/${route}/${name}${query}`);
      const cells = page.locator('.jp-CodeCell .cm-content:visible');
      await expect(cells).toHaveCount(3);
      const source = cells.nth(1);
      // Whole-line comment and a separately styled unit disprove Python highlighting.
      await expect(source.locator('span').filter({ hasText: /^\/\/ A minimal implicit ODE:/ })).toHaveCount(1);
      await expect(source.locator('span').filter({ hasText: /^s$/ })).toHaveCount(1);
      await expect(cells.nth(0).locator('span').filter({ hasText: /^import$/ })).toHaveCount(1);
      const eqiora = notebook.cells[1].source.join('');
      await source.fill('value = 42\nprint(value)');
      await expect(source.locator('span').filter({ hasText: /^42$/ })).toHaveCount(1);
      await source.fill(eqiora);
      await expect(source.locator('span').filter({ hasText: /^s$/ })).toHaveCount(1);
      await page.keyboard.press('ControlOrMeta+s');
      await expect.poll(async () => {
        const saved = await request.get(`/api/contents/${name}${query}`);
        const body = await saved.json();
        return body.content?.cells[1].source;
      }).toBe(eqiora);
      await page.reload();
      await expect(page.locator('.jp-CodeCell .cm-content:visible').nth(1).locator('span').filter({ hasText: /^s$/ })).toHaveCount(1);
      const saved = await (await request.get(`/api/contents/${name}${query}`)).json();
      expect(saved.content.cells.every((cell: { execution_count: unknown }) => cell.execution_count === null)).toBe(true);
    } finally {
      const sessions = await (await request.get(`/api/sessions${query}`)).json();
      for (const session of sessions) {
        if (session.path === name) await request.delete(`/api/sessions/${session.id}${query}`);
      }
      expect((await request.delete(`/api/contents/${name}${query}`)).ok()).toBe(true);
    }
  });
}
