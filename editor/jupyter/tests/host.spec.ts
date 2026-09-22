import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';

for (const route of ['lab/tree', 'notebooks']) {
  test(`${route}: source highlighting before execution, dynamic switching and saved reopen`, async ({ page, request }) => {
    const notebook = JSON.parse(await readFile(new URL('../decay.ipynb', import.meta.url), 'utf8'));
    notebook.cells.push(
      { cell_type: 'markdown', metadata: {}, source: ['%%eqiora\n# Markdown title\n// plain comment'] },
      { cell_type: 'raw', metadata: {}, source: ['%%eqiora\n// plain comment\nvalue = 42'] },
    );
    const name = `${route.replace('/', '-')}-${Date.now()}-test.ipynb`;
    const token = process.env.EQIORA_JUPYTER_TOKEN ?? 'eqiora-test';
    const query = `?token=${encodeURIComponent(token)}`;
    const created = await request.put(`/api/contents/${name}${query}`, { data: { type: 'notebook', content: notebook } });
    expect(created.ok()).toBeTruthy();
    try {
      await page.goto(`/${route}/${name}${query}`);
      const cells = page.locator('.jp-CodeCell .cm-content:visible');
      await expect(cells).toHaveCount(3, { timeout: 30000 });
      const markdown = page.locator('.jp-MarkdownCell:visible');
      await markdown.dblclick();
      const markdownEditor = markdown.locator('.cm-content');
      await expect(markdownEditor).toBeVisible();
      // The magic-looking header must not replace Markdown or Raw languages.
      await expect(markdownEditor.locator('span').filter({ hasText: /^ Markdown title$/ })).toHaveCount(1);
      await expect(markdownEditor.locator('span').filter({ hasText: /^\/\/ plain comment$/ })).toHaveCount(0);
      await expect(page.locator('.jp-RawCell:visible .cm-content span')).toHaveCount(0);
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
      expect(saved.content.cells.filter((cell: { cell_type: string }) => cell.cell_type === 'code').every((cell: { execution_count: unknown }) => cell.execution_count === null)).toBe(true);
    } finally {
      const sessions = await (await request.get(`/api/sessions${query}`)).json();
      for (const session of sessions) {
        if (session.path === name) await request.delete(`/api/sessions/${session.id}${query}`);
      }
      expect((await request.delete(`/api/contents/${name}${query}`)).ok()).toBe(true);
    }
  });

  test(`${route}: kernel completion edits an unexecuted source cell`, async ({ page, request }) => {
    const notebook = JSON.parse(await readFile(new URL('../decay.ipynb', import.meta.url), 'utf8'));
    notebook.cells[0].source = ['%load_ext eqiora.jupyter\nprint("Eqiora completion ready")'];
    const partial = '%%eqiora model\nmodel Cell(){parameter speed:1=2;variable x:1;relation law{x=spe';
    notebook.cells[1].source = [partial];
    const name = `${route.replace('/', '-')}-${Date.now()}-completion.ipynb`;
    const token = process.env.EQIORA_JUPYTER_TOKEN ?? 'eqiora-test';
    const query = `?token=${encodeURIComponent(token)}`;
    expect((await request.put(`/api/contents/${name}${query}`, {
      data: { type: 'notebook', content: notebook },
    })).ok()).toBeTruthy();
    try {
      await page.goto(`/${route}/${name}${query}`);
      const cells = page.locator('.jp-CodeCell .cm-content:visible');
      await expect(cells).toHaveCount(3, { timeout: 30000 });
      await expect.poll(() => cells.nth(0).innerText()).toBe(notebook.cells[0].source[0]);
      await cells.nth(0).click();
      await cells.nth(0).press('Shift+Enter');
      await expect(page.locator('.jp-CodeCell:visible').nth(0).locator('.jp-OutputArea')).toContainText('Eqiora completion ready', { timeout: 30000 });
      const source = cells.nth(1);
      await source.click();
      await source.press('ControlOrMeta+End');
      await source.press('Tab');
      await expect(page.locator('.jp-Completer:visible')).toContainText('speed');
      await page.keyboard.press('Enter');
      await expect.poll(() => source.innerText()).toBe(partial + 'ed');
      await page.keyboard.press('ControlOrMeta+s');
      await expect.poll(async () => {
        const saved = await (await request.get(`/api/contents/${name}${query}`)).json();
        return saved.content.cells[1].source;
      }).toBe(partial + 'ed');
      const saved = await (await request.get(`/api/contents/${name}${query}`)).json();
      expect(saved.content.cells[1].execution_count).toBeNull();
    } finally {
      const sessions = await (await request.get(`/api/sessions${query}`)).json();
      for (const session of sessions) {
        if (session.path === name) await request.delete(`/api/sessions/${session.id}${query}`);
      }
      expect((await request.delete(`/api/contents/${name}${query}`)).ok()).toBe(true);
    }
  });
}
