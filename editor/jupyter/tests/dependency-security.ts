import { expect, test } from '@playwright/test';

test('locked Jupyter sanitizer preserves allowed HTML and rejects unsafe markup', async ({ page }) => {
  await page.goto('/tests/dependency.html');
  await page.waitForFunction(() => 'dependencyProbe' in window);
  const observed = await page.evaluate(async () => {
    const { Sanitizer } = (window as any).dependencyProbe;
    const sanitizer = new Sanitizer();
    const dirty = '<p onclick="alert(1)">safe <strong>value</strong>'
      + '<a href="https://example.org/">good</a><a href="javascript:alert(1)">bad</a>'
      + '<script>alert(1)</script></p>';
    const document = new DOMParser().parseFromString(sanitizer.sanitize(dirty), 'text/html');
    // Reparse the sanitizer output as a browser does: a solidus textarea close
    // must not turn text into an active event-bearing element (GHSA-jxwj-j7wr-gfrw).
    const mutation = new DOMParser().parseFromString(sanitizer.sanitize(
      '<textarea>safe</textarea/><img src=x onerror=alert(1)>'
    ), 'text/html');
    return {
      strong: document.querySelector('strong')?.textContent,
      links: [...document.querySelectorAll('a')].map(a => a.getAttribute('href')),
      active: document.querySelectorAll('script, [onclick], [onerror]').length,
      mutationActive: mutation.querySelectorAll('script, [onerror]').length,
    };
  });
  expect(observed).toEqual({ strong: 'value', links: ['https://example.org/', null], active: 0, mutationActive: 0 });
});

test('locked sanitizer enforces scheme and discard rules with explicit optional tags', async ({ page }) => {
  await page.goto('/tests/dependency.html');
  await page.waitForFunction(() => 'dependencyProbe' in window);
  const observed = await page.evaluate(async () => {
    const { Sanitizer } = (window as any).dependencyProbe;
    const sanitizer = new Sanitizer();
    const refresh = sanitizer.sanitize('<meta http-equiv="refresh" content="0;url=javascript:alert(1)">', {
      allowedTags: ['meta'], allowedAttributes: { meta: ['http-equiv', 'content'] },
    });
    const animation = sanitizer.sanitize('<svg><animate attributeName="href" values="#safe;javascript:alert(1)"></animate></svg>', {
      allowedTags: ['svg', 'animate'], allowedAttributes: { animate: ['attributename', 'values'] },
    });
    const discarded = sanitizer.sanitize('<div><noscript></div><img src="https://example.org/beacon"></noscript>', {
      allowedTags: ['div', 'img'], allowedAttributes: { img: ['src'] },
      nonTextTags: ['script', 'style', 'textarea', 'option', 'noscript'],
    });
    return { refresh, animation, discarded };
  });
  expect(observed.refresh).not.toContain('javascript:');
  expect(observed.animation).not.toContain('javascript:');
  expect(observed.discarded).not.toContain('<img');
});

test('Jupyter Mermaid renders a mathematical fraction with patched KaTeX', async ({ page }) => {
  await page.goto('/tests/dependency.html');
  await page.waitForFunction(() => 'dependencyProbe' in window);
  const observed = await page.evaluate(async () => {
    const { MermaidManager, katex } = (window as any).dependencyProbe;
    const manager = new MermaidManager();
    const result = await manager.renderSvg('flowchart LR\n A["$$x = \\frac{1}{2}$$"] --> B["value"]');
    const svg = new DOMParser().parseFromString(result.svg, 'image/svg+xml');
    const figure = manager.makeMermaidFigure(result);
    document.body.appendChild(figure);
    const image = figure.querySelector('img')!;
    await image.decode();
    const math = document.createElement('div');
    math.innerHTML = katex.renderToString('x=\\frac{1}{2}', { displayMode: true });
    document.body.appendChild(math);
    await document.fonts.ready;
    const htmlMath = math.querySelector('.katex-html')!;
    return {
      mermaid: manager.getMermaidVersion(),
      fraction: svg.querySelector('mfrac')?.textContent?.replace(/\s/g, ''),
      imageWidth: image.naturalWidth,
      htmlMathVisible: htmlMath.getBoundingClientRect().height > 0,
      htmlMathFont: getComputedStyle(math.querySelector('.katex')!).fontFamily,
    };
  });
  expect(observed.mermaid).toBe('11.17.2');
  expect(observed.fraction).toBe('12');
  expect(observed.imageWidth).toBeGreaterThan(0);
  expect(observed.htmlMathVisible).toBe(true);
  expect(observed.htmlMathFont).toContain('KaTeX_Main');
});

test('KaTeX ignores inherited trust rather than enabling unsafe links', async ({ page }) => {
  await page.goto('/tests/dependency.html');
  await page.waitForFunction(() => 'dependencyProbe' in window);
  const unsafeLinks = await page.evaluate(async () => {
    const { katex } = (window as any).dependencyProbe;
    Object.defineProperty(Object.prototype, 'trust', { configurable: true, writable: true, value: true });
    try {
      const html = katex.renderToString('\\href{javascript:alert(1)}{bad}', { output: 'mathml' });
      return new DOMParser().parseFromString(html, 'text/html').querySelectorAll('[href]').length;
    } finally {
      delete (Object.prototype as { trust?: boolean }).trust;
    }
  });
  expect(unsafeLinks).toBe(0);
});
