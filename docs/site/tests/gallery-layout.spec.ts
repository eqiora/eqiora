import { expect, test } from '@playwright/test';
import { rejectExternalRequests } from './support';

for (const width of [390, 768, 1440]) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`gallery cards align at ${width}px in ${colorScheme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      const external = await rejectExternalRequests(page);
      await page.goto('/gallery/');
      const cards = page.locator('.eq-gallery-card');
      await expect(cards).toHaveCount(3);
      const boxes = () => cards.evaluateAll((nodes) => nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        const media = node.querySelector('.eq-gallery-card__media')!.getBoundingClientRect();
        const body = node.querySelector('.eq-gallery-card__body')!.getBoundingClientRect();
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height,
          media: media.height, text: body.y - rect.y };
      }));
      const assertAligned = (values: Awaited<ReturnType<typeof boxes>>) => {
        for (const a of values) {
          expect(Math.abs(a.width - values[0].width)).toBeLessThan(1);
          expect(Math.abs(a.height - values[0].height)).toBeLessThan(1);
          expect(Math.abs(a.media - values[0].media)).toBeLessThan(1);
          expect(Math.abs(a.text - values[0].text)).toBeLessThan(1);
          for (const b of values.filter((b) => Math.abs(a.y - b.y) < 1)) {
            expect(Math.abs(a.height - b.height)).toBeLessThan(1);
          }
        }
      };
      await page.locator('.eq-gallery-card img').evaluateAll(async (images) => {
        await Promise.all(images.map((image) => (image as HTMLImageElement).decode()));
      });
      assertAligned(await boxes());

      // Exercise the same real components with different intrinsic ratios and
      // delayed loading, not a parallel imitation of the gallery CSS.
      await page.route('**/gallery-ratio-*.svg', async (route) => {
        const index = Number(route.request().url().match(/ratio-(\d)/)![1]);
        await new Promise((resolve) => setTimeout(resolve, [180, 30, 100][index]));
        const [w, h] = [[1200, 300], [600, 600], [300, 900]][index];
        await route.fulfill({ contentType: 'image/svg+xml', body:
          `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="steelblue"/></svg>` });
      });
      await cards.evaluateAll((nodes) => nodes.forEach((node, index) => {
        const image = node.querySelector('img')!;
        image.loading = 'eager';
        image.src = `/gallery-ratio-${index}.svg`;
        if (index === 1) node.querySelector('.eq-gallery-card__title')!.textContent += ' with a longer descriptive title';
      }));
      const before = await boxes();
      assertAligned(before);
      await page.locator('.eq-gallery-card img').evaluateAll(async (images) => {
        await Promise.all(images.map((image) => (image as HTMLImageElement).decode()));
      });
      const after = await boxes();
      assertAligned(after);
      expect(after).toEqual(before);
      expect(external).toEqual([]);
    });
  }
}

test('current full figures retain their aspect and the pressure generation is unique', async ({ page }) => {
  const external = await rejectExternalRequests(page);
  for (const width of [390, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const colorScheme of ['light', 'dark'] as const) {
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      for (const entry of ['exact-cylinder-steady-stokes', 'mixed-boundary-elasticity', 'karman-vortex-street']) {
        await page.goto(`/gallery/${entry}/`);
        const figures = page.locator('.eq-result-figure img, .eq-gallery-motion__still');
        for (const figure of await figures.all()) {
          await figure.scrollIntoViewIfNeeded();
          await expect(figure).toBeVisible();
          const sizes = await figure.evaluate(async (element) => {
            const image = element as HTMLImageElement;
            await image.decode();
            const box = image.getBoundingClientRect();
            return { actual: box.width / box.height, natural: image.naturalWidth / image.naturalHeight };
          });
          expect(Math.abs(sizes.actual - sizes.natural)).toBeLessThan(0.01);
        }
        expect(await figures.count()).toBeGreaterThan(0);
        if (entry === 'exact-cylinder-steady-stokes') {
          await expect(page.locator('#pressure-visualization img')).toHaveCount(1);
          await expect(page.locator('#pressure-visualization img')).toHaveAttribute('src', /exact-cylinder-pressure-presentation/);
        }
      }
    }
  }
  expect(external).toEqual([]);
});
