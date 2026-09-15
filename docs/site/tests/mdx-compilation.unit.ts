import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { mdxToJs } from 'satteri';

test('standard source lookup compiles its imports and following prose as MDX', () => {
  const source = readFileSync(new URL('../src/content/docs/reference/standard-packages/index.mdx', import.meta.url), 'utf8');
  assert.doesNotThrow(() => mdxToJs(source));
});

test('Kármán vortex-street gallery source compiles as MDX', () => {
  const source = readFileSync(new URL('../src/content/docs/gallery/karman-vortex-street.mdx', import.meta.url), 'utf8');
  assert.doesNotThrow(() => mdxToJs(source));
});
