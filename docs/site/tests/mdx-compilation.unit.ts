import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';
import { mdxToJs } from 'satteri';
import { katexMathPlugin } from '../src/plugins/katex.ts';

test('standard source lookup compiles its imports and following prose as MDX', () => {
  const source = readFileSync(new URL('../src/content/docs/reference/standard-packages/index.mdx', import.meta.url), 'utf8');
  assert.doesNotThrow(() => mdxToJs(source));
});

test('Kármán vortex-street gallery source compiles as MDX', () => {
  const source = readFileSync(new URL('../src/content/docs/gallery/karman-vortex-street.mdx', import.meta.url), 'utf8');
  assert.doesNotThrow(() => mdxToJs(source));
});

const learn = new URL('../src/content/docs/learn/', import.meta.url);
for (const path of readdirSync(learn, { recursive: true }).filter((name) => name.endsWith('.mdx'))) {
  test(`Learn chapter ${path} compiles as MDX`, () => {
    assert.doesNotThrow(() => mdxToJs(readFileSync(new URL(path, learn), 'utf8'), { features: { math: true }, mdastPlugins: [katexMathPlugin] }));
  });
}
