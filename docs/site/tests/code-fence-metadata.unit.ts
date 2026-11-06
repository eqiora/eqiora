import assert from 'node:assert/strict';
import test from 'node:test';
import { markdownToHast, mdxToHast, type HastNode } from 'satteri';
import { preserveCodeFenceMetadata, restoreCodeFenceMetadata } from '../src/plugins/code-fence-metadata.ts';

const source = [
  'Inline `geometry.rectangle("<&{}>")` stays inline.',
  '',
  '```python title="geometry & boundaries.py" {1}',
  'print("<&{}>")',
  '```',
  '',
  '```rust title="mesh <&>.rs" {2}',
  'let text = "<&{}>";',
  '```',
  '',
  '```',
  'unlabelled source',
  '```',
].join('\n');

for (const [name, compile] of [['Markdown', markdownToHast], ['MDX', mdxToHast]] as const) {
  test(`${name} keeps fence declarations through raw HTML conversion and leaves inline code alone`, () => {
    const tree = compile(source, {
      features: { rawHtml: true },
      mdastPlugins: [preserveCodeFenceMetadata],
      hastPlugins: [restoreCodeFenceMetadata],
    });
    const codes: Array<Extract<HastNode, { type: 'element' }>> = [];
    const visit = (node: HastNode): void => {
      if (node.type === 'element' && node.tagName === 'code') codes.push(node);
      if ('children' in node) for (const child of node.children) visit(child);
    };
    visit(tree);
    assert.equal(codes.length, 4);
    assert.equal(codes[0].data, undefined);
    assert.deepEqual(codes[0].properties, {});
    assert.deepEqual(codes.slice(1).map(({ data }) => data), [
      { lang: 'python', meta: 'title="geometry & boundaries.py" {1}' },
      { lang: 'rust', meta: 'title="mesh <&>.rs" {2}' },
      undefined,
    ]);
    assert.deepEqual(codes.map((code) => code.children.map((child) => child.type === 'text' ? child.value : '').join('')), [
      'geometry.rectangle("<&{}>")',
      'print("<&{}>")\n',
      'let text = "<&{}>";\n',
      'unlabelled source\n',
    ]);
    for (const { properties } of codes) {
      assert.ok(Object.keys(properties).every((key) => !key.startsWith('dataEqioraFence')));
    }
  });
}
