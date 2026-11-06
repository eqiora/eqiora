import katex from 'katex';
import type { MdastPluginDefinition } from 'satteri';

function renderMath(source: string, displayMode: boolean): string {
  return katex.renderToString(source, {
    displayMode,
    throwOnError: true,
    output: 'htmlAndMathml',
    trust: false,
  });
}

export const katexMathPlugin: MdastPluginDefinition = {
  name: 'eqiora-build-time-katex',
  math(node) {
    const rendered = renderMath(node.value, node.type === 'math');
    return {
      mdxExpressions: false,
      raw:
        '<div class="eqiora-math-region" role="region" aria-label="Displayed equation" tabindex="0">' +
        rendered +
        '</div>',
    };
  },
  inlineMath(node) {
    return { raw: renderMath(node.value, node.type === 'math'), mdxExpressions: false };
  },
};
