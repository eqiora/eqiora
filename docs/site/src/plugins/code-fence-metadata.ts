import type { HastPluginDefinition, MdastPluginDefinition } from 'satteri';

const METADATA = 'dataEqioraFenceMetadata';

// rawHtml reparses HAST through HTML, which retains properties but drops data.
// Keep the fence declaration through that conversion for syntax renderers.
export const preserveCodeFenceMetadata: MdastPluginDefinition = {
  name: 'eqiora-preserve-code-fence-metadata',
  code(node, context) {
    context.setField(node, 'data', {
      ...node.data,
      hProperties: {
        ...node.data?.hProperties,
        [METADATA]: JSON.stringify({ lang: node.lang ?? undefined, meta: node.meta ?? undefined }),
      },
    });
  },
};

export const restoreCodeFenceMetadata: HastPluginDefinition = {
  name: 'eqiora-restore-code-fence-metadata',
  element: {
    filter: ['code'],
    visit(node) {
      const { [METADATA]: encoded, ...properties } = node.properties;
      if (encoded === undefined) return;
      if (typeof encoded !== 'string') {
        throw new Error('Code fence metadata must survive HTML conversion as a string');
      }
      const metadata: unknown = JSON.parse(encoded);
      if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)
        || Object.entries(metadata).some(([key, value]) => !['lang', 'meta'].includes(key) || typeof value !== 'string')) {
        throw new Error('Code fence metadata must contain only declared language and options');
      }
      return { ...node, properties, data: { ...node.data, ...metadata } };
    },
  },
};
