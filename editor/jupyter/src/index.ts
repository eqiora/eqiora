import type { JupyterFrontEndPlugin } from '@jupyterlab/application';
import { EditorExtensionRegistry, IEditorExtensionRegistry } from '@jupyterlab/codemirror';
import { eqioraMagic } from './language.js';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@eqiora/jupyter:source-cells',
  description: 'Highlight Eqiora source cells with the canonical Eqiora syntax grammar.',
  autoStart: true,
  requires: [IEditorExtensionRegistry],
  activate: (_app, editors: IEditorExtensionRegistry) => {
    editors.addExtension({
      name: '@eqiora/jupyter:source-cells',
      factory: ({ inline, model }) => inline
        ? EditorExtensionRegistry.createImmutableExtension(eqioraMagic(model.sharedModel.getSource()))
        : null,
    });
  },
};
export default plugin;
