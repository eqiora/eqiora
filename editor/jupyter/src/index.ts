import type { JupyterFrontEndPlugin } from '@jupyterlab/application';
import { EditorExtensionRegistry, IEditorExtensionRegistry } from '@jupyterlab/codemirror';
import { INotebookTracker } from '@jupyterlab/notebook';
import { eqioraMagic } from './language.js';
import { eqioraHover } from './hover.js';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@eqiora/jupyter:source-cells',
  description: 'Highlight Eqiora source cells and show compiler-owned hover information.',
  autoStart: true,
  requires: [IEditorExtensionRegistry, INotebookTracker],
  activate: (_app, editors: IEditorExtensionRegistry, notebooks: INotebookTracker) => {
    editors.addExtension({
      name: '@eqiora/jupyter:source-cells',
      factory: ({ inline, model }) => inline
        && 'cell_type' in model.sharedModel && model.sharedModel.cell_type === 'code'
        ? EditorExtensionRegistry.createImmutableExtension([
          eqioraMagic(model.sharedModel.getSource()), eqioraHover(model, notebooks),
        ])
        : null,
    });
  },
};
export default plugin;
