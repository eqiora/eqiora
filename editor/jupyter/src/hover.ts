import type { IEditorExtensionFactory } from '@jupyterlab/codemirror';
import type { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { closeHoverTooltips, EditorView, hoverTooltip, ViewPlugin, type Tooltip, type ViewUpdate } from '@codemirror/view';
import { isEqioraCell } from './language.js';

type Model = IEditorExtensionFactory.IOptions['model'];

// A cloned view may share its cell model. Never borrow the active notebook's kernel.
function owner(tracker: INotebookTracker, model: Model): NotebookPanel | null {
  const matches: NotebookPanel[] = [];
  tracker.forEach(panel => {
    if (!panel.isDisposed && panel.model && Array.from(panel.model.cells).some(
      cell => cell.sharedModel === model.sharedModel,
    )) matches.push(panel);
  });
  return matches.length === 1 ? matches[0] : null;
}

/** Private, transient transport to the extension-loaded owning Python kernel. */
export function eqioraHover(model: Model, tracker: INotebookTracker) {
  const lifetime = ViewPlugin.fromClass(class {
    alive = true;
    cancel: (hide?: boolean) => void = () => {};
    update(update: ViewUpdate) { if (update.docChanged) this.cancel(); }
    destroy() { this.alive = false; this.cancel(); }
  });
  return [lifetime, hoverTooltip((view: EditorView, pos: number) => {
    const state = view.plugin(lifetime)!;
    state.cancel();
    const doc = view.state.doc;
    const source = doc.toString();
    if (!isEqioraCell(source) || pos <= source.indexOf('\n') || source.indexOf('\n') < 0
      || source.length > 256 * 1024 || new TextEncoder().encode(source).length > 256 * 1024
      || model.sharedModel.getSource() !== source) return null;
    const panel = owner(tracker, model);
    const kernel = panel?.sessionContext.session?.kernel;
    if (!panel || !kernel || kernel.isDisposed || kernel.connectionStatus !== 'connected'
      || kernel.status !== 'idle') return null;
    const cursor = Array.from(source.slice(0, pos)).length;
    return new Promise<Tooltip | null>(resolve => {
      const comm = kernel.createComm('eqiora.source_hover');
      let active = true;
      let replied = false;
      let remoteClosed = false;
      let timer: ReturnType<typeof setTimeout>;
      const current = () => active && state.alive && view.state.doc === doc
        && model.sharedModel.getSource() === source && owner(tracker, model) === panel
        && panel.sessionContext.session?.kernel === kernel && !kernel.isDisposed
        && kernel.connectionStatus === 'connected' && ['idle', 'busy'].includes(kernel.status);
      const cancel = (hide = false) => {
        if (!active) return;
        active = false;
        clearTimeout(timer);
        model.sharedModel.changed.disconnect(changed);
        panel.disposed.disconnect(changed);
        panel.sessionContext.kernelChanged.disconnect(changed);
        kernel.statusChanged.disconnect(status);
        kernel.connectionStatusChanged.disconnect(connection);
        kernel.disposed.disconnect(changed);
        comm.onClose = () => {};
        if (!comm.isDisposed && !remoteClosed && !kernel.isDisposed
          && kernel.connectionStatus === 'connected' && ['idle', 'busy'].includes(kernel.status)) {
          comm.close().dispose();
        }
        comm.dispose();
        resolve(null);
        if (hide) queueMicrotask(() => {
          if (state.alive) view.dispatch({ effects: closeHoverTooltips });
        });
      };
      const changed = () => cancel(true);
      const status = () => { if (!['idle', 'busy'].includes(kernel.status)) cancel(true); };
      const connection = () => { if (kernel.connectionStatus !== 'connected') cancel(true); };
      state.cancel = cancel;
      model.sharedModel.changed.connect(changed);
      panel.disposed.connect(changed);
      panel.sessionContext.kernelChanged.connect(changed);
      kernel.statusChanged.connect(status);
      kernel.connectionStatusChanged.connect(connection);
      kernel.disposed.connect(changed);
      timer = setTimeout(() => cancel(), 3000);
      comm.onClose = () => { remoteClosed = true; if (!replied) cancel(); };
      comm.onMsg = message => {
        if (replied || !current()) { cancel(); return; }
        const result = message.content.data.result;
        if (!Array.isArray(result) || result.length !== 3) { cancel(); return; }
        const [start, end, text] = result;
        const points = Array.from(source);
        if (typeof start !== 'number' || typeof end !== 'number'
          || !Number.isInteger(start) || !Number.isInteger(end) || start < 0
          || start > cursor || end < cursor || end <= start || end > points.length
          || typeof text !== 'string' || !text || text.length > 8192
          || Array.from(text).length > 4096) { cancel(); return; }
        replied = true;
        resolve({
          pos: points.slice(0, start).join('').length,
          end: points.slice(0, end).join('').length,
          above: true,
          create: () => {
            // If CodeMirror discarded the pending hover, create never runs;
            // its original deadline still releases the lifecycle subscriptions.
            clearTimeout(timer);
            const dom = document.createElement('div');
            dom.className = 'eqiora-source-hover';
            dom.style.whiteSpace = 'pre-wrap';
            dom.style.maxWidth = '40em';
            dom.textContent = current() ? text : '';
            return { dom, destroy: () => cancel() };
          },
        });
      };
      comm.open({ source, cursor }).done.catch(() => { if (!replied) cancel(); });
    });
  }, { hideOnChange: true })];
}
