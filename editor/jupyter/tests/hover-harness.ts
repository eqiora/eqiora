import { EditorState } from '@codemirror/state';
import { activateHover, EditorView } from '@codemirror/view';
import { eqioraHover } from '../src/hover';

class Signal {
  slots = new Set<() => void>();
  connect(slot: () => void) { this.slots.add(slot); }
  disconnect(slot: () => void) { this.slots.delete(slot); }
  emit() { for (const slot of [...this.slots]) slot(); }
}
let view: EditorView;
let source: string;
let requests: any[];
let kernel: any;
let panel: any;
let shared: any;
let panels: any[];
function open(text: string) {
  view?.destroy();
  source = text;
  requests = [];
  shared = { getSource: () => source, changed: new Signal() };
  kernel = {
    status: 'idle', connectionStatus: 'connected', isDisposed: false,
    statusChanged: new Signal(), connectionStatusChanged: new Signal(), disposed: new Signal(),
    createComm: (target: string) => {
      const comm: any = {
        target, isDisposed: false, onClose: () => {}, onMsg: () => {}, closed: false,
        open(data: any) { comm.data = data; return { done: Promise.resolve() }; },
        close() {
          if (kernel.isDisposed) throw new Error('Cannot close');
          comm.closed = true; comm.onClose(); return { dispose() {} };
        },
        dispose() { comm.isDisposed = true; },
      };
      requests.push(comm);
      return comm;
    },
  };
  panel = {
    isDisposed: false, disposed: new Signal(), model: { cells: [{ sharedModel: shared }] },
    sessionContext: { session: { kernel }, kernelChanged: new Signal() },
  };
  panels = [panel];
  view = new EditorView({ parent: document.querySelector('#editor')!, state: EditorState.create({
    doc: source,
    extensions: [eqioraHover({ sharedModel: shared } as any, { forEach: (fn: any) => panels.forEach(fn) } as any)],
  }) });
}
Object.assign(window, { hover: {
  open,
  request: (position: number) => activateHover(view, position, 1),
  requests: () => requests.map(({ target, data, closed, isDisposed }) => ({ target, data, closed, isDisposed })),
  reply: (result: unknown, index = requests.length - 1) => requests[index].onMsg({ content: { data: { result } } }),
  remoteClose: () => requests.at(-1).onClose(),
  edit: (text: string) => {
    source = text;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
    shared.changed.emit();
  },
  restart: () => { kernel.status = 'restarting'; kernel.statusChanged.emit(); kernel.status = 'idle'; },
  switchKernel: () => { panel.sessionContext.session.kernel = { ...kernel }; panel.sessionContext.kernelChanged.emit(); },
  detach: () => { panels = []; },
  ambiguous: () => { panels = [panel, { ...panel }]; },
  disconnect: () => { kernel.connectionStatus = 'disconnected'; kernel.connectionStatusChanged.emit(); },
  busy: () => { kernel.status = 'busy'; kernel.statusChanged.emit(); },
  disposeKernel: () => { kernel.isDisposed = true; kernel.disposed.emit(); },
  destroy: () => view.destroy(),
} });
