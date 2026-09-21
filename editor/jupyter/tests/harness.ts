import { EditorState } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { python } from '@codemirror/lang-python';
import { syntaxHighlighting, defaultHighlightStyle, syntaxTree, language } from '@codemirror/language';
import { eqioraMagic } from '../src/language';

let view: EditorView;
function open(source: string) {
  view?.destroy();
  view = new EditorView({
    parent: document.querySelector('#editor')!,
    state: EditorState.create({
      doc: source,
      extensions: [python(), syntaxHighlighting(defaultHighlightStyle), eqioraMagic(source)],
    }),
  });
}
function replace(source: string) {
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
}
function tokens() {
  const result: { name: string; text: string }[] = [];
  syntaxTree(view.state).iterate({ enter(node) {
    if (node.type.name !== 'Document' && !node.node.firstChild)
      result.push({ name: node.name, text: view.state.sliceDoc(node.from, node.to) });
  } });
  return result;
}
Object.assign(window, { editor: { open, replace, tokens, language: () => view.state.facet(language)?.name } });
open('x = 1\nprint(x)');
