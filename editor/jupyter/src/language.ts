import { HighlightStyle, StreamLanguage, syntaxHighlighting, type StreamParser } from '@codemirror/language';
import { Compartment, EditorState, Prec, type Extension } from '@codemirror/state';
import { createJavaScriptRegexEngine } from '@shikijs/engine-javascript';
import { INITIAL, Registry, type IRawGrammar, type IToken, type StateStack } from '@shikijs/vscode-textmate';
import { tags } from '@lezer/highlight';
import grammarSource from './grammar.json';

const engine = createJavaScriptRegexEngine();
const registry = new Registry({
  onigLib: {
    createOnigScanner: engine.createScanner,
    createOnigString: engine.createString,
  },
  loadGrammar: (scope) => scope === grammarSource.scopeName ? grammarSource as IRawGrammar : null,
});
const grammar = registry.loadGrammar(grammarSource.scopeName)!;

// This maps presentation scopes, never Eqiora names or semantic rules.
function style(scopes: readonly string[]): string | null {
  for (const scope of [...scopes].reverse()) {
    if (scope.startsWith('comment.')) return 'comment';
    if (scope.startsWith('keyword.operator.')) return 'operator';
    if (scope.startsWith('keyword.') || scope.startsWith('storage.')) return 'keyword';
    if (scope.startsWith('support.type.') || scope.startsWith('entity.name.type.')) return 'typeName';
    if (scope.startsWith('support.function.')) return 'builtin';
    if (scope.startsWith('constant.numeric.')) return 'number';
    if (scope.startsWith('constant.')) return 'atom';
    if (scope.startsWith('variable.') || scope.startsWith('entity.')) return 'variableName';
    if (scope.startsWith('punctuation.')) return 'punctuation';
  }
  return null;
}

interface State {
  firstLine: boolean;
  stack: StateStack;
  tokens: readonly IToken[];
  index: number;
}

const parser: StreamParser<State> = {
  name: 'eqiora',
  startState: () => ({ firstLine: true, stack: INITIAL, tokens: [], index: 0 }),
  copyState: (state) => ({ ...state }),
  blankLine(state) { state.firstLine = false; },
  token(stream, state) {
    if (stream.sol()) {
      const header = state.firstLine && isEqioraCell(stream.string);
      state.firstLine = false;
      if (header) {
        stream.skipToEnd();
        return 'meta';
      }
      const result = grammar.tokenizeLine(stream.string, state.stack);
      state.stack = result.ruleStack;
      state.tokens = result.tokens;
      state.index = 0;
    }
    while (state.index < state.tokens.length && state.tokens[state.index].endIndex <= stream.pos) {
      state.index++;
    }
    const token = state.tokens[state.index];
    if (!token) { stream.skipToEnd(); return null; }
    stream.pos = Math.min(token.endIndex, stream.string.length);
    return style(token.scopes);
  },
};

export const eqioraLanguage = StreamLanguage.define(parser);

export function isEqioraCell(source: string): boolean {
  return /^%%eqiora(?:[\t ]|\r?$)/.test(source.split('\n', 1)[0]);
}

/** Overrides the host language only while this cell has an Eqiora magic header. */
export function eqioraMagic(initialSource: string): Extension {
  const language = new Compartment();
  const selected = (source: string) => isEqioraCell(source) ? [
    Prec.highest(eqioraLanguage),
    syntaxHighlighting(HighlightStyle.define([{
      tag: tags.typeName, color: 'var(--jp-mirror-editor-builtin-color, #0550ae)',
    }], { scope: eqioraLanguage })),
  ] : [];
  return [
    language.of(selected(initialSource)),
    EditorState.transactionExtender.of((transaction) => {
      if (!transaction.docChanged) return null;
      const before = transaction.startState.doc.line(1).text;
      const after = transaction.newDoc.line(1).text;
      return isEqioraCell(before) === isEqioraCell(after)
        ? null : { effects: language.reconfigure(selected(after)) };
    }),
  ];
}
