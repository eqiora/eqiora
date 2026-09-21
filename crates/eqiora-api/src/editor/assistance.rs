//! Syntax recovery and declaration queries for editor assistance.
use super::{EditorSnapshot, EditorSymbol, EditorSymbolKind, EditorWorkspaceSnapshot};
use eqiora_lang::TextRange;
use std::collections::BTreeMap;

mod builtins;
mod cursor;
mod declarations;
mod query;
#[cfg(test)]
mod tests;
mod vocabulary;

/// Syntactic position used to select applicable completion families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Context {
    /// A declaration or its modifier may begin here.
    Declaration,
    /// A type or definition is expected after a colon.
    Type,
    /// A value expression is being authored.
    Expression,
    /// A canonical module path is being imported.
    Import,
    /// A qualified member is being selected.
    Member,
    /// A named binding in an instance argument list.
    Argument,
}

/// Completion at one UTF-8 cursor position in an immutable source snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Completion {
    /// Exact full token range to replace, including any qualification.
    pub range: TextRange,
    /// Visible authored candidates matching the prefix.
    pub items: Vec<EditorSymbol>,
}

impl EditorSnapshot {
    /// Recover the qualified spelling and UTF-8 range under a hover cursor.
    #[must_use]
    pub fn name_at(&self, offset: u32) -> Option<(String, TextRange)> {
        if self.source.len() > Self::MAX_SOURCE_BYTES {
            return None;
        }
        let (name, start, end) = cursor::name_at(&self.source, offset, false)?;
        Some((name, TextRange::new(start, end)))
    }

    /// Query documented builtin vocabulary and current local declarations.
    /// Returns the replacement range and matching symbols, recovering incomplete source.
    /// Returns `None` inside comments/notation or at an invalid byte position.
    #[must_use]
    pub fn completion(&self, offset: u32) -> Option<(TextRange, Vec<EditorSymbol>)> {
        query::Query::local(self)
            .completion(offset)
            .map(|c| (c.range, c.items))
    }

    /// Resolve documented vocabulary or a visible authored spelling for hover or signature help.
    #[must_use]
    pub fn assistance(&self, offset: u32, name: &str) -> Option<EditorSymbol> {
        query::Query::local(self).resolve(offset, name)
    }

    /// Recover the callable name, argument index and named binding at the cursor,
    /// including missing delimiters.
    #[must_use]
    pub fn call_at(&self, offset: u32) -> Option<(String, usize, Option<String>)> {
        if self.source.len() > Self::MAX_SOURCE_BYTES {
            return None;
        }
        cursor::call_at(&self.source, offset).map(|call| (call.name, call.argument, call.named))
    }
}

impl EditorWorkspaceSnapshot {
    /// Query documented vocabulary, declarations and canonical modules in the current graph.
    /// Returns the replacement range and matching symbols.
    /// Recovery never grants executable validity or crosses a private boundary.
    /// Prepared Model parameter initializers, component parameter bindings and
    /// scalar connection endpoints rank compatible contracts first, unknown
    /// candidates next, and incompatible contracts last. Only complete simple
    /// references are ranked. Clocked/spatial endpoints, arithmetic operands,
    /// unsupported types and failed analysis retain ordinary name completion.
    #[must_use]
    pub fn completion(&self, file: &str, offset: u32) -> Option<(TextRange, Vec<EditorSymbol>)> {
        query::Query::workspace(self, file)?
            .completion(offset)
            .map(|c| (c.range, c.items))
    }

    /// Resolve documented vocabulary or an authored local, imported or exposed member spelling.
    #[must_use]
    pub fn assistance(&self, file: &str, offset: u32, name: &str) -> Option<EditorSymbol> {
        query::Query::workspace(self, file)?.resolve(offset, name)
    }
}

fn contains(symbol: &EditorSymbol, source: &str, offset: u32) -> bool {
    let range = symbol.range();
    range.start() <= offset
        && (offset < range.end()
            || (offset == range.end()
                && offset as usize == source.len()
                && cursor::tokens(&source[range.start() as usize..])
                    .iter()
                    .fold(0i32, |n, t| {
                        use eqiora_lang::TokenKind as K;
                        n + match t.kind() {
                            K::LeftBrace | K::LeftParen => 1,
                            K::RightBrace | K::RightParen => -1,
                            _ => 0,
                        }
                    })
                    > 0))
}

fn visible<'a>(
    symbols: &'a [EditorSymbol],
    source: &str,
    offset: u32,
    out: &mut BTreeMap<&'a str, &'a EditorSymbol>,
) {
    for symbol in symbols {
        out.insert(symbol.name(), symbol);
    }
    for symbol in symbols.iter().filter(|s| contains(s, source, offset)) {
        visible(symbol.children(), source, offset, out);
    }
}

fn documented(
    name: &str,
    detail: &str,
    documentation: &str,
    kind: EditorSymbolKind,
) -> EditorSymbol {
    let mut symbol = EditorSymbol::leaf(kind, name, TextRange::default());
    symbol.detail = Some(detail.into());
    symbol.help = Some(documentation.into());
    symbol
}

// This cache is a deterministic projection of the exact admitted input graph.
// Compare the inputs rather than requiring compiler implementation internals to
// become part of the editor snapshot's equality contract.
#[derive(Clone, Debug)]
pub(super) struct PreparedCompletion {
    pub(super) input: std::sync::Arc<eqiora_compiler::ResolvedHierarchyInput>,
    pub(super) analysis: std::sync::Arc<eqiora_compiler::AnalyzedResolvedHierarchy>,
    pub(super) file: String,
}
impl PartialEq for PreparedCompletion {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.file == other.file
    }
}
