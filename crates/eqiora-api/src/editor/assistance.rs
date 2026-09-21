//! Syntax recovery and declaration queries for editor assistance.
use super::{EditorSnapshot, EditorSymbol, EditorSymbolKind, EditorWorkspaceSnapshot};
use eqiora_lang::{DocComment, TextRange};
use std::collections::BTreeMap;

mod cursor;
mod declarations;
mod query;
#[cfg(test)]
mod tests;
pub use cursor::EditorCall;

/// Syntactic position used to select applicable completion families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditorCompletionContext {
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

/// One declared formal, including whether an instance may bind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorParameter {
    /// Source name.
    pub name: String,
    /// Complete authored signature entry.
    pub detail: String,
    /// Sanitized declaration documentation.
    pub documentation: Option<DocComment>,
    /// `Some(true)` for required bindings, `Some(false)` for defaulted bindings,
    /// and `None` for owned endpoints.
    pub required: Option<bool>,
}

/// An authored declaration or insertion candidate from the current snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorCandidate {
    /// Visible spelling, including any qualifier.
    pub name: String,
    /// Source declaration head.
    pub detail: String,
    /// Sanitized declaration documentation.
    pub documentation: Option<DocComment>,
    /// Editor declaration category.
    pub kind: EditorSymbolKind,
    /// Insertion replacing the completion's exact range.
    pub insert_text: String,
    /// Declared callable signature, when known.
    pub parameters: Option<Vec<EditorParameter>>,
    /// Required/defaulted classification for a named-argument candidate.
    pub required: Option<bool>,
}

/// Completion at one UTF-8 cursor position in an immutable source snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorCompletion {
    /// Exact full token range to replace, including any qualification.
    pub range: TextRange,
    /// Source spelling before the cursor, for filtering builtin candidates too.
    pub prefix: String,
    /// Syntactic candidate family.
    pub context: EditorCompletionContext,
    /// Visible authored candidates matching the prefix.
    pub items: Vec<EditorCandidate>,
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

    /// Query current local declarations, recovering incomplete source.
    /// Returns `None` inside comments/notation or at an invalid byte position.
    #[must_use]
    pub fn completion(&self, offset: u32) -> Option<EditorCompletion> {
        query::Query::local(self).completion(offset)
    }

    /// Resolve a visible authored spelling for hover or signature help.
    #[must_use]
    pub fn assistance(&self, offset: u32, name: &str) -> Option<EditorCandidate> {
        query::Query::local(self).resolve(offset, name)
    }

    /// Recover the callable containing the cursor, including missing delimiters.
    #[must_use]
    pub fn call_at(&self, offset: u32) -> Option<EditorCall> {
        if self.source.len() > Self::MAX_SOURCE_BYTES {
            return None;
        }
        cursor::call_at(&self.source, offset)
    }
}

impl EditorWorkspaceSnapshot {
    /// Query declarations and canonical modules in the current graph.
    /// Recovery never grants executable validity or crosses a private boundary.
    #[must_use]
    pub fn completion(&self, file: &str, offset: u32) -> Option<EditorCompletion> {
        query::Query::workspace(self, file)?.completion(offset)
    }

    /// Resolve an authored local, imported or exposed member spelling.
    #[must_use]
    pub fn assistance(&self, file: &str, offset: u32, name: &str) -> Option<EditorCandidate> {
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
