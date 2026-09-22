//! Same-file navigation projects compiler-owned declaration identities.
use super::{EditorWorkspaceSnapshot, cursor};
use crate::editor::EditorPosition;
use eqiora_lang::{TextRange, TokenKind};

impl EditorWorkspaceSnapshot {
    /// Resolve a simple Model field or parameter reference to its same-file
    /// declaration name. Invalid snapshots, positions, qualified members and
    /// nested binder scopes return no location; recovery grants no navigation.
    #[must_use]
    pub fn local_definition_at_position(
        &self,
        file: &str,
        position: EditorPosition,
    ) -> Option<TextRange> {
        if !self.diagnostics().is_empty() {
            return None;
        }
        let snapshot = self.document(file)?;
        let offset = snapshot.byte_offset(position)?;
        let semantics = snapshot.semantics.as_ref()?;
        let (name, declaration) = semantics.analysis.local_definition(file, offset)?;
        if snapshot.name_at(offset)?.0 != name {
            return None;
        }
        let source = snapshot
            .source
            .get(declaration.start() as usize..declaration.end() as usize)?;
        cursor::tokens(source).windows(2).find_map(|pair| {
            let token = &pair[0];
            (token.kind() == eqiora_lang::TokenKind::Identifier
                && token.text() == name
                && pair[1].kind() == eqiora_lang::TokenKind::Colon)
                .then(|| {
                    TextRange::new(
                        declaration.start() + token.range().start(),
                        declaration.start() + token.range().end(),
                    )
                })
        })
    }

    /// Find simple value references to a same-file Model field or parameter.
    /// The cursor must name its declaration or an admitted value reference.
    /// Results have exact identifier ranges in source order; declarations are
    /// included only when requested. A supported unused declaration returns an
    /// empty list. Invalid snapshots and unsupported scopes return `None`.
    /// References inside nested binders and qualified names are omitted; this
    /// bounded query is not a complete rename or cross-file reference index.
    #[must_use]
    pub fn local_references_at_position(
        &self,
        file: &str,
        position: EditorPosition,
        include_declaration: bool,
    ) -> Option<Vec<TextRange>> {
        if !self.diagnostics().is_empty() {
            return None;
        }
        let snapshot = self.document(file)?;
        let offset = snapshot.byte_offset(position)?;
        let (name, cursor_range) = snapshot.name_at(offset)?;
        let semantics = snapshot.semantics.as_ref()?;
        let (declaration, expressions) =
            semantics.analysis.local_references(file, offset, &name)?;
        // A single source token sweep handles parenthesized Name expressions
        // without re-lexing the whole source for every occurrence.
        let tokens = cursor::tokens(&snapshot.source);
        let declaration_name = tokens.windows(2).find_map(|pair| {
            let token = &pair[0];
            (declaration.start() <= token.range().start()
                && pair[1].range().end() <= declaration.end()
                && token.kind() == TokenKind::Identifier
                && token.text() == name
                && pair[1].kind() == TokenKind::Colon)
                .then_some(token.range())
        })?;
        let mut expressions = expressions.iter().peekable();
        let mut ranges = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            while expressions
                .peek()
                .is_some_and(|range| range.end() <= token.range().start())
            {
                expressions.next();
            }
            if token.kind() == TokenKind::Identifier
                && token.text() == name
                && index
                    .checked_sub(1)
                    .and_then(|i| tokens.get(i))
                    .is_none_or(|previous| previous.kind() != TokenKind::Dot)
                && tokens
                    .get(index + 1)
                    .is_none_or(|next| next.kind() != TokenKind::Dot)
                && expressions.peek().is_some_and(|range| {
                    range.start() <= token.range().start() && token.range().end() <= range.end()
                })
            {
                ranges.push(token.range());
            }
        }
        if cursor_range != declaration_name && !ranges.contains(&cursor_range) {
            return None;
        }
        if include_declaration {
            let index = ranges.partition_point(|range| range.start() < declaration_name.start());
            ranges.insert(index, declaration_name);
        }
        Some(ranges)
    }
}
