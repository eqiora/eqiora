//! Navigation projects compiler-owned declaration identities and source owners.
use super::{EditorWorkspaceSnapshot, cursor};
use crate::editor::{EditorPosition, EditorSymbol, EditorSymbolKind};
use eqiora_core::Span;
use eqiora_lang::{TextRange, Token, TokenKind};

impl EditorWorkspaceSnapshot {
    /// Resolve a prepared Model/Component Field/Parameter/Port/Clock value reference or a Model child's
    /// public Port reference to the exact declaration name in its source file.
    /// The cursor must cover the terminal identifier, not a qualifier or dot.
    /// Deeper members, nested binders and invalid snapshots return no location;
    /// lexical recovery never grants navigation. Clock targets include owned periodic
    /// declarations and prepared signature requirements, without inferring a borrowed
    /// schedule. Prepared Field requirements also retain their own source target,
    /// without inferring occurrence-specific type/support facts. Exact authored
    /// activation-name tokens use the same Clock target.
    #[must_use]
    pub fn value_definition_at_position(
        &self,
        file: &str,
        position: EditorPosition,
    ) -> Option<eqiora_core::Span> {
        if !self.diagnostics().is_empty() {
            return None;
        }
        let snapshot = self.document(file)?;
        let offset = snapshot.byte_offset(position)?;
        let semantics = snapshot.semantics.as_ref()?;
        let (name, declaration) = semantics.analysis.value_definition(file, offset)?;
        if !cursor::at_value_name(&snapshot.source, offset, name) {
            return None;
        }
        let terminal = name.rsplit('.').next()?;
        let target = self.document(&declaration.file)?;
        let source = target
            .source
            .get(declaration.start as usize..declaration.end as usize)?;
        let tokens = source_tokens(source);
        let range = declaration_name(&tokens, TextRange::new(0, source.len() as u32), terminal)?;
        Some(Span {
            file: declaration.file,
            start: declaration.start + range.start(),
            end: declaration.start + range.end(),
        })
    }

    /// Find references to a prepared Model/Component Field/Parameter/Port/Clock declaration or
    /// a Model child's public Port across prepared declaration scopes.
    /// The cursor must be on its declaration name or a terminal value-reference
    /// token. Multiple instance spellings may share one source declaration;
    /// these results do not identify physical occurrences or support rename.
    /// Clock results cover retained value uses and authored activation-name tokens,
    /// including prepared Clock requirements. Field requirements retain their own
    /// declaration references. Caller binding labels are not value uses.
    /// Results use source-qualified exact identifier spans, sorted by file and
    /// offset. Include the declaration once only when requested. An admitted
    /// unused declaration returns `Some([])`; unknown targets, private child/deeper
    /// members, binder cursors and invalid/recovering snapshots return `None`.
    #[must_use]
    pub fn value_references_at_position(
        &self,
        file: &str,
        position: EditorPosition,
        include_declaration: bool,
    ) -> Option<Vec<Span>> {
        if !self.diagnostics().is_empty() {
            return None;
        }
        let snapshot = self.document(file)?;
        let offset = snapshot.byte_offset(position)?;
        let tokens = source_tokens(&snapshot.source);
        let token = tokens.iter().find(|token| {
            token.kind() == TokenKind::Identifier
                && token.range().start() <= offset
                && offset < token.range().end()
        })?;
        let cursor = span(file, token.range());
        let semantics = snapshot.semantics.as_ref()?;
        let (name, declaration) =
            if let Some((name, declaration)) = semantics.analysis.value_definition(file, offset) {
                (name.rsplit('.').next()?.to_owned(), declaration)
            } else {
                let symbol = declared_at(snapshot.symbols(), &tokens, offset)?;
                (symbol.name().to_owned(), span(file, symbol.range()))
            };
        if token.text() != name {
            return None;
        }
        // Lexical declaration selection is only a query key; the compiler must
        // have admitted this exact whole declaration before any result exists.
        let expressions = semantics.analysis.value_references(&declaration)?;
        let mut by_file = std::collections::BTreeMap::<String, Vec<Span>>::new();
        for expression in expressions {
            by_file
                .entry(expression.file.clone())
                .or_default()
                .push(expression);
        }
        by_file.entry(declaration.file.clone()).or_default();
        let mut input_tokens = Some(tokens);
        let mut declaration_token = None;
        let mut references = Vec::new();
        for (origin, expressions) in by_file {
            // Reuse the input tokens and sweep each other participating file
            // once, including parenthesized names and spaced qualified paths.
            let tokens = if origin == file {
                input_tokens.take()?
            } else {
                source_tokens(&self.document(&origin)?.source)
            };
            if origin == declaration.file {
                declaration_token = declaration_name(
                    &tokens,
                    TextRange::new(declaration.start, declaration.end),
                    &name,
                )
                .map(|range| span(&origin, range));
            }
            let mut index = 0;
            for expression in expressions {
                while tokens
                    .get(index)
                    .is_some_and(|token| token.range().end() <= expression.start)
                {
                    index += 1;
                }
                let mut terminal = None;
                while let Some(token) = tokens
                    .get(index)
                    .filter(|token| token.range().start() < expression.end)
                {
                    if token.kind() == TokenKind::Identifier
                        && token.range().end() <= expression.end
                    {
                        terminal = Some(token);
                    }
                    index += 1;
                }
                let terminal = terminal?;
                if terminal.text() != name {
                    return None;
                }
                references.push(span(&origin, terminal.range()));
            }
        }
        let declaration_token = declaration_token?;
        // A qualifier with the same spelling as its terminal member still has
        // a different range. Neither qualifiers nor units acquire references.
        if cursor != declaration_token && !references.contains(&cursor) {
            return None;
        }
        if include_declaration {
            references.push(declaration_token);
        }
        references.sort_by(|left, right| {
            (&left.file, left.start, left.end).cmp(&(&right.file, right.start, right.end))
        });
        references.dedup();
        Some(references)
    }
}

fn source_tokens(source: &str) -> Vec<Token> {
    let mut tokens = cursor::tokens(source);
    tokens.retain(|token| token.kind() != TokenKind::Notation);
    tokens
}

fn span(file: &str, range: TextRange) -> Span {
    Span {
        file: file.to_owned(),
        start: range.start(),
        end: range.end(),
    }
}

fn declaration_name(tokens: &[Token], range: TextRange, name: &str) -> Option<TextRange> {
    let start = tokens.partition_point(|token| token.range().start() < range.start());
    let end = tokens.partition_point(|token| token.range().start() < range.end());
    let pair = tokens[start..end]
        .windows(2)
        .find(|pair| matches!(pair[1].kind(), TokenKind::Colon | TokenKind::Equal))?;
    (pair[0].kind() == TokenKind::Identifier && pair[0].text() == name).then_some(pair[0].range())
}

fn declared_at<'a>(
    symbols: &'a [EditorSymbol],
    tokens: &[Token],
    offset: u32,
) -> Option<&'a EditorSymbol> {
    let symbol = symbols
        .iter()
        .find(|symbol| symbol.range().start() <= offset && offset < symbol.range().end())?;
    if matches!(
        symbol.kind(),
        EditorSymbolKind::Field
            | EditorSymbolKind::Parameter
            | EditorSymbolKind::Port
            | EditorSymbolKind::Clock
    ) && declaration_name(tokens, symbol.range(), symbol.name())
        .is_some_and(|range| range.start() <= offset && offset < range.end())
    {
        return Some(symbol);
    }
    declared_at(symbol.children(), tokens, offset)
}
