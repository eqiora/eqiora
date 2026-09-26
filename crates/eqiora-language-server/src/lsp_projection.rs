use eqiora::api::{EditorPosition, EditorSnapshot, EditorSymbol, EditorSymbolKind};
use eqiora::{Diagnostic as EqioraDiagnostic, Severity};
use lsp_types::{NumberOrString, Position, Range, SymbolKind};

pub(crate) fn lsp_diagnostic(
    snapshot: &EditorSnapshot,
    diagnostic: &EqioraDiagnostic,
) -> lsp_types::Diagnostic {
    let range = diagnostic
        .source_span()
        .and_then(|span| {
            source_range(
                snapshot,
                usize::try_from(span.start).ok()?,
                usize::try_from(span.end).ok()?,
            )
            .ok()
        })
        .unwrap_or_default();
    lsp_types::Diagnostic {
        range,
        severity: Some(match diagnostic.severity() {
            Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
            Severity::Warning => lsp_types::DiagnosticSeverity::WARNING,
            Severity::Note => lsp_types::DiagnosticSeverity::INFORMATION,
        }),
        code: Some(NumberOrString::String(diagnostic.code().to_string())),
        source: Some("eqiora".to_owned()),
        message: diagnostic.message().to_owned(),
        ..lsp_types::Diagnostic::default()
    }
}

pub(crate) fn source_range(
    snapshot: &EditorSnapshot,
    start: usize,
    end: usize,
) -> Result<Range, String> {
    let start = u32::try_from(start).map_err(|_| "source offset exceeds u32".to_owned())?;
    let end = u32::try_from(end).map_err(|_| "source offset exceeds u32".to_owned())?;
    let start = snapshot
        .position(start)
        .ok_or_else(|| "source start is not an exact UTF-8 boundary".to_owned())?;
    let end = snapshot
        .position(end)
        .ok_or_else(|| "source end is not an exact UTF-8 boundary".to_owned())?;
    Ok(Range::new(lsp_position(start), lsp_position(end)))
}

pub(crate) fn symbol_range(
    snapshot: &EditorSnapshot,
    symbol: &EditorSymbol,
) -> Result<Range, String> {
    source_range(
        snapshot,
        usize::try_from(symbol.range().start()).map_err(|_| "source offset exceeds usize")?,
        usize::try_from(symbol.range().end()).map_err(|_| "source offset exceeds usize")?,
    )
}

fn lsp_position(position: EditorPosition) -> Position {
    Position::new(position.line(), position.character())
}

pub(crate) const fn editor_position(position: Position) -> EditorPosition {
    EditorPosition::new(position.line, position.character)
}

pub(crate) const fn symbol_kind(kind: EditorSymbolKind) -> SymbolKind {
    match kind {
        EditorSymbolKind::Import => SymbolKind::NAMESPACE,
        EditorSymbolKind::Dimension => SymbolKind::TYPE_PARAMETER,
        EditorSymbolKind::Enum => SymbolKind::ENUM,
        EditorSymbolKind::Record => SymbolKind::STRUCT,
        EditorSymbolKind::FiniteSpace => SymbolKind::TYPE_PARAMETER,
        EditorSymbolKind::EnumMember => SymbolKind::ENUM_MEMBER,
        EditorSymbolKind::IndexSet => SymbolKind::ARRAY,
        EditorSymbolKind::Observable => SymbolKind::VARIABLE,
        EditorSymbolKind::Property | EditorSymbolKind::Parameter => SymbolKind::PROPERTY,
        EditorSymbolKind::Material => SymbolKind::OBJECT,
        EditorSymbolKind::Connector => SymbolKind::INTERFACE,
        EditorSymbolKind::Component | EditorSymbolKind::Model => SymbolKind::CLASS,
        EditorSymbolKind::Operator => SymbolKind::FUNCTION,
        EditorSymbolKind::Domain | EditorSymbolKind::Support => SymbolKind::NAMESPACE,
        EditorSymbolKind::Let | EditorSymbolKind::Formal => SymbolKind::CONSTANT,
        EditorSymbolKind::Field => SymbolKind::FIELD,
        EditorSymbolKind::Port => SymbolKind::INTERFACE,
        EditorSymbolKind::Clock | EditorSymbolKind::Event => SymbolKind::EVENT,
        EditorSymbolKind::Relation => SymbolKind::OPERATOR,
        EditorSymbolKind::Instance => SymbolKind::OBJECT,
        _ => SymbolKind::OBJECT,
    }
}

pub(crate) const fn symbol_label(kind: EditorSymbolKind) -> &'static str {
    match kind {
        EditorSymbolKind::Import => "Import",
        EditorSymbolKind::Dimension => "Dimension",
        EditorSymbolKind::Enum => "Enum",
        EditorSymbolKind::Record => "Record",
        EditorSymbolKind::FiniteSpace => "Finite space",
        EditorSymbolKind::EnumMember => "Enum member",
        EditorSymbolKind::IndexSet => "Index set",
        EditorSymbolKind::Observable => "Observable",
        EditorSymbolKind::Property => "Property",
        EditorSymbolKind::Material => "Material",
        EditorSymbolKind::Connector => "Connector",
        EditorSymbolKind::Component => "Component",
        EditorSymbolKind::Operator => "Operator",
        EditorSymbolKind::Model => "Model",
        EditorSymbolKind::Domain => "Domain",
        EditorSymbolKind::Parameter => "Parameter",
        EditorSymbolKind::Let => "Let",
        EditorSymbolKind::Formal => "Formal",
        EditorSymbolKind::Support => "Support",
        EditorSymbolKind::Field => "Field",
        EditorSymbolKind::Port => "Port",
        EditorSymbolKind::Clock => "Clock",
        EditorSymbolKind::Event => "Event",
        EditorSymbolKind::Relation => "Relation",
        EditorSymbolKind::Instance => "Instance",
        _ => "Declaration",
    }
}
