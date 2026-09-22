use super::{EditorSnapshot, EditorSymbol, EditorSymbolKind, cursor};
use eqiora_lang::{Document, SignatureItem};

// The first exact identifier in an AST declaration is its name. Testing this
// token, rather than the whole declaration, excludes initializer binder uses.
pub(super) fn at_name(snapshot: &EditorSnapshot, symbol: &EditorSymbol, offset: u32) -> bool {
    let Some(source) = snapshot
        .source
        .get(symbol.range().start() as usize..symbol.range().end() as usize)
    else {
        return false;
    };
    cursor::tokens(source)
        .iter()
        .find(|token| {
            token.kind() == eqiora_lang::TokenKind::Identifier && token.text() == symbol.name()
        })
        .is_some_and(|token| {
            symbol.range().start() + token.range().start() <= offset
                && offset < symbol.range().start() + token.range().end()
        })
}

pub(super) fn candidate(snapshot: &EditorSnapshot, symbol: &EditorSymbol) -> Option<EditorSymbol> {
    let text = snapshot
        .source
        .get(symbol.range().start() as usize..symbol.range().end() as usize)?;
    let callable = matches!(
        symbol.kind(),
        EditorSymbolKind::Operator | EditorSymbolKind::Component | EditorSymbolKind::Model
    );
    let end = cursor::head_end(text, callable);
    let signature = snapshot
        .syntax
        .as_ref()
        .and_then(|doc| signature(doc, symbol.name()));
    let parameters = callable.then(|| {
        symbol
            .children()
            .iter()
            .filter(|p| p.range().start() < symbol.range().start() + end as u32)
            .filter_map(|p| {
                let mut formal = p.clone();
                formal.detail = Some(cursor::clean(
                    snapshot
                        .source
                        .get(p.range().start() as usize..p.range().end() as usize)?,
                ));
                formal.binding_required = signature
                    .and_then(|items| items.iter().find(|item| item.name() == p.name()))
                    .and_then(|item| match item {
                        SignatureItem::Parameter(value) => Some(value.default().is_none()),
                        SignatureItem::Support(_)
                        | SignatureItem::Field(_)
                        | SignatureItem::Clock(_)
                        | SignatureItem::Property(_)
                        | SignatureItem::Input(_) => Some(true),
                        _ => None,
                    })
                    .or_else(|| (symbol.kind() == EditorSymbolKind::Operator).then_some(true));
                Some(formal)
            })
            .collect()
    });
    let mut candidate = symbol.clone();
    candidate.detail = Some(cursor::clean(&text[..end]));
    candidate.callable = callable;
    candidate.children = parameters.unwrap_or_default();
    Some(candidate)
}

pub(super) fn signature<'a>(document: &'a Document, name: &str) -> Option<&'a [SignatureItem]> {
    document
        .components()
        .iter()
        .find(|c| c.name() == name)
        .map(|c| c.signature())
        .or_else(|| {
            document
                .models()
                .iter()
                .find(|m| m.name() == name)
                .map(|m| m.signature())
        })
}

pub(super) fn exported(snapshot: &EditorSnapshot, symbol: &EditorSymbol) -> bool {
    cursor::tokens(&snapshot.source[symbol.range().start() as usize..symbol.range().end() as usize])
        .first()
        .is_some_and(|t| t.text() == "public")
}

pub(super) fn simple(name: &str, detail: String, kind: EditorSymbolKind) -> EditorSymbol {
    let mut symbol = EditorSymbol::leaf(kind, name, eqiora_lang::TextRange::default());
    symbol.detail = Some(detail);
    symbol
}

pub(super) fn target(document: &Document, symbol: &EditorSymbol) -> Option<String> {
    use eqiora_lang::{ComponentItem as C, Item as I, PortSyntax};
    let mut instances = Vec::new();
    let mut ports = Vec::new();
    for model in document.models() {
        instances.extend(model.items().iter().filter_map(|i| {
            if let I::Instance(i) = i {
                Some(i)
            } else {
                None
            }
        }));
        ports.extend(model.items().iter().filter_map(|i| {
            if let I::Port(p) = i {
                Some((p.range(), p.syntax()))
            } else {
                None
            }
        }));
        ports.extend(model.signature().iter().filter_map(|i| {
            if let SignatureItem::Port(p) = i {
                Some((p.range(), p.syntax()))
            } else {
                None
            }
        }));
    }
    for component in document.components() {
        instances.extend(component.items().iter().filter_map(|i| {
            if let C::Instance(i) = i {
                Some(i)
            } else {
                None
            }
        }));
        ports.extend(component.items().iter().filter_map(|i| {
            if let C::Port(p) = i {
                Some((p.range(), p.syntax()))
            } else {
                None
            }
        }));
        ports.extend(component.signature().iter().filter_map(|i| {
            if let SignatureItem::Port(p) = i {
                Some((p.range(), p.syntax()))
            } else {
                None
            }
        }));
    }
    if let Some(instance) = instances.into_iter().find(|i| i.range() == symbol.range()) {
        return Some(instance.definition().as_str().into());
    }
    ports
        .into_iter()
        .find(|(range, _)| *range == symbol.range())
        .and_then(|(_, syntax)| match syntax {
            PortSyntax::ScalarPhysicalConnector { connector }
            | PortSyntax::FieldPhysical { connector, .. } => Some(connector.as_str().into()),
            _ => None,
        })
}

pub(super) fn members(snapshot: &EditorSnapshot, symbol: &EditorSymbol) -> Vec<EditorSymbol> {
    let Some(document) = snapshot.syntax.as_ref() else {
        return Vec::new();
    };
    if let Some(signature) = signature(document, symbol.name()) {
        return symbol
            .children()
            .iter()
            .filter(|s| {
                signature.iter().any(|item| {
                    matches!(
                        item,
                        SignatureItem::Input(_)
                            | SignatureItem::Output(_)
                            | SignatureItem::Port(_)
                            | SignatureItem::PortFamily(_)
                    ) && item.range() == s.range()
                })
            })
            .filter_map(|s| candidate(snapshot, s))
            .collect();
    }
    if let Some(connector) = document
        .connectors()
        .iter()
        .find(|c| c.range() == symbol.range())
    {
        use eqiora_lang::ConnectorSyntax;
        let names = match connector.syntax() {
            ConnectorSyntax::ScalarPhysical {
                across_name,
                through_name,
                across_type,
                through_type,
            } => vec![
                ("across", across_name.as_str(), across_type.range()),
                ("through", through_name.as_str(), through_type.range()),
            ],
            ConnectorSyntax::FieldPhysical { trace, flux, .. } => vec![
                ("trace", trace.name(), trace.dimension().range()),
                ("flux", flux.name(), flux.dimension().range()),
            ],
            _ => Vec::new(),
        };
        return names
            .into_iter()
            .map(|(role, name, range)| {
                simple(
                    name,
                    format!(
                        "{role} {name}: {}",
                        cursor::clean(
                            &snapshot.source[range.start() as usize..range.end() as usize]
                        )
                    ),
                    EditorSymbolKind::Field,
                )
            })
            .collect();
    }
    symbol
        .children()
        .iter()
        .filter_map(|s| candidate(snapshot, s))
        .collect()
}
