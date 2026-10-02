use super::ServerState;
use crate::lsp_projection::{source_range, symbol_kind, symbol_label, symbol_range};
use eqiora::api::{EditorSnapshot, EditorSymbol, EditorSymbolKind};
use lsp_types::{
    DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, InitializeParams, Location,
    SymbolInformation, SymbolKind, Uri,
};

#[derive(Default)]
pub(super) struct Options {
    hierarchical: bool,
    extended_kinds: bool,
}

impl Options {
    pub(super) fn from_initialize(params: &InitializeParams) -> Self {
        let capabilities = params
            .capabilities
            .text_document
            .as_ref()
            .and_then(|text| text.document_symbol.as_ref());
        Self {
            hierarchical: capabilities
                .and_then(|caps| caps.hierarchical_document_symbol_support)
                .unwrap_or(false),
            // The presence of valueSet guarantees graceful unknown-kind handling,
            // including an empty set. Without it only File through Array are safe.
            extended_kinds: capabilities
                .and_then(|caps| caps.symbol_kind.as_ref())
                .and_then(|kinds| kinds.value_set.as_ref())
                .is_some(),
        }
    }

    fn kind(&self, kind: EditorSymbolKind) -> SymbolKind {
        let kind = symbol_kind(kind);
        if self.extended_kinds {
            return kind;
        }
        match kind {
            SymbolKind::TYPE_PARAMETER | SymbolKind::STRUCT | SymbolKind::OBJECT => {
                SymbolKind::CLASS
            }
            SymbolKind::ENUM_MEMBER => SymbolKind::CONSTANT,
            SymbolKind::EVENT | SymbolKind::OPERATOR => SymbolKind::FUNCTION,
            _ => kind,
        }
    }
}

pub(super) fn document_symbols(
    params: DocumentSymbolParams,
    state: &ServerState,
) -> Result<DocumentSymbolResponse, String> {
    let snapshot = state.snapshot(&params.text_document.uri)?;
    if state.symbol_options.hierarchical {
        let symbols = snapshot
            .symbols()
            .iter()
            .map(|symbol| lsp_symbol(snapshot, symbol, &state.symbol_options))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DocumentSymbolResponse::Nested(symbols))
    } else {
        let mut symbols = Vec::new();
        flat_symbols(
            snapshot,
            snapshot.symbols(),
            &params.text_document.uri,
            None,
            &state.symbol_options,
            &mut symbols,
        )?;
        Ok(DocumentSymbolResponse::Flat(symbols))
    }
}

#[allow(deprecated)]
fn flat_symbols(
    snapshot: &EditorSnapshot,
    symbols: &[EditorSymbol],
    uri: &Uri,
    container: Option<&str>,
    options: &Options,
    output: &mut Vec<SymbolInformation>,
) -> Result<(), String> {
    for symbol in symbols {
        output.push(SymbolInformation {
            name: symbol.name().to_owned(),
            kind: options.kind(symbol.kind()),
            tags: None,
            deprecated: None,
            location: Location {
                uri: uri.clone(),
                range: symbol_range(snapshot, symbol)?,
            },
            container_name: container.map(str::to_owned),
        });
        flat_symbols(
            snapshot,
            symbol.children(),
            uri,
            Some(symbol.name()),
            options,
            output,
        )?;
    }
    Ok(())
}

#[allow(deprecated)]
fn lsp_symbol(
    snapshot: &EditorSnapshot,
    symbol: &EditorSymbol,
    options: &Options,
) -> Result<DocumentSymbol, String> {
    let range = symbol_range(snapshot, symbol)?;
    let children = symbol
        .children()
        .iter()
        .map(|child| lsp_symbol(snapshot, child, options))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DocumentSymbol {
        name: symbol.name().to_owned(),
        detail: Some(
            symbol
                .detail()
                .unwrap_or_else(|| symbol_label(symbol.kind()))
                .to_owned(),
        ),
        kind: options.kind(symbol.kind()),
        tags: None,
        deprecated: None,
        range,
        selection_range: match symbol.name_range() {
            Some(name) => source_range(snapshot, name.start() as usize, name.end() as usize)?,
            None => range,
        },
        children: (!children.is_empty()).then_some(children),
    })
}
