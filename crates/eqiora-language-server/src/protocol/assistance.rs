//! Documentation shared by hover, completion and signature help.
use std::collections::BTreeMap;

use eqiora::api::{EditorService, EditorSymbol, EditorSymbolKind};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse, CompletionTextEdit,
    Documentation, Hover, HoverContents, MarkupContent, MarkupKind, ParameterInformation,
    ParameterLabel, SignatureHelp, SignatureHelpParams, SignatureInformation, TextEdit, Uri,
};

use super::{ServerState, document};
use crate::lsp_projection::{editor_position, source_range};

mod builtins;
mod syntax;
#[cfg(test)]
mod tests;

struct Parameter {
    name: String,
    label: String,
    documentation: Option<String>,
}

struct Entry {
    name: String,
    label: String,
    documentation: Option<String>,
    kind: CompletionItemKind,
    parameters: Option<Vec<Parameter>>,
}

fn markdown(value: String) -> MarkupContent {
    MarkupContent {
        kind: MarkupKind::Markdown,
        value,
    }
}

impl Entry {
    fn hover(&self) -> Hover {
        let fence = "`".repeat(
            self.label
                .split(|c| c != '`')
                .map(str::len)
                .max()
                .unwrap_or(0)
                .saturating_add(1)
                .max(3),
        );
        let mut value = format!("{fence}eqiora\n{}\n{fence}", self.label);
        if let Some(doc) = &self.documentation {
            value.push_str("\n\n");
            value.push_str(doc);
        }
        Hover {
            contents: HoverContents::Markup(markdown(value)),
            range: None,
        }
    }

    fn signature(self, argument: usize, named: Option<&str>) -> Option<SignatureHelp> {
        let parameters = self.parameters?;
        let active = named
            .and_then(|name| parameters.iter().position(|p| p.name == name))
            .unwrap_or(argument);
        let active = (!parameters.is_empty())
            .then(|| u32::try_from(active.min(parameters.len() - 1)).unwrap_or(0));
        let mut cursor = self.label.find('(').map_or(0, |index| index + 1);
        let parameters = parameters
            .into_iter()
            .map(|p| {
                let label = self.label[cursor..].find(&p.label).map_or_else(
                    || ParameterLabel::Simple(p.label.clone()),
                    |relative| {
                        let start = cursor + relative;
                        cursor = start + p.label.len();
                        ParameterLabel::LabelOffsets([
                            self.label[..start].encode_utf16().count() as u32,
                            self.label[..cursor].encode_utf16().count() as u32,
                        ])
                    },
                );
                ParameterInformation {
                    label,
                    documentation: p
                        .documentation
                        .map(|s| Documentation::MarkupContent(markdown(s))),
                }
            })
            .collect();
        Some(SignatureHelp {
            signatures: vec![SignatureInformation {
                label: self.label,
                documentation: self
                    .documentation
                    .map(|s| Documentation::MarkupContent(markdown(s))),
                parameters: Some(parameters),
                active_parameter: active,
            }],
            active_signature: Some(0),
            active_parameter: active,
        })
    }
}

fn symbol_entry(symbol: &EditorSymbol, source: &str, origin: u32) -> Option<Entry> {
    let start = symbol.range().start().checked_sub(origin)? as usize;
    let end = symbol.range().end().checked_sub(origin)? as usize;
    let text = source.get(start..end)?;
    let callable = matches!(
        symbol.kind(),
        EditorSymbolKind::Operator | EditorSymbolKind::Component | EditorSymbolKind::Model
    );
    let head_end = syntax::head_end(text, callable);
    let parameters = callable.then(|| {
        symbol
            .children()
            .iter()
            .filter(|p| (p.range().start() - symbol.range().start()) < head_end as u32)
            .filter_map(|p| {
                let label = syntax::clean(source.get(
                    (p.range().start() - origin) as usize..(p.range().end() - origin) as usize,
                )?);
                Some(Parameter {
                    name: p.name().into(),
                    label,
                    documentation: p.doc_comment().map(|doc| doc.markdown()),
                })
            })
            .collect()
    });
    Some(Entry {
        name: symbol.name().into(),
        label: syntax::clean(&text[..head_end]),
        documentation: symbol.doc_comment().map(|doc| doc.markdown()),
        kind: if callable {
            CompletionItemKind::FUNCTION
        } else {
            CompletionItemKind::VARIABLE
        },
        parameters,
    })
}

fn local_entries(
    symbols: &[EditorSymbol],
    source: &str,
    offset: u32,
    entries: &mut BTreeMap<String, Entry>,
) {
    for symbol in symbols {
        if let Some(entry) = symbol_entry(symbol, source, 0) {
            entries.insert(entry.name.clone(), entry);
        }
    }
    // Only descend into the cursor's lexical owner; sibling members are private.
    for symbol in symbols
        .iter()
        .filter(|s| s.range().start() <= offset && offset < s.range().end())
    {
        local_entries(symbol.children(), source, offset, entries);
    }
}

fn entries(state: &ServerState, uri: &Uri, offset: u32) -> Result<BTreeMap<String, Entry>, String> {
    let open = document(state, uri)?;
    let mut entries: BTreeMap<_, _> = builtins::entries()
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();
    local_entries(
        open.snapshot().symbols(),
        &open.source,
        offset,
        &mut entries,
    );
    Ok(entries)
}

fn resolved_entry(state: &ServerState, uri: &Uri, offset: u32) -> Option<Entry> {
    let (workspace, file) = state.resolved(uri)?;
    let (definition, source) = workspace.hover(file, offset)?;
    let service = EditorService::new("signature", 0, source.to_owned());
    let symbol = service.current().symbols().first()?;
    let mut entry = symbol_entry(symbol, source, 0)?;
    entry.documentation = definition.doc_comment().map(|doc| doc.markdown());
    Some(entry)
}

pub(super) fn hover(
    uri: &Uri,
    position: lsp_types::Position,
    state: &ServerState,
) -> Result<Option<Hover>, String> {
    let open = document(state, uri)?;
    let Some(offset) = open.snapshot().byte_offset(editor_position(position)) else {
        return Ok(None);
    };
    let Some((name, start, end)) = syntax::name_at(&open.source, offset, false) else {
        return Ok(None);
    };
    let Some(entry) = entries(state, uri, offset)?.remove(&name) else {
        return Ok(None);
    };
    let mut hover = entry.hover();
    hover.range = Some(source_range(open.snapshot(), start as usize, end as usize)?);
    Ok(Some(hover))
}

pub(super) fn completion(
    params: CompletionParams,
    state: &ServerState,
) -> Result<CompletionResponse, String> {
    let uri = &params.text_document_position.text_document.uri;
    let open = document(state, uri)?;
    let empty = || CompletionResponse::Array(Vec::new());
    let Some(offset) = open
        .snapshot()
        .byte_offset(editor_position(params.text_document_position.position))
    else {
        return Ok(empty());
    };
    let Some((prefix, start, end)) = syntax::name_at(&open.source, offset, true) else {
        return Ok(empty());
    };
    let range = source_range(open.snapshot(), start as usize, end as usize)?;
    let items = entries(state, uri, offset)?
        .into_values()
        .filter(|e| e.name.starts_with(&prefix))
        .map(|e| CompletionItem {
            label: e.name.clone(),
            kind: Some(e.kind),
            detail: Some(e.label),
            documentation: e
                .documentation
                .map(|doc| Documentation::MarkupContent(markdown(doc))),
            text_edit: Some(CompletionTextEdit::Edit(TextEdit {
                range,
                new_text: e.name,
            })),
            ..Default::default()
        })
        .collect();
    Ok(CompletionResponse::Array(items))
}

pub(super) fn signature_help(
    params: SignatureHelpParams,
    state: &ServerState,
) -> Result<Option<SignatureHelp>, String> {
    let uri = &params.text_document_position_params.text_document.uri;
    let open = document(state, uri)?;
    let Some(offset) = open.snapshot().byte_offset(editor_position(
        params.text_document_position_params.position,
    )) else {
        return Ok(None);
    };
    let Some(call) = syntax::call_at(&open.source, offset) else {
        return Ok(None);
    };
    let entry = resolved_entry(state, uri, call.start)
        .or_else(|| entries(state, uri, offset).ok()?.remove(&call.name));
    Ok(entry.and_then(|entry| entry.signature(call.argument, call.named.as_deref())))
}
