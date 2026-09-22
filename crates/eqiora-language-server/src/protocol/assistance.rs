//! Documentation shared by hover, completion and signature help.
use eqiora::api::EditorSymbol;
use eqiora::api::EditorSymbolKind;
use eqiora::language::{Notation, NotationLabel, NotationProfile};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse, CompletionTextEdit,
    Documentation, Hover, HoverContents, MarkupContent, MarkupKind, ParameterInformation,
    ParameterLabel, SignatureHelp, SignatureHelpParams, SignatureInformation, TextEdit, Uri,
};

use super::{ServerState, document};
use crate::lsp_projection::{editor_position, source_range};

#[cfg(test)]
mod tests;

struct Parameter {
    name: String,
    label: String,
    documentation: Option<String>,
}

pub(super) fn append_notation(value: &mut String, notation: Option<&Notation>) {
    if let Some(notation) = notation {
        let label = NotationLabel::from_notation(notation).render(NotationProfile::Plain);
        // The admitted plain alphabet has no code-span delimiter or executable markup.
        value.push_str(&format!("\n\nNotation: `{label}`"));
    }
}

struct Entry {
    name: String,
    label: String,
    documentation: Option<String>,
    kind: CompletionItemKind,
    notation: Option<Notation>,
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
        append_notation(&mut value, self.notation.as_ref());
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

fn authored(candidate: EditorSymbol) -> Entry {
    Entry {
        name: candidate.name().into(),
        label: candidate.detail().unwrap_or(candidate.name()).into(),
        documentation: candidate.documentation(),
        notation: candidate.notation().cloned(),
        kind: match candidate.kind() {
            EditorSymbolKind::Operator | EditorSymbolKind::Component | EditorSymbolKind::Model => {
                CompletionItemKind::FUNCTION
            }
            EditorSymbolKind::Import => CompletionItemKind::MODULE,
            EditorSymbolKind::Keyword => CompletionItemKind::KEYWORD,
            EditorSymbolKind::Let => CompletionItemKind::CONSTANT,
            EditorSymbolKind::Enum => CompletionItemKind::ENUM,
            EditorSymbolKind::EnumMember => CompletionItemKind::ENUM_MEMBER,
            EditorSymbolKind::Record => CompletionItemKind::STRUCT,
            EditorSymbolKind::Dimension
            | EditorSymbolKind::FiniteSpace
            | EditorSymbolKind::ValueType => CompletionItemKind::CLASS,
            _ => CompletionItemKind::VARIABLE,
        },
        parameters: candidate.parameters().map(|ps| {
            ps.iter()
                .map(|p| Parameter {
                    name: p.name().into(),
                    label: p.detail().unwrap_or(p.name()).into(),
                    documentation: p.documentation(),
                })
                .collect()
        }),
    }
}

fn entry(state: &ServerState, uri: &Uri, offset: u32, name: &str) -> Option<Entry> {
    let open = document(state, uri).ok()?;
    // An available workspace owns both positive and negative scope results.
    // Retrying its rejection through a lexical snapshot would resurrect a
    // declaration at a keyword, unit or other unsupported cursor position.
    let candidate = if let Some((workspace, file)) = state.resolved(uri) {
        workspace.assistance(file, offset, name)
    } else {
        open.snapshot().assistance(offset, name)
    };
    candidate.map(authored)
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
    let Some((name, range)) = open.snapshot().name_at(offset) else {
        return Ok(None);
    };
    let Some(entry) = entry(state, uri, offset, &name) else {
        return Ok(None);
    };
    let mut hover = entry.hover();
    hover.range = Some(source_range(
        open.snapshot(),
        range.start() as usize,
        range.end() as usize,
    )?);
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
    let completion = state
        .resolved(uri)
        .and_then(|(w, file)| w.completion(file, offset))
        .or_else(|| open.snapshot().completion(offset));
    let Some((range, candidates)) = completion else {
        return Ok(empty());
    };
    let range = source_range(
        open.snapshot(),
        range.start() as usize,
        range.end() as usize,
    )?;
    let items = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let insertion = candidate.insert_text().to_owned();
            let required = candidate.required();
            completion_item(authored(candidate), insertion, required, index, range)
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
    let Some((name, argument, named)) = open.snapshot().call_at(offset) else {
        return Ok(None);
    };
    let entry = entry(state, uri, offset, &name);
    Ok(entry.and_then(|entry| entry.signature(argument, named.as_deref())))
}

fn completion_item(
    e: Entry,
    insertion: String,
    required: Option<bool>,
    index: usize,
    range: lsp_types::Range,
) -> CompletionItem {
    let status = required.map(|r| if r { "required" } else { "defaulted" });
    CompletionItem {
        label: e.name.clone(),
        kind: Some(e.kind),
        detail: Some(status.map_or(e.label.clone(), |s| format!("{} ({s})", e.label))),
        sort_text: Some(format!("{index:010}")),
        documentation: e
            .documentation
            .map(|doc| Documentation::MarkupContent(markdown(doc))),
        text_edit: Some(CompletionTextEdit::Edit(TextEdit {
            range,
            new_text: insertion,
        })),
        ..Default::default()
    }
}
