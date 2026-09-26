use eqiora::api::{EditorDefinition, EditorSymbolKind, EditorWorkspaceSnapshot};
use eqiora::language::Notation;
use lsp_types::{
    DocumentHighlight, DocumentHighlightKind, DocumentHighlightParams, GotoDefinitionParams,
    GotoDefinitionResponse, Hover, HoverContents, HoverParams, Location, MarkupContent, MarkupKind,
    ReferenceContext, ReferenceParams, Uri,
};

use super::{ServerState, document};
use crate::lsp_projection::{editor_position, source_range, symbol_label};

pub(super) fn definition(
    params: GotoDefinitionParams,
    state: &ServerState,
) -> Result<Option<GotoDefinitionResponse>, String> {
    let uri = &params.text_document_position_params.text_document.uri;
    document(state, uri)?;
    let Some((workspace, file)) = state.resolved(uri) else {
        return Ok(None);
    };
    let position = editor_position(params.text_document_position_params.position);
    if let Some(span) = workspace.value_definition_at_position(file, position) {
        let target_uri = state
            .uri_for_file(uri, &span.file)
            .ok_or("resolved definition URI is unavailable")?;
        let snapshot = workspace
            .document(&span.file)
            .ok_or("source is unavailable")?;
        return Ok(Some(GotoDefinitionResponse::Scalar(Location::new(
            target_uri,
            source_range(snapshot, span.start as usize, span.end as usize)?,
        ))));
    }
    let Some(definition) = workspace.definition_for_reference_at_position(file, position) else {
        return Ok(None);
    };
    Ok(Some(GotoDefinitionResponse::Scalar(definition_location(
        state, workspace, uri, definition,
    )?)))
}

pub(super) fn references(
    params: ReferenceParams,
    state: &ServerState,
) -> Result<Vec<Location>, String> {
    let uri = &params.text_document_position.text_document.uri;
    document(state, uri)?;
    let Some((workspace, file)) = state.resolved(uri) else {
        return Ok(Vec::new());
    };
    let position = editor_position(params.text_document_position.position);
    if let Some(ranges) =
        workspace.value_references_at_position(file, position, params.context.include_declaration)
    {
        return ranges
            .into_iter()
            .map(|span| {
                let reference_uri = state
                    .uri_for_file(uri, &span.file)
                    .ok_or("resolved reference URI is unavailable")?;
                let snapshot = workspace
                    .document(&span.file)
                    .ok_or("source is unavailable")?;
                Ok(Location::new(
                    reference_uri,
                    source_range(snapshot, span.start as usize, span.end as usize)?,
                ))
            })
            .collect();
    }
    let Some((target, _source)) = workspace.hover_at_position(file, position) else {
        return Ok(Vec::new());
    };

    let mut locations = Vec::new();
    if params.context.include_declaration {
        locations.push(definition_location(state, workspace, uri, target)?);
    }
    for reference in workspace.references() {
        if same_definition(reference.definition(), target) {
            let reference_uri = state
                .uri_for_file(uri, reference.file())
                .ok_or_else(|| "resolved reference URI is unavailable".to_owned())?;
            let snapshot = workspace
                .document(reference.file())
                .ok_or_else(|| "resolved reference document is unavailable".to_owned())?;
            locations.push(Location::new(
                reference_uri,
                source_range(
                    snapshot,
                    usize::try_from(reference.range().start())
                        .map_err(|_| "reference offset exceeds usize")?,
                    usize::try_from(reference.range().end())
                        .map_err(|_| "reference offset exceeds usize")?,
                )?,
            ));
        }
    }
    Ok(locations)
}

pub(super) fn document_highlights(
    params: DocumentHighlightParams,
    state: &ServerState,
) -> Result<Vec<DocumentHighlight>, String> {
    let uri = params
        .text_document_position_params
        .text_document
        .uri
        .clone();
    let mut highlights = references(
        ReferenceParams {
            text_document_position: params.text_document_position_params,
            context: ReferenceContext {
                include_declaration: true,
            },
            work_done_progress_params: params.work_done_progress_params,
            partial_result_params: params.partial_result_params,
        },
        state,
    )?
    .into_iter()
    .filter(|location| location.uri == uri)
    .map(|location| DocumentHighlight {
        range: location.range,
        // An Eqiora relation does not imply an assignment or a read/write role.
        kind: Some(DocumentHighlightKind::TEXT),
    })
    .collect::<Vec<_>>();
    highlights.sort_by_key(|highlight| (highlight.range.start, highlight.range.end));
    highlights.dedup_by_key(|highlight| highlight.range);
    Ok(highlights)
}

fn same_definition(left: &EditorDefinition, right: &EditorDefinition) -> bool {
    left.namespace() == right.namespace()
        && left.path() == right.path()
        && left.kind() == right.kind()
}

fn definition_location(
    state: &ServerState,
    workspace: &EditorWorkspaceSnapshot,
    source_uri: &Uri,
    definition: &EditorDefinition,
) -> Result<Location, String> {
    let uri = state
        .uri_for_file(source_uri, definition.file())
        .ok_or_else(|| "resolved definition URI is unavailable".to_owned())?;
    let snapshot = workspace
        .document(definition.file())
        .ok_or_else(|| "resolved definition document is unavailable".to_owned())?;
    let range = definition.name_range().unwrap_or(definition.range());
    Ok(Location::new(
        uri,
        source_range(
            snapshot,
            usize::try_from(range.start()).map_err(|_| "definition offset exceeds usize")?,
            usize::try_from(range.end()).map_err(|_| "definition offset exceeds usize")?,
        )?,
    ))
}

pub(super) fn hover(params: HoverParams, state: &ServerState) -> Result<Option<Hover>, String> {
    let uri = &params.text_document_position_params.text_document.uri;
    document(state, uri)?;
    let Some((workspace, file)) = state.resolved(uri) else {
        return super::assistance::hover(uri, params.text_document_position_params.position, state);
    };
    let position = editor_position(params.text_document_position_params.position);
    let Some((definition, source)) = workspace.hover_at_position(file, position) else {
        return super::assistance::hover(uri, params.text_document_position_params.position, state);
    };
    let documentation = definition.doc_comment().map(|doc| doc.markdown());
    Ok(Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown_hover(
                definition.kind(),
                definition.path(),
                source,
                documentation.as_deref(),
                definition.notation(),
                definition.namespace(),
                definition.file(),
            ),
        }),
        range: None,
    }))
}

fn markdown_hover(
    kind: EditorSymbolKind,
    path: &str,
    source: &str,
    documentation: Option<&str>,
    notation: Option<&Notation>,
    namespace: &[String],
    file: &str,
) -> String {
    let literal = format!("{source}\n// Origin namespace: {namespace:?}\n// Source file: {file:?}");
    let longest_run = literal
        .split(|character| character != '`')
        .map(str::len)
        .max()
        .unwrap_or_default();
    let fence = "`".repeat(longest_run.saturating_add(1).max(3));
    let mut detail = format!(
        "**{}** `{path}`\n\n{fence}eqiora\n{literal}\n{fence}",
        symbol_label(kind)
    );
    super::assistance::append_notation(&mut detail, notation);
    match documentation {
        Some(prose) => format!("{prose}\n\n{detail}"),
        None => detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_keeps_sanitized_prose_outside_a_source_derived_safe_fence() {
        let source = "public component C() { // ``` hostile fence\n}";
        let prose = "Summary&#46;\n\n\\[run\\](command&#58;delete)\n\\<script\\>";
        let rendered = markdown_hover(
            EditorSymbolKind::Component,
            "C",
            source,
            Some(prose),
            None,
            &["local".into()],
            "src/main.eqi",
        );
        assert!(rendered.starts_with(prose));
        assert!(rendered.contains("\n````eqiora\npublic component C"));
        assert!(rendered.ends_with("\n````"));
        assert!(!rendered.contains("[run](command:"));
    }
    #[test]
    fn hover_origin_preserves_opaque_segment_boundaries_inside_the_safe_fence() {
        let namespace = vec![
            "pkg".into(),
            "a::b".into(),
            "opaque````\n[run](command:run)".into(),
        ];
        let rendered = markdown_hover(
            EditorSymbolKind::Component,
            "main.Part",
            "public component Part(){}",
            None,
            None,
            &namespace,
            "src/quoted\"file.eqi",
        );
        assert!(rendered.contains("\n`````eqiora\n"));
        assert!(rendered.ends_with("\n`````"));
        assert!(rendered.contains("[\"pkg\", \"a::b\", \"opaque````\\n[run](command:run)\"]"));
        assert!(rendered.contains("Source file: \"src/quoted\\\"file.eqi\""));
        assert!(!rendered.contains("\n[run]"));
        let split = markdown_hover(
            EditorSymbolKind::Component,
            "main.Part",
            "public component Part(){}",
            None,
            None,
            &["pkg".into(), "a".into(), "b".into()],
            "src/main.eqi",
        );
        assert!(split.contains("[\"pkg\", \"a\", \"b\"]"));
        assert_ne!(rendered, split);
    }
}
