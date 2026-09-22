//! Read-only rich editor projection. All mathematics comes from the compiler.

use eqiora::{api::MathRendering, kernel::KernelNode, language::NotationProfile};
use lsp_types::TextDocumentIdentifier;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ServerState, document, source_range};

mod plan;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct InspectParams {
    text_document: TextDocumentIdentifier,
    model: Option<String>,
    #[serde(default)]
    fingerprint: bool,
    plan: Option<String>,
}

pub(super) fn inspect(params: InspectParams, state: &ServerState) -> Result<Value, String> {
    if params
        .plan
        .as_ref()
        .is_some_and(|plan| plan.len() > 2 * 1024 * 1024)
    {
        return Err("Plan exceeds the 2 MiB editor admission limit".to_owned());
    }
    let uri = &params.text_document.uri;
    let open = document(state, uri)?;
    let (workspace, file) = state
        .resolved(uri)
        .ok_or("resolved workspace is unavailable")?;
    let snapshot = workspace.document(file).ok_or("source is unavailable")?;
    let models = snapshot
        .symbols()
        .iter()
        .filter(|symbol| symbol.kind() == eqiora::api::EditorSymbolKind::Model)
        .map(|symbol| symbol.name())
        .collect::<Vec<_>>();
    let mut result = json!({"version": open.version, "models": models, "model": null,
        "nodes": [], "edges": [], "equations": [], "fingerprint": null, "plan": null, "errors": []});
    let Some(selected) = params.model.as_deref().or_else(|| models.first().copied()) else {
        return Ok(result);
    };
    if !models.contains(&selected) {
        return Err("select a Model declared in the current document".to_owned());
    }
    result["model"] = json!(selected);
    let compiled = match workspace.compile_model(file, selected) {
        Ok(compiled) => compiled,
        Err(errors) => {
            result["errors"] = json!(
                errors
                    .iter()
                    .map(|error| error.message())
                    .collect::<Vec<_>>()
            );
            return Ok(result);
        }
    };
    if compiled.program().nodes().len() > 4096 || compiled.program().edges().len() > 16384 {
        return Err(
            "model exceeds the rich editor view limit (4096 nodes / 16384 edges)".to_owned(),
        );
    }
    if let Some(bytes) = params.plan {
        result["plan"] = plan::project(bytes.as_bytes(), &compiled)?;
    }
    let mut nodes = Vec::new();
    let mut equations = Vec::new();
    let mut errors = Vec::new();
    for node in compiled.program().nodes() {
        let id = node.id();
        let names = compiled
            .aliases()
            .iter()
            .filter(|(_, target)| **target == id)
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        let locations = compiled
            .notation()
            .iter()
            .filter(|entry| entry.graph_id() == Some(id))
            .filter_map(|entry| {
                let span = entry.definition_span()?;
                let target = workspace.document(&span.file)?;
                let target_uri = state.uri_for_file(uri, &span.file)?;
                let range = source_range(target, span.start as usize, span.end as usize).ok()?;
                Some(json!({"uri": target_uri, "range": range}))
            })
            .collect::<Vec<_>>();
        let value_type = match node {
            KernelNode::Field(field) => Some(field.value_type()),
            KernelNode::Parameter(parameter) => Some(parameter.value_type()),
            KernelNode::Port(port) => port.signal_contract().map(|(_, value)| value),
            _ => None,
        };
        let type_text = value_type
            .and_then(|value| MathRendering::value_type(value, NotationProfile::Plain).ok())
            .map(|rendered| rendered.plain().to_owned());
        nodes.push(
            json!({"id": id.to_string(), "kind": format!("{:?}", node.kind()),
            "names": names, "locations": locations, "valueType": type_text,
            "boundary": compiled.program().boundary().contains(&id),
            "details": format!("{node:?}")}),
        );
        if matches!(node, KernelNode::Relation(_)) {
            match compiled.render_equations(id, NotationProfile::Latex) {
                Ok(renderings) => {
                    for (index, rendered) in renderings.iter().enumerate() {
                        equations.push(json!({"relation": id.to_string(), "index": index,
                            "latex": rendered.text(), "plain": rendered.plain(),
                            "speech": rendered.speech(), "fallback": rendered.used_fallback(),
                            "references": rendered.references().iter().filter_map(|reference| reference.graph_id().map(|target| target.to_string())).collect::<Vec<_>>()}));
                    }
                }
                Err(error) => errors.push(error.message().to_owned()),
            }
        }
    }
    result["nodes"] = json!(nodes);
    result["equations"] = json!(equations);
    result["edges"] = json!(compiled.program().edges().iter().map(|edge| json!({
        "from": edge.from().to_string(), "to": edge.to().to_string(), "kind": format!("{:?}", edge.kind())
    })).collect::<Vec<_>>());
    if params.fingerprint {
        match compiled.structural_fingerprint() {
            Ok(fingerprint) => result["fingerprint"] = json!(fingerprint.to_string()),
            Err(error) => errors.push(error.message().to_owned()),
        }
    }
    result["errors"] = json!(errors);
    if serde_json::to_vec(&result)
        .map_err(|error| error.to_string())?
        .len()
        > 4 * 1024 * 1024
    {
        return Err("model exceeds the 4 MiB rich editor response limit".to_owned());
    }
    Ok(result)
}
