//! Studio transport for the shared compile/check control contract.
//!
//! Canonical compilation is owned by `eqiora::api::ModelDocument::compile`;
//! `eqiora::control` adapts that operation to control-v2. This module only
//! decodes the closed control request, projects an accepted immutable Model
//! for Studio, and admits that Model to the bounded local cache.

use eqiora::control::{
    CompileOutcomeV2, CompileRequestV2, CompileResponseV2, ControlDiagnosticSourceV2,
    ControlDiagnosticV2, ControlSeverityV2, execute_compile_v2,
};
use serde::Serialize;
use tauri::State;

use super::{
    AppState, DiagnosticDto, DocumentProjection, PROTOCOL, SourceSpanDto, project_document,
};

/// Native adapter response. The shared control response remains intact; the
/// Studio projection is an explicitly separate, read-only adapter result.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompileCommandEnvelope {
    protocol: &'static str,
    control: Option<CompileResponseV2>,
    projection: Option<DocumentProjection>,
    diagnostics: Vec<DiagnosticDto>,
}

impl CompileCommandEnvelope {
    fn control_rejection(diagnostic: ControlDiagnosticV2) -> Self {
        Self {
            protocol: PROTOCOL,
            control: None,
            projection: None,
            diagnostics: vec![control_diagnostic(diagnostic)],
        }
    }

    fn adapter_rejection(control: CompileResponseV2, diagnostic: DiagnosticDto) -> Self {
        Self {
            protocol: PROTOCOL,
            control: Some(control),
            projection: None,
            diagnostics: vec![diagnostic],
        }
    }
}

#[tauri::command]
pub(crate) fn compile_model(
    request_json: String,
    state: State<'_, AppState>,
) -> CompileCommandEnvelope {
    compile_request(&request_json, state.inner())
}

fn compile_request(request_json: &str, state: &AppState) -> CompileCommandEnvelope {
    // Tauri deliberately admits encoded JSON here. The shared decoder owns
    // encoded-size and unknown-field checks before allocating source content
    // or entering compilation.
    let request = match CompileRequestV2::from_json(request_json.as_bytes()) {
        Ok(request) => request,
        Err(diagnostic) => return CompileCommandEnvelope::control_rejection(diagnostic),
    };

    let (control, document) = execute_compile_v2(&request).into_parts();
    let Some(document) = document else {
        debug_assert!(matches!(
            control.outcome(),
            CompileOutcomeV2::Rejected { .. }
        ));
        return CompileCommandEnvelope {
            protocol: PROTOCOL,
            control: Some(control),
            projection: None,
            diagnostics: Vec::new(),
        };
    };

    let digest = match control.outcome() {
        CompileOutcomeV2::Accepted { model } => model.digest().to_owned(),
        CompileOutcomeV2::Rejected { .. } => {
            unreachable!("a rejected control execution cannot expose a Model document")
        }
    };
    let projection = match project_document(&document, digest.clone()) {
        Ok(projection) => projection,
        Err(diagnostic) => return CompileCommandEnvelope::adapter_rejection(control, *diagnostic),
    };

    // Cache admission is last. Invalid requests, compiler rejection, and
    // projection rejection leave the prior immutable lineage untouched.
    match state.documents.lock() {
        Ok(mut documents) => documents.reset(digest, document),
        Err(_) => {
            return CompileCommandEnvelope::adapter_rejection(
                control,
                super::studio_error("ST0001", "native document cache is unavailable"),
            );
        }
    }
    CompileCommandEnvelope {
        protocol: PROTOCOL,
        control: Some(control),
        projection: Some(projection),
        diagnostics: Vec::new(),
    }
}

fn control_diagnostic(diagnostic: ControlDiagnosticV2) -> DiagnosticDto {
    DiagnosticDto {
        source: match diagnostic.source() {
            ControlDiagnosticSourceV2::Control => "control",
            ControlDiagnosticSourceV2::Kernel => "kernel",
        },
        severity: match diagnostic.severity() {
            ControlSeverityV2::Error => "error",
            ControlSeverityV2::Warning => "warning",
            ControlSeverityV2::Note => "note",
        },
        code: diagnostic.code().to_owned(),
        message: diagnostic.message().to_owned(),
        graph_path: diagnostic.graph_path().map(|path| path.join("/")),
        span: diagnostic.span().map(|span| SourceSpanDto {
            file: span.file().to_owned(),
            start: span.start(),
            end: span.end(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use eqiora::control::{COMPILE_COMMAND_V1, CONTROL_PROTOCOL_V2, CompileOutcomeV2};
    use serde_json::json;

    use super::{AppState, compile_request};

    const SOURCE: &str = r#"
model decay(parameter rate: 1 / s = 1[1 / s]) {
  state x: 1;
  initial { x = 1; }
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;
    const ACCEPTED_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/accepted-v2.json"
    );
    const REJECTED_SOURCE_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/rejected-source-v2.json"
    );
    const RETIRED_PROTOCOL_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/retired-v1.json"
    );
    const UNKNOWN_PROTOCOL_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/unknown-protocol-v2.json"
    );
    const FORBIDDEN_SELECTION_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/forbidden-model-wire-v2.json"
    );
    const FORBIDDEN_FEATURE_REQUEST: &str = include_str!(
        "../../../verify/interfaces/control-plane-compile-check/models/forbidden-required-features-v2.json"
    );

    fn request(request_id: &str, source: &str) -> String {
        json!({
            "protocol": CONTROL_PROTOCOL_V2,
            "command": COMPILE_COMMAND_V1,
            "requestId": request_id,
            "filename": "model.eqi",
            "source": source,
        })
        .to_string()
    }

    #[test]
    fn accepted_control_execution_is_projected_and_cached_once() {
        let state = AppState::default();
        let response = compile_request(ACCEPTED_REQUEST, &state);
        let projection = response.projection.expect("Studio projection");
        let control = response.control.expect("shared control response");

        assert_eq!(control.request_id(), "shared-accepted-v2");
        let CompileOutcomeV2::Accepted { model } = control.outcome() else {
            panic!("accepted fixture must return one Model descriptor");
        };
        assert_eq!(model.schema(), "eqiora.model-envelope/v26");
        assert_eq!(
            model.transaction_schema(),
            "eqiora.model-transaction-envelope/v26"
        );
        assert!(response.diagnostics.is_empty());
        assert!(state.documents.lock().unwrap().contains(&projection.digest));
    }

    #[test]
    fn every_rejection_preserves_the_previously_admitted_lineage() {
        let state = AppState::default();
        let accepted = compile_request(&request("studio.compile:1", SOURCE), &state);
        assert!(matches!(
            accepted.control.as_ref().map(|response| response.outcome()),
            Some(CompileOutcomeV2::Accepted { .. })
        ));
        let digest = accepted.projection.unwrap().digest;

        let mut unknown: serde_json::Value =
            serde_json::from_str(&request("studio.compile:2", SOURCE)).unwrap();
        unknown["unexpected"] = json!(true);
        let rejected = compile_request(&unknown.to_string(), &state);
        assert!(rejected.control.is_none());
        assert_eq!(rejected.diagnostics[0].source, "control");
        assert!(state.documents.lock().unwrap().contains(&digest));

        let rejected = compile_request(REJECTED_SOURCE_REQUEST, &state);
        assert!(matches!(
            rejected.control.as_ref().map(|response| response.outcome()),
            Some(CompileOutcomeV2::Rejected { .. })
        ));
        assert!(rejected.projection.is_none());
        assert!(state.documents.lock().unwrap().contains(&digest));
    }

    #[test]
    fn retired_selection_inputs_fail_before_cache_mutation() {
        let state = AppState::default();
        let accepted = compile_request(&request("studio.compile:1", SOURCE), &state);
        let digest = accepted.projection.unwrap().digest;

        for fixture in [FORBIDDEN_SELECTION_REQUEST, FORBIDDEN_FEATURE_REQUEST] {
            let rejected = compile_request(fixture, &state);
            assert!(rejected.control.is_none());
            assert_eq!(rejected.diagnostics[0].code, "EQ0901");
            assert!(state.documents.lock().unwrap().contains(&digest));
        }

        for fixture in [RETIRED_PROTOCOL_REQUEST, UNKNOWN_PROTOCOL_REQUEST] {
            let rejected = compile_request(fixture, &state);
            assert!(rejected.control.is_none());
            assert_eq!(rejected.diagnostics[0].code, "EQ0001");
            assert!(state.documents.lock().unwrap().contains(&digest));
        }
    }
}
