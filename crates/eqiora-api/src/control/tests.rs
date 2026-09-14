use serde_json::Value;

use super::*;

const SOURCE: &str = r#"model decay() {
  state x: 1; initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;

#[test]
fn current_request_round_trips_in_frozen_member_order() {
    let request = CompileRequestV2::new("compile-1", "decay.eqi", SOURCE).unwrap();
    let bytes = request.canonical_json().unwrap();
    assert_eq!(
        String::from_utf8(bytes.clone()).unwrap(),
        format!(
            r#"{{"protocol":"{CONTROL_PROTOCOL_V2}","command":"{COMPILE_COMMAND_V1}","requestId":"compile-1","filename":"decay.eqi","source":{}}}"#,
            serde_json::to_string(SOURCE).unwrap()
        )
    );
    assert_eq!(CompileRequestV2::from_json(&bytes).unwrap(), request);
}

#[test]
fn dispatch_precedes_closed_dto_admission() {
    let retired = include_bytes!(
        "../../../../verify/interfaces/control-plane-compile-check/models/retired-v1.json"
    );
    let diagnostic = CompileRequestV2::from_json(retired).unwrap_err();
    assert_eq!(diagnostic.code(), "EQ0001");
    assert_eq!(
        diagnostic.message(),
        "unsupported control protocol `eqiora.control/v1`; expected `eqiora.control/v2`"
    );

    for bytes in [
        include_bytes!(
            "../../../../verify/interfaces/control-plane-compile-check/models/forbidden-model-wire-v2.json"
        )
        .as_slice(),
        include_bytes!(
            "../../../../verify/interfaces/control-plane-compile-check/models/forbidden-required-features-v2.json"
        )
        .as_slice(),
    ] {
        assert_eq!(
            CompileRequestV2::from_json(bytes).unwrap_err().code(),
            "EQ0901"
        );
    }
}

#[test]
fn accepted_execution_links_the_same_document_and_schema_facts() {
    let request = CompileRequestV2::new("compile-1", "decay.eqi", SOURCE).unwrap();
    let execution = execute_compile_v2(&request);
    let document = execution.document().expect("accepted document");
    let CompileOutcomeV2::Accepted { model } = execution.response().outcome() else {
        panic!("valid source must be accepted")
    };
    assert_eq!(execution.response().request_id(), "compile-1");
    assert_eq!(model.schema(), "eqiora.model-envelope/v26");
    assert_eq!(
        model.transaction_schema(),
        "eqiora.model-transaction-envelope/v26"
    );
    assert_eq!(model.digest(), document.digest().unwrap());
    assert_eq!(model.semantic_revision(), document.program().revision().0);
    assert_eq!(
        CompileResponseV2::from_json(&execution.response().canonical_json().unwrap()).unwrap(),
        *execution.response()
    );
}

#[test]
fn rejected_execution_has_kernel_diagnostics_and_no_document() {
    let request = CompileRequestV2::new("compile-1", "bad.eqi", "not a model").unwrap();
    let execution = execute_compile_v2(&request);
    assert!(execution.document().is_none());
    let CompileOutcomeV2::Rejected { diagnostics } = execution.response().outcome() else {
        panic!("invalid source must be rejected")
    };
    assert!(!diagnostics.is_empty());
    assert_eq!(diagnostics[0].source(), ControlDiagnosticSourceV2::Kernel);
    assert!(diagnostics[0].code().starts_with("EQ"));
}

#[test]
fn committed_schema_is_the_independent_promoted_oracle() {
    let schema: Value = serde_json::from_str(COMPILE_V2_SCHEMA_JSON).unwrap();
    let definitions = &schema["$defs"];
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(definitions["request"]["additionalProperties"], false);
    assert_eq!(definitions["response"]["additionalProperties"], false);
    assert_eq!(definitions["diagnostic"]["additionalProperties"], false);
    assert_eq!(
        definitions["request"]["x-eqiora-maxEncodedUtf8Bytes"],
        MAX_COMPILE_REQUEST_BYTES_V2
    );
    assert_eq!(
        definitions["response"]["x-eqiora-maxEncodedUtf8Bytes"],
        MAX_COMPILE_RESPONSE_BYTES_V2
    );
    for key in ["maxLength", "x-eqiora-maxUtf8Bytes"] {
        assert_eq!(
            definitions["request"]["properties"]["source"][key],
            MAX_COMPILE_SOURCE_BYTES_V2
        );
        assert_eq!(
            definitions["request"]["properties"]["filename"][key],
            MAX_COMPILE_FILENAME_BYTES_V2
        );
        assert_eq!(
            definitions["diagnostic"]["properties"]["message"][key],
            MAX_CONTROL_DIAGNOSTIC_MESSAGE_BYTES_V2
        );
        assert_eq!(
            definitions["diagnostic"]["properties"]["graphPath"]["oneOf"][0]["items"][key],
            MAX_CONTROL_TEXT_MEMBER_BYTES_V2
        );
        assert_eq!(
            definitions["sourceSpan"]["properties"]["file"][key],
            MAX_CONTROL_TEXT_MEMBER_BYTES_V2
        );
        assert_eq!(
            definitions["patch"]["properties"]["summary"][key],
            MAX_CONTROL_TEXT_MEMBER_BYTES_V2
        );
    }
    assert_eq!(
        definitions["requestId"]["maxLength"],
        MAX_CONTROL_REQUEST_ID_BYTES_V2
    );
    assert_eq!(
        definitions["rejectedOutcome"]["properties"]["diagnostics"]["maxItems"],
        MAX_CONTROL_DIAGNOSTICS_V2
    );
    assert_eq!(
        definitions["diagnostic"]["properties"]["graphPath"]["oneOf"][0]["maxItems"],
        MAX_CONTROL_GRAPH_PATH_SEGMENTS_V2
    );
}

#[test]
fn bounded_dispatch_identity_never_echoes_oversized_content() {
    let marker = "eqiora-oracle-prelude-marker";
    let protocol = format!("{}{}", "x".repeat(129), marker);
    let request = format!(
        r#"{{"protocol":{},"command":"{COMPILE_COMMAND_V1}"}}"#,
        serde_json::to_string(&protocol).unwrap()
    );
    let diagnostic = CompileRequestV2::from_json(request.as_bytes()).unwrap_err();
    assert_eq!(diagnostic.code(), "EQ0901");
    assert!(!diagnostic.message().contains(marker));
}

#[test]
fn request_encoder_enforces_the_same_encoded_byte_bound_as_dispatch() {
    let request = CompileRequestV2::new(
        "compile-1",
        "escaped.eqi",
        "\"".repeat(MAX_COMPILE_SOURCE_BYTES_V2),
    )
    .unwrap();
    let diagnostic = request.canonical_json().unwrap_err();
    assert_eq!(diagnostic.code(), "EQ0901");
    assert_eq!(
        diagnostic.message(),
        format!("compile/check request exceeds {MAX_COMPILE_REQUEST_BYTES_V2} encoded bytes")
    );
}

#[test]
fn response_model_identity_uses_the_frozen_character_bound() {
    let value = serde_json::json!({
        "protocol": CONTROL_PROTOCOL_V2,
        "command": COMPILE_COMMAND_V1,
        "requestId": "compile-1",
        "outcome": {
            "status": "accepted",
            "model": {
                "schema": "eqiora.model-envelope/v26",
                "transactionSchema": "eqiora.model-transaction-envelope/v26",
                "digest": "0".repeat(64),
                "modelId": "π".repeat(128),
                "semanticRevision": 1
            }
        }
    });
    CompileResponseV2::from_json(&serde_json::to_vec(&value).unwrap())
        .expect("the v2 schema bounds modelId by Unicode characters");
}

#[test]
fn oversized_kernel_diagnostic_projects_to_the_exact_overflow() {
    let diagnostic = eqiora_core::Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_ARTIFACT,
        "x".repeat(MAX_CONTROL_DIAGNOSTIC_MESSAGE_BYTES_V2 + 1),
    );
    assert!(ControlDiagnosticV2::from_kernel(diagnostic).is_err());

    let overflow = ControlDiagnosticV2::diagnostics_overflow();
    assert_eq!(overflow.source(), ControlDiagnosticSourceV2::Control);
    assert_eq!(overflow.severity(), ControlSeverityV2::Error);
    assert_eq!(overflow.code(), "EQ0901");
    assert_eq!(
        overflow.message(),
        "compile/check diagnostics exceed the control v2 response limits"
    );
    assert!(overflow.graph_path().is_none());
    assert!(overflow.span().is_none());
    assert!(overflow.patch().is_none());
}
