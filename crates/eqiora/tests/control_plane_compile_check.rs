use eqiora::api::ModelDocument;
use eqiora::control::{
    COMPILE_COMMAND_V1, CONTROL_PROTOCOL_V2, CompileOutcomeV2, CompileRequestV2, CompileResponseV2,
    ControlDiagnosticSourceV2, execute_compile_v2,
};
use serde_json::Value;

const CONTROL_EXECUTOR_SOURCE: &str =
    include_str!("../../eqiora-api/src/control/compile_response.rs");
const PYTHON_ADAPTER_SOURCE: &str = include_str!("../../eqiora-python/src/lib.rs");

#[path = "control_plane_compile_check/contract.rs"]
mod contract;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!(
            "../../../verify/interfaces/control-plane-compile-check/models/",
            $name
        ))
    };
}

const ACCEPTED: &[u8] = fixture!("accepted-v2.json");
const REJECTED_SOURCE: &[u8] = fixture!("rejected-source-v2.json");
const RETIRED: &[u8] = fixture!("retired-v1.json");
const UNKNOWN_PROTOCOL: &[u8] = fixture!("unknown-protocol-v2.json");
const UNKNOWN_COMMAND: &[u8] = fixture!("unknown-command-v2.json");
const FORBIDDEN_MODEL_SELECTION: &[u8] = fixture!("forbidden-model-wire-v2.json");
const FORBIDDEN_FEATURE_LIST: &[u8] = fixture!("forbidden-required-features-v2.json");
const EXPECTED: &str =
    include_str!("../../../verify/interfaces/control-plane-compile-check/expected/contract.json");

fn expected() -> Value {
    serde_json::from_str(EXPECTED).unwrap()
}

fn rejection<'a>(expected: &'a Value, name: &str) -> &'a Value {
    expected["rejections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("the frozen contract omits rejection `{name}`"))
}

#[test]
fn accepted_fixture_links_one_execution_and_preserves_structural_meaning() {
    let expected = expected();
    let accepted = &expected["accepted"];
    let linkage = &expected["sameExecutionLinkage"];
    let relation = &expected["structuralRelation"];
    assert_eq!(
        expected["schema"],
        "eqiora.verify.control-plane-compile-check/v2"
    );
    let request = CompileRequestV2::from_json(ACCEPTED).unwrap();
    let first = execute_compile_v2(&request);
    let document = first.document().expect("accepted document");
    let CompileOutcomeV2::Accepted { model } = first.response().outcome() else {
        panic!("accepted fixture must produce a Model descriptor")
    };
    let reference = document.artifact_reference().unwrap();
    let public_schema: Value =
        serde_json::from_str(eqiora::control::COMPILE_V2_SCHEMA_JSON).unwrap();
    let model_properties = &public_schema["$defs"]["model"]["properties"];
    assert_eq!(first.response().request_id(), accepted["requestId"]);
    assert_eq!(model.schema(), model_properties["schema"]["const"]);
    assert_eq!(
        model.transaction_schema(),
        model_properties["transactionSchema"]["const"]
    );
    assert_eq!(model.digest(), reference.artifact().as_str());
    assert_eq!(model.model_id(), reference.model().to_string());
    assert_eq!(model.semantic_revision(), accepted["semanticRevision"]);
    assert_eq!(linkage["request"], accepted["request"]);
    assert_eq!(linkage["documentPresent"], first.document().is_some());
    assert_eq!(linkage["echoesRequestId"], true);

    let second = execute_compile_v2(&request);
    let second_document = second.document().unwrap();
    let ordinary = ModelDocument::compile(request.filename(), request.source()).unwrap();
    let documents = [document, second_document, &ordinary];
    for document in documents {
        assert_eq!(
            document.structural_fingerprint().unwrap().generation(),
            eqiora::api::SemanticFingerprintGeneration::V22
        );
    }
    assert_all_equal(documents.map(|value| value.digest().unwrap()));
    assert_all_equal(documents.map(|value| value.program().model()));
    assert_eq!(
        document.structural_fingerprint().unwrap(),
        second_document.structural_fingerprint().unwrap()
    );
    assert_eq!(
        document.structural_fingerprint().unwrap(),
        ordinary.structural_fingerprint().unwrap()
    );
    assert_eq!(relation["request"], accepted["request"]);
    assert_eq!(relation["semanticRevision"], model.semantic_revision());
    assert_eq!(
        relation["equalFields"],
        serde_json::json!(["modelId", "digest", "structuralSemanticFingerprint"])
    );

    let response: Value =
        serde_json::from_slice(&first.response().canonical_json().unwrap()).unwrap();
    for field in expected["forbiddenResponseFields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(
            response.get(field).is_none(),
            "`{field}` entered control data"
        );
    }
}

#[test]
fn one_transport_neutral_operation_owns_both_adapters() {
    let _operation: fn(&str, &str) -> Result<ModelDocument, Vec<eqiora::Diagnostic>> =
        ModelDocument::compile;

    let control = rust_function(CONTROL_EXECUTOR_SOURCE, "pub fn execute_compile_v2(");
    assert_eq!(
        control.matches("ModelDocument::compile").count(),
        1,
        "control-v2 must invoke the transport-neutral operation exactly once"
    );
    assert!(
        control.contains("ModelDocument::compile(request.filename(), request.source())"),
        "control-v2 must forward the one admitted filename/source pair unchanged"
    );
    assert_no_lower_compiler_entrypoint(control, "control-v2");

    let python = rust_function(PYTHON_ADAPTER_SOURCE, "fn compile(");
    assert_eq!(
        python.matches("ModelDocument::compile(").count(),
        1,
        "Python compile must invoke the transport-neutral operation exactly once"
    );
    let compact_python = python
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let detached_call = rust_call_expression(&compact_python, "py.detach");
    assert_eq!(
        detached_call.matches("ModelDocument::compile(").count(),
        1,
        "the py.detach call expression itself must own the operation invocation"
    );
    assert_no_lower_compiler_entrypoint(python, "Python");
}

#[test]
fn detached_call_ownership_predicate_rejects_post_detach_compilation() {
    let accepted = r#"
        fn compile(py: Python<'_>, filename: &str, source: &str) {
            py.detach(move || ModelDocument::compile(filename, source));
        }
    "#;
    assert_eq!(
        rust_call_expression(accepted, "py.detach")
            .matches("ModelDocument::compile")
            .count(),
        1
    );

    let post_detach_mutant = r#"
        fn compile(py: Python<'_>, filename: &str, source: &str) {
            py.detach(|| ());
            ModelDocument::compile(filename, source);
        }
    "#;
    assert_eq!(
        rust_call_expression(post_detach_mutant, "py.detach")
            .matches("ModelDocument::compile")
            .count(),
        0,
        "a compile call after a completed detach expression must not satisfy ownership"
    );
}

#[test]
fn admitted_source_failure_is_a_closed_response_without_a_document() {
    let expected = expected();
    let frozen = rejection(&expected, "rejected-source");
    let request = CompileRequestV2::from_json(REJECTED_SOURCE).unwrap();
    let execution = execute_compile_v2(&request);
    assert!(execution.document().is_none());
    let CompileOutcomeV2::Rejected { diagnostics } = execution.response().outcome() else {
        panic!("empty source must be rejected")
    };
    assert_eq!(execution.response().request_id(), frozen["requestId"]);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(frozen["stage"], "compilation");
    assert_eq!(frozen["documentPresent"], execution.document().is_some());
    assert_eq!(frozen["diagnosticSource"], "kernel");
    assert_eq!(diagnostics[0].source(), ControlDiagnosticSourceV2::Kernel);
    assert_eq!(diagnostics[0].code(), frozen["diagnosticCode"]);
    assert_eq!(
        frozen["messageNonempty"],
        !diagnostics[0].message().is_empty()
    );
}

#[test]
fn dispatch_rejections_precede_closed_dto_admission() {
    let expected = expected();
    for (name, bytes) in [
        ("retired-protocol", RETIRED),
        ("unknown-protocol", UNKNOWN_PROTOCOL),
        ("unknown-command", UNKNOWN_COMMAND),
    ] {
        let frozen = rejection(&expected, name);
        let diagnostic = CompileRequestV2::from_json(bytes).unwrap_err();
        assert_eq!(frozen["stage"], "dispatch-prelude");
        assert_eq!(frozen["standaloneDiagnostic"], true);
        assert_eq!(diagnostic.source(), ControlDiagnosticSourceV2::Control);
        assert_eq!(diagnostic.code(), frozen["diagnosticCode"]);
        assert_eq!(diagnostic.message(), frozen["message"]);
        let value = serde_json::to_value(&diagnostic).unwrap();
        for field in expected["dispatcherNullDiagnosticFields"]
            .as_array()
            .unwrap()
        {
            assert!(value[field.as_str().unwrap()].is_null());
        }
    }
    for (name, bytes) in [
        ("forbidden-model-selection", FORBIDDEN_MODEL_SELECTION),
        ("forbidden-feature-list", FORBIDDEN_FEATURE_LIST),
    ] {
        let frozen = rejection(&expected, name);
        let diagnostic = CompileRequestV2::from_json(bytes).unwrap_err();
        assert_eq!(frozen["stage"], "dto-admission");
        assert_eq!(diagnostic.source(), ControlDiagnosticSourceV2::Control);
        assert_eq!(diagnostic.code(), frozen["diagnosticCode"]);
        assert!(!diagnostic.message().is_empty());
    }
    for witness in expected["precedenceWitnesses"].as_array().unwrap() {
        let diagnostic = CompileRequestV2::from_json(contract::named_fixture(
            witness["request"].as_str().unwrap(),
        ))
        .unwrap_err();
        assert_eq!(witness["expectedStage"], "dispatch-prelude");
        assert_eq!(diagnostic.code(), witness["expectedCode"]);
    }
}

#[test]
fn generated_request_resource_falsifiers_fail_closed() {
    let expected = expected();
    let accepted: Value = serde_json::from_slice(ACCEPTED).unwrap();
    for boundary in expected["generatedResourceBoundaries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|boundary| boundary["member"] != "request")
    {
        let member = boundary["member"].as_str().unwrap();
        let mut mutant = accepted.clone();
        mutant[member] =
            Value::String("x".repeat(boundary["falsifier"].as_u64().unwrap() as usize));
        assert_eq!(
            CompileRequestV2::from_json(&serde_json::to_vec(&mutant).unwrap())
                .unwrap_err()
                .code(),
            boundary["diagnosticCode"]
        );
    }

    let encoded = expected["generatedResourceBoundaries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|boundary| boundary["member"] == "request")
        .unwrap();
    let mut padded = serde_json::to_vec(&accepted).unwrap();
    padded.resize(encoded["falsifier"].as_u64().unwrap() as usize, b' ');
    assert_eq!(
        CompileRequestV2::from_json(&padded).unwrap_err().code(),
        encoded["diagnosticCode"]
    );
}

#[test]
fn dispatch_prelude_boundaries_are_generated_from_the_frozen_contract() {
    let expected = expected();
    let prelude = &expected["dispatchPrelude"];
    let marker = prelude["contentMarker"].as_str().unwrap();
    for falsifier in prelude["overflow"]["falsifiers"].as_array().unwrap() {
        let characters = falsifier["characters"].as_u64().unwrap() as usize;
        let utf8_bytes = falsifier["utf8Bytes"].as_u64().unwrap() as usize;
        let marker_characters = marker.chars().count();
        let mut value = if utf8_bytes == characters {
            "x".repeat(characters - marker_characters)
        } else {
            let mut value = "é".to_owned();
            value.push_str(&"x".repeat(characters - marker_characters - 1));
            value
        };
        value.push_str(marker);
        assert_eq!(value.chars().count(), characters);
        assert_eq!(value.len(), utf8_bytes);

        let mut request = serde_json::json!({
            "protocol": CONTROL_PROTOCOL_V2,
            "command": COMPILE_COMMAND_V1,
        });
        request[falsifier["member"].as_str().unwrap()] = Value::String(value);
        let diagnostic =
            CompileRequestV2::from_json(&serde_json::to_vec(&request).unwrap()).unwrap_err();
        assert_eq!(diagnostic.code(), prelude["overflow"]["diagnosticCode"]);
        assert!(!diagnostic.message().contains(marker));
    }

    let at_bound = &prelude["atBound"];
    let value = "x".repeat(at_bound["characters"].as_u64().unwrap() as usize);
    let request = serde_json::json!({
        "protocol": value,
        "command": COMPILE_COMMAND_V1,
    });
    let diagnostic =
        CompileRequestV2::from_json(&serde_json::to_vec(&request).unwrap()).unwrap_err();
    assert_eq!(diagnostic.code(), at_bound["diagnosticCode"]);
}

#[test]
fn frozen_diagnostic_overflow_is_one_closed_decodable_response() {
    let expected = expected();
    let overflow = &expected["diagnosticOverflow"];
    let request_id = "overflow-contract";
    let bytes = serde_json::to_vec(&serde_json::json!({
        "protocol": CONTROL_PROTOCOL_V2,
        "command": COMPILE_COMMAND_V1,
        "requestId": request_id,
        "outcome": {
            "status": overflow["outcome"],
            "diagnostics": [{
                "source": overflow["diagnosticSource"],
                "severity": overflow["diagnosticSeverity"],
                "code": overflow["diagnosticCode"],
                "message": overflow["message"],
                "graphPath": null,
                "span": null,
                "patch": null,
            }],
        },
    }))
    .unwrap();
    let response = CompileResponseV2::from_json(&bytes).unwrap();
    assert_eq!(response.request_id(), request_id);
    let CompileOutcomeV2::Rejected { diagnostics } = response.outcome() else {
        panic!("the frozen overflow cannot expose a Model descriptor")
    };
    assert_eq!(diagnostics.len(), overflow["diagnosticCount"]);
    assert_eq!(diagnostics[0].code(), overflow["diagnosticCode"]);
    assert_eq!(diagnostics[0].message(), overflow["message"]);
    let value = serde_json::to_value(&diagnostics[0]).unwrap();
    for field in expected["dispatcherNullDiagnosticFields"]
        .as_array()
        .unwrap()
    {
        assert!(value[field.as_str().unwrap()].is_null());
    }
    assert_eq!(overflow["partialKernelDiagnosticSerialized"], false);
    assert_eq!(overflow["truncated"], false);
    assert_eq!(overflow["echoesRequestId"], true);
    assert_eq!(overflow["modelDescriptorPresent"], false);
    assert_eq!(overflow["documentPresent"], false);
    assert_eq!(
        overflow["triggers"],
        serde_json::json!([
            "message",
            "graphPath",
            "patchSummary",
            "diagnosticCount",
            "response"
        ])
    );
}

fn assert_all_equal<T: std::fmt::Debug + PartialEq>(values: [T; 3]) {
    assert_eq!(&values[0], &values[1]);
    assert_eq!(&values[0], &values[2]);
    assert_eq!(&values[1], &values[2]);
}

fn rust_function<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("source omits function signature {signature:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("function {signature:?} has no body"));
    let mut depth = 0_u32;
    for (offset, byte) in source[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {signature:?} has an unterminated body")
}

fn rust_call_expression<'a>(source: &'a str, callee: &str) -> &'a str {
    let start = source
        .find(callee)
        .unwrap_or_else(|| panic!("source omits call to {callee:?}"));
    let suffix = &source[start + callee.len()..];
    let open_offset = suffix
        .find('(')
        .unwrap_or_else(|| panic!("call to {callee:?} has no argument list"));
    assert!(
        suffix[..open_offset].trim().is_empty(),
        "{callee:?} is not immediately followed by one call argument list"
    );
    let open = start + callee.len() + open_offset;
    let mut depth = 0_u32;
    for (offset, byte) in source[open..].bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("call to {callee:?} has an unterminated argument list")
}

fn assert_no_lower_compiler_entrypoint(function: &str, adapter: &str) {
    for forbidden in [
        "eqiora::compiler::",
        "eqiora_compiler",
        "compiler::compile(",
        "lower_module(",
        "lower_model(",
    ] {
        assert!(
            !function.contains(forbidden),
            "{adapter} bypasses ModelDocument::compile through {forbidden:?}"
        );
    }
}
