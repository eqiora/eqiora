use eqiora_artifact::{ImplicitTimeRunManifestV1, TimeDecoderLimits, TimeRunManifestV1};
use eqiora_core::Diagnostic;
use serde_json::{Value, json};

fn manifest(implicit: bool, method: &str) -> Value {
    let mut value = json!({
        "schema": if implicit { "eqiora.implicit-time-run-manifest/v1" } else { "eqiora.time-run-manifest/v1" },
        "encoding": "eqiora.canonical-json/v1",
        "model_sha256": "0".repeat(64),
        "semantic_revision": 1,
        "lowering_sha256": "1".repeat(64),
        "plan": {
            "method": method,
            "start_time": 0.0,
            "initial_step": 0.25,
            "relative_tolerance": 0.01,
            "absolute_tolerances": [0.001],
            "output_times": [0.5, 1.0]
        },
        "execution": {
            "backend": "test-time-producer",
            "backend_version": "1",
            "method": method,
            "equation_class": if implicit { json!("general-implicit-dae") } else { json!({"kind": "explicit-ode"}) },
            "initial_condition": "provided"
        },
        "output_sha256": []
    });
    if implicit {
        value["input_initial_data_sha256"] = json!("2".repeat(64));
        value["accepted_initial_data_sha256"] = json!("3".repeat(64));
    }
    value
}

fn canonical(implicit: bool, value: &Value) -> Result<Vec<u8>, Diagnostic> {
    let bytes = serde_json::to_vec(value).unwrap();
    if implicit {
        ImplicitTimeRunManifestV1::from_json(&bytes, Default::default())?.canonical_json()
    } else {
        TimeRunManifestV1::from_json(&bytes, Default::default())?.canonical_json()
    }
}

#[test]
fn both_run_families_preserve_the_current_plan_leaf() {
    for implicit in [false, true] {
        let value = manifest(implicit, "bdf");
        let bytes = canonical(implicit, &value).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains(concat!(
            "\"plan\":{\"method\":\"bdf\",\"start_time\":0.0,\"initial_step\":0.25,",
            "\"relative_tolerance\":0.01,\"absolute_tolerances\":[0.001],\"output_times\":[0.5,1.0]}"
        )));
    }
}

#[test]
fn each_run_family_keeps_its_method_admission() {
    for (implicit, accepted, rejected) in [
        (false, "tsitouras45", "implicit-euler"),
        (true, "implicit-euler", "tsitouras45"),
    ] {
        canonical(implicit, &manifest(implicit, accepted)).unwrap();
        assert!(canonical(implicit, &manifest(implicit, rejected)).is_err());
    }
}

#[test]
fn implicit_plan_retains_its_state_dimension_budget() {
    let bytes = serde_json::to_vec(&manifest(true, "bdf")).unwrap();
    let mut limits = TimeDecoderLimits {
        max_time_state_dimension: 1,
        ..Default::default()
    };
    ImplicitTimeRunManifestV1::from_json(&bytes, limits).unwrap();
    limits.max_time_state_dimension = 0;
    assert!(ImplicitTimeRunManifestV1::from_json(&bytes, limits).is_err());
}

#[test]
fn both_run_families_reject_invalid_shared_controls() {
    for implicit in [false, true] {
        for (field, invalid) in [
            ("initial_step", json!(0.0)),
            ("relative_tolerance", json!(-0.01)),
            ("absolute_tolerances", json!([])),
            ("absolute_tolerances", json!([0.0])),
            ("output_times", json!([])),
            ("output_times", json!([0.0, 1.0])),
            ("output_times", json!([1.0, 0.5])),
            ("output_times", json!([0.5, 0.5])),
            ("unknown_control", json!(1)),
        ] {
            let mut value = manifest(implicit, "bdf");
            value["plan"][field] = invalid;
            assert!(canonical(implicit, &value).is_err(), "{implicit}: {field}");
        }
    }
}
