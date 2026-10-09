//! The harmonic request is a Formulation sidecar, not another transient Model.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

fn source() -> &'static str {
    include_str!("../../../docs/language/harmonic-rc.md")
        .split_once("```eqiora\n")
        .unwrap()
        .1
        .split_once("```")
        .unwrap()
        .0
}

#[test]
fn harmonic_request_retains_the_original_unknowns_and_initial_condition() {
    let source = source();
    let original = format!(
        "{} }}",
        source.split_once("  form harmonic_response").unwrap().0
    );
    let original = CompiledModel::compile_selected("rc.eqi", &original, "RC", &[]).unwrap();
    let harmonic = CompiledModel::compile_selected("rc.eqi", source, "RC", &[]).unwrap();
    assert_eq!(original.transaction().ops(), harmonic.transaction().ops());
    let form = harmonic.authored_formulations().next().unwrap();
    assert_eq!(harmonic.authored_formulations().len(), 1);
    let projection = form.projection();
    assert_eq!(
        projection,
        &AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap()
    );
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("rc.eqi", source)
            .into_document()
            .unwrap(),
    );
    let replayed = CompiledModel::compile_selected("rc.eqi", &formatted, "RC", &[]).unwrap();
    assert_eq!(
        projection,
        replayed
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
    );
}

#[test]
fn harmonic_request_owns_frequency_excitation_and_amplitude_mappings() {
    let model = CompiledModel::compile_selected("rc.eqi", source(), "RC", &[]).unwrap();
    let form = model.authored_formulations().next().unwrap();
    let request = form.projection().harmonic_request().unwrap();
    assert_eq!(request.convention(), "negative-exponential");
    assert_eq!(request.normalization(), "peak");
    assert_eq!(request.relations().len(), 1);
    assert_eq!(request.excitations().len(), 1);
    assert_eq!(
        request
            .amplitudes()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["voltage_hat", "current_hat"]
    );
    assert_eq!(form.trials().len(), 2);
    assert!(form.projection().equations().is_empty());
    assert_eq!(
        form.projection().implication(),
        "harmonic-response-satisfies-original-relations"
    );
}

#[test]
fn harmonic_request_rejects_wrong_types_missing_inputs_and_unknown_dependencies() {
    for (from, to, diagnostic) in [
        (
            "complex<V> for voltage",
            "complex<A> for voltage",
            "must preserve the original real Field",
        ),
        ("excitation source = drive;", "", "without an excitation"),
        (
            "angular_frequency = omega",
            "angular_frequency = drive",
            "dimension 1/time",
        ),
        (
            "excitation source = drive;",
            "excitation source = math.complex(voltage,0[V]);",
            "closed Parameters",
        ),
        (
            "amplitude current_hat: complex<A> for current;",
            "",
            "without an amplitude mapping",
        ),
        (
            "complex<A> for current;",
            "complex<A> for voltage;",
            "original unknown mappings must be distinct",
        ),
    ] {
        let source = source().replace(from, to);
        let errors = CompiledModel::compile_selected("rc.eqi", &source, "RC", &[]).expect_err(to);
        assert!(format!("{errors:?}").contains(diagnostic), "{errors:?}");
    }
}

#[test]
fn harmonic_request_accepts_inline_dimensioned_frequency_and_excitation() {
    let source = source()
        .replace("angular_frequency = omega", "angular_frequency = 1000[1/s]")
        .replace(
            "excitation source = drive;",
            "excitation source = math.complex(1[V],0[V]);",
        );
    let model = CompiledModel::compile_selected("rc.eqi", &source, "RC", &[]).unwrap();
    assert!(
        model
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .harmonic_request()
            .is_some()
    );
}
