//! Probability law, bound outcome and typed moments survive the ordinary artifact path.
use eqiora_artifact::{ModelDecoderLimits, ModelEnvelope, StructuralSemanticFingerprint};
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::AxisBounds;
use eqiora_sem::KernelProgram;

fn admit(compiled: CompiledModel) -> Result<KernelProgram, Vec<eqiora_core::Diagnostic>> {
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model)
}

fn replay(program: &KernelProgram) -> KernelProgram {
    let fingerprint = StructuralSemanticFingerprint::from_program(program).unwrap();
    let bytes = ModelEnvelope::from_program(program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let replay = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default())
        .unwrap()
        .to_program()
        .unwrap();
    assert_eq!(
        StructuralSemanticFingerprint::from_program(&replay).unwrap(),
        fingerprint
    );
    let old = String::from_utf8(bytes)
        .unwrap()
        .replace("eqiora.model-envelope/v50", "eqiora.model-envelope/v48");
    assert!(ModelEnvelope::from_json(old.as_bytes(), ModelDecoderLimits::default()).is_err());
    replay
}

#[test]
fn finite_moments_keep_bound_outcomes_arity_units_and_population_meaning() {
    let source = "enum Outcome {Low,High} model M() { distribution law:normalize_mass(Outcome, weights=[1,3]); observable x:m=random_variable(law, if outcome(law)==Outcome.Low then 0[m] else 2[m]); observable mean:m=expectation(x); observable variance:m^2=variance(x); observable covariance:m^2=covariance(x,x); }";
    let compiled = eqiora_compiler::compile("probability.eqi", source)
        .unwrap()
        .remove(0);
    let moments = ["mean", "variance", "covariance"]
        .map(|name| compiled.symbols().get(name).unwrap().downcast().unwrap());
    let original = admit(compiled).unwrap();
    let restored = replay(&original);
    for program in [&original, &restored] {
        for (id, expected) in moments.into_iter().zip([1.5, 0.75, 0.75]) {
            let actual = program
                .evaluate_finite_observable(id, &mut |_| {
                    panic!("exact law needs no sampled inputs")
                })
                .unwrap();
            assert_eq!(actual.real_scalar_value().unwrap().value(), expected);
        }
    }
}

#[test]
fn bounded_density_preserves_measure_and_rejects_wrong_units() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(3.0, length)).unwrap();
    for (density, valid) in [
        ("1[1/m]", true),
        ("1[m]", false),
        ("0[1/m]", false),
        ("-1[1/m]", false),
    ] {
        let source = format!(
            "model M(support position:interval(m)) {{ distribution law:normalize_density(measure(position), density={density}); coordinate position_value:m on position from position[0]; observable x:m=random_variable(law,position_value); observable mean:m=expectation(x); }}"
        );
        let compiled = CompiledModel::compile_selected(
            "density.eqi",
            &source,
            "M",
            &[("position", StaticBindingValue::CoordinateInterval(bounds))],
        )
        .unwrap();
        let program = admit(compiled);
        if valid {
            replay(&program.unwrap());
        } else {
            assert!(program.is_err(), "{density}");
        }
    }
}

#[test]
fn probability_identity_ignores_local_names_but_retains_weights_and_moment_kind() {
    let source = "enum Outcome {Low,High} model M() { distribution law:normalize_mass(Outcome,weights=[1,3]); observable x:m=random_variable(law,if outcome(law)==Outcome.Low then 0[m] else 2[m]); observable reduced:m^2=variance(x); }";
    let fingerprint = |source: &str| {
        let compiled = eqiora_compiler::compile("identity.eqi", source)
            .unwrap()
            .remove(0);
        StructuralSemanticFingerprint::from_program(&admit(compiled).unwrap()).unwrap()
    };
    let original = fingerprint(source);
    let renamed = source
        .replace("Outcome", "Choice")
        .replace("law", "distribution_name")
        .replace("observable x:", "observable quantity:")
        .replace("variance(x)", "variance(quantity)")
        .replace("reduced", "spread")
        .replace("model M()", "model Renamed()");
    assert_eq!(original, fingerprint(&renamed));
    assert_ne!(
        original,
        fingerprint(&source.replace("weights=[1,3]", "weights=[3,1]"))
    );
    // Cov(x,x) has the same scalar answer as Var(x), but is a distinct retained
    // two-input operation. Structural identity does not erase that meaning.
    assert_ne!(
        original,
        fingerprint(&source.replace("variance(x)", "covariance(x,x)"))
    );
}

#[test]
fn probability_wire_cannot_drop_bindings_or_change_moment_arity_and_units() {
    let source = "enum Outcome {Low,High} model M() { distribution law:normalize_mass(Outcome,weights=[1,3]); observable x:m=random_variable(law,if outcome(law)==Outcome.Low then 0[m] else 2[m]); observable mean:m=expectation(x); observable paired:m^2=covariance(x,x); }";
    let compiled = eqiora_compiler::compile("tamper.eqi", source)
        .unwrap()
        .remove(0);
    let original = admit(compiled).unwrap();
    let bytes = ModelEnvelope::from_program(&original)
        .unwrap()
        .canonical_json()
        .unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (kind, moment, replacement) in [
        ("random-variable", None, serde_json::json!({"kind":"value"})),
        (
            "probability-moment",
            Some("expectation"),
            serde_json::json!("variance"),
        ),
        (
            "probability-moment",
            Some("covariance"),
            serde_json::json!("variance"),
        ),
    ] {
        let mut altered = original.clone();
        let definition = altered["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .map(|node| &mut node["definition"])
            .find(|definition| {
                definition["reduction"]["kind"] == kind
                    && moment.is_none_or(|moment| definition["reduction"]["moment"] == moment)
            })
            .expect("ordinary valid reduction precedes the mutation");
        if moment.is_some() {
            definition["reduction"]["moment"] = replacement;
        } else {
            definition["reduction"] = replacement;
        }
        let bytes = serde_json::to_vec(&altered).unwrap();
        let rejected = match ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()) {
            Err(_) => true,
            Ok(envelope) => envelope.to_program().is_err(),
        };
        assert!(rejected, "{kind} {moment:?}");
    }
}
