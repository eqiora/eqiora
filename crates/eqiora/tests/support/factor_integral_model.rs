//! Factor-selected integrals retain input, measure and output independently in Model replay.
use eqiora_artifact::ModelEnvelope;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::{AxisBounds, KernelNode, ObservableReduction};
use eqiora_sem::KernelProgram;

const SOURCE: &str = "model Distribution(support position:interval(m), support velocity:interval(m/s), support other:interval(m/s)) {
    support phase:product(position,velocity);
    variable f:s/m^2 on phase;
    relation retain on phase { f=0[s/m^2]; }
    observable density:1/m on position=integral(f,measure(velocity));
    observable twice:1/m on position=2*density;
    observable mass:1=integral(f,measure(phase));
    observable composed:1=integral(density,measure(position));
}";

fn compile(source: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let interval = |time| {
        let unit = DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap();
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(-2.0, unit), DynQuantity::new(4.0, unit)).unwrap(),
        )
    };
    CompiledModel::compile_selected(
        "factor-integrals.eqi",
        source,
        "Distribution",
        &[
            ("position", interval(0)),
            ("velocity", interval(-1)),
            ("other", interval(-1)),
        ],
    )
}

#[test]
fn partial_integral_replays_distinct_input_measure_and_output_supports() {
    let compiled = compile(SOURCE).unwrap();
    let symbols = compiled.symbols().clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let bytes = ModelEnvelope::from_program(&program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let replay = ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    assert_eq!(program, replay);
    for name in ["density", "twice", "mass", "composed"] {
        let id = symbols.get(name).unwrap();
        let Some(KernelNode::Observable(definition)) = replay.node(id) else {
            panic!("Observable");
        };
        let typed = replay.typed_observable(definition.id()).unwrap();
        if name == "twice" {
            assert!(
                replay
                    .evaluate_finite_observable(definition.id(), &mut |_| None)
                    .unwrap_err()
                    .message()
                    .contains("output support")
            );
        }

        if name == "density" {
            let ObservableReduction::SpatialIntegral { input, domain, .. } = definition.reduction()
            else {
                panic!("integral");
            };
            assert_eq!(input.erase(), symbols.get("phase").unwrap());
            assert_eq!(domain.erase(), symbols.get("velocity").unwrap());
            assert_eq!(
                *typed
                    .node_type(definition.expression().roots()[0])
                    .unwrap()
                    .support
                    .as_ref()
                    .unwrap()
                    .domain(),
                input.erase()
            );
        }
        let outputs = replay
            .edges()
            .iter()
            .filter(|edge| edge.from() == id && edge.kind() == EdgeKind::DefinedOn)
            .map(|edge| edge.to())
            .collect::<Vec<_>>();
        assert_eq!(
            outputs,
            if matches!(name, "mass" | "composed") {
                vec![]
            } else {
                vec![symbols.get("position").unwrap()]
            }
        );
    }
}

#[test]
fn source_rejects_wrong_factor_output_units_and_implicit_normalization() {
    for (from, to, diagnostic) in [
        ("measure(velocity)", "measure(other)", "foreign"),
        (
            "density:1/m on position",
            "density:1/m on velocity",
            "remaining factors",
        ),
        (
            "density:1/m on position",
            "density:1/m",
            "remaining factors",
        ),
        (
            "density:1/m on position",
            "density:s/m^2 on position",
            "declared type",
        ),
        ("mass:1=", "mass:1 on position=", "full integral"),
    ] {
        let errors = compile(&SOURCE.replace(from, to)).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(diagnostic)),
            "{to}: {errors:?}"
        );
    }
}

#[test]
fn integral_factor_binding_is_alpha_invariant_and_replay_rejects_stale_input() {
    fn program(source: &str) -> KernelProgram {
        let (transaction, model, _) = compile(source).unwrap().into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
    }
    let original = program(SOURCE);
    let renamed = program(
        &SOURCE
            .replace("phase", "renamed_phase")
            .replace("density", "number_density"),
    );
    assert_eq!(
        eqiora_artifact::StructuralSemanticFingerprint::from_program(&original).unwrap(),
        eqiora_artifact::StructuralSemanticFingerprint::from_program(&renamed).unwrap(),
    );
    let bytes = ModelEnvelope::from_program(&original)
        .unwrap()
        .canonical_json()
        .unwrap();
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let reduction = wire["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find_map(|node| {
            let reduction = node.get_mut("definition")?.get_mut("reduction")?;
            (reduction["kind"] == "volume-integral" && reduction["input"] != reduction["domain"])
                .then_some(reduction)
        })
        .unwrap();
    reduction["input"] = reduction["domain"].clone();
    let replay = ModelEnvelope::from_json(&serde_json::to_vec(&wire).unwrap(), Default::default());
    let errors = match replay {
        Ok(envelope) => envelope.to_program().unwrap_err(),
        Err(error) => vec![error],
    };
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("input support")),
        "{errors:?}"
    );
}

#[test]
fn an_undifferentiated_observable_is_not_silently_treated_as_an_independent_constant() {
    let errors = eqiora_compiler::compile("partial-observable.eqi", "model M() { parameter x:1=2; variable anchor:1; relation r {anchor=x;} observable doubled:1=2*x; observable slope:1=partial(doubled,wrt=x); }").unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("differentiation through its retained definition")),
        "{errors:?}"
    );
}
