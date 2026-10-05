//! Source-derived witnesses must replay every authored and companion equation.
use eqiora_artifact::{
    GeneralImplicitTimeLoweringEnvelopeV2, ImplicitTimeCheckpointEnvelopeV1, ModelEnvelope,
    TimeLoweringEnvelopeV2,
};
use eqiora_compiler::compile;
use eqiora_core::{Id, entity::kinds};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_sem::KernelProgram;
use eqiora_time::{
    ConstantDerivativeMatrixProof, DaeVariableKind, GeneralImplicitLoweringProof,
    GeneralImplicitReason, TimeLoweringProof,
};

fn oscillator(parameter_mass: bool) -> (KernelProgram, Id<kinds::Relation>, Id<kinds::Field>) {
    let mass = if parameter_mass { "mass" } else { "1" };
    let source = format!(
        "model Oscillator() {{ parameter mass:1=1; state x:m; initial {{ x=1[m]; derivative(x)=2[m/s]; }} relation motion {{ {mass}*derivative(derivative(x))+1[1/s]*derivative(x)+4[1/s^2]*x=0[m/s^2]; }} }}"
    );
    let (transaction, model, symbols) = compile("oscillator.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let relation = symbols.get("motion").unwrap().downcast().unwrap();
    let field = symbols.get("x").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        relation,
        field,
    )
}

#[test]
fn constant_projection_replays_source_order_and_companion_matrix() {
    let (program, relation, field) = oscillator(false);
    let model = ModelEnvelope::from_program(&program).unwrap();
    // Residuals [x'' + x'/s + 4*x/s², D(x) - velocity], rates [D(x), D(velocity)].
    let make = |coefficients| {
        TimeLoweringProof::new(
            relation,
            vec![(field, 0), (field, 1)],
            ConstantDerivativeMatrixProof::new(2, coefficients).unwrap(),
        )
        .unwrap()
    };
    let proof = make(vec![0., 1., 1., 0.]);
    let envelope = TimeLoweringEnvelopeV2::from_proof(&model, &program, &proof).unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let decoded = TimeLoweringEnvelopeV2::from_json(&bytes, Default::default()).unwrap();
    decoded.validate_against(&model, &program).unwrap();
    assert_eq!(decoded.proof().unwrap(), proof);
    // Both forged matrices have full rank; rank/class alone cannot prove linkage.
    for coefficients in [vec![0., 2., 1., 0.], vec![0., 1., 2., 0.]] {
        assert!(TimeLoweringEnvelopeV2::from_proof(&model, &program, &make(coefficients)).is_err());
    }
}

#[test]
fn implicit_checkpoint_replays_lower_derivatives_and_companion_residuals() {
    let (program, relation, field) = oscillator(true);
    let model = ModelEnvelope::from_program(&program).unwrap();
    let proof = GeneralImplicitLoweringProof::new(
        relation,
        vec![(field, 0), (field, 1)],
        vec![DaeVariableKind::Differential; 2],
        GeneralImplicitReason::NonconstantDerivativeJacobian,
    )
    .unwrap();
    let lowering =
        GeneralImplicitTimeLoweringEnvelopeV2::from_proof(&model, &program, &proof).unwrap();
    let decoded = GeneralImplicitTimeLoweringEnvelopeV2::from_json(
        &lowering.canonical_json().unwrap(),
        Default::default(),
    )
    .unwrap();
    decoded.validate_against(&model, &program).unwrap();
    let checkpoint = ImplicitTimeCheckpointEnvelopeV1::from_accepted_pair(
        &decoded,
        &program,
        0.,
        vec![1., 2.],
        vec![2., -6.],
        0.,
    )
    .unwrap();
    let restored = ImplicitTimeCheckpointEnvelopeV1::from_json(
        &checkpoint.canonical_json().unwrap(),
        Default::default(),
    )
    .unwrap();
    restored.validate_against(&decoded, &program).unwrap();
    // The authored equation still vanishes, but D(x) != velocity.
    assert!(
        ImplicitTimeCheckpointEnvelopeV1::from_accepted_pair(
            &decoded,
            &program,
            0.,
            vec![1., 2.],
            vec![3., -6.],
            0.
        )
        .is_err()
    );
    let mut forged: serde_json::Value =
        serde_json::from_slice(&checkpoint.canonical_json().unwrap()).unwrap();
    forged["derivative"][0] = 3.into();
    let forged = ImplicitTimeCheckpointEnvelopeV1::from_json(
        &serde_json::to_vec(&forged).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(forged.validate_against(&decoded, &program).is_err());
}

#[test]
fn coordinate_replay_has_no_fixture_order_ceiling() {
    for order in [1_u32, 2, 3, 5, 6, 9, 16] {
        let derivative = (0..order).fold("x".to_owned(), |value, _| format!("derivative({value})"));
        let source = format!(
            "model Evolution() {{ state x:m; relation motion {{ {derivative}+1[1/s^{order}]*x=0[m/s^{order}]; }} }}"
        );
        let (transaction, model_id, symbols) = compile("order.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let field = symbols.get("x").unwrap().downcast().unwrap();
        let relation = symbols.get("motion").unwrap().downcast().unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let replay = ModelEnvelope::from_json(&model.canonical_json().unwrap(), Default::default())
            .unwrap()
            .to_program()
            .unwrap();
        let n = order as usize;
        let mut coefficients = vec![0.; n * n];
        coefficients[n - 1] = 1.;
        for row in 1..n {
            coefficients[row * n + row - 1] = 1.;
        }
        let proof = TimeLoweringProof::new(
            relation,
            (0..order).map(|k| (field, k)).collect(),
            ConstantDerivativeMatrixProof::new(n, coefficients).unwrap(),
        )
        .unwrap();
        let lowering = TimeLoweringEnvelopeV2::from_proof(&model, &replay, &proof).unwrap();
        let restored = TimeLoweringEnvelopeV2::from_json(
            &lowering.canonical_json().unwrap(),
            Default::default(),
        )
        .unwrap();
        restored.validate_against(&model, &replay).unwrap();
        assert_eq!(restored.proof().unwrap(), proof);
    }
}

#[test]
fn current_model_epoch_requires_explicit_positive_derivative_orders() {
    let (program, _, _) = oscillator(false);
    let model = ModelEnvelope::from_program(&program).unwrap();
    let bytes = model.canonical_json().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(original["schema"], "eqiora.model-envelope/v41");
    ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    let mut old = original.clone();
    old["schema"] = "eqiora.model-envelope/v40".into();
    assert!(
        ModelEnvelope::from_json(&serde_json::to_vec(&old).unwrap(), Default::default()).is_err()
    );

    fn change_orders(value: &mut serde_json::Value, order: Option<u32>) -> usize {
        match value {
            serde_json::Value::Object(object) => {
                if object.get("kind").and_then(serde_json::Value::as_str) == Some("derivative") {
                    if let Some(order) = order {
                        object.insert("order".to_owned(), order.into());
                    } else {
                        object.remove("order");
                    }
                    1
                } else {
                    object
                        .values_mut()
                        .map(|value| change_orders(value, order))
                        .sum()
                }
            }
            serde_json::Value::Array(values) => values
                .iter_mut()
                .map(|value| change_orders(value, order))
                .sum(),
            _ => 0,
        }
    }
    for order in [None, Some(0)] {
        let mut forged = original.clone();
        assert!(change_orders(&mut forged, order) >= 2);
        assert!(
            ModelEnvelope::from_json(&serde_json::to_vec(&forged).unwrap(), Default::default())
                .is_err()
        );
    }
}
