//! Process identity, physical coefficient types and exact offline Model replay.
use eqiora_artifact::{ModelEnvelope, StructuralSemanticFingerprint};
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    ModelView,
    kernel::{
        ActivationDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, NoiseProcess, RelationDef,
        RelationMeaning, StochasticCalculus, SymbolRef,
    },
};
use eqiora_sem::KernelProgram;

fn relaxation(
    calculus: StochasticCalculus,
    covariance: f64,
    diffusion_order: i32,
) -> Result<KernelProgram, Vec<Diagnostic>> {
    let state = Id::new();
    let noise = Id::new();
    let law = Id::new();
    let noise_activation = Id::new();
    let law_activation = Id::new();
    let model = OntologyId::new();
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let volts = DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).unwrap();
    let mut process = ExprDagBuilder::new();
    let covariance = process
        .constant(DynQuantity::new(covariance, DimExponents::DIMENSIONLESS))
        .unwrap();
    let mut coefficients = ExprDagBuilder::new();
    let value = coefficients.symbol(SymbolRef::Field(state)).unwrap();
    let mean = coefficients.constant(DynQuantity::new(1., volts)).unwrap();
    let rate = coefficients
        .constant(DynQuantity::new(2., time.pow(-1, 1).unwrap()))
        .unwrap();
    let sigma = coefficients
        .constant(DynQuantity::new(
            0.5,
            time.pow(-diffusion_order, 2).unwrap(),
        ))
        .unwrap();
    let difference = coefficients.sub(mean, value).unwrap();
    let drift = coefficients.mul(rate, difference).unwrap();
    // Multiplicative noise distinguishes the written Ito and Stratonovich laws.
    let diffusion = coefficients.mul(sigma, value).unwrap();
    let nodes: Vec<KernelNode> = vec![
        FieldDef::new(
            state,
            ValueType::scalar(ScalarDomain::Real, volts).unwrap(),
            FieldRole::State,
        )
        .into(),
        RelationDef::noise(
            noise,
            process.finish([covariance]).unwrap(),
            NoiseProcess::Wiener,
            covariance,
            false,
        )
        .unwrap()
        .into(),
        RelationDef::stochastic(
            law,
            coefficients.finish([drift, diffusion]).unwrap(),
            state,
            Some(calculus),
            Some(drift),
            vec![(noise, diffusion)],
            vec![],
        )
        .unwrap()
        .into(),
        ActivationDef::continuous(noise_activation).into(),
        ActivationDef::continuous(law_activation).into(),
    ];
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("stochastic physical law");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (noise_activation.erase(), noise.erase(), EdgeKind::Activates),
        (law_activation.erase(), law.erase(), EdgeKind::Activates),
        (law.erase(), noise.erase(), EdgeKind::DependsOn),
        (law.erase(), state.erase(), EdgeKind::DependsOn),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model)
}

#[test]
fn exact_process_and_calculus_reopen_without_a_solver() {
    let mut fingerprints = std::collections::HashSet::new();
    for calculus in [StochasticCalculus::Ito, StochasticCalculus::Stratonovich] {
        let program = relaxation(calculus, 1., 1).unwrap();
        let bytes = ModelEnvelope::from_program(&program)
            .unwrap()
            .canonical_json()
            .unwrap();
        let decoded = ModelEnvelope::from_json(&bytes, Default::default()).unwrap();
        assert_eq!(decoded.to_program().unwrap(), program);
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert!(
            fingerprints.insert(StructuralSemanticFingerprint::from_program(&program).unwrap())
        );
        for node in program.nodes() {
            if let KernelNode::Relation(relation) = node {
                assert!(
                    program
                        .numerical_residuals(relation.id().erase())
                        .unwrap_err()
                        .message()
                        .contains("explicit stochastic execution")
                );
            }
        }
        assert_eq!(
            ModelEnvelope::from_program(&program)
                .unwrap()
                .canonical_json()
                .unwrap(),
            bytes
        );
        // Alpha-renaming fresh native occurrences cannot change mathematical structure.
        assert_eq!(
            StructuralSemanticFingerprint::from_program(&program).unwrap(),
            StructuralSemanticFingerprint::from_program(&relaxation(calculus, 1., 1).unwrap())
                .unwrap()
        );
    }
}

#[test]
fn diffusion_needs_square_root_time_and_covariance_cannot_be_negative() {
    for (covariance, order, reason) in [(-1., 1, "positive semidefinite"), (1., 2, "time exponent")]
    {
        let errors = relaxation(StochasticCalculus::Ito, covariance, order).unwrap_err();
        assert!(
            errors.iter().any(|error| error.message().contains(reason)),
            "{errors:?}"
        );
    }
}

#[test]
fn process_replay_rejects_foreign_noise_and_changed_coefficient_attribution() {
    let program = relaxation(StochasticCalculus::Ito, 1., 1).unwrap();
    let bytes = ModelEnvelope::from_program(&program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for foreign in [false, true] {
        let mut mutant = original.clone();
        let meaning = &mut mutant["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["definition"]["meaning"]["kind"] == "stochastic")
            .unwrap()["definition"]["meaning"];
        if foreign {
            meaning["diffusions"][0]["group"]["ulid"] = serde_json::json!(
                Id::<eqiora_core::entity::kinds::Relation>::new()
                    .ulid()
                    .to_string()
            );
        } else {
            meaning["diffusions"][0]["coefficient"] = meaning["drift"].clone();
        }
        let bytes = serde_json::to_vec(&mutant).unwrap();
        match ModelEnvelope::from_json(&bytes, Default::default()) {
            Err(_) => {}
            Ok(envelope) => assert!(envelope.to_program().is_err()),
        }
    }
    assert!(program.nodes().any(|node| matches!(node,KernelNode::Relation(relation) if matches!(relation.meaning(),RelationMeaning::Noise { .. }))));
}
