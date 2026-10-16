//! Ordinary source authoring must retain process meaning through offline replay.
use eqiora_artifact::{ModelEnvelope, ModelTransactionEnvelope};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::{KernelNode, RelationMeaning};
use eqiora_sem::KernelProgram;

const RELAXATION: &str = r#"
model Relaxation(
  parameter rate: 1/s = 2[1/s],
  parameter mean: V = 1[V],
  parameter sigma: V/s^(1/2) = 0.5[V/s^(1/2)]
) {
  noise driving: wiener(covariance = 1);
  state value: V;
  initial { value = 2[V]; }
  law relaxation for value {
    calculus ito;
    drift rate * (mean - value);
    diffusion driving = sigma;
  }
}
"#;

const BIRTH_DEATH: &str = r#"
model Population(parameter birth: 1/s = 2[1/s], parameter death: 1/s = 3[1/s]) {
  state count: integer;
  initial { count = 0; }
  noise events: poisson(intensity = [birth, if count > 0 then death * to_real(count) else 0[1/s]]);
  law population for count { jump events = [1, -1]; }
}
"#;

fn program(source: &str) -> Result<KernelProgram, String> {
    let compiled = eqiora_compiler::compile("process.eqi", source)
        .map_err(|e| format!("{e:?}"))?
        .remove(0);
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).map_err(|e| format!("{e:?}"))?;
    KernelProgram::from_snapshot(&store.snapshot(), model).map_err(|e| format!("{e:?}"))
}

#[test]
fn continuous_and_integer_jump_sources_reopen_without_random_evaluation() {
    for source in [RELAXATION, BIRTH_DEATH] {
        let program = program(source).unwrap();
        let envelope = ModelEnvelope::from_program(&program).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let mut displaced: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        displaced["schema"] = serde_json::json!("eqiora.model-envelope/v46");
        assert!(
            ModelEnvelope::from_json(&serde_json::to_vec(&displaced).unwrap(), Default::default())
                .is_err()
        );
        let reopened = ModelEnvelope::from_json(&bytes, Default::default())
            .unwrap()
            .to_program()
            .unwrap();
        assert_eq!(program, reopened);
        let (transaction, model) = envelope.to_transaction().unwrap();
        let transaction_bytes = ModelTransactionEnvelope::from_transaction(&transaction)
            .unwrap()
            .canonical_json()
            .unwrap();
        let restored = ModelTransactionEnvelope::from_json(&transaction_bytes, Default::default())
            .unwrap()
            .to_transaction()
            .unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(restored).unwrap();
        assert_eq!(
            KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
            program
        );
        for node in program.nodes() {
            if let KernelNode::Relation(relation) = node
                && matches!(
                    relation.meaning(),
                    RelationMeaning::Noise { .. } | RelationMeaning::Stochastic { .. }
                )
            {
                for _ in 0..2 {
                    let error = program
                        .numerical_residuals(relation.id().erase())
                        .unwrap_err();
                    assert!(
                        error.message().contains("explicit stochastic execution"),
                        "{error:?}"
                    );
                }
            }
        }
        assert_eq!(
            bytes,
            ModelEnvelope::from_program(&program)
                .unwrap()
                .canonical_json()
                .unwrap()
        );
    }
}

#[test]
fn source_rejects_wrong_process_units_identity_and_calculus() {
    for (source, reason) in [
        (
            RELAXATION.replace("covariance = 1", "covariance = -1"),
            "positive semidefinite",
        ),
        (RELAXATION.replace("V/s^(1/2)", "V/s"), "time exponent"),
        (
            RELAXATION.replace("diffusion driving", "diffusion foreign"),
            "foreign",
        ),
        (
            RELAXATION.replace("calculus ito;", "calculus automatic;"),
            "calculus",
        ),
        (
            RELAXATION.replace("wiener(covariance = 1)", "poisson(intensity = 1[1/s])"),
            "interchanged",
        ),
        (
            BIRTH_DEATH.replace("2[1/s]", "-2[1/s]").replace(
                "[birth, if count > 0 then death * to_real(count) else 0[1/s]]",
                "[birth, death]",
            ),
            "nonnegative",
        ),
    ] {
        let error = program(&source).unwrap_err();
        assert!(error.contains(reason), "expected {reason}: {error}");
    }
}

#[test]
fn finite_correlated_channels_and_nested_instances_retain_nominal_groups() {
    let correlated = RELAXATION
        .replace("covariance = 1", "covariance = [[1, 0.5], [0.5, 1]]")
        .replace(
            "diffusion driving = sigma",
            "diffusion driving = [sigma, sigma]",
        );
    program(&correlated).unwrap();
    assert!(
        program(&correlated.replace("[sigma, sigma]", "sigma"))
            .unwrap_err()
            .contains("channel shape")
    );
    assert!(
        program(&correlated.replace("0.5", "2"))
            .unwrap_err()
            .contains("positive semidefinite")
    );

    let component = RELAXATION
        .replace("model Relaxation", "component Relaxation")
        .replace("covariance = 1", "covariance = 1, independent = true");
    let source = format!(
        "{component} model Pair() {{ instance left: Relaxation(); instance right: Relaxation(); }}"
    );
    let model = program(&source).unwrap();
    let groups = model
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation)
                if matches!(relation.meaning(), RelationMeaning::Noise { .. }) =>
            {
                Some(relation.id().erase())
            }
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(groups.len(), 2);
    let targets = model
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) => match relation.meaning() {
                RelationMeaning::Stochastic { diffusions, .. } => Some(diffusions[0].0.erase()),
                _ => None,
            },
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(targets, groups);
    assert!(
        program(&source.replace(", independent = true", ""))
            .unwrap_err()
            .contains("explicit independence")
    );
}

#[test]
fn stochastic_state_cannot_gain_a_classical_time_derivative_or_second_owner() {
    for addition in [
        "observable velocity: V/s = derivative(value);",
        "relation velocity { derivative(value) = 0[V/s]; }",
        "law other for value { calculus ito; drift rate*(mean-value); diffusion driving=sigma; }",
    ] {
        let source =
            RELAXATION.replacen("state value: V;", &format!("state value: V; {addition}"), 1);
        let error = program(&source).unwrap_err();
        assert!(
            error.contains("ordinary physical-time derivative")
                || error.contains("conflicting stochastic Laws"),
            "{error}"
        );
    }
}
