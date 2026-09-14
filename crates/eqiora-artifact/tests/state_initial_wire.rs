use eqiora_artifact::{ModelEnvelope, ModelTransactionEnvelope, StructuralSemanticFingerprint};
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{FieldRole, KernelNode};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};

fn program(source: &str) -> KernelProgram {
    let compiled = compile("initial-wire.eqi", source).unwrap().remove(0);
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

const DECAY: &str = "model M() { state x: 1; parameter k: 1/s=1; initial { x=2; } relation dynamics { derivative(x)+k*x=0; } }";

#[test]
fn current_wire_replays_roles_initial_relations_and_before_tick_values() {
    let model = program(DECAY);
    let envelope = ModelEnvelope::from_program(&model).unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["schema"], "eqiora.model-envelope/v26");
    let replay = ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    assert_eq!(replay, model);
    assert!(
        replay.nodes().any(
            |node| matches!(node, KernelNode::Field(field) if field.role() == FieldRole::State)
        )
    );
    assert_eq!(
        replay
            .nodes()
            .filter(|node| matches!(node, KernelNode::Relation(relation) if relation.is_initial()))
            .count(),
        1
    );
    let config = ReferenceConfig::new(0.0, 0.1).unwrap();
    let initial = Interpreter::new().initialize(&replay, config).unwrap();
    assert!(
        (initial
            .fields()
            .values()
            .next()
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value()
            - 2.0)
            .abs()
            < 1e-8
    );
    assert!((initial.derivatives().values().next().unwrap() + 2.0).abs() < 1e-8);
}

#[test]
fn semantic_identity_binds_role_and_initial_mathematics_but_not_numerical_seed() {
    let model = program(DECAY);
    let identity = StructuralSemanticFingerprint::from_program(&model).unwrap();
    assert_eq!(
        identity.generation().as_str(),
        "eqiora.structural-semantic-fingerprint/v21"
    );
    let before = ModelEnvelope::from_program(&model)
        .unwrap()
        .digest()
        .unwrap();
    for guess in [-5.0, 4.0] {
        Interpreter::new()
            .initialize(
                &model,
                ReferenceConfig::new(0.0, 0.1)
                    .unwrap()
                    .with_initial_guess(guess)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            ModelEnvelope::from_program(&model)
                .unwrap()
                .digest()
                .unwrap(),
            before
        );
        assert_eq!(
            StructuralSemanticFingerprint::from_program(&model).unwrap(),
            identity
        );
    }
    let changed = program(&DECAY.replace("x=2", "x=3"));
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&changed).unwrap(),
        identity
    );
    let variable = program("model M() { variable x: 1; relation r { x=0; } }");
    let state = program("model M() { state x: 1; relation r { x=0; } }");
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&variable).unwrap(),
        StructuralSemanticFingerprint::from_program(&state).unwrap()
    );
}

#[test]
fn displaced_versions_payloads_and_unmarked_relations_are_rejected() {
    let model = program(DECAY);
    let wire: serde_json::Value = serde_json::from_slice(
        &ModelEnvelope::from_program(&model)
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    let mut stale = wire.clone();
    stale["schema"] = serde_json::json!("eqiora.model-envelope/v12");
    assert!(
        ModelEnvelope::from_json(&serde_json::to_vec(&stale).unwrap(), Default::default()).is_err()
    );
    let mut payload = wire.clone();
    let field = payload["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["definition"]["kind"] == "field")
        .unwrap();
    field["definition"]["initial"] = serde_json::json!(2.0);
    assert!(
        ModelEnvelope::from_json(&serde_json::to_vec(&payload).unwrap(), Default::default())
            .is_err()
    );
    let mut marker = wire;
    let relation = marker["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["definition"]["kind"] == "relation")
        .unwrap();
    relation["definition"]
        .as_object_mut()
        .unwrap()
        .remove("initial");
    assert!(
        ModelEnvelope::from_json(&serde_json::to_vec(&marker).unwrap(), Default::default())
            .is_err()
    );
    let field = model
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    let mut edit = Transaction::new("forbidden Field value payload");
    edit.push(Op::SetValue {
        target: field.erase(),
        value: eqiora_core::ValueLiteral::try_from(eqiora_core::DynQuantity::new(
            2.0,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))
        .unwrap(),
    });
    assert!(ModelTransactionEnvelope::from_transaction(&edit).is_err());
}

#[test]
fn initial_block_keeps_independent_supports_without_spatial_execution_claim() {
    let source = "model M() { domain a=box(0,1,0,1); domain b=box(0,2,0,1); state x:1; state u:vector<1,2> on a; state v:vector<1,2> on b; initial { x=1; u=0; v=0; } }";
    let model = program(source);
    let envelope = ModelEnvelope::from_program(&model).unwrap();
    let replay = envelope.to_program().unwrap();
    let initial = replay
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) if relation.is_initial() => Some(relation.id()),
            _ => None,
        })
        .unwrap();
    replay.typed_relation_residual(initial).unwrap();
    assert!(
        Interpreter::new()
            .initialize(&replay, ReferenceConfig::new(0.0, 0.1).unwrap())
            .is_err()
    );
    for condition in ["u=v", "u=x", "u=1"] {
        let invalid = source.replace("u=0", condition);
        assert!(
            compile("invalid-initial.eqi", &invalid).is_err(),
            "{condition}"
        );
    }
    let scalar = program("model M() { domain a=box(0,1); state x:1 on a; initial { x=0; } }");
    let errors = Interpreter::new()
        .initialize(&scalar, ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap_err();
    assert!(errors.iter().any(|error| {
        error
            .message()
            .contains("does not realize distributed Fields")
    }));
}

#[test]
fn clocked_state_replay_keeps_pre_first_tick_distinct_from_first_observation() {
    let model = program(
        "model M() { clock tick=periodic(1[s] / 10, phase = 0[s] / 1); state x:1 at tick; initial { pre(x)=3; } relation r at tick { next(x)=pre(x)+1; } }",
    );
    let replay = ModelEnvelope::from_program(&model)
        .unwrap()
        .to_program()
        .unwrap();
    let config = ReferenceConfig::new(0.0, 0.1).unwrap();
    let initial = Interpreter::new().initialize(&replay, config).unwrap();
    let (&field, value) = initial.fields().iter().next().unwrap();
    assert_eq!(value.real_scalar_value().unwrap().value(), 3.0);
    let observed = Interpreter::new().run(&replay, config).unwrap();
    assert_eq!(observed.last_value(field).unwrap().value(), 4.0);
}
