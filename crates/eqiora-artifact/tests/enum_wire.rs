//! Nominal enum declarations and exact tags survive the ordinary artifact lifecycle.
use eqiora_artifact::{
    ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope, StructuralSemanticFingerprint,
};
use eqiora_core::{Id, OntologyId, ValueLiteral};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, EnumDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, ParameterDef,
        RelationDef,
    },
};
use eqiora_sem::KernelProgram;
use serde_json::{Value, json};

fn fixture(
    members: [&str; 2],
    tag: u32,
    shared: bool,
) -> (KernelProgram, Id<eqiora_core::entity::kinds::Parameter>) {
    let first = EnumDef::new(Id::new(), members.map(str::to_owned)).unwrap();
    let second = EnumDef::new(Id::new(), members.map(str::to_owned)).unwrap();
    let parameter = Id::new();
    let field = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let mut dag = ExprDagBuilder::new();
    let member = dag
        .constant(ValueLiteral::enum_value(first.value_type(), 0).unwrap())
        .unwrap();
    let zero = dag
        .constant(eqiora_core::DynQuantity::new(
            0.0,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))
        .unwrap();
    let nodes: Vec<KernelNode> = vec![
        RelationDef::new(relation, dag.finish([member, member, zero, zero]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
        first.clone().into(),
        second.clone().into(),
        ParameterDef::new(
            parameter,
            ValueLiteral::enum_value(first.value_type(), tag).unwrap(),
        )
        .into(),
        FieldDef::new(
            field,
            if shared {
                first.value_type()
            } else {
                second.value_type()
            },
            FieldRole::State,
        )
        .into(),
    ];
    let model = OntologyId::<Model>::new();
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("nominal enum fixture");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: eqiora_graph::EdgeKind::Activates,
    });
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        parameter,
    )
}

#[test]
fn enum_zero_tag_and_edited_current_tag_round_trip_without_numeric_payloads() {
    let (program, parameter) = fixture(["Off", "On"], 0, true);
    let envelope = ModelEnvelope::from_program(&program).unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let value = &json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["definition"]["kind"] == "parameter")
        .unwrap()["definition"]["value"];
    assert_eq!(value["components"], json!({"kind":"enum","tag":0}));
    assert_eq!(value["value_type"]["domain"], json!("enum"));
    assert_eq!(value["value_type"]["basis"]["count"], json!(2));
    let restored = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()).unwrap();
    assert_eq!(restored.to_program().unwrap(), program);
    let (seed, model) = restored.to_transaction().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(seed).unwrap();
    // Select the exact Parameter type, not whichever equal-looking declaration sorts first.
    let parameter_type = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(value) if value.id() == parameter => {
                Some(value.value().value_type().clone())
            }
            _ => None,
        })
        .unwrap();
    let mut edit = Transaction::new("enum tag edit");
    edit.push(Op::SetValue {
        target: parameter.erase(),
        value: ValueLiteral::enum_value(parameter_type, 1).unwrap(),
    });
    let wire = ModelTransactionEnvelope::from_transaction(&edit)
        .unwrap()
        .canonical_json()
        .unwrap();
    let replay = ModelTransactionEnvelope::from_json(&wire, ModelDecoderLimits::default())
        .unwrap()
        .to_transaction()
        .unwrap();
    store.commit(replay).unwrap();
    let edited = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    assert_eq!(
        edited.typed_value(parameter.erase()).unwrap().enum_tag(),
        Some(1)
    );
    let original_literal = edited
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(value) if value.id() == parameter => Some(value.value()),
            _ => None,
        })
        .unwrap();
    assert_eq!(original_literal.enum_tag(), Some(0));
    let current = ModelEnvelope::from_program(&edited).unwrap();
    assert_eq!(
        ModelEnvelope::from_json(
            &current.canonical_json().unwrap(),
            ModelDecoderLimits::default()
        )
        .unwrap()
        .to_program()
        .unwrap(),
        edited
    );
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&program).unwrap(),
        StructuralSemanticFingerprint::from_program(&edited).unwrap()
    );
}

#[test]
fn enum_identity_binds_order_tag_and_shared_versus_distinct_nominal_owners() {
    let mut fingerprints = std::collections::BTreeSet::new();
    for members in [["Off", "On"], ["On", "Off"]] {
        for tag in [0, 1] {
            for shared in [false, true] {
                assert!(
                    fingerprints.insert(
                        StructuralSemanticFingerprint::from_program(
                            &fixture(members, tag, shared).0
                        )
                        .unwrap()
                    )
                );
            }
        }
    }
}

#[test]
fn enum_reopen_rejects_numeric_spellings_and_wrong_nominal_profiles() {
    let mut original: Value = serde_json::from_slice(
        &ModelEnvelope::from_program(&fixture(["Off", "On"], 0, true).0)
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    // Decoder admission must validate nominal literals even outside the root closure.
    // Fingerprint positives above instead use a fully reachable expression DAG.
    let relation = original["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["definition"]["kind"] == "relation")
        .unwrap();
    relation["definition"]["expression"]["roots"]
        .as_array_mut()
        .unwrap()
        .drain(..2);
    // Removing an equality also removes its descriptor; keep the nominal-literal
    // falsifier focused on unreachable expression payload admission.
    relation["definition"]["meaning"]["conditions"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    let unused = ModelEnvelope::from_json(
        &serde_json::to_vec(&original).unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap()
    .to_program()
    .unwrap();
    assert!(
        StructuralSemanticFingerprint::from_program(&unused)
            .unwrap_err()
            .message()
            .contains("unreachable expression nodes")
    );
    for mutation in 0..10 {
        let mut invalid = original.clone();
        let nodes = invalid["nodes"].as_array_mut().unwrap();
        let parameter = nodes
            .iter_mut()
            .find(|node| node["definition"]["kind"] == "parameter")
            .unwrap();
        let value = &mut parameter["definition"]["value"];
        match mutation {
            0 => value["components"] = json!({"kind":"zero"}),
            1 => value["components"] = json!({"kind":"integer","values":[1]}),
            2 => value["components"] = json!({"kind":"dense","values":[[1.0,0.0]]}),
            3 => value["components"]["tag"] = json!(2),
            4 => value["value_type"]["basis"]["count"] = json!(3),
            5 => {
                value["value_type"]["basis"]["definition"]["ulid"] = json!(
                    Id::<eqiora_core::entity::kinds::Enum>::new()
                        .ulid()
                        .to_string()
                )
            }
            6 => value["value_type"]["basis"] = json!({"kind":"ordinary"}),
            7 => value["value_type"]["dimension"][1] = json!([1, 1]),
            8 => {
                let field = nodes
                    .iter_mut()
                    .find(|node| node["definition"]["kind"] == "field")
                    .unwrap();
                field["definition"]["value_type"]["basis"]["count"] = json!(3);
            }
            _ => {
                let relation = nodes
                    .iter_mut()
                    .find(|node| node["definition"]["kind"] == "relation")
                    .unwrap();
                relation["definition"]["expression"]["nodes"][0]["value"]["value_type"]["basis"]
                    ["count"] = json!(3);
            }
        }
        let rejected = match ModelEnvelope::from_json(
            &serde_json::to_vec(&invalid).unwrap(),
            ModelDecoderLimits::default(),
        ) {
            Err(_) => true,
            Ok(envelope) => envelope.to_program().is_err(),
        };
        assert!(rejected, "mutation {mutation}");
    }
}
