//! Exact pure-operator constraints survive the existing model and transaction lifecycle.

use eqiora_artifact::{
    ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope, StructuralSemanticFingerprint,
};
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::pure_operator::{CalculusBuilder, CalculusNode, PureValueClass};
use eqiora_schema::{
    ModelView,
    kernel::{
        ActivationDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, RelationDef, SymbolRef,
    },
};
use eqiora_sem::KernelProgram;
use serde_json::{Value, json};

fn program(dimension: DimExponents, constrained: bool) -> KernelProgram {
    let field = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let model = OntologyId::new();
    let class = if constrained {
        PureValueClass::from_value_type(ValueType::scalar(ScalarDomain::Real, dimension).unwrap())
            .unwrap()
    } else {
        PureValueClass::invariant_scalar()
    };
    let mut calculus = CalculusBuilder::new([class.clone()], class).unwrap();
    let component = calculus
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let definition = calculus.finish(component).unwrap();
    let mut dag = ExprDagBuilder::new();
    let output = dag.symbol(SymbolRef::Field(field)).unwrap();
    let argument = dag.constant(DynQuantity::new(2.0, dimension)).unwrap();
    let selected = dag.pure_operator(&definition, [argument]).unwrap();
    let nodes: Vec<KernelNode> = vec![
        FieldDef::new(
            field,
            ValueType::scalar(ScalarDomain::Real, dimension).expect("valid fixture scalar type"),
            FieldRole::Variable,
        )
        .into(),
        RelationDef::new(relation, dag.finish([output, selected]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("constrained operator fixture");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (activation.erase(), relation.erase(), EdgeKind::Activates),
        (relation.erase(), field.erase(), EdgeKind::DependsOn),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn relation_expression(json: &mut Value) -> &mut Value {
    &mut json["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["definition"]["kind"] == "relation")
        .unwrap()["definition"]["expression"]
}

fn length() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
}

#[test]
fn constraints_round_trip_through_model_and_transaction() {
    for constrained in [false, true] {
        let original = program(length(), constrained);
        let envelope = ModelEnvelope::from_program(&original).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let mut json: Value = serde_json::from_slice(&bytes).unwrap();
        let definition = &relation_expression(&mut json)["definitions"][0];
        // The constrained identity has one complete ValueType; the generic
        // template keeps explicitly unresolved constraints. Both apply to the
        // same real, length-valued operand, but they are different contracts.
        let expected = if constrained {
            json!({
                "kind": "exact",
                "value_type": {
                    "domain": "real",
                    "dimension": [[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                    "shape": [], "frame": "invariant", "array_rank": 0,
                    "basis": {"kind": "ordinary"}
                }
            })
        } else {
            json!({"kind": "invariant-scalar", "dimension": null, "scalar_domain": null})
        };
        assert_eq!(definition["formals"][0], expected);
        assert_eq!(definition["result"], expected);
        let decoded = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()).unwrap();
        assert_eq!(decoded.to_program().unwrap(), original);
        let (seed, model) = decoded.to_transaction().unwrap();
        let wire = ModelTransactionEnvelope::from_transaction(&seed)
            .unwrap()
            .canonical_json()
            .unwrap();
        let replay = ModelTransactionEnvelope::from_json(&wire, ModelDecoderLimits::default())
            .unwrap()
            .to_transaction()
            .unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(replay).unwrap();
        assert_eq!(
            KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
            original
        );
    }
}

#[test]
fn constraint_presence_changes_identity_without_changing_operand_types() {
    let unconstrained = program(length(), false);
    let constrained = program(length(), true);
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&unconstrained).unwrap(),
        StructuralSemanticFingerprint::from_program(&constrained).unwrap()
    );
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    assert_ne!(
        StructuralSemanticFingerprint::from_program(&constrained).unwrap(),
        StructuralSemanticFingerprint::from_program(&program(time, true)).unwrap()
    );
}

#[test]
fn malformed_missing_and_tampered_constraints_are_rejected() {
    let bytes = ModelEnvelope::from_program(&program(length(), true))
        .unwrap()
        .canonical_json()
        .unwrap();
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    for field in ["formals", "result"] {
        for mutation in 0..4 {
            let mut invalid = original.clone();
            let definition = &mut relation_expression(&mut invalid)["definitions"][0];
            let class = if field == "formals" {
                &mut definition[field][0]["value_type"]
            } else {
                &mut definition[field]["value_type"]
            };
            match mutation {
                0 => {
                    assert!(class.as_object_mut().unwrap().remove("dimension").is_some());
                }
                1 => {
                    class["dimension"][1] = json!([1, 0]);
                }
                2 => {
                    class["dimension"][1] = json!([2, 2]);
                }
                _ => {
                    class["dimension"] = Value::Null;
                }
            }
            assert!(
                ModelEnvelope::from_json(
                    &serde_json::to_vec(&invalid).unwrap(),
                    ModelDecoderLimits::default()
                )
                .is_err(),
                "{field} mutation {mutation}"
            );
        }
    }
}

#[test]
fn scalar_domain_constraints_are_mandatory_checked_and_identity_bound() {
    let original: Value = serde_json::from_slice(
        &ModelEnvelope::from_program(&program(length(), true))
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    for formal in [false, true] {
        for domain in [
            None,
            Some(Value::Null),
            Some(json!("integer")),
            Some(json!("boolean")),
            Some(json!("complex")),
        ] {
            let mut invalid = original.clone();
            let definition = &mut relation_expression(&mut invalid)["definitions"][0];
            let class = if formal {
                &mut definition["formals"][0]["value_type"]
            } else {
                &mut definition["result"]["value_type"]
            };
            match domain {
                None => {
                    assert!(class.as_object_mut().unwrap().remove("domain").is_some());
                }
                Some(value) => {
                    class["domain"] = value;
                }
            }
            assert!(
                ModelEnvelope::from_json(
                    &serde_json::to_vec(&invalid).unwrap(),
                    ModelDecoderLimits::default()
                )
                .is_err()
            );
        }
    }
}

#[test]
fn exact_and_generic_contracts_reject_missing_payloads_and_obsolete_constraints() {
    for constrained in [false, true] {
        let original: Value = serde_json::from_slice(
            &ModelEnvelope::from_program(&program(length(), constrained))
                .unwrap()
                .canonical_json()
                .unwrap(),
        )
        .unwrap();
        for formal in [false, true] {
            let fields: &[&str] = if constrained {
                &["value_type"]
            } else {
                &["dimension", "scalar_domain"]
            };
            for field in fields {
                let mut invalid = original.clone();
                let definition = &mut relation_expression(&mut invalid)["definitions"][0];
                let class = if formal {
                    &mut definition["formals"][0]
                } else {
                    &mut definition["result"]
                };
                assert!(class.as_object_mut().unwrap().remove(*field).is_some());
                assert!(
                    ModelEnvelope::from_json(
                        &serde_json::to_vec(&invalid).unwrap(),
                        ModelDecoderLimits::default()
                    )
                    .is_err()
                );
            }
            if constrained {
                let mut invalid = original.clone();
                let definition = &mut relation_expression(&mut invalid)["definitions"][0];
                let class = if formal {
                    &mut definition["formals"][0]
                } else {
                    &mut definition["result"]
                };
                // Fully specified constraints must not retain the displaced
                // generic encoding, even when their mathematical meaning agrees.
                *class = json!({"kind":"invariant-scalar", "scalar_domain":"real",
                    "dimension":[[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]]});
                assert!(
                    ModelEnvelope::from_json(
                        &serde_json::to_vec(&invalid).unwrap(),
                        ModelDecoderLimits::default()
                    )
                    .is_err()
                );
            }
        }
    }
}
