//! Exact storage correspondence is admitted independently of balance construction.

use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora::kernel::{ConservationTerms, KernelNode, RelationDef, RelationMeaning};
use eqiora::sem::KernelProgram;

const HEAT: &str = r#"
model HeatedBody() {
  domain body = box(0, 1, 0, 1, 0, 1);
  state temperature: K on body;
  parameter capacity: kg / m / s ^ 2 / K = 1000000;
  parameter conductivity: kg * m / s ^ 3 / K = 10;
  parameter heating: kg / m / s ^ 3 = 1000;
  initial { temperature = 300 [K]; }
  law heat_balance on body {
    storage capacity * temperature;
    flux -conductivity * grad(temperature);
    source heating;
  }
}
"#;

#[test]
fn three_dimensional_heat_storage_admits_and_replays_exact_physical_terms() {
    let model = compile("heat.eqi", HEAT).unwrap().pop().unwrap();
    let (transaction, model_id, _) = model.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let envelope = eqiora::artifact::ModelEnvelope::from_program(&kernel).unwrap();
    let replayed = eqiora::artifact::ModelEnvelope::from_json(
        &envelope.canonical_json().unwrap(),
        eqiora::artifact::ModelDecoderLimits::default(),
    )
    .unwrap();
    assert_eq!(envelope.digest().unwrap(), replayed.digest().unwrap());
    let replay = replayed.to_program().unwrap();
    for program in [&kernel, &replay] {
        let relation = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(relation)
                    if matches!(relation.meaning(), RelationMeaning::Conservation(_)) =>
                {
                    Some(relation)
                }
                _ => None,
            })
            .unwrap();
        let RelationMeaning::Conservation(terms) = relation.meaning() else {
            unreachable!()
        };
        let (storage, accumulation) = terms.storage().unwrap();
        relation
            .expression()
            .verify_time_derivative(storage, accumulation)
            .unwrap();
    }
}

#[test]
fn native_storage_tampering_fails_despite_valid_balance_and_matching_dimensions() {
    let source = r#"model M() {
      domain body=box(0,1);
      state q:1 on body;
      parameter diffusivity:m^2/s=1;
      initial {q=1;}
      law balance on body {storage q*q; flux -diffusivity*grad(q); source 1 [1/s];}
    }"#;
    let model = compile("storage.eqi", source).unwrap().pop().unwrap();
    let (original, model_id, symbols) = model.into_parts();
    let q = symbols.get("q").unwrap();
    for tamper in [false, true] {
        let mut transaction = Transaction::new("storage correspondence falsifier");
        for op in original.ops() {
            let mut op = op.clone();
            if tamper
                && let Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } = &op
                && let RelationMeaning::Conservation(terms) = relation.meaning()
            {
                let (storage, accumulation) = terms.storage().unwrap();
                let Some(eqiora::kernel::ExprNode::Mul(value, _)) =
                    relation.expression().node(storage)
                else {
                    panic!("q*q storage")
                };
                assert!(
                    matches!(relation.expression().node(*value), Some(eqiora::kernel::ExprNode::Symbol(eqiora::kernel::SymbolRef::Field(id))) if id.erase() == q)
                );
                let terms = ConservationTerms::new(
                    Some((*value, accumulation)),
                    terms.flux(),
                    terms.source(),
                );
                op = Op::DefineKernelNode {
                    node: RelationDef::conservation(
                        relation.id(),
                        relation.expression().clone(),
                        terms,
                    )
                    .unwrap()
                    .into(),
                };
            }
            transaction.push(op);
        }
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let result = KernelProgram::from_snapshot(&store.snapshot(), model_id);
        if tamper {
            let errors = result.unwrap_err();
            assert!(
                errors.iter().any(|error| error
                    .to_string()
                    .contains("storage accumulation correspondence")),
                "{errors:?}"
            );
        } else {
            result.unwrap();
        }
    }
}

#[test]
fn native_canceled_storage_still_requires_continuous_state_fields() {
    use eqiora::kernel::{ExprDagBuilder, FieldDef, FieldRole, SymbolRef};
    let model = compile(
        "canceled.eqi",
        r#"model M() {
      domain body=box(0,1);
      state q:1 on body;
      parameter diffusivity:m^2/s=1;
      law balance on body {storage q; flux -diffusivity*grad(q); source 1 [1/s];}
    }"#,
    )
    .unwrap()
    .pop()
    .unwrap();
    let (original, model_id, symbols) = model.into_parts();
    let q = symbols.get("q").unwrap().downcast().unwrap();
    let diffusivity = symbols.get("diffusivity").unwrap().downcast().unwrap();
    let mut dag = ExprDagBuilder::new();
    let field = dag.symbol(SymbolRef::Field(q)).unwrap();
    let coefficient = dag.symbol(SymbolRef::Parameter(diffusivity)).unwrap();
    let stored = dag.sub(field, field).unwrap();
    let inverse_time = eqiora::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let accumulation = dag
        .constant(eqiora::DynQuantity::new(0.0, inverse_time))
        .unwrap();
    let source = dag
        .constant(eqiora::DynQuantity::new(1.0, inverse_time))
        .unwrap();
    let gradient = dag.gradient(field).unwrap();
    let flux = dag.mul(coefficient, gradient).unwrap();
    let flux = dag.neg(flux).unwrap();
    let divergence = dag.divergence(flux).unwrap();
    let left = dag.add(accumulation, divergence).unwrap();
    let expression = dag.finish([left, source]).unwrap();
    let terms = ConservationTerms::new(Some((stored, accumulation)), flux, source);
    expression
        .verify_time_derivative(stored, accumulation)
        .unwrap();
    for variable in [false, true] {
        let mut transaction = Transaction::new("canceled storage eligibility");
        for op in original.ops() {
            let op = match op {
                Op::DefineKernelNode {
                    node: KernelNode::Relation(relation),
                } => Op::DefineKernelNode {
                    node: RelationDef::conservation(relation.id(), expression.clone(), terms)
                        .unwrap()
                        .into(),
                },
                Op::DefineKernelNode {
                    node: KernelNode::Field(field),
                } if variable => Op::DefineKernelNode {
                    node: FieldDef::new(
                        field.id(),
                        field.value_type().clone(),
                        FieldRole::Variable,
                    )
                    .into(),
                },
                _ => op.clone(),
            };
            transaction.push(op);
        }
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let result = KernelProgram::from_snapshot(&store.snapshot(), model_id);
        if variable {
            let errors = result.unwrap_err();
            assert!(
                errors.iter().any(|error| error
                    .to_string()
                    .contains("storage may read only continuous state Fields")),
                "{errors:?}"
            );
        } else {
            result.unwrap();
        }
    }
}
