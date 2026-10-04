//! Ordered dimensioned factors survive ordinary Model replay without Geometry or mesh.
use eqiora_artifact::{ModelEnvelope, ModelTransactionEnvelope, StructuralSemanticFingerprint};
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{ActivationDef, AxisBounds, DomainDef, ExprDagBuilder, KernelNode, RelationDef},
};
use eqiora_sem::KernelProgram;

fn fixture(
    reverse: bool,
    velocity_exponent: i32,
    upper: f64,
) -> (KernelProgram, ModelTransactionEnvelope) {
    let position = Id::new();
    let velocity = Id::new();
    let phase = Id::new();
    let relation = Id::new();
    let activation = Id::new();
    let model = OntologyId::<Model>::new();
    let interval = |id, time, upper| {
        DomainDef::coordinate_interval(
            id,
            AxisBounds::new(
                DynQuantity::new(
                    -1.0,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
                DynQuantity::new(
                    upper,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
            )
            .unwrap(),
        )
    };
    let factors = if reverse {
        vec![velocity, position]
    } else {
        vec![position, velocity]
    };
    let mut dag = ExprDagBuilder::new();
    let zero = dag
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let nodes: Vec<KernelNode> = vec![
        interval(position, 0, 1.0).into(),
        interval(velocity, velocity_exponent, upper).into(),
        DomainDef::coordinate_product(phase, factors)
            .unwrap()
            .into(),
        RelationDef::new(relation, dag.finish([zero, zero]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("coordinate product wire");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (phase.erase(), position.erase(), EdgeKind::DependsOn),
        (phase.erase(), velocity.erase(), EdgeKind::DependsOn),
        (activation.erase(), relation.erase(), EdgeKind::Activates),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    let envelope = ModelTransactionEnvelope::from_transaction(&transaction).unwrap();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        envelope,
    )
}

#[test]
fn factor_model_and_transaction_round_trip_exactly() {
    let (program, transaction) = fixture(false, -1, 2.0);
    let bytes = ModelEnvelope::from_program(&program)
        .unwrap()
        .canonical_json()
        .unwrap();
    let decoded = ModelEnvelope::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.to_program().unwrap(), program);
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    let wire = transaction.canonical_json().unwrap();
    let decoded = ModelTransactionEnvelope::from_json(&wire, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), wire);
    let mut store = InMemoryGraphStore::new();
    store.commit(decoded.to_transaction().unwrap()).unwrap();
    let replay = KernelProgram::from_snapshot(&store.snapshot(), program.model()).unwrap();
    assert_eq!(replay, program);
}

#[test]
fn factor_fingerprint_is_alpha_invariant_but_binds_order_units_and_bounds() {
    let fingerprint = |reverse, time, upper| {
        StructuralSemanticFingerprint::from_program(&fixture(reverse, time, upper).0).unwrap()
    };
    let original = fingerprint(false, -1, 2.0);
    assert_eq!(original, fingerprint(false, -1, 2.0));
    assert_ne!(original, fingerprint(true, -1, 2.0));
    assert_ne!(original, fingerprint(false, -2, 2.0));
    assert_ne!(original, fingerprint(false, -1, 3.0));
}

#[test]
fn source_distribution_measure_replays_through_semantic_model() {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    let interval = |time| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(
                DynQuantity::new(
                    -1.0,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
                DynQuantity::new(
                    1.0,
                    DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap(),
                ),
            )
            .unwrap(),
        )
    };
    let source = "operator identity(input x:scalar):scalar=x; model Distribution(support position:interval(m), support velocity:interval(m/s)) { support phase:product(position,velocity); variable f:s/m^2 on phase; relation retain on phase {identity(x=f)=0[s/m^2];} observable mass:1=integral(f,measure(phase)); }";
    let compiled = CompiledModel::compile_selected(
        "distribution.eqi",
        source,
        "Distribution",
        &[("position", interval(0)), ("velocity", interval(-1))],
    )
    .unwrap();
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
    assert!(program.nodes().any(|node| matches!(node, KernelNode::Observable(value) if value.value_type().dimension() == DimExponents::DIMENSIONLESS)));
    let wrong = source.replace("observable mass:1", "observable mass:s");
    assert!(
        CompiledModel::compile_selected(
            "wrong-measure.eqi",
            &wrong,
            "Distribution",
            &[("position", interval(0)), ("velocity", interval(-1))]
        )
        .is_err()
    );
}

#[test]
fn forwarded_coordinate_factors_keep_their_exact_enclosing_identity() {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    use eqiora_schema::kernel::DomainKind;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length)).unwrap();
    let source = "component Particle(support position:interval(m), support radius:interval(m)) { support phase:product(position,radius); variable density:1/m^2 on phase; relation retain on phase {density=0[1/m^2];} } model Root(support position:interval(m), support radius:interval(m)) { instance particle:Particle(position=position,radius=radius); }";
    let compiled = CompiledModel::compile_selected(
        "particle.eqi",
        source,
        "Root",
        &[
            ("position", StaticBindingValue::CoordinateInterval(bounds)),
            ("radius", StaticBindingValue::CoordinateInterval(bounds)),
        ],
    )
    .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let factors = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CoordinateInterval { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        factors.len(),
        2,
        "equal interval bounds do not alias position and radius"
    );
    let products = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain) => match domain.kind() {
                DomainKind::CoordinateProduct { factors } => Some(factors),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(products.len(), 1);
    assert_eq!(
        products[0]
            .iter()
            .map(|id| id.erase())
            .collect::<std::collections::BTreeSet<_>>(),
        factors
    );
}
