//! Ordered physical interface identity survives replay without a conserving law.
use eqiora_artifact::{ModelDecoderLimits, ModelEnvelope, StructuralSemanticFingerprint};
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, entity::kinds};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, AxisBounds, BoundarySide, DomainDef, DomainKind, ExprDagBuilder, KernelNode,
        RelationDef,
    },
};
use eqiora_sem::KernelProgram;

fn fixture(reverse: bool) -> (KernelProgram, Id<kinds::Domain>, [Id<kinds::Domain>; 2]) {
    let regions = [Id::<kinds::Domain>::new(), Id::new()];
    let boundaries = [Id::<kinds::Domain>::new(), Id::new()];
    let interface = Id::<kinds::Domain>::new();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let volume = |id, lower, upper| {
        DomainDef::cartesian_box(
            id,
            vec![
                AxisBounds::new(
                    DynQuantity::new(lower, length),
                    DynQuantity::new(upper, length),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    };
    let ordered = if reverse {
        [boundaries[1], boundaries[0]]
    } else {
        boundaries
    };
    let mut nodes = [
        volume(regions[0], 0.0, 1.0),
        volume(regions[1], 1.0, 3.0),
        DomainDef::cartesian_boundary(boundaries[0], 0, BoundarySide::Upper),
        DomainDef::cartesian_boundary(boundaries[1], 0, BoundarySide::Lower),
        DomainDef::physical_interface(interface, ordered).unwrap(),
    ]
    .map(KernelNode::from)
    .to_vec();
    // Model admission requires a Relation; this global identity imposes no interface law.
    let relation = Id::new();
    let activation = Id::new();
    let mut expression = ExprDagBuilder::new();
    let zero = expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    nodes.push(
        RelationDef::new(relation, expression.finish([zero, zero]).unwrap())
            .unwrap()
            .into(),
    );
    nodes.push(ActivationDef::continuous(activation).into());
    let model = OntologyId::<Model>::new();
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("physical interface wire");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for side in 0..2 {
        transaction.push(Op::Connect {
            from: boundaries[side].erase(),
            to: regions[side].erase(),
            edge: EdgeKind::BoundaryOf,
        });
        transaction.push(Op::Connect {
            from: interface.erase(),
            to: boundaries[side].erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: EdgeKind::Activates,
    });
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        interface,
        ordered,
    )
}

#[test]
fn ordered_boundaries_roundtrip_without_a_connection() {
    for reverse in [false, true] {
        let (program, interface, boundaries) = fixture(reverse);
        let bytes = ModelEnvelope::from_program(&program)
            .unwrap()
            .canonical_json()
            .unwrap();
        let envelope = ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()).unwrap();
        let replay = envelope.to_program().unwrap();
        let Some(KernelNode::Domain(domain)) = replay.node(interface.erase()) else {
            panic!("missing interface");
        };
        assert_eq!(domain.kind(), &DomainKind::PhysicalInterface { boundaries });
        assert!(
            !replay
                .edges()
                .iter()
                .any(|edge| edge.kind() == EdgeKind::Connects)
        );
        assert_eq!(envelope.canonical_json().unwrap(), bytes);
    }
}

#[test]
fn fingerprint_retains_orientation_but_not_generated_identifiers() {
    let fingerprint =
        |reverse| StructuralSemanticFingerprint::from_program(&fixture(reverse).0).unwrap();
    assert_eq!(fingerprint(false), fingerprint(false));
    assert_ne!(fingerprint(false), fingerprint(true));
}

#[test]
fn current_wire_requires_the_boundary_pair_and_rejects_the_previous_epoch() {
    let bytes = ModelEnvelope::from_program(&fixture(false).0)
        .unwrap()
        .canonical_json()
        .unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut previous = wire.clone();
    previous["schema"] = "eqiora.model-envelope/v42".into();
    assert!(
        ModelEnvelope::from_json(
            &serde_json::to_vec(&previous).unwrap(),
            ModelDecoderLimits::default(),
        )
        .is_err()
    );
    let mut missing = wire;
    let domain = missing["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["definition"]["domain"]["kind"] == "physical-interface")
        .expect("physical interface wire definition");
    domain["definition"]["domain"]
        .as_object_mut()
        .unwrap()
        .remove("boundaries");
    assert!(
        ModelEnvelope::from_json(
            &serde_json::to_vec(&missing).unwrap(),
            ModelDecoderLimits::default(),
        )
        .is_err()
    );
}
