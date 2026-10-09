//! Model-owned one-sided values share only their explicitly selected interface.
use eqiora_core::{
    DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType, entity::kinds,
};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::{
    Model, ModelView,
    kernel::{
        ActivationDef, AxisBounds, BoundarySide, DomainDef, ExprDagBuilder, FieldDef, FieldRole,
        KernelNode, RelationDef, RepresentationDef, SymbolRef, typing::SpatialSupport,
    },
};
use eqiora_sem::KernelProgram;

#[test]
fn authored_scalar_jump_types_on_the_exact_two_sided_interface() {
    let regions = [Id::<kinds::Domain>::new(), Id::new()];
    let boundaries = [Id::<kinds::Domain>::new(), Id::new()];
    let interface = Id::<kinds::Domain>::new();
    let fields = [Id::<kinds::Field>::new(), Id::new()];
    let representation = Id::new();
    let relation = Id::new();
    let activation = Id::new();
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
    let mut dag = ExprDagBuilder::new();
    let left = dag.symbol(SymbolRef::Field(fields[0])).unwrap();
    let right = dag.symbol(SymbolRef::Field(fields[1])).unwrap();
    let left_trace = dag.trace(left, interface).unwrap();
    let right_trace = dag.trace(right, interface).unwrap();
    let jump = dag.sub(left_trace, right_trace).unwrap();
    let zero = dag
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let mut nodes = vec![
        KernelNode::from(volume(regions[0], 0.0, 1.0)),
        volume(regions[1], 1.0, 3.0).into(),
        DomainDef::cartesian_boundary(boundaries[0], 0, BoundarySide::Upper).into(),
        DomainDef::cartesian_boundary(boundaries[1], 0, BoundarySide::Lower).into(),
        DomainDef::physical_interface(interface, boundaries)
            .unwrap()
            .into(),
        RepresentationDef::continuum(representation).into(),
        RelationDef::new(relation, dag.finish([jump, zero]).unwrap())
            .unwrap()
            .into(),
        ActivationDef::continuous(activation).into(),
    ];
    for field in fields {
        nodes.push(
            FieldDef::new(
                field,
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                FieldRole::Variable,
            )
            .into(),
        );
    }
    let model = OntologyId::<Model>::new();
    let view = ModelView::new(model, nodes.iter().map(KernelNode::id), []).unwrap();
    let mut transaction = Transaction::new("authored interface scalar equation");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for side in 0..2 {
        for (from, to, edge) in [
            (
                boundaries[side].erase(),
                regions[side].erase(),
                EdgeKind::BoundaryOf,
            ),
            (
                interface.erase(),
                boundaries[side].erase(),
                EdgeKind::DependsOn,
            ),
            (
                fields[side].erase(),
                regions[side].erase(),
                EdgeKind::DefinedOn,
            ),
            (
                fields[side].erase(),
                representation.erase(),
                EdgeKind::DefinedOn,
            ),
            (relation.erase(), fields[side].erase(), EdgeKind::DependsOn),
        ] {
            transaction.push(Op::Connect { from, to, edge });
        }
    }
    transaction.push(Op::Connect {
        from: relation.erase(),
        to: interface.erase(),
        edge: EdgeKind::AppliesOn,
    });
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: EdgeKind::Activates,
    });
    transaction.push(Op::DefineOntologyView { view: view.into() });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let typed = program.typed_relation_residual(relation).unwrap();
    let expected = SpatialSupport::PhysicalInterface {
        domain: interface.erase(),
        boundaries: Box::new(boundaries.map(Id::erase)),
        parents: Box::new(regions.map(Id::erase)),
        dimensions: 1,
    };
    for value in [left_trace, right_trace, jump] {
        assert_eq!(
            typed.node_type(value).unwrap().support,
            Some(expected.clone())
        );
    }
    assert!(
        !program
            .edges()
            .iter()
            .any(|edge| edge.kind() == EdgeKind::Connects)
    );
}
