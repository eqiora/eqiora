use std::collections::BTreeSet;

use eqiora_artifact::{ModelEnvelope, ModelTransactionEnvelope};
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, OntologyId, RawId};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ConnectionDef, ConnectionSemantics, DomainDef, ExprDagBuilder, KernelNode,
    ParameterDef, PortDef, RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;

#[derive(Clone, Copy)]
struct ClosureIds {
    domain: Id<kinds::Domain>,
    parameter: Id<kinds::Parameter>,
    activation: Id<kinds::Activation>,
    joined_connections: [Id<kinds::Connection>; 2],
    isolated_connection: Id<kinds::Connection>,
    ports: [Id<kinds::Port>; 6],
    relations: [Id<kinds::Relation>; 5],
    unrelated_relation: Id<kinds::Relation>,
    model: OntologyId<Model>,
}

fn ids() -> ClosureIds {
    let mut joined_connections = [Id::new(), Id::new()];
    joined_connections.sort_by_key(|id: &Id<kinds::Connection>| id.erase());
    ClosureIds {
        domain: Id::new(),
        parameter: Id::new(),
        activation: Id::new(),
        joined_connections,
        isolated_connection: Id::new(),
        ports: [
            Id::new(),
            Id::new(),
            Id::new(),
            Id::new(),
            Id::new(),
            Id::new(),
        ],
        relations: [Id::new(), Id::new(), Id::new(), Id::new(), Id::new()],
        unrelated_relation: Id::new(),
        model: OntologyId::new(),
    }
}

fn relation(id: Id<kinds::Relation>, symbols: &[SymbolRef], subtract: bool) -> RelationDef {
    let mut expression = ExprDagBuilder::new();
    let mut values = symbols
        .iter()
        .map(|symbol| expression.symbol(*symbol).unwrap())
        .collect::<Vec<_>>();
    let root = if subtract {
        expression.sub(values.remove(0), values.remove(0)).unwrap()
    } else {
        values.remove(0)
    };
    let zero = expression
        .constant(eqiora_core::DynQuantity::new(
            0.0,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))
        .unwrap();
    RelationDef::new(id, expression.finish([root, zero]).unwrap()).unwrap()
}

fn closure_transaction(ids: ClosureIds, unrelated: bool, reversed: bool) -> Transaction {
    let dimension = DimExponents::DIMENSIONLESS;
    let mut nodes = vec![
        DomainDef::scalar_physical(
            ids.domain,
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                .expect("valid fixture scalar type"),
            eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                .expect("valid fixture scalar type"),
        )
        .unwrap()
        .into(),
        ParameterDef::new(
            ids.parameter,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                    .expect("valid fixture scalar type"),
                2.0,
            )
            .unwrap(),
        )
        .into(),
        ActivationDef::continuous(ids.activation).into(),
        ConnectionDef::new(ids.joined_connections[0], ConnectionSemantics::Conserving).into(),
        ConnectionDef::new(ids.joined_connections[1], ConnectionSemantics::Conserving).into(),
        ConnectionDef::new(ids.isolated_connection, ConnectionSemantics::Conserving).into(),
    ];
    nodes.extend(
        ids.ports
            .iter()
            .copied()
            .map(|port| KernelNode::from(PortDef::scalar_physical(port, ids.domain))),
    );
    nodes.extend([
        relation(
            ids.relations[0],
            &[
                SymbolRef::Across(ids.ports[0]),
                SymbolRef::Parameter(ids.parameter),
            ],
            true,
        )
        .into(),
        relation(
            ids.relations[1],
            &[
                SymbolRef::Across(ids.ports[1]),
                SymbolRef::Across(ids.ports[2]),
            ],
            true,
        )
        .into(),
        relation(ids.relations[2], &[SymbolRef::Through(ids.ports[3])], false).into(),
        relation(
            ids.relations[3],
            &[
                SymbolRef::Across(ids.ports[4]),
                SymbolRef::Parameter(ids.parameter),
            ],
            true,
        )
        .into(),
        relation(ids.relations[4], &[SymbolRef::Through(ids.ports[5])], false).into(),
    ]);
    if unrelated {
        nodes.push(
            relation(
                ids.unrelated_relation,
                &[
                    SymbolRef::Parameter(ids.parameter),
                    SymbolRef::Parameter(ids.parameter),
                ],
                true,
            )
            .into(),
        );
    }
    if reversed {
        nodes.reverse();
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let view = ModelView::new(ids.model, members, []).unwrap();

    let owners = [
        (ids.relations[0], ids.ports[0]),
        (ids.relations[1], ids.ports[1]),
        (ids.relations[1], ids.ports[2]),
        (ids.relations[2], ids.ports[3]),
        (ids.relations[3], ids.ports[4]),
        (ids.relations[4], ids.ports[5]),
    ];
    let dependencies = [
        (ids.relations[0].erase(), ids.ports[0].erase()),
        (ids.relations[0].erase(), ids.parameter.erase()),
        (ids.relations[1].erase(), ids.ports[1].erase()),
        (ids.relations[1].erase(), ids.ports[2].erase()),
        (ids.relations[2].erase(), ids.ports[3].erase()),
        (ids.relations[3].erase(), ids.ports[4].erase()),
        (ids.relations[3].erase(), ids.parameter.erase()),
        (ids.relations[4].erase(), ids.ports[5].erase()),
    ];
    let memberships = [
        (ids.joined_connections[0], ids.ports[0]),
        (ids.joined_connections[0], ids.ports[1]),
        (ids.joined_connections[1], ids.ports[2]),
        (ids.joined_connections[1], ids.ports[3]),
        (ids.isolated_connection, ids.ports[4]),
        (ids.isolated_connection, ids.ports[5]),
    ];
    let mut edges = Vec::new();
    for (relation, port) in owners {
        edges.push((relation.erase(), port.erase(), EdgeKind::HasPort));
    }
    for (relation, dependency) in dependencies {
        edges.push((relation, dependency, EdgeKind::DependsOn));
    }
    if unrelated {
        edges.push((
            ids.unrelated_relation.erase(),
            ids.parameter.erase(),
            EdgeKind::DependsOn,
        ));
    }
    for relation in ids.relations {
        edges.push((
            ids.activation.erase(),
            relation.erase(),
            EdgeKind::Activates,
        ));
    }
    if unrelated {
        edges.push((
            ids.activation.erase(),
            ids.unrelated_relation.erase(),
            EdgeKind::Activates,
        ));
    }
    for (connection, port) in memberships {
        edges.push((connection.erase(), port.erase(), EdgeKind::Connects));
    }
    if reversed {
        edges.reverse();
    }

    let mut transaction = Transaction::new("two-junction physical closure");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in edges {
        transaction.push(Op::Connect { from, to, edge });
    }
    transaction.push(Op::DefineOntologyView { view: view.into() });
    transaction
}

fn program_from_transaction(transaction: Transaction, model: OntologyId<Model>) -> KernelProgram {
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn relation_ids(program: &eqiora_sem::ComposedResidualSystem) -> BTreeSet<RawId> {
    program
        .relations()
        .iter()
        .map(|group| group.owner().erase())
        .collect()
}

#[test]
fn bridge_closure_is_canonical_and_parameter_sharing_does_not_join_subsystems() {
    let ids = ids();
    let forward_transaction = closure_transaction(ids, true, false);
    let wire = ModelTransactionEnvelope::from_transaction(&forward_transaction).unwrap();
    let decoded =
        ModelTransactionEnvelope::from_json(&wire.canonical_json().unwrap(), Default::default())
            .unwrap();
    let forward = program_from_transaction(decoded.to_transaction().unwrap(), ids.model);
    let reversed = program_from_transaction(closure_transaction(ids, true, true), ids.model);
    let without_unrelated =
        program_from_transaction(closure_transaction(ids, false, false), ids.model);

    let selected_high = forward
        .compose_scalar_physical_subsystem(ids.joined_connections[1])
        .unwrap();
    assert_eq!(
        selected_high.subsystem().connection(),
        ids.joined_connections[0]
    );
    assert_eq!(
        selected_high
            .junctions()
            .iter()
            .map(|junction| junction.owner())
            .collect::<Vec<_>>(),
        ids.joined_connections
    );
    assert_eq!(
        relation_ids(&selected_high),
        ids.relations[..3]
            .iter()
            .map(|relation| relation.erase())
            .collect()
    );

    let isolated = forward
        .compose_scalar_physical_subsystem(ids.isolated_connection)
        .unwrap();
    assert_eq!(isolated.junctions().len(), 1);
    assert_eq!(
        relation_ids(&isolated),
        ids.relations[3..]
            .iter()
            .map(|relation| relation.erase())
            .collect()
    );
    assert_eq!(selected_high.parameters(), &[ids.parameter]);
    assert_eq!(isolated.parameters(), &[ids.parameter]);
    assert!(!relation_ids(&selected_high).contains(&ids.unrelated_relation.erase()));

    assert_eq!(
        selected_high,
        reversed
            .compose_scalar_physical_subsystem(ids.joined_connections[1])
            .unwrap()
    );
    assert_eq!(
        selected_high,
        without_unrelated
            .compose_scalar_physical_subsystem(ids.joined_connections[1])
            .unwrap()
    );
    assert_eq!(
        ModelEnvelope::from_program(&forward)
            .unwrap()
            .canonical_json()
            .unwrap(),
        ModelEnvelope::from_program(&reversed)
            .unwrap()
            .canonical_json()
            .unwrap()
    );
    assert_ne!(
        ModelEnvelope::from_program(&forward)
            .unwrap()
            .digest()
            .unwrap(),
        ModelEnvelope::from_program(&without_unrelated)
            .unwrap()
            .digest()
            .unwrap()
    );
}
