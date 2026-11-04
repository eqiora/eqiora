use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ValueShape};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, AxisBounds, BoundaryPairing, BoundaryPhysicalConnector, BoundarySide,
    ConnectionDef, ConnectionSemantics, DomainDef, ExprDagBuilder, KernelNode, PortDef,
    RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{BoundaryJunctionGeometry, KernelProgram};

fn length(value: f64) -> DynQuantity {
    DynQuantity::new(
        value,
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
    )
}

#[derive(Debug)]
struct InterfaceFixture {
    program: KernelProgram,
    connection: Id<kinds::Connection>,
}

fn interface_program(
    right_start: f64,
    semantics: ConnectionSemantics,
    share_parent: bool,
    channels: Option<u32>,
) -> Result<InterfaceFixture, Vec<eqiora_core::Diagnostic>> {
    interface_program_with_geometry(right_start, semantics, share_parent, channels, None)
}

fn interface_program_with_geometry(
    right_start: f64,
    semantics: ConnectionSemantics,
    share_parent: bool,
    channels: Option<u32>,
    geometry: Option<(&eqiora_geometry::CanonicalGeometryV1, [&str; 2], [&str; 2])>,
) -> Result<InterfaceFixture, Vec<eqiora_core::Diagnostic>> {
    interface_program_with_geometry_contract(
        right_start,
        semantics,
        share_parent,
        channels,
        geometry,
        2,
        true,
    )
}

fn interface_program_with_geometry_contract(
    right_start: f64,
    semantics: ConnectionSemantics,
    share_parent: bool,
    channels: Option<u32>,
    geometry: Option<(&eqiora_geometry::CanonicalGeometryV1, [&str; 2], [&str; 2])>,
    spatial_extent: u32,
    same_connector: bool,
) -> Result<InterfaceFixture, Vec<eqiora_core::Diagnostic>> {
    let connector = Id::<kinds::Domain>::new();
    let left_volume = Id::<kinds::Domain>::new();
    let right_volume = Id::<kinds::Domain>::new();
    let left_boundary = Id::<kinds::Domain>::new();
    let right_boundary = Id::<kinds::Domain>::new();
    let left_port = Id::<kinds::Port>::new();
    let right_port = Id::<kinds::Port>::new();
    let left_relation = Id::<kinds::Relation>::new();
    let right_relation = Id::<kinds::Relation>::new();
    let left_activation = Id::<kinds::Activation>::new();
    let right_activation = Id::<kinds::Activation>::new();
    let connection = Id::<kinds::Connection>::new();
    let model = OntologyId::<Model>::new();

    let velocity = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let traction = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let trace_type = eqiora_core::ValueType::shaped(
        eqiora_core::ScalarDomain::Real,
        velocity,
        ValueShape::new([spatial_extent]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let trace_type = match channels {
        Some(extent) => trace_type.array(extent).unwrap(),
        None => trace_type,
    };
    let connector_contract = BoundaryPhysicalConnector::new(
        trace_type.clone(),
        trace_type
            .clone()
            .with_dimension(traction)
            .expect("valid real tensor dimension"),
        BoundaryPairing::EuclideanBoundaryDuality,
    )
    .unwrap();

    let residuals = |port: Id<kinds::Port>| {
        let mut expression = ExprDagBuilder::new();
        let trace = expression.symbol(SymbolRef::PortTrace(port)).unwrap();
        let flux = expression.symbol(SymbolRef::PortFlux(port)).unwrap();
        {
            let trace_zero = expression
                .constant(eqiora_core::ValueLiteral::from_real(trace_type.clone(), 0.0).unwrap())
                .unwrap();
            let flux_zero = expression
                .constant(
                    eqiora_core::ValueLiteral::from_real(
                        trace_type
                            .clone()
                            .with_dimension(traction)
                            .expect("valid real tensor dimension"),
                        0.0,
                    )
                    .unwrap(),
                )
                .unwrap();
            expression.finish([trace, trace_zero, flux, flux_zero])
        }
        .unwrap()
    };

    let mut nodes = vec![
        KernelNode::from(DomainDef::boundary_physical(
            connector,
            connector_contract.clone(),
        )),
        KernelNode::from(
            DomainDef::cartesian_box(
                left_volume,
                vec![
                    AxisBounds::new(length(0.0), length(1.0)).unwrap(),
                    AxisBounds::new(length(0.0), length(1.0)).unwrap(),
                ],
            )
            .unwrap(),
        ),
    ];
    if !share_parent {
        nodes.push(KernelNode::from(
            DomainDef::cartesian_box(
                right_volume,
                vec![
                    AxisBounds::new(length(right_start), length(2.0)).unwrap(),
                    AxisBounds::new(length(0.0), length(1.0)).unwrap(),
                ],
            )
            .unwrap(),
        ));
    }
    nodes.extend([
        KernelNode::from(DomainDef::cartesian_boundary(
            left_boundary,
            0,
            BoundarySide::Upper,
        )),
        KernelNode::from(DomainDef::cartesian_boundary(
            right_boundary,
            0,
            BoundarySide::Lower,
        )),
        KernelNode::from(PortDef::boundary_physical(
            left_port,
            connector,
            left_boundary,
        )),
        KernelNode::from(PortDef::boundary_physical(
            right_port,
            connector,
            right_boundary,
        )),
        KernelNode::from(RelationDef::new(left_relation, residuals(left_port)).unwrap()),
        KernelNode::from(RelationDef::new(right_relation, residuals(right_port)).unwrap()),
        KernelNode::from(ActivationDef::continuous(left_activation)),
        KernelNode::from(ActivationDef::continuous(right_activation)),
        KernelNode::from(ConnectionDef::new(connection, semantics)),
    ]);
    if let Some((artifact, regions, boundaries)) = geometry {
        for node in &mut nodes {
            let id = node.id();
            if id == left_volume.erase() || id == right_volume.erase() {
                *node = DomainDef::geometry_region(
                    id.downcast().unwrap(),
                    eqiora_schema::kernel::GeometryDigest::new(artifact.digest_bytes()),
                    regions[usize::from(id == right_volume.erase() && !share_parent)],
                )
                .unwrap()
                .into();
            } else if id == left_boundary.erase() || id == right_boundary.erase() {
                *node = DomainDef::geometry_boundary(
                    id.downcast().unwrap(),
                    boundaries[usize::from(id == right_boundary.erase())],
                )
                .unwrap()
                .into();
            }
        }
    }
    if !same_connector {
        let other_connector = Id::<kinds::Domain>::new();
        for node in &mut nodes {
            if node.id() == right_port.erase() {
                *node =
                    PortDef::boundary_physical(right_port, other_connector, right_boundary).into();
            }
        }
        nodes.push(DomainDef::boundary_physical(other_connector, connector_contract).into());
    }
    let mut transaction = Transaction::new("two field-valued physical boundaries");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for (from, to, edge) in [
        (
            left_boundary.erase(),
            left_volume.erase(),
            EdgeKind::BoundaryOf,
        ),
        (
            right_boundary.erase(),
            if share_parent {
                left_volume.erase()
            } else {
                right_volume.erase()
            },
            EdgeKind::BoundaryOf,
        ),
        (left_relation.erase(), left_port.erase(), EdgeKind::HasPort),
        (
            right_relation.erase(),
            right_port.erase(),
            EdgeKind::HasPort,
        ),
        (
            left_relation.erase(),
            left_port.erase(),
            EdgeKind::DependsOn,
        ),
        (
            right_relation.erase(),
            right_port.erase(),
            EdgeKind::DependsOn,
        ),
        (
            left_relation.erase(),
            left_boundary.erase(),
            EdgeKind::AppliesOn,
        ),
        (
            right_relation.erase(),
            right_boundary.erase(),
            EdgeKind::AppliesOn,
        ),
        (
            left_activation.erase(),
            left_relation.erase(),
            EdgeKind::Activates,
        ),
        (
            right_activation.erase(),
            right_relation.erase(),
            EdgeKind::Activates,
        ),
        (connection.erase(), left_port.erase(), EdgeKind::Connects),
        (connection.erase(), right_port.erase(), EdgeKind::Connects),
    ] {
        transaction.push(Op::Connect { from, to, edge });
    }
    let members = transaction
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node.id()),
            _ => None,
        })
        .collect::<Vec<_>>();
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });

    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = match geometry {
        Some((artifact, _, _)) => {
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[artifact])
        }
        None => KernelProgram::from_snapshot(&store.snapshot(), model),
    };
    program.map(|program| InterfaceFixture {
        program,
        connection,
    })
}

#[test]
fn coincident_2d_vector_interface_is_admitted_componentwise() {
    let fixture = interface_program(1.0, ConnectionSemantics::Conserving, false, None)
        .expect("coincident interface must validate");
    assert_eq!(fixture.program.nodes().count(), 12);

    let junction = fixture
        .program
        .compose_boundary_physical_junction(fixture.connection)
        .expect("validated interface must compose");
    let typed = junction.typed();
    assert_eq!(typed.expression().roots().len(), 2);
    for root in typed.expression().roots() {
        let root_type = typed.node_type(*root).expect("every root is typed");
        assert_eq!(root_type.shape().extents()[0].get(), 2);
        assert_eq!(root_type.frame(), ValueFrame::SpatialCartesian);
    }
}

#[test]
fn noncoincident_cartesian_interface_fails_closed() {
    let diagnostics = interface_program(1.25, ConnectionSemantics::Conserving, false, None)
        .expect_err("separated boundaries must fail");
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("one coincident Cartesian boundary")
    }));
}

#[test]
fn boundary_junction_preserves_arrays_of_spatial_vectors() {
    let fixture = interface_program(1.0, ConnectionSemantics::Conserving, false, Some(3)).unwrap();
    let junction = fixture
        .program
        .compose_boundary_physical_junction(fixture.connection)
        .unwrap();
    for root in junction.typed().expression().roots() {
        let root_type = junction.typed().node_type(*root).unwrap();
        assert_eq!(root_type.value_type.array_rank(), 1);
        assert_eq!(
            root_type
                .shape()
                .extents()
                .iter()
                .map(|n| n.get())
                .collect::<Vec<_>>(),
            vec![3, 2]
        );
        assert_eq!(root_type.frame(), ValueFrame::SpatialCartesian);
    }
}

#[test]
fn opposite_sides_of_one_parent_form_a_spatial_periodic_junction() {
    let fixture = interface_program(0.0, ConnectionSemantics::SpatialPeriodic, true, None)
        .expect("opposite sides of one parent must validate");
    assert_eq!(fixture.program.nodes().count(), 11);

    let junction = fixture
        .program
        .compose_boundary_physical_junction(fixture.connection)
        .expect("validated periodic pair must compose");
    let BoundaryJunctionGeometry::CartesianPeriodic(identification) = junction.geometry() else {
        panic!("spatial-periodic junction must retain its derived chart map");
    };
    assert_eq!(identification.ambient_dimension(), 2);
    assert_eq!(identification.normal_axis(), 0);
    assert_eq!(identification.period(), 1.0);
    assert_eq!(identification.tangential_intervals(), &[(0.0, 1.0)]);
    assert_eq!(junction.typed().expression().roots().len(), 2);
}

#[test]
fn spatial_periodic_connection_rejects_distinct_parents() {
    let diagnostics = interface_program(1.0, ConnectionSemantics::SpatialPeriodic, false, None)
        .expect_err("periodic pair across distinct parents must fail");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.message().contains("one exact parent Domain") })
    );
}

fn periodic_geometry() -> eqiora_geometry::CanonicalGeometryV1 {
    let graph = eqiora_geometry::GeometryGraph::new();
    let rectangle = graph.rectangle([2.0, 5.0], [-1.0, 3.0]).unwrap();
    let boundaries = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &std::collections::BTreeMap::from([
                ("domain".to_owned(), vec![rectangle.region().into()]),
                ("left".to_owned(), vec![boundaries[0].into()]),
                ("right".to_owned(), vec![boundaries[1].into()]),
                ("top".to_owned(), vec![boundaries[3].into()]),
                ("bottom".to_owned(), vec![boundaries[2].into()]),
            ]),
        )
        .unwrap()
}

#[test]
fn exact_geometry_periodic_pair_composes_after_canonical_round_trip() {
    let original = periodic_geometry();
    let artifact = eqiora_geometry::CanonicalGeometryV1::decode_planar_rectangle_v2_canonical(
        original.canonical_bytes(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(original.digest_bytes(), artifact.digest_bytes());
    let fixture = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::SpatialPeriodic,
        true,
        Some(3),
        Some((&artifact, ["domain", "domain"], ["right", "left"])),
    )
    .unwrap();
    let junction = fixture
        .program
        .compose_boundary_physical_junction(fixture.connection)
        .unwrap();
    let BoundaryJunctionGeometry::CartesianPeriodic(chart) = junction.geometry() else {
        panic!("Geometry periodic pair lost its translation");
    };
    assert_eq!(chart.normal_axis(), 0);
    assert_eq!(chart.period(), 3.0);
    assert_eq!(chart.ambient_dimension(), 2);
    assert_eq!(junction.dag().roots().len(), 2);
}

#[test]
fn geometry_periodic_pair_rejects_nonopposite_and_stale_selections() {
    let artifact = periodic_geometry();
    for upper in ["left", "top", "missing"] {
        let diagnostics = interface_program_with_geometry(
            0.0,
            ConnectionSemantics::SpatialPeriodic,
            true,
            None,
            Some((&artifact, ["domain", "domain"], [upper, "left"])),
        )
        .expect_err("invalid exact periodic selection must fail");
        let expected = if upper == "missing" {
            "absent from its parent artifact"
        } else {
            "explicit spatial-periodic pair"
        };
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains(expected)),
            "{upper}: {diagnostics:?}"
        );
    }
    let diagnostics = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::SpatialPeriodic,
        false,
        None,
        Some((&artifact, ["domain", "domain"], ["right", "left"])),
    )
    .expect_err("equal geometry does not equate distinct parent identities");
    assert!(
        diagnostics.iter().any(|diagnostic| diagnostic
            .message()
            .contains("explicit spatial-periodic pair")),
        "{diagnostics:?}"
    );
}

#[test]
fn exact_geometry_coincident_contract_retains_parent_and_selection_identity() {
    let artifact = periodic_geometry();
    let fixture = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::Conserving,
        true,
        Some(3),
        Some((&artifact, ["domain", "domain"], ["left", "left"])),
    )
    .unwrap();
    assert!(matches!(
        fixture
            .program
            .compose_boundary_physical_junction(fixture.connection)
            .unwrap()
            .geometry(),
        BoundaryJunctionGeometry::Coincident
    ));
    for (share_parent, selection) in [(false, "left"), (true, "right")] {
        let errors = interface_program_with_geometry(
            0.0,
            ConnectionSemantics::Conserving,
            share_parent,
            None,
            Some((&artifact, ["domain", "domain"], [selection, "left"])),
        )
        .unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .message()
                .contains("exact same-support selection pair")),
            "{errors:?}"
        );
    }
}

#[test]
fn authored_diagonal_interface_admits_exact_opposite_parent_boundary_ports() {
    use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        vec![
            PlanarFace::new(vec![0, 2, 1], vec![]),
            PlanarFace::new(vec![1, 2, 3], vec![]),
        ],
        vec![
            NamedEntitySet::new("first", 2, vec![0]),
            NamedEntitySet::new("second", 2, vec![1]),
            NamedEntitySet::new("first_interface", 1, vec![1]),
            NamedEntitySet::new("second_interface", 1, vec![3]),
        ],
        1e-12,
    )
    .unwrap();
    let artifact = CanonicalGeometryV1::from_region(&region).unwrap();
    let fixture = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::Conserving,
        false,
        None,
        Some((
            &artifact,
            ["first", "second"],
            ["first_interface", "second_interface"],
        )),
    )
    .unwrap();
    let junction = fixture
        .program
        .compose_boundary_physical_junction(fixture.connection)
        .unwrap();
    assert_eq!(junction.typed().expression().roots().len(), 2);
    let rejected = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::Conserving,
        false,
        None,
        Some((
            &artifact,
            ["first", "second"],
            ["second_interface", "first_interface"],
        )),
    );
    assert!(rejected.is_err());
}

#[test]
fn curved_and_grouped_same_support_pairs_need_no_cartesian_chart() {
    use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};
    let artifact = CanonicalGeometryV1::from_circular_hole(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("fluid", 2, vec![0]),
            NamedEntitySet::new("inlet", 1, vec![0]),
            NamedEntitySet::new("outlet", 1, vec![1]),
            NamedEntitySet::new("circle", 1, vec![4]),
            NamedEntitySet::new("walls", 1, vec![2, 3]),
        ],
    )
    .unwrap();
    for boundary in ["circle", "walls"] {
        let fixture = interface_program_with_geometry(
            0.0,
            ConnectionSemantics::Conserving,
            true,
            None,
            Some((&artifact, ["fluid", "fluid"], [boundary, boundary])),
        )
        .unwrap();
        assert!(matches!(
            fixture
                .program
                .compose_boundary_physical_junction(fixture.connection)
                .unwrap()
                .geometry(),
            BoundaryJunctionGeometry::Coincident
        ));
        for (semantics, share_parent, boundaries, extent, same_connector) in [
            (
                ConnectionSemantics::SpatialPeriodic,
                true,
                [boundary, boundary],
                2,
                true,
            ),
            (
                ConnectionSemantics::Conserving,
                false,
                [boundary, boundary],
                2,
                true,
            ),
            (
                ConnectionSemantics::Conserving,
                true,
                ["circle", "walls"],
                2,
                true,
            ),
            (
                ConnectionSemantics::Conserving,
                true,
                [boundary, boundary],
                3,
                true,
            ),
            (
                ConnectionSemantics::Conserving,
                true,
                [boundary, boundary],
                2,
                false,
            ),
        ] {
            assert!(
                interface_program_with_geometry_contract(
                    0.0,
                    semantics,
                    share_parent,
                    None,
                    Some((&artifact, ["fluid", "fluid"], boundaries)),
                    extent,
                    same_connector
                )
                .is_err()
            );
        }
    }
}

#[test]
fn same_nonaffine_metadata_does_not_prove_membership_in_the_parent() {
    use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        vec![
            PlanarFace::new(vec![0, 2, 1], vec![]),
            PlanarFace::new(vec![1, 2, 3], vec![]),
        ],
        vec![
            NamedEntitySet::new("second", 2, vec![1]),
            NamedEntitySet::new("first_edge", 1, vec![0]),
        ],
        1e-12,
    )
    .unwrap();
    let artifact = CanonicalGeometryV1::from_region(&region).unwrap();
    assert!(!artifact.selection_is_boundary_of(
        artifact.entity_set("first_edge").unwrap(),
        artifact.entity_set("second").unwrap()
    ));
    let errors = interface_program_with_geometry(
        0.0,
        ConnectionSemantics::Conserving,
        true,
        None,
        Some((
            &artifact,
            ["second", "second"],
            ["first_edge", "first_edge"],
        )),
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("same-support selection pair")),
        "{errors:?}"
    );
}
