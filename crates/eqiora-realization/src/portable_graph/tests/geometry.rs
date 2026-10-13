use super::*;

pub(super) fn fixture(harmonic_count: usize) -> PortableRealizationGraph {
    let domains = [Id::new(), Id::new(), Id::new()];
    let fields = [Id::new(), Id::new(), Id::new()];
    let regions = domains.iter().zip(fields).map(|(&domain, field)| {
        crate::DomainFieldDiscretization::new(
            domain,
            [FieldSpaceBinding::new(
                field,
                Space::continuous_lagrange(NonZeroU16::MIN),
            )],
            [],
        )
        .unwrap()
    });
    let edges = [(0, 1), (0, 2), (1, 2)];
    let connections = [Id::new(), Id::new(), Id::new()];
    let quotients = edges.iter().zip(connections).map(|(&(a, b), connection)| {
        crate::ConformingTraceQuotient::new(
            crate::ConformingTraceSource::ConservingConnection(connection),
            crate::TraceFieldEndpoint::new(domains[a], fields[a]),
            crate::TraceFieldEndpoint::new(domains[b], fields[b]),
        )
        .unwrap()
    });
    let graph = PortableRealizationGraph::spatial_regions(
        RealizationLineage::explicit(
            OntologyId::new(),
            SemanticRevision::new(1),
            RealizationRevision::new(1),
        ),
        regions,
        quotients,
        None,
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(2).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        LinearOperatorProperties::General,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        solver(LinearSolver::BiConjugateGradientStabilized),
        None,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
        SystemScaling::Dimensional,
    )
    .unwrap();
    let interfaces = edges
        .iter()
        .zip(connections)
        .filter_map(|(&(a, b), connection)| {
            ((a < harmonic_count) != (b < harmonic_count)).then_some(connection)
        });
    let motion = crate::P1HarmonicMeshMotionPolicy::new(
        domains[..harmonic_count].iter().copied(),
        domains[harmonic_count..]
            .iter()
            .copied()
            .zip(fields[harmonic_count..].iter().copied()),
        interfaces,
        AleGeometryQualityGate::new(0.05).unwrap(),
        solver(LinearSolver::ConjugateGradient),
    )
    .unwrap();
    graph
        .with_harmonic_geometry(
            &motion,
            domains[..harmonic_count].iter().copied(),
            fields[..harmonic_count].iter().map(|field| {
                BackwardEulerRelationStep::new(Id::new(), *field, DynQuantity::new(0.1, TIME))
                    .unwrap()
            }),
        )
        .unwrap()
}

fn solver(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-9, 1e-12, NonZeroUsize::new(100).unwrap()).unwrap()
}

#[test]
fn plural_harmonic_and_driven_inventories_round_trip() {
    for harmonic_count in [1, 2] {
        let graph = fixture(harmonic_count);
        let GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            ..
        } = &graph.geometry_actions[0];
        assert_eq!(harmonic_domains.len(), harmonic_count);
        assert_eq!(drivers.len(), 3 - harmonic_count);
        assert_eq!(interfaces.len(), 2); // Third Connection stays inside one group.
        let bytes = graph.to_bytes().unwrap();
        assert_eq!(PortableRealizationGraph::from_bytes(&bytes).unwrap(), graph);
        let old = String::from_utf8(bytes).unwrap().replace(
            "portable-realization-graph/v5",
            "portable-realization-graph/v4",
        );
        assert!(PortableRealizationGraph::from_bytes(old.as_bytes()).is_err());
    }
}

#[test]
fn malformed_geometry_inventories_fail_closed() {
    let graph = fixture(2);
    for variant in 0..9 {
        let mut bad = graph.clone();
        let GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            duration,
            ..
        } = &mut bad.geometry_actions[0];
        match variant {
            0 => harmonic_domains.clear(),
            1 => harmonic_domains.reverse(),
            2 => harmonic_domains.push(harmonic_domains[0]),
            3 => drivers.push(drivers[0]),
            4 => interfaces.pop().map(|_| ()).unwrap(),
            5 => interfaces[0] = Id::new(),
            6 => harmonic_domains[0] = DomainDiscretizationId::new(100),
            7 => *duration = DynQuantity::new(0.0, TIME),
            8 => {
                harmonic_domains.push(bad.fields[drivers[0].index()].domain);
                harmonic_domains.sort();
            }
            _ => unreachable!(),
        }
        assert!(bad.to_bytes().is_err(), "variant {variant}");
    }
    let graph = fixture(1);
    let mut bad = graph.clone();
    let GeometryActionNode::P1HarmonicExtension { drivers, .. } = &mut bad.geometry_actions[0];
    drivers.reverse();
    assert!(bad.to_bytes().is_err());
}

#[test]
fn connection_coverage_uses_identity_and_exact_endpoint_domains() {
    let graph = fixture(2);
    let selected = graph.transformations.iter().find(|node| {
        let TransformationNode::ConformingTraceQuotient { source, .. } = node else { return false; };
        let GeometryActionNode::P1HarmonicExtension { interfaces, .. } = &graph.geometry_actions[0];
        matches!(source, crate::ConformingTraceSource::ConservingConnection(id) if interfaces.contains(id))
    }).copied().unwrap();
    let TransformationNode::ConformingTraceQuotient {
        source,
        mut endpoints,
    } = selected
    else {
        unreachable!()
    };
    endpoints.reverse();
    let mut repeated = graph.clone();
    repeated
        .transformations
        .push(TransformationNode::ConformingTraceQuotient { source, endpoints });
    repeated.systems[0]
        .transformations
        .push(TransformationId::new(repeated.transformations.len() - 1));
    repeated.validate().unwrap(); // The same exact Connection can quotient multiple traces.

    let mut bad = repeated;
    let foreign = (0..3)
        .map(FieldRepresentationId::new)
        .find(|id| !endpoints.contains(id))
        .unwrap();
    endpoints[0] = foreign;
    *bad.transformations.last_mut().unwrap() =
        TransformationNode::ConformingTraceQuotient { source, endpoints };
    assert!(
        bad.to_bytes()
            .unwrap_err()
            .to_string()
            .contains("inconsistent endpoint Domains")
    );

    let mut bad = graph.clone();
    let GeometryActionNode::P1HarmonicExtension { interfaces, .. } = &mut bad.geometry_actions[0];
    let internal = bad
        .transformations
        .iter()
        .find_map(|node| match node {
            TransformationNode::ConformingTraceQuotient {
                source: crate::ConformingTraceSource::ConservingConnection(id),
                ..
            } if !interfaces.contains(id) => Some(*id),
            _ => None,
        })
        .unwrap();
    interfaces.push(internal);
    interfaces.sort_by_key(|id| id.ulid());
    assert!(
        bad.to_bytes()
            .unwrap_err()
            .to_string()
            .contains("exactly cover cross-group")
    );
}

#[test]
fn foreign_driver_and_configuration_references_fail_closed() {
    let graph = fixture(2);
    for variant in 0..5 {
        let mut bad = graph.clone();
        let GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            ..
        } = &mut bad.geometry_actions[0];
        let domain = harmonic_domains[0];
        match variant {
            0 => drivers[0] = FieldRepresentationId::new(100),
            1 => drivers.clear(),
            2 => bad.domains[domain.index()].configuration = DomainConfiguration::FixedGeometry,
            3 => {
                bad.domains[domain.index()].configuration =
                    DomainConfiguration::CurrentAleGeometry {
                        action: GeometryActionId::new(100),
                    }
            }
            4 => {
                harmonic_domains.remove(0);
            }
            _ => unreachable!(),
        }
        assert!(bad.to_bytes().is_err(), "variant {variant}");
    }
    let mut bad = fixture(1);
    let GeometryActionNode::P1HarmonicExtension { drivers, .. } = &bad.geometry_actions[0];
    bad.fields[drivers[1].index()].domain = bad.fields[drivers[0].index()].domain;
    assert!(
        bad.to_bytes()
            .unwrap_err()
            .to_string()
            .contains("disjoint Domain groups")
    );
}

#[test]
fn pullbacks_require_unique_fields_matching_configuration_and_time() {
    let graph = fixture(2);
    for variant in 0..5 {
        let mut bad = graph.clone();
        let selected = bad
            .transformations
            .iter()
            .position(|node| matches!(node, TransformationNode::GclCompatibleAlePullback { .. }))
            .unwrap();
        let TransformationNode::GclCompatibleAlePullback { velocity, .. } =
            bad.transformations[selected]
        else {
            unreachable!()
        };
        match variant {
            0 => {
                bad.transformations.push(bad.transformations[selected]);
            }
            1 => {
                bad.domains[bad.fields[velocity.index()].domain.index()].configuration =
                    DomainConfiguration::ReferenceConfiguration;
            }
            2 => {
                bad.transformations[selected] = TransformationNode::GclCompatibleAlePullback {
                    relation: Id::new(),
                    velocity,
                    geometry: GeometryActionId::new(0),
                };
            }
            3 => {
                let GeometryActionNode::P1HarmonicExtension { duration, .. } =
                    &mut bad.geometry_actions[0];
                *duration = DynQuantity::new(0.2, TIME);
            }
            4 => {
                bad.transformations.retain(|node| {
                    !matches!(node, TransformationNode::GclCompatibleAlePullback { .. })
                });
            }
            _ => unreachable!(),
        }
        bad.systems[0].transformations = (0..bad.transformations.len())
            .map(TransformationId::new)
            .collect();
        assert!(bad.to_bytes().is_err(), "variant {variant}");
    }
}

#[test]
fn wire_rejects_noncanonical_geometry_inventory_and_excessive_nested_counts() {
    let graph = fixture(2);
    let text = String::from_utf8(graph.to_bytes().unwrap()).unwrap();
    let GeometryActionNode::P1HarmonicExtension {
        harmonic_domains, ..
    } = &graph.geometry_actions[0];
    let [a, b] = harmonic_domains.as_slice() else {
        unreachable!()
    };
    let original = format!("\"harmonic_domains\":[{},{}]", a.index(), b.index());
    assert!(text.contains(&original));
    for (replacement, reason) in [
        (
            format!("\"harmonic_domains\":[{},{}]", b.index(), a.index()),
            "strictly ordered",
        ),
        (
            format!("\"harmonic_domains\":[{}]", vec!["0"; 100_001].join(",")),
            "inventory exceeds",
        ),
    ] {
        let bytes = text.replace(&original, &replacement);
        assert!(
            PortableRealizationGraph::from_bytes(bytes.as_bytes())
                .unwrap_err()
                .to_string()
                .contains(reason)
        );
    }
}
