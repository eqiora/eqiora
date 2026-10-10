use super::*;

fn graph(
    space: Space,
    dimension: usize,
    points: usize,
    quotient: bool,
) -> Result<PortableRealizationGraph, Diagnostic> {
    let domains = [Id::new(), Id::new()];
    let fields = [Id::new(), Id::new()];
    let count = if quotient { 2 } else { 1 };
    let regions = (0..count)
        .map(|i| {
            crate::DomainFieldDiscretization::new(
                domains[i],
                [FieldSpaceBinding::new(fields[i], space)],
                [],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quotients = if quotient {
        vec![crate::ConformingTraceQuotient::new(
            crate::ConformingTraceSource::ConservingConnection(Id::new()),
            crate::TraceFieldEndpoint::new(domains[0], fields[0]),
            crate::TraceFieldEndpoint::new(domains[1], fields[1]),
        )?]
    } else {
        vec![]
    };
    let quadrature = if space == Space::continuous_lagrange(NonZeroU16::MIN) {
        QuadraturePolicy::SimplexCentroid
    } else {
        QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: NonZeroUsize::new(dimension).unwrap(),
            points_per_axis: NonZeroUsize::new(points).unwrap(),
        }
    };
    PortableRealizationGraph::linear_regions(
        RealizationLineage::explicit(
            OntologyId::new(),
            SemanticRevision::new(1),
            RealizationRevision::new(1),
        ),
        regions,
        quotients,
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([7; 32]),
            },
            quadrature,
        ),
        LinearOperatorProperties::General,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-10,
            1e-12,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
}

#[test]
fn moment_graphs_preserve_functionals_and_check_their_numerical_profile_on_replay() {
    for space in [Space::tetrahedral_edge(), Space::tetrahedral_face()] {
        let graph = graph(space, 3, 3, false).unwrap();
        let bytes = graph.to_bytes().unwrap();
        let replay = PortableRealizationGraph::from_bytes(&bytes).unwrap();
        assert_eq!(replay, graph);
        assert_eq!(replay.fields()[0].space(), space);
        assert_eq!(replay.digest().unwrap(), graph.digest().unwrap());
        let text = String::from_utf8(bytes).unwrap();
        for (before, after) in [
            ("\"spatial_dimension\":3", "\"spatial_dimension\":2"),
            ("\"points_per_axis\":3", "\"points_per_axis\":2"),
        ] {
            let mutated = text.replace(before, after);
            assert_ne!(text, mutated);
            let error = PortableRealizationGraph::from_bytes(mutated.as_bytes()).unwrap_err();
            assert!(error.message().contains("tetrahedral moments"));
        }
        for old in ["v1", "v2"] {
            let stale = text.replace(
                "portable-realization-graph/v4",
                &format!("portable-realization-graph/{old}"),
            );
            assert!(PortableRealizationGraph::from_bytes(stale.as_bytes()).is_err());
        }
        let mut other = graph.clone();
        other.fields[0].space = if space == Space::tetrahedral_edge() {
            Space::tetrahedral_face()
        } else {
            Space::tetrahedral_edge()
        };
        assert_ne!(other.digest().unwrap(), graph.digest().unwrap());
        let mut wrong_mesh = graph.clone();
        wrong_mesh.domains[0].discretization = Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::MIN,
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        );
        assert!(wrong_mesh.to_bytes().is_err());
        let error = self::graph(space, 3, 3, true).unwrap_err();
        assert!(
            error
                .message()
                .contains("unadmitted transformations or constraints")
        );
        let mut constrained = graph;
        constrained.systems[0].operator_properties = LinearOperatorProperties::SymmetricIndefinite;
        constrained.systems[0]
            .blocks
            .push(SystemBlock::ConstraintMultiplier(
                AlgebraicConstraint::ZeroIntegral {
                    field: constrained.fields[0].field,
                },
            ));
        assert!(
            constrained
                .to_bytes()
                .unwrap_err()
                .message()
                .contains("unadmitted transformations or constraints")
        );
    }
    graph(Space::continuous_lagrange(NonZeroU16::MIN), 3, 3, true).unwrap();
    let mut scalar = graph(Space::continuous_lagrange(NonZeroU16::MIN), 3, 3, false).unwrap();
    scalar.systems[0].operator_properties = LinearOperatorProperties::SymmetricIndefinite;
    scalar.systems[0]
        .blocks
        .push(SystemBlock::ConstraintMultiplier(
            AlgebraicConstraint::ZeroIntegral {
                field: scalar.fields[0].field,
            },
        ));
    scalar.to_bytes().unwrap();
    assert!(graph(Space::tetrahedral_edge(), 2, 3, false).is_err());
    assert!(graph(Space::tetrahedral_face(), 3, 2, false).is_err());
}

#[test]
fn planar_nodal_duffy_graph_replays_and_checks_its_discretization_profile() {
    let mut graph = graph(Space::continuous_lagrange(NonZeroU16::MIN), 3, 3, false).unwrap();
    graph.domains[0].discretization = Discretization::new(
        DiscretizationMethod::ContinuousGalerkin,
        graph.domains[0].discretization.mesh(),
        QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: NonZeroUsize::new(2).unwrap(),
            points_per_axis: NonZeroUsize::new(3).unwrap(),
        },
    );
    let bytes = graph.to_bytes().unwrap();
    assert_eq!(PortableRealizationGraph::from_bytes(&bytes).unwrap(), graph);
    // A generic graph can represent other profiles (including 3D FSI).
    // The linear-region constructor's Space/discretization owner admits this
    // narrower profile; Common Plan replay also reconstructs the exact graph.
    for (dimension, points) in [(3, 3), (2, 1)] {
        let invalid = Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            graph.domains[0].discretization.mesh(),
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(dimension).unwrap(),
                points_per_axis: NonZeroUsize::new(points).unwrap(),
            },
        );
        assert!(
            invalid
                .validate_space(Space::continuous_lagrange(NonZeroU16::MIN))
                .unwrap_err()
                .message()
                .contains("planar P1")
        );
    }
}
