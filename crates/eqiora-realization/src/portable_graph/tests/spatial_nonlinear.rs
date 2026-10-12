use super::*;

#[test]
fn spatial_nonlinear_root_preserves_policy_and_linear_graph_inventory() {
    let lineage = RealizationLineage::explicit(
        OntologyId::new(),
        SemanticRevision::new(1),
        RealizationRevision::new(1),
    );
    let domains = [Id::new(), Id::new()];
    let fields = [Id::new(), Id::new()];
    let regions = domains
        .iter()
        .zip(fields)
        .map(|(&domain, field)| {
            crate::DomainFieldDiscretization::new(
                domain,
                [FieldSpaceBinding::new(
                    field,
                    Space::continuous_lagrange(NonZeroU16::MIN),
                )],
                [AlgebraicConstraint::ZeroIntegral { field }],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let quotient = crate::ConformingTraceQuotient::new(
        crate::ConformingTraceSource::ConservingConnection(Id::new()),
        crate::TraceFieldEndpoint::new(domains[0], fields[0]),
        crate::TraceFieldEndpoint::new(domains[1], fields[1]),
    )
    .unwrap();
    let build = |nonlinear, properties, algorithm| {
        PortableRealizationGraph::spatial_regions(
            lineage,
            regions.clone(),
            [quotient],
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
            properties,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
            SolverPlan::new(algorithm, 1e-9, 1e-12, NonZeroUsize::new(100).unwrap()).unwrap(),
            nonlinear,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
            SystemScaling::Dimensional,
        )
    };
    let policy = NonlinearSolvePlan::new(1e-7, 1e-10, NonZeroUsize::new(17).unwrap(), 9).unwrap();
    let graph = build(
        Some(policy),
        LinearOperatorProperties::General,
        LinearSolver::BiConjugateGradientStabilized,
    )
    .unwrap();
    let SolveRoot::Nonlinear(root) = graph.root() else {
        panic!("explicit nonlinear policy must own the solve root");
    };
    let node = graph.nonlinear_solve(root).unwrap();
    assert_eq!(graph.nonlinear_solves().len(), 1);
    assert_eq!(node.plan(), policy);
    assert_eq!(node.residual_system(), AlgebraicSystemId::new(0));
    assert_eq!(node.linearization(), LinearSolveId::new(0));
    assert_eq!(
        graph.systems()[0].operator_properties(),
        LinearOperatorProperties::General
    );
    assert_eq!(graph.systems()[0].blocks().len(), 4);
    assert_eq!(graph.transformations().len(), 1);
    assert!(matches!(
        graph.transformations()[0],
        TransformationNode::ConformingTraceQuotient { .. }
    ));
    let bytes = graph.to_bytes().unwrap();
    let replay = PortableRealizationGraph::from_bytes(&bytes).unwrap();
    assert_eq!(replay, graph);
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    assert_eq!(replay.digest().unwrap(), graph.digest().unwrap());

    let linear = build(
        None,
        LinearOperatorProperties::SymmetricIndefinite,
        LinearSolver::MinimumResidual,
    )
    .unwrap();
    assert!(matches!(linear.root(), SolveRoot::Linear(_)));
    assert!(linear.nonlinear_solves().is_empty());
    assert_eq!(linear.domains(), graph.domains());
    assert_eq!(linear.fields(), graph.fields());
    assert_eq!(linear.transformations(), graph.transformations());
    assert_eq!(linear.systems()[0].blocks(), graph.systems()[0].blocks());
    assert_ne!(linear.digest().unwrap(), graph.digest().unwrap());

    let different_policy =
        NonlinearSolvePlan::new(1e-6, 1e-10, NonZeroUsize::new(17).unwrap(), 9).unwrap();
    assert_ne!(
        build(
            Some(different_policy),
            LinearOperatorProperties::General,
            LinearSolver::BiConjugateGradientStabilized
        )
        .unwrap()
        .digest()
        .unwrap(),
        graph.digest().unwrap()
    );
    // No generic nonlinear tangent symmetry follows from its current coefficients.
    let error = build(
        Some(policy),
        LinearOperatorProperties::SymmetricIndefinite,
        LinearSolver::MinimumResidual,
    )
    .unwrap_err();
    assert!(error.message().contains("general Jacobian"));
    // An explicit general tangent cannot be sent to a symmetry-only solver.
    let error = build(
        Some(policy),
        LinearOperatorProperties::General,
        LinearSolver::MinimumResidual,
    )
    .unwrap_err();
    assert!(error.message().contains("incompatible"));
    // The nonlinear extension does not broaden the existing linear multiplier contract.
    assert!(
        build(
            None,
            LinearOperatorProperties::General,
            LinearSolver::BiConjugateGradientStabilized
        )
        .is_err()
    );
}
