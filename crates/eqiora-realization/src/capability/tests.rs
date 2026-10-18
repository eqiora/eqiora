use super::*;

fn solver(algorithm: LinearSolver, properties: LinearOperatorProperties) -> SolverCapability {
    SolverCapability {
        scalar_domain: eqiora_core::ScalarDomain::Real,
        algorithm,
        operator_properties: properties,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }
}

fn plan(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(algorithm, 1.0e-10, 1.0e-12, NonZeroUsize::MIN)
        .expect("test solver plan is valid")
}

#[test]
fn exact_tuple_rejects_an_invalid_solver_property_pair() {
    let context = RealizationCapabilityContext::new(
        SpatialCapability::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        ),
        VectorLayoutKind::Replicated,
        TargetCapability::HostCpu {
            maximum_threads: NonZeroUsize::MIN,
        },
        ScheduleCapability::Offline,
    );
    let error = RealizationCapability::new(
        context,
        solver(
            LinearSolver::ConjugateGradient,
            LinearOperatorProperties::General,
        ),
    )
    .unwrap_err();
    assert_eq!(
        error.code(),
        eqiora_core::diagnostic::codes::INVALID_REALIZATION
    );
}

#[test]
fn complex_solver_support_does_not_imply_spatial_realization_support() {
    let context = RealizationCapabilityContext::new(
        SpatialCapability::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        ),
        VectorLayoutKind::Replicated,
        TargetCapability::HostCpu {
            maximum_threads: NonZeroUsize::MIN,
        },
        ScheduleCapability::Offline,
    );
    let mut tuple = solver(
        LinearSolver::BiConjugateGradientStabilized,
        LinearOperatorProperties::General,
    );
    RealizationCapability::new(context, tuple).unwrap();
    tuple.scalar_domain = eqiora_core::ScalarDomain::Complex;
    SolverCapabilities::exact([tuple]).unwrap();
    let error = RealizationCapability::new(context, tuple).unwrap_err();
    assert!(error.message().contains("real scalar operators"));
}

#[test]
fn real_time_schedule_admission_is_exact() {
    let admitted = ScheduleCapability::RealTime {
        priority: 3,
        deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
    };
    assert!(admitted.supports(ExecutionSchedule::RealTime {
        priority: 3,
        deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
    }));
    assert!(!admitted.supports(ExecutionSchedule::RealTime {
        priority: 4,
        deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
    }));
    assert!(!admitted.supports(ExecutionSchedule::RealTime {
        priority: 3,
        deadline_ns: NonZeroU64::new(20_000).expect("deadline is non-zero"),
    }));
    assert!(!admitted.supports(ExecutionSchedule::Offline));
}

#[test]
fn exact_admission_tuples_reject_recombined_axes() {
    let dimension = NonZeroUsize::new(2).expect("two is non-zero");
    let host_fem = RealizationCapability::new(
        RealizationCapabilityContext::new(
            SpatialCapability::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(dimension),
            ),
            VectorLayoutKind::Replicated,
            TargetCapability::HostCpu {
                maximum_threads: NonZeroUsize::MIN,
            },
            ScheduleCapability::Offline,
        ),
        solver(
            LinearSolver::ConjugateGradient,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        ),
    )
    .expect("host FEM tuple is coherent");
    let cuda_fvm = RealizationCapability::new(
        RealizationCapabilityContext::new(
            SpatialCapability::new(
                DiscretizationMethod::CellCenteredFiniteVolume,
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(dimension),
            ),
            VectorLayoutKind::Distributed,
            TargetCapability::CudaGpu { device: 3 },
            ScheduleCapability::Offline,
        ),
        solver(
            LinearSolver::BiConjugateGradientStabilized,
            LinearOperatorProperties::General,
        ),
    )
    .expect("CUDA FVM tuple is coherent");
    let capabilities =
        RealizationCapabilities::exact([host_fem, cuda_fvm]).expect("two exact tuples");

    assert!(
        capabilities
            .supports_components(
                RealizationRequirements::new(
                    dimension,
                    ScalarType::F64,
                    VectorLayoutKind::Replicated,
                ),
                DiscretizationMethod::ContinuousGalerkin,
                MeshKind::GeneratedCartesian,
                plan(LinearSolver::ConjugateGradient),
                Some(LinearOperatorProperties::SymmetricPositiveDefinite),
                Target::HostCpu {
                    threads: NonZeroUsize::MIN,
                },
                ExecutionSchedule::Offline,
            )
            .is_ok()
    );
    assert!(
        capabilities
            .supports_components(
                RealizationRequirements::new(
                    dimension,
                    ScalarType::F64,
                    VectorLayoutKind::Distributed,
                ),
                DiscretizationMethod::CellCenteredFiniteVolume,
                MeshKind::GeneratedCartesian,
                plan(LinearSolver::BiConjugateGradientStabilized),
                Some(LinearOperatorProperties::General),
                Target::CudaGpu { device: 3 },
                ExecutionSchedule::Offline,
            )
            .is_ok()
    );

    let recombined = capabilities.supports_components(
        RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated),
        DiscretizationMethod::CellCenteredFiniteVolume,
        MeshKind::GeneratedCartesian,
        plan(LinearSolver::BiConjugateGradientStabilized),
        Some(LinearOperatorProperties::General),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    );
    assert!(recombined.is_err());
}
#[test]
fn spatial_projection_keeps_real_tuples_from_a_mixed_provider_inventory() {
    let inventory = SolverCapabilities::reference();
    let build = |solver| {
        RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::MIN),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
    };
    let projected = build(inventory.clone()).unwrap();
    let expected = inventory
        .combinations()
        .iter()
        .copied()
        .filter(|tuple| tuple.scalar_domain == eqiora_core::ScalarDomain::Real)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        projected
            .combinations()
            .iter()
            .map(|tuple| tuple.solver())
            .collect::<BTreeSet<_>>(),
        expected
    );
    let complex = SolverCapabilities::exact(
        inventory
            .combinations()
            .iter()
            .copied()
            .filter(|tuple| tuple.scalar_domain == eqiora_core::ScalarDomain::Complex),
    )
    .unwrap();
    assert!(build(complex).is_err());
}
