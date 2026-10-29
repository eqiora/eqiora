//! Literal existing MPI tuples: metadata selection only, no MPI initialization.
use super::*;

fn availability() -> SolverProviderAvailability {
    const LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("mpi-rs", "0.8.2")];
    let provider = ExecutionProvider::new(
        crate::ExecutionId::new("eqiora.mpi"),
        env!("CARGO_PKG_VERSION"),
        LIBRARIES,
    );
    SolverProviderAvailability::new(
        MPI_PROVIDER,
        provider,
        crate::ExecutionReport::distributed(provider.id(), NonZeroUsize::new(2).unwrap()),
        crate::SolverCapabilities::exact(
            [
                (
                    LinearSolver::ConjugateGradient,
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                    PreconditionerPolicy::Jacobi,
                    ReductionPolicy::Reproducible,
                ),
                (
                    LinearSolver::ConjugateGradient,
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                    PreconditionerPolicy::Jacobi,
                    ReductionPolicy::Fast,
                ),
                (
                    LinearSolver::MinimumResidual,
                    LinearOperatorProperties::SymmetricIndefinite,
                    PreconditionerPolicy::Identity,
                    ReductionPolicy::Reproducible,
                ),
            ]
            .map(
                |(algorithm, operator_properties, preconditioner, reduction)| SolverCapability {
                    scalar_domain: ScalarDomain::Real,
                    algorithm,
                    operator_properties,
                    preconditioner,
                    reduction,
                    scalar_type: ScalarType::F64,
                },
            ),
        )
        .unwrap(),
    )
    .unwrap()
}

fn profile(
    properties: LinearOperatorProperties,
    diagonal: Option<bool>,
    reduction: Option<ReductionPolicy>,
) -> SolverPlanningProfile {
    let a = availability();
    SolverPlanningProfile::canonical_csr(properties, diagonal, reduction)
        .with_execution(a.execution_provider, a.execution)
        .unwrap()
        .with_size(SolverPlanningSize::exact(3, 7))
}

fn select(
    profile: SolverPlanningProfile,
    objective: SolverPlanningObjective,
    a: &[SolverProviderAvailability],
) -> Result<SolverSelection, Diagnostic> {
    SolverSelection::resolve_available::<f64>(
        profile,
        objective,
        2e-9,
        3e-12,
        NonZeroUsize::new(73).unwrap(),
        a,
    )
}

#[test]
pub(crate) fn declared_mpi_selection_preserves_literal_tuples_and_unknown_costs() {
    for (objective, properties, reduction, algorithm, preconditioner, id) in [
        (
            SolverPlanningObjective::Robust,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            ReductionPolicy::Reproducible,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            "eqiora.mpi.cg-spd-jacobi-reproducible-f64",
        ),
        (
            SolverPlanningObjective::Fast,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            ReductionPolicy::Fast,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            "eqiora.mpi.cg-spd-jacobi-fast-f64",
        ),
        (
            SolverPlanningObjective::Fast,
            LinearOperatorProperties::SymmetricIndefinite,
            ReductionPolicy::Reproducible,
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            "eqiora.mpi.minres-indefinite-identity-reproducible-f64",
        ),
    ] {
        let input = profile(properties, Some(true), None);
        let choice = select(input.clone(), objective, &[availability()]).unwrap();
        assert_eq!(choice.selected_candidate_id(), id);
        assert_eq!(
            choice.solver_plan(),
            SolverPlan::new(algorithm, 2e-9, 3e-12, NonZeroUsize::new(73).unwrap())
                .unwrap()
                .with_preconditioner(preconditioner)
                .with_reduction(reduction)
        );
        assert_eq!(choice.solver_provider(), MPI_PROVIDER);
        assert_eq!(choice.execution(), availability().execution);
        for cost in choice.costs() {
            assert_eq!(
                (
                    cost.setup_work(),
                    cost.solve_work(),
                    cost.acceptance_work(),
                    cost.total_work(),
                    cost.numeric_payload_bytes()
                ),
                (None, None, None, None, None)
            );
        }
        assert!(
            choice
                .reasons()
                .any(|(_, reason)| reason == "cost.execution-overhead-unmodeled")
        );
        assert_eq!(
            choice.reasons().collect::<Vec<_>>(),
            select(input, objective, &[availability()])
                .unwrap()
                .reasons()
                .collect::<Vec<_>>()
        );
    }
    // An explicit reduction is admission, even when ranking asks for the other intent.
    let choice = select(
        profile(
            LinearOperatorProperties::SymmetricPositiveDefinite,
            Some(true),
            Some(ReductionPolicy::Reproducible),
        ),
        SolverPlanningObjective::Fast,
        &[availability()],
    )
    .unwrap();
    assert_eq!(
        choice.solver_plan().reduction(),
        ReductionPolicy::Reproducible
    );
}

#[test]
pub(crate) fn declared_mpi_requirements_cannot_be_ranked_away() {
    let spd = profile(
        LinearOperatorProperties::SymmetricPositiveDefinite,
        Some(true),
        None,
    );
    assert!(
        select(spd.clone(), SolverPlanningObjective::Fast, &[])
            .unwrap_err()
            .message()
            .contains("provider-unavailable")
    );
    assert!(
        select(
            profile(
                LinearOperatorProperties::SymmetricPositiveDefinite,
                None,
                None
            ),
            SolverPlanningObjective::Fast,
            &[availability()]
        )
        .unwrap_err()
        .message()
        .contains("diagonal")
    );
    assert!(
        select(
            profile(
                LinearOperatorProperties::SymmetricIndefinite,
                None,
                Some(ReductionPolicy::Fast)
            ),
            SolverPlanningObjective::Fast,
            &[availability()]
        )
        .is_err()
    );
    assert!(
        select(
            profile(LinearOperatorProperties::General, Some(true), None),
            SolverPlanningObjective::Fast,
            &[availability()]
        )
        .is_err()
    );
    let capped = spd.clone().with_cost_request(
        SolverCostRequest::default()
            .with_numeric_payload_limit(NonZeroUsize::new(1_000_000).unwrap()),
    );
    assert!(
        select(
            capped,
            SolverPlanningObjective::LowMemory,
            &[availability()]
        )
        .is_err()
    );
    let mut foreign = availability();
    foreign.provider = CUDA_PROVIDER;
    assert!(select(spd.clone(), SolverPlanningObjective::Robust, &[foreign]).is_err());
    let mut foreign_execution = availability();
    foreign_execution.execution = crate::ExecutionReport::distributed(
        foreign_execution.execution_provider.id(),
        NonZeroUsize::new(4).unwrap(),
    );
    assert!(
        select(spd, SolverPlanningObjective::Robust, &[foreign_execution])
            .unwrap_err()
            .message()
            .contains("execution-mismatch")
    );
}

#[test]
pub(crate) fn selected_mpi_provider_snapshot_rejects_replacement_even_if_tuple_survives() {
    let availability = availability();
    let choice = select(
        profile(
            LinearOperatorProperties::SymmetricPositiveDefinite,
            Some(true),
            None,
        ),
        SolverPlanningObjective::Robust,
        std::slice::from_ref(&availability),
    )
    .unwrap();
    choice
        .require_declared_provider(availability.provider, &availability.capabilities)
        .unwrap();
    assert!(
        choice
            .require_declared_provider(CUDA_PROVIDER, &availability.capabilities)
            .is_err()
    );
    let only_selected = crate::SolverCapabilities::exact([SolverCapability {
        scalar_domain: ScalarDomain::Real,
        scalar_type: ScalarType::F64,
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Jacobi,
        reduction: ReductionPolicy::Reproducible,
    }])
    .unwrap();
    assert!(
        choice
            .require_declared_provider(availability.provider, &only_selected)
            .is_err()
    );
}
