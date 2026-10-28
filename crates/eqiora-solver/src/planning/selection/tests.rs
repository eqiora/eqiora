use super::*;

fn deployment() -> (ExecutionProvider, crate::ExecutionReport) {
    // Declared metadata only: these tests do not discover or execute a device.
    let provider = ExecutionProvider::new(
        crate::ExecutionId::new("eqiora.cuda.single-device"),
        env!("CARGO_PKG_VERSION"),
        &[],
    );
    (provider, crate::ExecutionReport::cuda(provider.id(), 2))
}

fn available() -> SolverProviderAvailability {
    let (provider, execution) = deployment();
    SolverProviderAvailability::new(
        CUDA_PROVIDER,
        provider,
        execution,
        crate::SolverCapabilities::exact([
            SolverCapability {
                scalar_domain: ScalarDomain::Real,
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: ScalarDomain::Real,
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: ScalarDomain::Real,
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
        ])
        .unwrap(),
    )
    .unwrap()
}

fn profile(
    properties: LinearOperatorProperties,
    diagonal: Option<bool>,
    reduction: Option<ReductionPolicy>,
) -> SolverPlanningProfile {
    let (provider, execution) = deployment();
    SolverPlanningProfile::canonical_csr(properties, diagonal, reduction)
        .with_execution(provider, execution)
        .unwrap()
        .with_size(SolverPlanningSize::exact(3, 7))
}

fn select(
    profile: SolverPlanningProfile,
    objective: SolverPlanningObjective,
    availability: &[SolverProviderAvailability],
) -> Result<SolverSelection, Diagnostic> {
    SolverSelection::resolve::<f64>(
        profile,
        objective,
        2e-9,
        3e-12,
        NonZeroUsize::new(73).unwrap(),
        availability,
    )
}

#[test]
pub(crate) fn declared_cuda_selection_has_literal_tuples_and_unknown_costs() {
    // Literal independent policy requirements, not values read from the catalog.
    for (properties, algorithm, preconditioner, id) in [
        (
            LinearOperatorProperties::SymmetricPositiveDefinite,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            "eqiora.cuda.cg-spd-jacobi-fast-f64",
        ),
        (
            LinearOperatorProperties::General,
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
            "eqiora.cuda.bicgstab-general-identity-fast-f64",
        ),
        (
            LinearOperatorProperties::SymmetricIndefinite,
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
            "eqiora.cuda.minres-indefinite-identity-fast-f64",
        ),
    ] {
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let selection = select(
                profile(properties, Some(true), None),
                objective,
                &[available()],
            )
            .unwrap();
            let expected = SolverPlan::new(algorithm, 2e-9, 3e-12, NonZeroUsize::new(73).unwrap())
                .unwrap()
                .with_preconditioner(preconditioner)
                .with_reduction(ReductionPolicy::Fast);
            assert_eq!(selection.solver_plan(), expected);
            assert_eq!(selection.selected_candidate_id(), id);
            assert_eq!(selection.execution(), deployment().1);
            assert_eq!(selection.solver_provider(), CUDA_PROVIDER);
            assert_eq!(selection.costs().len(), 1);
            let cost = selection.costs()[0];
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
            assert!(
                selection
                    .reasons()
                    .any(|pair| pair == (id, "cost.execution-overhead-unmodeled"))
            );
            assert_eq!(
                selection.reasons().collect::<Vec<_>>(),
                select(
                    profile(properties, Some(true), None),
                    objective,
                    &[available()]
                )
                .unwrap()
                .reasons()
                .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
pub(crate) fn declared_cuda_requirements_cannot_be_ranked_away() {
    let general = profile(LinearOperatorProperties::General, None, None);
    let error = |profile, available: &[SolverProviderAvailability]| {
        select(profile, SolverPlanningObjective::Fast, available)
            .unwrap_err()
            .message()
            .to_owned()
    };
    assert!(error(general.clone(), &[]).contains("catalog.provider-unavailable"));
    assert!(
        error(
            profile(
                LinearOperatorProperties::SymmetricPositiveDefinite,
                None,
                None
            ),
            &[available()]
        )
        .contains("diagonal")
    );
    assert!(
        error(
            profile(
                LinearOperatorProperties::General,
                None,
                Some(ReductionPolicy::Reproducible)
            ),
            &[available()]
        )
        .contains("reduction")
    );
    let capped = general.clone().with_cost_request(
        SolverCostRequest::default()
            .with_numeric_payload_limit(NonZeroUsize::new(usize::MAX).unwrap()),
    );
    assert!(error(capped, &[available()]).contains("resource.numeric-payload-bound-unproved"));
    let mut wrong_execution = available();
    wrong_execution.execution = crate::ExecutionReport::cuda(deployment().0.id(), 3);
    assert!(error(general.clone(), &[wrong_execution]).contains("catalog.execution-mismatch"));
    let mut stale = available();
    stale.provider = SolverProvider::new(CUDA_PROVIDER.id(), "stale", &[]);
    assert!(error(general.clone(), &[stale]).contains("catalog.provider-unavailable"));
    assert!(
        error(general.clone(), &[available(), available()])
            .contains("availability.duplicate-provider")
    );
    assert!(
        SolverSelection::resolve::<num_complex::Complex64>(
            general,
            SolverPlanningObjective::Fast,
            2e-9,
            3e-12,
            NonZeroUsize::new(73).unwrap(),
            &[available()]
        )
        .unwrap_err()
        .message()
        .contains("catalog.unsupported-profile")
    );
}

#[test]
pub(crate) fn declared_host_selection_preserves_existing_reference_choice() {
    use crate::LinearSolverBackend;
    let backend = crate::REFERENCE_LINEAR_SOLVER;
    let availability = SolverProviderAvailability::new(
        REFERENCE_PROVIDER,
        SERIAL_EXECUTION_PROVIDER,
        crate::ExecutionReport::host_serial(),
        <crate::reference::ReferenceLinearSolver as LinearSolverBackend<f64>>::capabilities(
            &backend,
        ),
    )
    .unwrap();
    for objective in [
        SolverPlanningObjective::Robust,
        SolverPlanningObjective::Fast,
        SolverPlanningObjective::LowMemory,
    ] {
        let selected = select(
            SolverPlanningProfile::general_canonical_csr(),
            objective,
            std::slice::from_ref(&availability),
        )
        .unwrap();
        assert_eq!(
            selected.selected_candidate_id(),
            "eqiora.reference.bicgstab-general-identity-reproducible-f64"
        );
        assert_eq!(
            selected.solver_plan().reduction(),
            ReductionPolicy::Reproducible
        );
        assert_eq!(selected.execution(), crate::ExecutionReport::host_serial());
    }
}
