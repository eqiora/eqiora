//! Private solver-decision plane for common root resolution.
//!
//! These functions select one exact admitted solver/provider tuple. They do
//! not recognize mathematics, choose a Formulation, admit spatial resources,
//! construct a Realization, or execute numerical work.

use super::*;

/// Admit explicit caller intent against facts owned by mathematical admission.
/// `Some` supplies a known diagonal fact; `None` makes no structural claim.
pub(crate) fn resolve_linear(
    request: CommonLinearRequest,
    properties: LinearOperatorProperties,
    complete_diagonal: Option<bool>,
    size: Option<eqiora_solver::SolverPlanningSize>,
    required_reduction: Option<ReductionPolicy>,
    structure: Option<eqiora_solver::AlgebraicStructure>,
    supplied_backend: &dyn LinearSolverBackend,
) -> Result<NativeLinearPolicy, Diagnostic> {
    let profile = planning_profile(
        request,
        properties,
        complete_diagonal,
        size,
        required_reduction,
        structure,
    )?;
    resolve_typed_linear(
        request,
        properties,
        profile,
        &REFERENCE_LINEAR_SOLVER,
        Some(supplied_backend),
    )
}

fn planning_profile(
    request: CommonLinearRequest,
    properties: LinearOperatorProperties,
    complete_diagonal: Option<bool>,
    size: Option<eqiora_solver::SolverPlanningSize>,
    required_reduction: Option<ReductionPolicy>,
    structure: Option<eqiora_solver::AlgebraicStructure>,
) -> Result<eqiora_solver::HostSerialSolverProfile, Diagnostic> {
    let profile = eqiora_solver::HostSerialSolverProfile::canonical_csr(
        properties,
        complete_diagonal,
        required_reduction,
    );
    let profile = match size {
        Some(size) => profile.with_size(size),
        None => profile,
    };
    let profile = match request.cost() {
        Some(cost) => {
            // No current common admission owner supplies a known retained preparation horizon.
            // A forecast is not authority to amortize setup across independent Runs/solves.
            if cost.structures() != cost.solves() || cost.numeric_preparations() != cost.solves() {
                return Err(invalid(
                    "common cost planning has no admitted retained preparation horizon; count each preparation independently",
                ));
            }
            profile.with_cost_request(cost)
        }
        None => profile,
    };
    let profile = match structure {
        Some(structure) => profile.with_structure(structure)?,
        None => profile,
    };
    Ok(profile)
}

pub(super) fn resolve_typed_linear<S>(
    request: CommonLinearRequest,
    properties: LinearOperatorProperties,
    profile: eqiora_solver::HostSerialSolverProfile,
    reference_backend: &dyn LinearSolverBackend<S>,
    supplied_backend: Option<&dyn LinearSolverBackend<S>>,
) -> Result<NativeLinearPolicy, Diagnostic>
where
    S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync,
{
    if let Some((plan, provider)) = request.exact_request() {
        let backend = exact_backend(
            provider,
            request.execution(),
            reference_backend,
            supplied_backend,
        )?;
        if request.execution() != (SERIAL_EXECUTION_PROVIDER, NonZeroUsize::MIN)
            && request
                .cost()
                .and_then(|cost| cost.numeric_payload_limit())
                .is_some()
        {
            return Err(invalid(
                "resource.numeric-payload-bound-unproved: this execution has no proven numeric-payload bound",
            ));
        }
        profile.require_plan(plan)?;
        backend
            .capabilities()
            .require_problem(plan, S::DOMAIN, S::STORAGE, properties)?;
        profile.require_resources(plan, backend.provider(), S::DOMAIN, S::STORAGE)?;
        let mut selected = NativeLinearPolicy::exact(plan, backend)?;
        if request.execution() == (SERIAL_EXECUTION_PROVIDER, NonZeroUsize::MIN) {
            selected.planning_costs = vec![profile.estimate_cost(
                "manual",
                plan,
                backend.provider(),
                S::DOMAIN,
                S::STORAGE,
            )];
        }
        selected.planning_profile = Some(profile);
        return Ok(selected);
    }
    if request.execution() != (SERIAL_EXECUTION_PROVIDER, NonZeroUsize::MIN) {
        return Err(invalid(
            "automatic solver ranking is implemented only for host-serial execution",
        ));
    }
    let supplied_backend = supplied_backend.filter(|backend| {
        backend.default_execution_provider() == SERIAL_EXECUTION_PROVIDER
            && backend.default_execution() == eqiora_solver::ExecutionReport::host_serial()
    });
    let objective = request
        .objective()
        .expect("linear intent is exact or program-controlled");
    let decision = eqiora_solver::ResolvedHostSerialSolverPlan::resolve(
        profile,
        objective,
        request.relative_tolerance(),
        request.absolute_tolerance(),
        request.maximum_iterations(),
        reference_backend,
        supplied_backend,
    )?;
    let backend = exact_backend(
        decision.solver_provider(),
        request.execution(),
        reference_backend,
        supplied_backend,
    )?;
    NativeLinearPolicy::exact(decision.solver_plan(), backend)?.with_planning(&decision)
}

/// Only exact Jacobi requests need a new structural fact in this admission slice.
/// Automatic selection retains its existing independently admitted profile.
pub(super) fn requests_diagonal(request: CommonLinearRequest) -> bool {
    request
        .exact_request()
        .is_some_and(|(plan, _)| plan.preconditioner() == PreconditionerPolicy::Jacobi)
}

/// Ask the canonical operator's existing diagonal owner about structural availability.
/// A stored zero is available here; numerical Jacobi validity remains execution-owned.
pub(super) fn complete_diagonal<S>(
    system: &eqiora_solver::CanonicalCsrSystemView<S>,
) -> Result<bool, Diagnostic>
where
    S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync,
{
    use eqiora_solver::LinearOperator;
    let mut diagonal = vec![S::zero(); system.rows()];
    Ok(system.diagonal(&mut diagonal)? == eqiora_solver::DiagonalAvailability::Available)
}

/// Resolve against the selected providers' typed complex implementations.
/// Structural facts come from the same mathematical owner as real admission.
pub(super) fn resolve_complex_linear(
    request: CommonLinearRequest,
    properties: LinearOperatorProperties,
    complete_diagonal: Option<bool>,
    size: Option<eqiora_solver::SolverPlanningSize>,
    required_reduction: Option<ReductionPolicy>,
    structure: Option<eqiora_solver::AlgebraicStructure>,
    supplied_backend: &dyn LinearSolverBackend,
) -> Result<NativeLinearPolicy, Diagnostic> {
    let profile = planning_profile(
        request,
        properties,
        complete_diagonal,
        size,
        required_reduction,
        structure,
    )?;
    let reference = typed_complex(&REFERENCE_LINEAR_SOLVER)?
        .expect("reference implements complex linear solves");
    // Manual selection only needs the selected provider's implementation.
    let supplied = if let Some((_, provider)) = request.exact_request() {
        Some(
            typed_complex(exact_backend(
                provider,
                request.execution(),
                &REFERENCE_LINEAR_SOLVER,
                Some(supplied_backend),
            )?)?
            .ok_or_else(|| {
                invalid("selected provider has no typed complex linear implementation")
            })?,
        )
    } else {
        typed_complex(supplied_backend)?
    };
    resolve_typed_linear(request, properties, profile, reference, supplied)
}

fn typed_complex(
    backend: &dyn LinearSolverBackend,
) -> Result<Option<&dyn LinearSolverBackend<num_complex::Complex64>>, Diagnostic> {
    let Some(complex) = backend.complex_backend() else {
        return Ok(None);
    };
    if complex.provider() != backend.provider()
        || complex.default_execution_provider() != backend.default_execution_provider()
        || complex.default_execution() != backend.default_execution()
    {
        return Err(invalid(
            "complex implementation belongs to a different provider",
        ));
    }
    Ok(Some(complex))
}

fn exact_backend<'a, S: eqiora_core::Scalar>(
    provider: SolverProvider,
    execution: (ExecutionProvider, NonZeroUsize),
    reference_backend: &'a dyn LinearSolverBackend<S>,
    supplied_backend: Option<&'a dyn LinearSolverBackend<S>>,
) -> Result<&'a dyn LinearSolverBackend<S>, Diagnostic> {
    execution.0.validate()?;
    let matches = |backend: &dyn LinearSolverBackend<S>| {
        backend.provider() == provider
            && backend.default_execution_provider() == execution.0
            && backend.default_execution()
                == eqiora_solver::ExecutionReport::host(execution.0.id(), execution.1)
    };
    if matches(reference_backend) {
        Ok(reference_backend)
    } else if let Some(supplied) = supplied_backend.filter(|backend| matches(*backend)) {
        Ok(supplied)
    } else {
        Err(invalid(
            "exact solver and execution provider identities, versions, libraries and workers must match an admitted backend",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_solver::{
        BackendId, LinearProblem, LinearSolution, ReplicatedLinearExecution, SolverCapability,
    };

    #[derive(Debug)]
    struct ResolveOnlySparseBackend;

    impl LinearSolverBackend for ResolveOnlySparseBackend {
        fn provider(&self) -> SolverProvider {
            SolverProvider::new(BackendId::new("eqiora.test-solver-decision"), "1", &[])
        }

        fn capabilities(&self) -> SolverCapabilities {
            SolverCapabilities::exact([
                SolverCapability {
                    scalar_domain: eqiora_core::ScalarDomain::Real,
                    algorithm: LinearSolver::SparseLu,
                    operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                    preconditioner: PreconditionerPolicy::Identity,
                    reduction: ReductionPolicy::Fast,
                    scalar_type: ScalarType::F64,
                },
                SolverCapability {
                    scalar_domain: eqiora_core::ScalarDomain::Real,
                    algorithm: LinearSolver::SparseLu,
                    operator_properties: LinearOperatorProperties::General,
                    preconditioner: PreconditionerPolicy::Identity,
                    reduction: ReductionPolicy::Fast,
                    scalar_type: ScalarType::F64,
                },
            ])
            .unwrap()
        }

        fn solve_with_execution(
            &self,
            _problem: &LinearProblem<'_>,
            _plan: SolverPlan,
            _execution: &dyn ReplicatedLinearExecution,
        ) -> Result<LinearSolution, Diagnostic> {
            unreachable!("solver decision tests never execute numerical work")
        }
    }

    fn exact(
        algorithm: LinearSolver,
        reduction: ReductionPolicy,
        provider: SolverProvider,
    ) -> CommonLinearRequest {
        let plan = SolverPlan::new(algorithm, 1e-8, 1e-10, NonZeroUsize::new(100).unwrap())
            .unwrap()
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(reduction);
        CommonLinearRequest::exact(plan, provider).unwrap()
    }

    #[test]
    fn canonical_solver_diagonal_fact_does_not_claim_numeric_jacobi_validity() {
        let storage = eqiora_assembly::CsrMatrix::from_sorted_csr(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![0., 1., 1., 0.],
        )
        .unwrap();
        let system = eqiora_solver::CanonicalCsrSystemView::new(
            &storage,
            &[1., 2.],
            LinearOperatorProperties::General,
        )
        .unwrap();
        assert!(complete_diagonal(&system).unwrap());
        let request = exact(
            LinearSolver::BiConjugateGradientStabilized,
            ReductionPolicy::Reproducible,
            REFERENCE_LINEAR_SOLVER.provider(),
        );
        let (plan, provider) = request.exact_request().unwrap();
        let request = CommonLinearRequest::exact(
            plan.with_preconditioner(PreconditionerPolicy::Jacobi),
            provider,
        )
        .unwrap();
        let decision = resolve_linear(
            request,
            LinearOperatorProperties::General,
            Some(true),
            None,
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let checked = decision
            .checked_backend(&REFERENCE_LINEAR_SOLVER, None)
            .unwrap();
        let error = LinearSolveRequest::new(&checked, decision.solver)
            .solve(&system.linear_problem().unwrap())
            .unwrap_err();
        assert!(
            error.message().contains("finite nonzero diagonal"),
            "{error}"
        );
        let absent = eqiora_assembly::CsrMatrix::from_sorted_csr(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            vec![1., 1.],
        )
        .unwrap();
        let absent = eqiora_solver::CanonicalCsrSystemView::new(
            &absent,
            &[1., 2.],
            LinearOperatorProperties::General,
        )
        .unwrap();
        assert!(!complete_diagonal(&absent).unwrap());
        let error = LinearSolveRequest::new(&checked, decision.solver)
            .solve(&absent.linear_problem().unwrap())
            .unwrap_err();
        assert!(error.message().contains("profile"), "{error}");
    }

    #[test]
    fn exact_intent_preserves_plan_provider_and_absent_objective_without_execution() {
        for (algorithm, reduction, properties, provider) in [
            (
                LinearSolver::ConjugateGradient,
                ReductionPolicy::Reproducible,
                LinearOperatorProperties::SymmetricPositiveDefinite,
                REFERENCE_LINEAR_SOLVER.provider(),
            ),
            (
                LinearSolver::MinimumResidual,
                ReductionPolicy::Reproducible,
                LinearOperatorProperties::SymmetricIndefinite,
                REFERENCE_LINEAR_SOLVER.provider(),
            ),
            (
                LinearSolver::SparseLu,
                ReductionPolicy::Fast,
                LinearOperatorProperties::SymmetricIndefinite,
                ResolveOnlySparseBackend.provider(),
            ),
        ] {
            let request = exact(algorithm, reduction, provider);
            let decision = resolve_linear(
                request,
                properties,
                Some(false),
                None,
                None,
                None,
                &ResolveOnlySparseBackend,
            )
            .unwrap();
            assert_eq!(decision.solver, request.exact_request().unwrap().0);
            assert_eq!(decision.provider, provider);
            assert_eq!(decision.planning_objective, None);
            assert!(decision.planning_audit_is_coherent());
        }
    }

    #[test]
    fn exact_intent_rejects_provider_substitution_and_incompatible_mathematics() {
        let reference = REFERENCE_LINEAR_SOLVER.provider();
        let stale = SolverProvider::new(reference.id(), "stale-release", reference.libraries());
        let request = exact(
            LinearSolver::ConjugateGradient,
            ReductionPolicy::Reproducible,
            stale,
        );
        assert!(
            resolve_linear(
                request,
                LinearOperatorProperties::SymmetricPositiveDefinite,
                Some(true),
                None,
                None,
                None,
                &ResolveOnlySparseBackend
            )
            .unwrap_err()
            .message()
            .contains("implementation version")
        );
        let request = exact(
            LinearSolver::ConjugateGradient,
            ReductionPolicy::Reproducible,
            reference,
        );
        assert!(
            resolve_linear(
                request,
                LinearOperatorProperties::SymmetricIndefinite,
                Some(true),
                None,
                None,
                None,
                &ResolveOnlySparseBackend
            )
            .is_err()
        );
        let (plan, provider) = request.exact_request().unwrap();
        let jacobi = CommonLinearRequest::exact(
            plan.with_preconditioner(PreconditionerPolicy::Jacobi),
            provider,
        )
        .unwrap();
        assert!(
            resolve_linear(
                jacobi,
                LinearOperatorProperties::SymmetricPositiveDefinite,
                Some(false),
                None,
                None,
                None,
                &ResolveOnlySparseBackend
            )
            .unwrap_err()
            .message()
            .contains("profile.complete-diagonal-required")
        );
    }
}
