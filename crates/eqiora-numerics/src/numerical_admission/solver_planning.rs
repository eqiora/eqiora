//! Private solver-decision plane for common root resolution.
//!
//! These functions select one exact admitted solver/provider tuple. They do
//! not recognize mathematics, choose a Formulation, admit spatial resources,
//! construct a Realization, or execute numerical work.

use super::*;

/// Admit explicit caller intent against facts owned by mathematical admission.
/// `Some` supplies a known diagonal fact; `None` makes no structural claim.
pub(super) fn resolve_linear(
    request: CommonLinearRequest,
    properties: LinearOperatorProperties,
    complete_diagonal: Option<bool>,
    required_reduction: Option<ReductionPolicy>,
    structure: Option<eqiora_solver::AlgebraicStructure>,
    supplied_backend: &dyn LinearSolverBackend,
) -> Result<NativeLinearPolicy, Diagnostic> {
    let profile = eqiora_solver::HostSerialSolverProfile::canonical_csr(
        properties,
        complete_diagonal,
        required_reduction,
    );
    let profile = match structure {
        Some(structure) => profile.with_structure(structure)?,
        None => profile,
    };
    if let Some((plan, provider)) = request.exact_request() {
        let backend = exact_backend(provider, supplied_backend)?;
        profile.require_plan(plan)?;
        backend
            .capabilities()
            .require_problem(plan, ScalarType::F64, properties)?;
        let mut selected = NativeLinearPolicy::exact(plan, backend)?;
        selected.planning_profile = Some(profile);
        return Ok(selected);
    }
    let objective = request
        .objective()
        .expect("linear intent is exact or program-controlled");
    let decision = eqiora_solver::plan_host_serial_solver_v2(
        profile,
        objective,
        request.relative_tolerance(),
        request.absolute_tolerance(),
        request.maximum_iterations(),
        &REFERENCE_LINEAR_SOLVER,
        supplied_backend,
    )?;
    let backend = exact_backend(decision.solver_provider(), supplied_backend)?;
    NativeLinearPolicy::exact(decision.solver_plan(), backend)?.with_planning(&decision)
}

fn exact_backend(
    provider: SolverProvider,
    supplied_backend: &dyn LinearSolverBackend,
) -> Result<&dyn LinearSolverBackend, Diagnostic> {
    if provider == REFERENCE_LINEAR_SOLVER.provider() {
        Ok(&REFERENCE_LINEAR_SOLVER)
    } else if provider == supplied_backend.provider() {
        Ok(supplied_backend)
    } else {
        Err(invalid(
            "exact solver provider identity, implementation version, and library inventory must match an admitted backend",
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
                    algorithm: LinearSolver::SparseLu,
                    operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                    preconditioner: PreconditionerPolicy::Identity,
                    reduction: ReductionPolicy::Fast,
                    scalar_type: ScalarType::F64,
                },
                SolverCapability {
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
                &ResolveOnlySparseBackend
            )
            .unwrap_err()
            .message()
            .contains("profile.complete-diagonal-required")
        );
    }
}
