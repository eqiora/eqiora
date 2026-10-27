use super::*;
use eqiora_solver::{
    BackendId, LinearProblem, LinearSolution, LinearSolver, LinearSolverBackend,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReplicatedLinearExecution, SolverCapabilities,
    SolverPlan, SolverProvider,
};
use std::num::NonZeroUsize;

fn policy(algorithm: LinearSolver) -> SolverPlan {
    SolverPlan::new(algorithm, 1e-11, 1e-13, NonZeroUsize::new(2_000).unwrap()).unwrap()
}

#[test]
fn canonical_projection_solves_both_classes_and_replays_raw_coefficients() {
    for algorithm in [
        LinearSolver::MinimumResidual,
        LinearSolver::BiConjugateGradientStabilized,
    ] {
        let plan = policy(algorithm);
        let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
        require_projection_solver(request).unwrap();
        for (matrix, rhs, properties) in [
            (
                [2., 0., 0., 4.],
                [2., 8.],
                LinearOperatorProperties::SymmetricPositiveDefinite,
            ),
            // Constraint x=1 and stationarity 2x+lambda=4 fix (x,lambda)=(1,2).
            (
                [2., 1., 1., 0.],
                [4., 1.],
                LinearOperatorProperties::SymmetricIndefinite,
            ),
        ] {
            let (values, report, residual) =
                solve_dense(matrix.to_vec(), &rhs, properties, request).unwrap();
            // ||A^-1||_infinity <=3 for these literal matrices; the requested
            // residual bound is below 1e-10, so 1e-9 also bounds coordinate error.
            for (actual, expected) in values.iter().zip([1., 2.]) {
                assert!((actual - expected).abs() < 1e-9);
            }
            assert_eq!(report.solver_plan(), plan);
            assert_eq!(report.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
            assert!(report.completed_iterations() <= plan.maximum_iterations().get());
            assert!(report.true_residual_norm() <= report.residual_target());
            assert_eq!(residual, residual_norm(&matrix, &values, &rhs).unwrap());
            assert!(residual_norm(&matrix, &[0., 0.], &rhs).unwrap() > 1.);
        }
    }
}

#[test]
fn capture_preserves_the_existing_per_matrix_resource_boundary() {
    assert_eq!(require_dense_dimension(4_000).unwrap(), 16_000_000);
    require_capture_budget(16_000_000, 16_000_000).unwrap();
    assert!(require_dense_dimension(4_001).is_err());
    assert!(require_capture_budget(16_000_001, 1).is_err());
    assert!(require_capture_budget(4, 5).is_err());
    assert!(require_dense_dimension(usize::MAX).is_err());
    assert!(require_capture_budget(usize::MAX, 1).is_err());
}

#[derive(Debug)]
struct OneClass(LinearOperatorProperties);
impl LinearSolverBackend for OneClass {
    fn provider(&self) -> SolverProvider {
        SolverProvider::new(BackendId::new("test.one-projection-class"), "0", &[])
    }
    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact(
            REFERENCE_LINEAR_SOLVER
                .capabilities()
                .combinations()
                .iter()
                .filter(|tuple| tuple.operator_properties == self.0)
                .copied(),
        )
        .unwrap()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("missing class must reject before any projection solve")
    }
}

#[test]
fn projection_requires_both_exact_provider_classes_before_execution() {
    let plan = policy(LinearSolver::BiConjugateGradientStabilized);
    for properties in [
        LinearOperatorProperties::SymmetricPositiveDefinite,
        LinearOperatorProperties::SymmetricIndefinite,
    ] {
        let backend = OneClass(properties);
        let error = require_projection_solver(LinearSolveRequest::new(&backend, plan)).unwrap_err();
        assert_eq!(
            error.code(),
            eqiora_core::diagnostic::codes::INVALID_REALIZATION
        );
    }
    for plan in [
        policy(LinearSolver::ConjugateGradient),
        plan.with_preconditioner(PreconditionerPolicy::Jacobi),
    ] {
        assert!(
            require_projection_solver(LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan))
                .is_err()
        );
    }
}
