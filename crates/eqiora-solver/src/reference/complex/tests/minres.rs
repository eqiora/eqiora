use super::*;
use crate::{ConvergenceReason, PreconditionerPolicy, REFERENCE_SOLVER_PROVIDER};
use eqiora_core::diagnostic::codes;
use std::sync::atomic::Ordering;

// Frobenius inverse bounds <1.5 (HPD), <1.2 (indefinite), with policy1e-12
// and RHS norm<14 bound solution error below3e-11. Fixed1e-10 permits roundoff.
const TOLERANCE: f64 = 1e-10;
struct Storage([Complex64; 4]);
impl CsrStorage<Complex64> for Storage {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }
    fn values(&self) -> &[Complex64] {
        &self.0
    }
}
fn plan(limit: usize, absolute: f64) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        0.,
        absolute,
        limit.try_into().unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
}
fn invoke(
    problem: &LinearProblem<'_, Complex64>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution<Complex64>,
) -> Result<LinearSolution<Complex64>, Diagnostic> {
    solve_real_coordinates(
        problem,
        plan,
        execution,
        problem.canonical_csr_system().unwrap(),
    )
}
fn assert_values(actual: &[Complex64], expected: &[Complex64]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((*actual - *expected).norm() < TOLERANCE);
    }
}

#[test]
fn hermitian_jacobi_preserves_normal_adjoint_initial_guesses_and_inertia() {
    let c = Complex64::new;
    let expected = [c(1., 1.), c(2., -1.)];
    for (off_diagonal, property, rhs) in [
        (
            1.,
            LinearOperatorProperties::HermitianPositiveDefinite,
            [c(5., 6.), c(3., -2.)],
        ),
        // det=-5, positive diagonal; congruence eigenvalues2.5,-.5.
        (
            3.,
            LinearOperatorProperties::Hermitian,
            [c(7., 10.), c(5., -4.)],
        ),
    ] {
        let storage = Storage([
            c(4., 0.),
            c(0., off_diagonal),
            c(0., -off_diagonal),
            c(1., 0.),
        ]);
        let source = CanonicalCsrSystemView::new(&storage, &rhs, property).unwrap();
        for orientation in [
            LinearOperatorOrientation::Normal,
            LinearOperatorOrientation::ConjugateTransposed,
        ] {
            let oriented = crate::Oriented::new(&source, orientation).unwrap();
            for initial in [expected, [c(0.25, -0.5), c(-0.25, 0.5)]] {
                let problem = LinearProblem::from_oriented_canonical(&oriented, &source, &rhs)
                    .unwrap()
                    .with_initial_guess(&initial)
                    .unwrap();
                let request = plan(4, 1e-12);
                let solution = invoke(&problem, request, &SERIAL_LINEAR_EXECUTION).unwrap();
                assert_values(solution.values(), &expected);
                let report = solution.report();
                assert_eq!(report.orientation(), orientation);
                assert_eq!(report.solver_plan(), request);
                assert_eq!(report.solver_provider(), REFERENCE_SOLVER_PROVIDER);
                assert!(report.true_residual_norm() <= 1e-12);
                assert!(report.completed_iterations() <= request.maximum_iterations().get());
                if initial == expected {
                    assert_eq!(solution.values(), &initial);
                    assert_eq!(report.completed_iterations(), 0);
                    assert_eq!(report.reason(), ConvergenceReason::InitialResidualSatisfied);
                } else {
                    assert_eq!(report.completed_iterations(), 2);
                }
            }
        }
    }
}

#[test]
fn typed_global_residual_precedes_physical_acceptance_and_rejection_continues() {
    let c = Complex64::new;
    let storage = Storage([c(4., 0.), c(1., 0.), c(1., 0.), c(1., 0.)]);
    let phase = c(1., 1.);
    let rhs = [5. * phase, 2. * phase];
    let source = CanonicalCsrSystemView::new(
        &storage,
        &rhs,
        LinearOperatorProperties::HermitianPositiveDefinite,
    )
    .unwrap();
    let execution = ObservedSerial::default();
    let physical = |values: &[Complex64]| {
        // This must run after a typed original action, within the recurrence,
        // rather than capturing a lowered candidate and checking only at return.
        assert!(execution.actions.load(Ordering::Relaxed) > 0);
        Ok(values.iter().all(|x| (*x - phase).norm() < TOLERANCE))
    };
    let problem = source
        .linear_problem()
        .unwrap()
        .with_candidate_check(&physical);
    let error = invoke(&problem, plan(1, 1.), &execution).unwrap_err();
    assert!(error.message().contains("plan limit of 1 iterations"));
    let solution = invoke(&problem, plan(2, 1.), &execution).unwrap();
    assert_values(solution.values(), &[phase, phase]);
    assert_eq!(solution.report().completed_iterations(), 2);
    assert!(execution.actions.load(Ordering::Relaxed) >= 4);
    let failure = |_: &[Complex64]| Err(invalid_realization("complex physical failure"));
    let problem = source
        .linear_problem()
        .unwrap()
        .with_candidate_check(&failure);
    assert_eq!(
        invoke(&problem, plan(2, 1.), &execution)
            .unwrap_err()
            .message(),
        "complex physical failure"
    );
}

#[test]
fn scaled_complex_norm_does_not_admit_zero_work_and_invalid_diagonals_fail() {
    let c = Complex64::new;
    let scale = 2f64.powi(80);
    let storage = Storage([c(scale, 0.), c(0., 0.), c(0., 0.), c(scale, 0.)]);
    let rhs = [c(1., 1.), c(0., 0.)];
    let source = CanonicalCsrSystemView::new(
        &storage,
        &rhs,
        LinearOperatorProperties::HermitianPositiveDefinite,
    )
    .unwrap();
    let solution = invoke(
        &source.linear_problem().unwrap(),
        plan(1, 2f64.powi(-30)),
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap();
    let expected = [c(2f64.powi(-80), 2f64.powi(-80)), c(0., 0.)];
    for (actual, expected) in solution.values().iter().zip(expected) {
        assert!((*actual - expected).norm() * scale <= 2f64.powi(-30));
    }
    assert_eq!(solution.report().completed_iterations(), 1);
    assert!(solution.report().initial_residual_norm() > 1.);
    assert!(solution.report().true_residual_norm() <= 2f64.powi(-30));
    for diagonal in [0., -1.] {
        let storage = Storage([c(diagonal, 0.), c(1., 0.), c(1., 0.), c(1., 0.)]);
        let source =
            CanonicalCsrSystemView::new(&storage, &rhs, LinearOperatorProperties::Hermitian)
                .unwrap();
        assert_eq!(
            invoke(
                &source.linear_problem().unwrap(),
                plan(2, 1e-12),
                &SERIAL_LINEAR_EXECUTION
            )
            .unwrap_err()
            .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
    }
}
