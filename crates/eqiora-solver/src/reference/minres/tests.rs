use super::*;
use crate::{LinearOperator, LinearOperatorProperties, SERIAL_LINEAR_EXECUTION};

mod validation;

// Fixed before execution: ||A^-1||_F<1.5 for SPD and <1.2 for indefinite.
// The 1e-12 policy gives an error bound below 1e-11 on these RHS; 1e-10
// leaves room for binary64 orthogonalization without fitting observed outputs.
const TOLERANCE: f64 = 1e-10;

#[derive(Debug)]
struct Dense<const N: usize>([[f64; N]; N]);
impl<const N: usize> LinearOperator for Dense<N> {
    type Scalar = f64;
    fn rows(&self) -> usize {
        N
    }
    fn columns(&self) -> usize {
        N
    }
    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        for (value, row) in output.iter_mut().zip(self.0) {
            *value = row.iter().zip(input).map(|(a, x)| a * x).sum();
        }
        Ok(())
    }
    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.0[i][i];
        }
        Ok(DiagonalAvailability::Available)
    }
}
fn plan(limit: usize, relative: f64, absolute: f64) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        relative,
        absolute,
        limit.try_into().unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
}
fn invoke(problem: &LinearProblem<'_>, plan: SolverPlan) -> Result<LinearSolution, Diagnostic> {
    solve(
        REFERENCE_SOLVER_PROVIDER,
        problem,
        plan,
        &SERIAL_LINEAR_EXECUTION,
    )
}
fn assert_values(actual: &[f64], expected: &[f64]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn jacobi_uses_symmetric_congruence_and_original_reports_for_spd_and_indefinite() {
    // B=[[1,.5],[.5,1]] for SPD. Left-only D^-1*A is nonsymmetric.
    // The indefinite B=[[1,1.5],[1.5,1]] has eigenvalues 2.5,-.5.
    for (operator, properties, rhs, expected) in [
        (
            Dense([[4., 1.], [1., 1.]]),
            LinearOperatorProperties::SymmetricPositiveDefinite,
            [5., 2.],
            [1., 1.],
        ),
        (
            Dense([[4., 1.], [1., 1.]]),
            LinearOperatorProperties::Symmetric,
            [5., 2.],
            [1., 1.],
        ),
        (
            Dense([[4., 3.], [3., 1.]]),
            LinearOperatorProperties::SymmetricIndefinite,
            [-2., 1.],
            [1., -2.],
        ),
    ] {
        let initial = [0.25, -0.5];
        let problem = LinearProblem::new(&operator, &rhs, properties)
            .unwrap()
            .with_initial_guess(&initial)
            .unwrap();
        let request = plan(2, 1e-12, 1e-12);
        let solution = invoke(&problem, request).unwrap();
        assert_values(solution.values(), &expected);
        let report = solution.report();
        // W^-1*(x-x0) is [1.5,1.5] (SPD) or [1.5,-1.5]
        // (indefinite), each an eigenvector of its symmetric congruence.
        assert_eq!(report.completed_iterations(), 1);
        assert_eq!(report.solver_plan(), request);
        assert_eq!(report.solver_provider(), REFERENCE_SOLVER_PROVIDER);
        assert_eq!(
            report.reason(),
            ConvergenceReason::ResidualToleranceSatisfied
        );
        let mut applied = [0.; 2];
        operator.apply(&initial, &mut applied).unwrap();
        let initial_norm = (rhs[0] - applied[0]).hypot(rhs[1] - applied[1]);
        assert!((report.initial_residual_norm() - initial_norm).abs() < 1e-14);
        operator.apply(solution.values(), &mut applied).unwrap();
        let residual = (rhs[0] - applied[0]).hypot(rhs[1] - applied[1]);
        assert!((report.true_residual_norm() - residual).abs() < 1e-14);
        assert_eq!(report.reported_residual_norm(), report.true_residual_norm());
        assert!(report.true_residual_norm() <= report.residual_target());
        assert!((report.residual_target() - 1e-12 * rhs[0].hypot(rhs[1])).abs() < 1e-25);
    }
}

#[test]
fn transformed_small_norm_cannot_admit_a_large_original_residual() {
    let magnitude = 2f64.powi(80);
    let operator = Dense([[magnitude, 0.], [0., magnitude]]);
    let rhs = [1., 0.];
    let problem = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    // ||W*b||=2^-40 < atol=2^-30, but ||b||=1 > atol. One step is required.
    let request = plan(1, 0., 2f64.powi(-30));
    let solution = invoke(&problem, request).unwrap();
    assert_eq!(solution.values(), &[2f64.powi(-80), 0.]);
    assert_eq!(solution.report().completed_iterations(), 1);
    assert_eq!(solution.report().initial_residual_norm(), 1.);
    assert_eq!(solution.report().residual_target(), 2f64.powi(-30));
    assert_eq!(solution.report().true_residual_norm(), 0.);
}

#[test]
fn exact_and_already_accepted_nonzero_initial_guesses_keep_zero_work() {
    let operator = Dense([[4., 1.], [1., 1.]]);
    let rhs = [5., 2.];
    for initial in [[1., 1.], [1., 1. + 1e-14]] {
        let problem = LinearProblem::new(
            &operator,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_initial_guess(&initial)
        .unwrap();
        let solution = invoke(&problem, plan(1, 1e-12, 1e-12)).unwrap();
        assert_eq!(solution.values(), &initial);
        assert_eq!(solution.report().completed_iterations(), 0);
        assert_eq!(
            solution.report().reason(),
            ConvergenceReason::InitialResidualSatisfied
        );
    }
}

#[test]
fn original_physical_acceptance_can_reject_an_iterate_and_continue_within_the_limit() {
    let operator = Dense([[4., 1.], [1., 1.]]);
    let rhs = [5., 2.];
    let problem = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let first = invoke(&problem, plan(1, 0., 1.)).unwrap();
    assert_values(first.values(), &[61. / 73., 488. / 365.]);
    let expected_norm = (117f64 / 365.).hypot(-63. / 365.);
    assert!((first.report().true_residual_norm() - expected_norm).abs() < 1e-14);
    let physical = |x: &[f64]| Ok(x.iter().all(|x| (x - 1.).abs() < TOLERANCE));
    assert!(!physical(first.values()).unwrap());
    let checked = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
    .with_candidate_check(&physical);
    let error = invoke(&checked, plan(1, 0., 1.)).unwrap_err();
    assert!(error.message().contains("plan limit of 1 iterations"));
    let solution = invoke(&checked, plan(2, 0., 1.)).unwrap();
    assert_values(solution.values(), &[1., 1.]);
    assert_eq!(solution.report().completed_iterations(), 2);
    let failure = |_: &[f64]| Err(invalid_realization("physical domain failure"));
    let error = invoke(&problem.with_candidate_check(&failure), plan(2, 0., 1.)).unwrap_err();
    assert_eq!(error.message(), "physical domain failure");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    let reject = |_: &[f64]| Ok(false);
    let problem = LinearProblem::new(
        &operator,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
    .with_candidate_check(&reject);
    let error = invoke(&problem, plan(2, 0., 1.)).unwrap_err();
    assert!(error.message().contains("Krylov space closed"));
}
