use super::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Matrix<const N: usize>([[f64; N]; N]);

impl<const N: usize> LinearOperator for Matrix<N> {
    type Scalar = f64;

    fn rows(&self) -> usize {
        N
    }

    fn columns(&self) -> usize {
        N
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        for (row, value) in self.0.iter().zip(output) {
            *value = row.iter().zip(input).map(|(a, x)| a * x).sum();
        }
        Ok(())
    }
}

fn plan(limit: usize) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-10,
        1e-12,
        NonZeroUsize::new(limit).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast)
}

#[test]
fn restart_recovers_a_nonsingular_system_after_recurrence_breakdown() {
    // det(A) = 2; substitution gives x = [1, -1]. From x0 = 0,
    // the first BiCGSTAB alpha is -1/3 and s is orthogonal to A*s,
    // so its omega substep breaks down despite the unique solution.
    for scale in [0.125, 1., 8.] {
        let operator = Matrix([[-scale, 0.], [-3. * scale, -2. * scale]]);
        let rhs = [-scale, -scale];
        let problem =
            LinearProblem::new(&operator, &rhs, LinearOperatorProperties::General).unwrap();
        let solution = FaerLinearSolver.solve(&problem, plan(32)).unwrap();
        assert!((solution.values()[0] - 1.).abs() < 1e-9);
        assert!((solution.values()[1] + 1.).abs() < 1e-9);
        assert!(solution.report().completed_iterations() <= 32);
        assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
        assert_eq!(solution.report().solver_provider(), FAER_SOLVER_PROVIDER);
    }
}

#[test]
fn restart_preserves_a_second_independent_manufactured_solution() {
    // det(A) = 4; row substitution gives x = [1, -1, 1].
    let operator = Matrix([[1., 2., 1.], [0., -2., -3.], [-2., 0., 2.]]);
    let problem =
        LinearProblem::new(&operator, &[0., -1., 0.], LinearOperatorProperties::General).unwrap();
    let solution = FaerLinearSolver.solve(&problem, plan(32)).unwrap();
    for (actual, expected) in solution.values().iter().zip([1., -1., 1.]) {
        assert!((actual - expected).abs() < 1e-9);
    }
    assert!(solution.report().completed_iterations() <= 32);
    assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
}

#[test]
fn restart_cannot_exceed_the_original_iteration_budget() {
    let operator = Matrix([[-1., 0.], [-3., -2.]]);
    let problem =
        LinearProblem::new(&operator, &[-1., -1.], LinearOperatorProperties::General).unwrap();
    let error = FaerLinearSolver.solve(&problem, plan(2)).unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("iteration budget"));
}

#[test]
fn an_exact_initial_guess_needs_no_krylov_iterations() {
    let operator = Matrix([[-1., 0.], [-3., -2.]]);
    let problem = LinearProblem::new(&operator, &[-1., -1.], LinearOperatorProperties::General)
        .unwrap()
        .with_initial_guess(&[1., -1.])
        .unwrap();
    let solution = FaerLinearSolver.solve(&problem, plan(1)).unwrap();
    assert_eq!(solution.report().completed_iterations(), 0);
    assert_eq!(solution.values(), &[1., -1.]);
}

#[test]
fn zero_rhs_does_not_misreport_a_changed_guess_as_initial_convergence() {
    let operator = Matrix([[1., 0.], [0., 1.]]);
    let problem = LinearProblem::new(&operator, &[0., 0.], LinearOperatorProperties::General)
        .unwrap()
        .with_initial_guess(&[1., -1.])
        .unwrap();
    let solution = FaerLinearSolver.solve(&problem, plan(1)).unwrap();
    assert_eq!(solution.values(), &[0., 0.]);
    assert_eq!(solution.report().completed_iterations(), 1);
}

#[test]
fn a_singular_inconsistent_system_does_not_claim_convergence() {
    let operator = Matrix([[0., 0.], [0., 0.]]);
    let problem =
        LinearProblem::new(&operator, &[1., 0.], LinearOperatorProperties::General).unwrap();
    let error = FaerLinearSolver.solve(&problem, plan(32)).unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("no true-residual progress"));
}

#[test]
fn an_operator_failure_during_a_cycle_is_not_reclassified_as_breakdown() {
    #[derive(Debug)]
    struct FailingOperator(AtomicUsize);

    impl LinearOperator for FailingOperator {
        type Scalar = f64;

        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if self.0.fetch_add(1, Ordering::Relaxed) >= 2 {
                return Err(solve_failed("operator domain failure"));
            }
            output.copy_from_slice(input);
            Ok(())
        }
    }

    let operator = FailingOperator(AtomicUsize::new(0));
    let problem =
        LinearProblem::new(&operator, &[1., 2.], LinearOperatorProperties::General).unwrap();
    let error = FaerLinearSolver.solve(&problem, plan(32)).unwrap_err();
    assert_eq!(error.message(), "operator domain failure");
}
