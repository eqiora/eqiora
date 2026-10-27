use super::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
struct Diagonal<const N: usize>([f64; N]);
impl<const N: usize> LinearOperator for Diagonal<N> {
    type Scalar = f64;
    fn rows(&self) -> usize {
        N
    }
    fn columns(&self) -> usize {
        N
    }
    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        for ((output, input), diagonal) in output.iter_mut().zip(input).zip(self.0) {
            *output = input * diagonal;
        }
        Ok(())
    }
}
fn plan(algorithm: LinearSolver, limit: usize, relative: f64) -> SolverPlan {
    SolverPlan::new(
        algorithm,
        relative,
        2f64.powi(-40),
        NonZeroUsize::new(limit).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast)
}
fn algorithms() -> [LinearSolver; 2] {
    [
        LinearSolver::ConjugateGradient,
        LinearSolver::BiConjugateGradientStabilized,
    ]
}

#[test]
fn physical_row_criterion_prevents_alpha_only_global_convergence() {
    let operator = Diagonal([1., 2.]);
    let large = 2f64.powi(30);
    let rhs = [large, 1.];
    let relative = 2f64.powi(-20);
    let criterion = |x: &[f64]| {
        Ok((1. - 2. * x[1]).abs() <= 2f64.powi(-40).max(relative * (2. * x[1].abs() + 1.)))
    };
    // Binary64 alpha rounds to1: x_alpha=(2^30,1), global residual1 <2^10,
    // but its second row fails. BiCGSTAB omega=1/2 produces (2^30,1/2)
    // during that SAME first iteration; CG needs its second iteration.
    assert!(!criterion(&[large, 1.]).unwrap());
    assert!(criterion(&[large, 0.5]).unwrap());
    for algorithm in algorithms() {
        let plain = LinearProblem::new(
            &operator,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let one = FaerLinearSolver
            .solve(&plain, plan(algorithm, 1, relative))
            .unwrap();
        assert_eq!(one.values(), &[large, 1.]);
        assert_eq!(one.report().completed_iterations(), 1);
        let checked = plain.with_candidate_check(&criterion);
        let limit = if algorithm == LinearSolver::ConjugateGradient {
            2
        } else {
            1
        };
        let solved = FaerLinearSolver
            .solve(&checked, plan(algorithm, limit, relative))
            .unwrap();
        assert_eq!(solved.values()[0], large);
        assert!(criterion(solved.values()).unwrap());
        assert_eq!(solved.report().completed_iterations(), limit);
        // The exact solution is (2^30,1/2); the adapter promises the original
        // residual/physical contracts, not exact floating-point coordinates.
        let residual = (1. - 2. * solved.values()[1]).abs();
        assert_eq!(solved.report().true_residual_norm(), residual);
        assert_eq!(
            solved.report().solver_plan(),
            plan(algorithm, limit, relative)
        );
        assert_eq!(solved.report().solver_provider(), FAER_SOLVER_PROVIDER);
    }
}

#[test]
fn exact_nonzero_rhs_alpha_exit_and_zero_rhs_shortcuts_are_preserved() {
    let operator = Diagonal([1., 1.]);
    let criterion = |_: &[f64]| Ok(true);
    for algorithm in algorithms() {
        let exact_guess = LinearProblem::new(
            &operator,
            &[1., 2.],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_initial_guess(&[1., 2.])
        .unwrap()
        .with_candidate_check(&criterion);
        let initial = FaerLinearSolver
            .solve(&exact_guess, plan(algorithm, 1, 1e-12))
            .unwrap();
        assert_eq!(initial.values(), &[1., 2.]);
        assert_eq!(initial.report().completed_iterations(), 0);
        // Faer BiCGSTAB returns from the exact alpha substep without callback.
        // A literal-zero library threshold would instead enter omega=0/0.
        let problem = LinearProblem::new(
            &operator,
            &[1., 2.],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        let solved = FaerLinearSolver
            .solve(&problem, plan(algorithm, 1, 1e-12))
            .unwrap();
        assert_eq!(solved.values(), &[1., 2.]);
        assert_eq!(solved.report().completed_iterations(), 1);
        for (guess, iterations) in [([0., 0.], 0), ([1., -1.], 1)] {
            let problem = LinearProblem::new(
                &operator,
                &[0., 0.],
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )
            .unwrap()
            .with_initial_guess(&guess)
            .unwrap()
            .with_candidate_check(&criterion);
            let solved = FaerLinearSolver
                .solve(&problem, plan(algorithm, 1, 1e-12))
                .unwrap();
            assert_eq!(solved.values(), &[0., 0.]);
            assert_eq!(solved.report().completed_iterations(), iterations);
        }
    }
}

#[test]
fn retained_candidate_survives_no_convergence_and_charges_the_whole_call() {
    let operator = Diagonal([1., 2., 3.]);
    let rhs = [1., 1., 1.];
    for algorithm in algorithms() {
        let criterion = |x: &[f64]| {
            if x[0] > 0.8 {
                return Err(solve_failed(
                    "later point must not be reassessed after capture",
                ));
            }
            Ok(x[0] > 0.)
        };
        let problem = LinearProblem::new(
            &operator,
            &rhs,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        let solved = FaerLinearSolver
            .solve(&problem, plan(algorithm, 2, 1.))
            .unwrap();
        // Both methods' first alpha point is (1/2,1/2,1/2), already jointly
        // accepted here. BiCGSTAB now publishes alpha before its omega point
        // (7/10,1/2,3/10), so the original first-accepted rule retains alpha.
        // Neither final residual is exactly zero; charge the whole limit2.
        let expected = [0.5, 0.5, 0.5];
        for (actual, expected) in solved.values().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
        let residual = rhs
            .iter()
            .zip(solved.values())
            .zip(operator.0)
            .fold(0f64, |norm, ((rhs, x), a)| norm.hypot(rhs - a * x));
        assert_eq!(solved.report().completed_iterations(), 2);
        assert!((solved.report().true_residual_norm() - residual).abs() < 1e-15);
        assert!((solved.report().reported_residual_norm() - residual).abs() < 1e-15);
    }
}

#[test]
fn method_errors_before_capture_propagate() {
    let operator = Diagonal([1., 2.]);
    let criterion = |x: &[f64]| {
        if x.iter().any(|v| *v != 0.) {
            Err(solve_failed("method domain failure"))
        } else {
            Ok(false)
        }
    };
    for algorithm in algorithms() {
        let problem = LinearProblem::new(
            &operator,
            &[1., 1.],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        let error = FaerLinearSolver
            .solve(&problem, plan(algorithm, 2, 1.))
            .unwrap_err();
        assert_eq!(error.message(), "method domain failure");
    }
}

#[test]
fn retained_candidate_does_not_swallow_a_later_actual_operator_failure() {
    #[derive(Debug)]
    struct FailingOperator {
        armed: AtomicBool,
        first: [f64; 3],
    }
    impl LinearOperator for FailingOperator {
        type Scalar = f64;
        fn rows(&self) -> usize {
            3
        }
        fn columns(&self) -> usize {
            3
        }
        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            if self.armed.load(Ordering::SeqCst) {
                return Err(solve_failed("actual later operator failure"));
            }
            for ((output, input), a) in output.iter_mut().zip(input).zip([1., 2., 3.]) {
                *output = a * input;
            }
            if input
                .iter()
                .zip(self.first)
                .all(|(x, expected)| (x - expected).abs() < 1e-14)
            {
                self.armed.store(true, Ordering::SeqCst);
            }
            Ok(())
        }
    }
    for algorithm in algorithms() {
        let first = if algorithm == LinearSolver::ConjugateGradient {
            [0.5, 0.5, 0.5]
        } else {
            [0.7, 0.5, 0.3]
        };
        let operator = FailingOperator {
            armed: AtomicBool::new(false),
            first,
        };
        let criterion = |x: &[f64]| Ok(x[0] > 0.);
        let problem = LinearProblem::new(
            &operator,
            &[1., 1., 1.],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        let error = FaerLinearSolver
            .solve(&problem, plan(algorithm, 2, 1.))
            .unwrap_err();
        assert_eq!(error.message(), "actual later operator failure");
    }
}

#[test]
fn cg_nonpositive_operator_error_remains_fatal_after_candidate_capture() {
    // The asserted SPD property is deliberately false: first curvature is2>0,
    // then CG detects negative curvature. A retained candidate cannot hide it.
    let operator = Diagonal([1., -2.]);
    let criterion = |x: &[f64]| Ok(x[0] > 0.);
    let problem = LinearProblem::new(
        &operator,
        &[2., 1.],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
    .with_candidate_check(&criterion);
    let error = FaerLinearSolver
        .solve(&problem, plan(LinearSolver::ConjugateGradient, 2, 10.))
        .unwrap_err();
    assert!(
        error.message().contains("NonPositiveDefiniteOperator"),
        "{error:?}"
    );
}
