use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

fn algorithms() -> [LinearSolver; 2] {
    [
        LinearSolver::ConjugateGradient,
        LinearSolver::BiConjugateGradientStabilized,
    ]
}
fn bounded(algorithm: LinearSolver, limit: usize, relative: f64) -> SolverPlan {
    SolverPlan::new(
        algorithm,
        relative,
        2f64.powi(-40),
        limit.try_into().unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast)
}

#[test]
fn initial_and_zero_rhs_paths_keep_the_first_accepted_point_and_original_budget() {
    fn check<S: KrylovScalar>(phase: S) {
        let operator = Matrix([[S::one(), S::zero()], [S::zero(), S::one()]]);
        let rhs = [S::zero(); 2];
        let tiny = phase * S::from(1e-14).unwrap();
        let properties = if S::DOMAIN == ScalarDomain::Complex {
            LinearOperatorProperties::HermitianPositiveDefinite
        } else {
            LinearOperatorProperties::SymmetricPositiveDefinite
        };
        for algorithm in algorithms() {
            for method in [false, true] {
                for (guess, iterations) in [
                    ([S::zero(); 2], 0),
                    ([tiny, -tiny], 0),
                    ([phase, -phase], 1),
                ] {
                    let criterion = |_: &[S]| Ok(true);
                    let mut problem = LinearProblem::new(&operator, &rhs, properties)
                        .unwrap()
                        .with_initial_guess(&guess)
                        .unwrap();
                    if method {
                        problem = problem.with_candidate_check(&criterion);
                    }
                    let request = SolverPlan::new(algorithm, 0., 1e-13, 1.try_into().unwrap())
                        .unwrap()
                        .with_reduction(ReductionPolicy::Fast);
                    let solution = invoke(&problem, request).unwrap();
                    assert_eq!(solution.report().completed_iterations(), iterations);
                    assert!(iterations <= request.maximum_iterations().get());
                    assert_eq!(
                        solution.report().reason(),
                        if iterations == 0 {
                            ConvergenceReason::InitialResidualSatisfied
                        } else {
                            ConvergenceReason::ResidualToleranceSatisfied
                        }
                    );
                    assert_eq!(
                        solution.values(),
                        if iterations == 0 { &guess } else { &rhs }
                    );
                }
            }
        }
    }
    check(1.);
    check(C::new(1., 1.));
}

#[test]
fn complex_physical_rows_and_global_norm_must_both_accept_within_the_original_limit() {
    let z = C::new(0., 0.);
    let source = Matrix([[C::new(1., 0.), z], [z, C::new(2., 0.)]]);
    let phase = C::new(1., 1.);
    let large = 2f64.powi(30);
    let rhs = [large * phase, phase];
    let relative = 2f64.powi(-20);
    let criterion = |x: &[C]| {
        Ok((phase - 2. * x[1]).norm()
            <= 2f64
                .powi(-40)
                .max(relative * (2. * x[1].norm() + phase.norm())))
    };
    assert!(!criterion(&[large * phase, phase]).unwrap());
    assert!(criterion(&[large * phase, 0.5 * phase]).unwrap());
    for algorithm in algorithms() {
        let plain = LinearProblem::new(
            &source,
            &rhs,
            LinearOperatorProperties::HermitianPositiveDefinite,
        )
        .unwrap();
        let global_only = invoke(&plain, bounded(algorithm, 1, relative)).unwrap();
        assert!(!criterion(global_only.values()).unwrap());
        let checked = plain.with_candidate_check(&criterion);
        if algorithm == LinearSolver::ConjugateGradient {
            assert!(invoke(&checked, bounded(algorithm, 1, relative)).is_err());
        }
        let limit = if algorithm == LinearSolver::ConjugateGradient {
            2
        } else {
            1
        };
        let solution = invoke(&checked, bounded(algorithm, limit, relative)).unwrap();
        assert!(criterion(solution.values()).unwrap());
        assert_eq!(solution.report().completed_iterations(), limit);
        assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
    }
}

#[test]
fn complex_first_joint_candidate_survives_later_points_and_no_convergence() {
    let z = C::new(0., 0.);
    let source = Matrix([
        [C::new(1., 0.), z, z],
        [z, C::new(2., 0.), z],
        [z, z, C::new(3., 0.)],
    ]);
    let phase = C::new(1., 1.);
    let rhs = [phase; 3];
    for algorithm in algorithms() {
        let criterion = |x: &[C]| {
            if x[0].re > 0.8 {
                return Err(solve_failed(
                    "later candidate must not replace the retained point",
                ));
            }
            Ok(x[0].re > 0.)
        };
        let problem = LinearProblem::new(
            &source,
            &rhs,
            LinearOperatorProperties::HermitianPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        let solution = invoke(&problem, bounded(algorithm, 2, 1.)).unwrap();
        // Multiplying the real diagonal problem by phase multiplies every
        // iterate by phase; conjugate-first products cancel |phase|².
        let first = if algorithm == LinearSolver::ConjugateGradient {
            [0.5, 0.5, 0.5]
        } else {
            [0.7, 0.5, 0.3]
        };
        assert_values(solution.values(), &first.map(|value| value * phase));
        assert_eq!(solution.report().completed_iterations(), 2);
    }
}

#[test]
fn complex_method_and_later_operator_diagnostics_propagate_without_reclassification() {
    let phase = C::new(1., 1.);
    let z = C::new(0., 0.);
    let source = Matrix([
        [C::new(1., 0.), z, z],
        [z, C::new(2., 0.), z],
        [z, z, C::new(3., 0.)],
    ]);
    for algorithm in algorithms() {
        let criterion = |x: &[C]| {
            if x.iter().any(|value| *value != z) {
                Err(solve_failed("complex method domain failure"))
            } else {
                Ok(false)
            }
        };
        let rhs = [phase; 3];
        let problem = LinearProblem::new(
            &source,
            &rhs,
            LinearOperatorProperties::HermitianPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        assert_eq!(
            invoke(&problem, bounded(algorithm, 2, 1.))
                .unwrap_err()
                .message(),
            "complex method domain failure"
        );
    }
    #[derive(Debug)]
    struct Failing {
        armed: AtomicBool,
        first: [C; 3],
    }
    impl LinearOperator for Failing {
        type Scalar = C;
        fn rows(&self) -> usize {
            3
        }
        fn columns(&self) -> usize {
            3
        }
        fn apply(&self, x: &[C], y: &mut [C]) -> Result<(), Diagnostic> {
            if self.armed.load(Ordering::SeqCst) {
                return Err(solve_failed("actual complex operator failure"));
            }
            for ((y, x), a) in y.iter_mut().zip(x).zip([1., 2., 3.]) {
                *y = a * x;
            }
            if x.iter()
                .zip(self.first)
                .all(|(x, first)| (*x - first).norm() < TOLERANCE)
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
        let source = Failing {
            armed: AtomicBool::new(false),
            first: first.map(|value| value * phase),
        };
        let criterion = |x: &[C]| Ok(x[0].re > 0.);
        let rhs = [phase; 3];
        let problem = LinearProblem::new(
            &source,
            &rhs,
            LinearOperatorProperties::HermitianPositiveDefinite,
        )
        .unwrap()
        .with_candidate_check(&criterion);
        assert_eq!(
            invoke(&problem, bounded(algorithm, 2, 1.))
                .unwrap_err()
                .message(),
            "actual complex operator failure"
        );
    }
}
