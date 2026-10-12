use super::*;
use eqiora_assembly::CsrMatrix;
use eqiora_solver::{
    LinearOperatorProperties, LinearProblem, LinearSolver, LinearSolverBackend,
    REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use std::cell::Cell;
use std::num::NonZeroUsize;

fn policy(iterations: usize, halvings: usize) -> NonlinearSolvePlan {
    NonlinearSolvePlan::new(
        0.0,
        1.0e-12,
        NonZeroUsize::new(iterations).unwrap(),
        halvings,
    )
    .unwrap()
}

fn solve_scalar(jacobian: f64, rhs: f64) -> Result<(Vec<f64>, SolveReport), Diagnostic> {
    let matrix = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![jacobian])?;
    let rhs = [rhs];
    let problem = LinearProblem::new(
        &matrix,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        0.0,
        1.0e-14,
        NonZeroUsize::new(2).unwrap(),
    )?;
    REFERENCE_LINEAR_SOLVER
        .solve(&problem, plan)
        .map(|solution| solution.into_parts())
}

#[test]
fn bounded_positive_root_backtracks_and_reuses_only_accepted_linearizations() {
    // x² = 1 on 0 < x < 2 has the independent solution x = 1. The
    // undamped first Newton point from 0.1 is 5.05, outside that domain.
    for retain_trial_linearization in [false, true] {
        let linearizations = Cell::new(0);
        let rejections = Cell::new(0);
        let (root, evidence) = solve_damped_newton(
            vec![0.1],
            policy(12, 8),
            |point| {
                linearizations.set(linearizations.get() + 1);
                Ok(point[0])
            },
            |point| Ok((point * point - 1.0).abs()),
            |_, point| solve_scalar(2.0 * point, 1.0 - point * point),
            |point| {
                if point[0] <= 0.0 || point[0] >= 2.0 {
                    rejections.set(rejections.get() + 1);
                    return Ok(None);
                }
                Ok(Some(NewtonTrial {
                    residual_norm: (point[0] * point[0] - 1.0).abs(),
                    linearization: retain_trial_linearization.then_some(point[0]),
                }))
            },
        )
        .unwrap();
        assert!((root - 1.0).abs() < 1.0e-12);
        assert!(rejections.get() >= 2);
        assert_eq!(evidence.initial_residual_norm, 0.99);
        assert_eq!(evidence.residual_target, 1.0e-12);
        assert_eq!(evidence.iterations, evidence.linear_solves.len());
        assert_eq!(
            linearizations.get(),
            if retain_trial_linearization {
                1
            } else {
                1 + evidence.iterations
            }
        );
    }
}

#[test]
fn converged_initial_point_does_not_request_a_correction_or_trial() {
    let (point, evidence) = solve_damped_newton(
        vec![1.0],
        policy(1, 0),
        |point| Ok(point[0]),
        |_| Ok(0.0),
        |_, _| panic!("no correction at a root"),
        |_| panic!("no trial at a root"),
    )
    .unwrap();
    assert_eq!(point, 1.0);
    assert_eq!(evidence.iterations, 0);
    assert!(evidence.linear_solves.is_empty());
}

#[test]
fn execution_errors_propagate_without_becoming_admissibility_rejections() {
    let trials = Cell::new(0);
    let error = solve_damped_newton(
        vec![0.0],
        policy(5, 8),
        |point| Ok(point[0]),
        |_| Ok(1.0),
        |_, _| solve_scalar(1.0, 1.0),
        |_| {
            trials.set(trials.get() + 1);
            Err(Diagnostic::error(
                codes::INVALID_DISCRETIZATION,
                "exact binding changed",
            ))
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_DISCRETIZATION);
    assert_eq!(error.message(), "exact binding changed");
    assert_eq!(trials.get(), 1);
}

#[test]
fn no_decrease_exhausts_the_exact_halving_bound_without_acceptance() {
    let trials = Cell::new(0);
    let linearizations = Cell::new(0);
    let error = solve_damped_newton(
        vec![0.0],
        policy(5, 3),
        |point| {
            linearizations.set(linearizations.get() + 1);
            Ok(point[0])
        },
        |_| Ok(1.0),
        |_, _| solve_scalar(1.0, 0.0),
        |_| {
            trials.set(trials.get() + 1);
            Ok(Some(NewtonTrial {
                residual_norm: 1.0,
                linearization: None,
            }))
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert_eq!(trials.get(), 4);
    assert_eq!(linearizations.get(), 1);
}

#[test]
fn an_improving_but_unconverged_step_does_not_exceed_the_iteration_bound() {
    // Newton's first point for x² = 1 from x = 2 is 5/4, whose
    // residual 9/16 cannot satisfy the requested tolerance.
    let trial_point = Cell::new(0.0);
    let error = solve_damped_newton(
        vec![2.0],
        policy(1, 0),
        |point| Ok(point[0]),
        |point| Ok((point * point - 1.0).abs()),
        |_, point| solve_scalar(2.0 * point, 1.0 - point * point),
        |point| {
            trial_point.set(point[0]);
            Ok(Some(NewtonTrial {
                residual_norm: (point[0] * point[0] - 1.0).abs(),
                linearization: Some(point[0]),
            }))
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert_eq!(trial_point.get(), 1.25);
}

#[test]
fn invalid_norms_and_incomplete_corrections_cannot_publish_a_root() {
    for norm in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            solve_damped_newton(
                vec![0.0],
                policy(1, 0),
                |point| Ok(point[0]),
                |_| Ok(norm),
                |_, _| panic!("invalid residual must fail before solving"),
                |_| panic!("invalid residual must fail before trials"),
            )
            .is_err()
        );
    }
    let error = solve_damped_newton(
        vec![0.0],
        policy(1, 0),
        |point| Ok(point[0]),
        |_| Ok(1.0),
        |_, _| {
            let (_, report) = solve_scalar(1.0, 1.0)?;
            Ok((Vec::new(), report))
        },
        |_| panic!("zip truncation must not create a trial"),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
}
