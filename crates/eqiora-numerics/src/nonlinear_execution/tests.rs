use super::*;
use eqiora_assembly::CsrMatrix;
use eqiora_realization::NonlinearSolvePlan;
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
    for (retain_trial_linearization, diagnostic_rejection) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let linearizations = Cell::new(0);
        let rejections = Cell::new(0);
        let (root, evidence) = solve_damped_newton(
            vec![0.1],
            policy(12, 8).into(),
            |point| {
                linearizations.set(linearizations.get() + 1);
                Ok(point[0])
            },
            |point| Ok((point * point - 1.0).abs()),
            |_, point| solve_scalar(2.0 * point, 1.0 - point * point),
            |point, _| {
                if point[0] <= 0.0 || point[0] >= 2.0 {
                    rejections.set(rejections.get() + 1);
                    return if diagnostic_rejection {
                        Err(Diagnostic::error(
                            codes::EXPRESSION_DOMAIN_VIOLATION,
                            "outside 0 < x < 2",
                        ))
                    } else {
                        Ok(None)
                    };
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
        assert_eq!(evidence.residual_target().unwrap(), 1.0e-12);
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
        policy(1, 0).into(),
        |point| Ok(point[0]),
        |_| Ok(0.0),
        |_, _| panic!("no correction at a root"),
        |_, _| panic!("no trial at a root"),
    )
    .unwrap();
    assert_eq!(point, 1.0);
    assert_eq!(evidence.iterations, 0);
    assert!(evidence.linear_solves.is_empty());
}

#[test]
fn execution_errors_propagate_without_becoming_admissibility_rejections() {
    for code in [
        codes::INVALID_DISCRETIZATION,
        codes::INVALID_OPERATOR_IR,
        codes::INVALID_LINEARIZATION,
        codes::NONFINITE_EVALUATION,
        codes::EXECUTION_CANCELLED,
    ] {
        let trials = Cell::new(0);
        let error = solve_damped_newton(
            vec![0.0],
            policy(5, 8).into(),
            |point| Ok(point[0]),
            |_| Ok(1.0),
            |_, _| solve_scalar(1.0, 1.0),
            |_, _| {
                trials.set(trials.get() + 1);
                Err(Diagnostic::error(code, "exact binding changed"))
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(error.message(), "exact binding changed");
        assert_eq!(trials.get(), 1);
    }
}

#[test]
fn no_decrease_exhausts_the_exact_halving_bound_without_acceptance() {
    let trials = Cell::new(0);
    let linearizations = Cell::new(0);
    let error = solve_damped_newton(
        vec![0.0],
        policy(5, 3).into(),
        |point| {
            linearizations.set(linearizations.get() + 1);
            Ok(point[0])
        },
        |_| Ok(1.0),
        |_, _| solve_scalar(1.0, 0.0),
        |_, _| {
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
        policy(1, 0).into(),
        |point| Ok(point[0]),
        |point| Ok((point * point - 1.0).abs()),
        |_, point| solve_scalar(2.0 * point, 1.0 - point * point),
        |point, _| {
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
                policy(1, 0).into(),
                |point| Ok(point[0]),
                |_| Ok(norm),
                |_, _| panic!("invalid residual must fail before solving"),
                |_, _| panic!("invalid residual must fail before trials"),
            )
            .is_err()
        );
    }
    let error = solve_damped_newton(
        vec![0.0],
        policy(1, 0).into(),
        |point| Ok(point[0]),
        |_| Ok(1.0),
        |_, _| {
            let (_, report) = solve_scalar(1.0, 1.0)?;
            Ok((Vec::new(), report))
        },
        |_, _| panic!("zip truncation must not create a trial"),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
}

#[test]
fn expression_domain_errors_are_recoverable_only_during_trials() {
    for fail_initial in [true, false] {
        let linearizations = Cell::new(0);
        let error = solve_damped_newton(
            vec![0.0],
            policy(5, 8).into(),
            |point| {
                linearizations.set(linearizations.get() + 1);
                if fail_initial || linearizations.get() > 1 {
                    Err(Diagnostic::error(
                        codes::EXPRESSION_DOMAIN_VIOLATION,
                        "undefined linearization",
                    ))
                } else {
                    Ok(point[0])
                }
            },
            |_| Ok(1.0),
            |_, _| solve_scalar(1.0, 1.0),
            |_, _| {
                Ok(Some(NewtonTrial {
                    residual_norm: 0.0,
                    linearization: None,
                }))
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::EXPRESSION_DOMAIN_VIOLATION);
        assert_eq!(linearizations.get(), if fail_initial { 1 } else { 2 });
    }
}

#[test]
fn expression_domain_rejections_exhaust_the_halving_bound_without_acceptance() {
    let trials = Cell::new(0);
    let linearizations = Cell::new(0);
    let error = solve_damped_newton(
        vec![0.0],
        policy(5, 3).into(),
        |point| {
            linearizations.set(linearizations.get() + 1);
            Ok(point[0])
        },
        |_| Ok(1.0),
        |_, _| solve_scalar(1.0, 1.0),
        |_, _| {
            trials.set(trials.get() + 1);
            Err(Diagnostic::error(
                codes::EXPRESSION_DOMAIN_VIOLATION,
                "no admissible trial",
            ))
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert_eq!(trials.get(), 4);
    assert_eq!(linearizations.get(), 1);
}

fn correction_options(absolute: &[f64], iterations: usize, halvings: usize) -> NewtonOptions<'_> {
    NewtonOptions {
        maximum_iterations: NonZeroUsize::new(iterations).unwrap(),
        maximum_line_search_steps: halvings,
        stopping: NewtonStopping::Correction {
            relative: 1e-6,
            absolute,
        },
    }
}

#[test]
fn correction_tolerance_does_not_accept_a_small_residual_without_an_update() {
    // Multiplying x-1=0 by 1e-10 makes its residual tiny, not its Newton step.
    let (point, evidence) = solve_damped_newton(
        vec![0.],
        correction_options(&[1e-6], 2, 0),
        |point| Ok(point[0]),
        |point| Ok(1e-10 * (point - 1.).abs()),
        |_, point| solve_scalar(1e-10, 1e-10 * (1. - point)),
        |point, current| {
            assert_eq!(*current, 0.);
            Ok(Some(NewtonTrial {
                residual_norm: 1e-10 * (point[0] - 1.).abs(),
                linearization: None,
            }))
        },
    )
    .unwrap();
    assert_eq!(point, 1.);
    assert_eq!(evidence.iterations, 1);
    assert!(evidence.residual_target().is_err());
    assert!(matches!(
        evidence.convergence,
        NewtonConvergence::ExactResidualZero
    ));
}

#[test]
fn small_full_correction_accepts_a_finite_trial_despite_residual_noise() {
    let (point, evidence) = solve_damped_newton(
        vec![1.],
        correction_options(&[1e-6], 1, 0),
        |point| Ok(point[0]),
        |_| Ok(1e-8),
        |_, _| solve_scalar(1., 1e-8),
        |_, _| {
            Ok(Some(NewtonTrial {
                residual_norm: 2e-8,
                linearization: None,
            }))
        },
    )
    .unwrap();
    assert_eq!(point, 1. + 1e-8);
    assert_eq!(evidence.linear_solves.len(), 1);
    let NewtonConvergence::Correction { weighted_norm, .. } = evidence.convergence else {
        panic!("correction acceptance");
    };
    assert!((weighted_norm - 0.005).abs() < 1e-15);
    assert!(evidence.residual_target().is_err());
}

#[test]
fn damping_a_large_correction_cannot_manufacture_convergence() {
    // The 1/16 update is below atol, but the full Newton step is ten times atol.
    let accepted = Cell::new(0.);
    let error = solve_damped_newton(
        vec![0.],
        correction_options(&[0.1], 1, 4),
        |point| {
            accepted.set(point[0]);
            Ok(point[0])
        },
        |_| Ok(1.),
        |_, _| solve_scalar(1., 1.),
        |point, _| {
            Ok((point[0] <= 0.0625).then_some(NewtonTrial {
                residual_norm: 0.5,
                linearization: None,
            }))
        },
    )
    .unwrap_err();
    assert_eq!(accepted.get(), 0.0625);
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("iterations"));
}

#[test]
fn correction_controls_validate_before_any_linearization() {
    for absolute in [
        vec![],
        vec![1e-6, 1e-6],
        vec![0.],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        let error = solve_damped_newton::<f64>(
            vec![0.],
            correction_options(&absolute, 1, 0),
            |_| panic!("invalid controls must fail before assembly"),
            |_| Ok(0.),
            |_, _| panic!("no solve"),
            |_, _| panic!("no trial"),
        )
        .unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    }
    let stopping = NewtonStopping::Correction {
        relative: 1e-6,
        absolute: &[1e-6, 2e-6],
    };
    // Weighted components 3 and 4 have RMS sqrt(25/2), not norm 5.
    assert!(
        (stopping
            .correction_norm(&[0., 0.], &[3e-6, 8e-6])
            .unwrap()
            .unwrap()
            - (12.5_f64).sqrt())
        .abs()
            < 1e-14
    );
}
