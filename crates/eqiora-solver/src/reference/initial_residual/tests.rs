use super::*;
use crate::{
    LinearOperator, LinearOperatorOrientation, LinearOperatorProperties, SERIAL_EXECUTION_PROVIDER,
};
use std::num::NonZeroUsize;

const ALGORITHMS: [LinearSolver; 3] = [
    LinearSolver::ConjugateGradient,
    LinearSolver::MinimumResidual,
    LinearSolver::BiConjugateGradientStabilized,
];

#[derive(Debug)]
struct ScalarAction {
    coefficient: f64,
    orientation: LinearOperatorOrientation,
}

impl LinearOperator for ScalarAction {
    type Scalar = f64;

    fn rows(&self) -> usize {
        1
    }

    fn columns(&self) -> usize {
        1
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = self.coefficient * input[0];
        Ok(())
    }

    fn orientation(&self) -> LinearOperatorOrientation {
        self.orientation
    }
}

#[test]
fn zero_iteration_reports_keep_exact_solution_policy_and_providers() {
    for algorithm in ALGORITHMS {
        for orientation in [
            LinearOperatorOrientation::Normal,
            LinearOperatorOrientation::Transposed,
        ] {
            let operator = ScalarAction {
                coefficient: 1.0,
                orientation,
            };
            for (rhs, initial, relative, absolute, residual, target) in [
                ([2.0], Some([2.0]), 0.0, 0.5, 0.0, 0.5),
                ([0.0], None, 0.0, 0.5, 0.0, 0.5),
                ([2.0], Some([1.5]), 0.0, 0.5, 0.5, 0.5),
                ([2.0], Some([1.5]), 0.25, 0.0, 0.5, 0.5),
            ] {
                let mut problem = LinearProblem::new(
                    &operator,
                    &rhs,
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                )
                .unwrap();
                if let Some(initial) = &initial {
                    problem = problem.with_initial_guess(initial).unwrap();
                }
                let plan =
                    SolverPlan::new(algorithm, relative, absolute, NonZeroUsize::MIN).unwrap();
                let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
                assert_eq!(solution.values(), &initial.unwrap_or([0.0]));
                let report = solution.report();
                assert_eq!(report.completed_iterations(), 0);
                assert_eq!(report.reason(), ConvergenceReason::InitialResidualSatisfied);
                assert_eq!(report.initial_residual_norm(), residual);
                assert_eq!(report.reported_residual_norm(), residual);
                assert_eq!(report.true_residual_norm(), residual);
                assert_eq!(report.residual_target(), target);
                assert_eq!(report.solver_plan(), plan);
                assert_eq!(report.orientation(), orientation);
                assert_eq!(report.solver_provider(), REFERENCE_SOLVER_PROVIDER);
                assert_eq!(report.execution_provider(), SERIAL_EXECUTION_PROVIDER);
                assert_eq!(report.verification_provider(), SERIAL_EXECUTION_PROVIDER);
            }
        }
    }
}

#[test]
fn initial_residual_just_above_each_target_enters_the_recurrence() {
    let operator = ScalarAction {
        coefficient: 1.0,
        orientation: LinearOperatorOrientation::Normal,
    };
    let initial = [1.5_f64.next_down()];
    let problem = LinearProblem::new(
        &operator,
        &[2.0],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
    .with_initial_guess(&initial)
    .unwrap();
    for algorithm in ALGORITHMS {
        for (relative, absolute) in [(0.0, 0.5), (0.25, 0.0)] {
            let plan = SolverPlan::new(algorithm, relative, absolute, NonZeroUsize::MIN).unwrap();
            let solution = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap();
            assert_eq!(solution.values(), &[2.0]);
            assert_eq!(solution.report().completed_iterations(), 1);
            assert_eq!(
                solution.report().reason(),
                ConvergenceReason::ResidualToleranceSatisfied
            );
            assert!(solution.report().initial_residual_norm() > 0.5);
            assert_eq!(solution.report().residual_target(), 0.5);
        }
    }
}

#[test]
fn every_method_rejects_nonfinite_initial_actions() {
    for coefficient in [f64::NAN, f64::INFINITY] {
        let operator = ScalarAction {
            coefficient,
            orientation: LinearOperatorOrientation::Normal,
        };
        let problem = LinearProblem::new(
            &operator,
            &[2.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        for algorithm in ALGORITHMS {
            let plan = SolverPlan::new(algorithm, 0.0, 0.5, NonZeroUsize::MIN).unwrap();
            let error = REFERENCE_LINEAR_SOLVER.solve(&problem, plan).unwrap_err();
            assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        }
    }
}
