use std::num::NonZeroUsize;

use eqiora::Diagnostic;
use eqiora::diagnostic::codes;
use eqiora::solver::{
    DiagonalAvailability, LinearOperator, LinearOperatorProperties, LinearProblem, LinearSolution,
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;

const DIMENSION: usize = 64;
const CONTRASTS: [f64; 4] = [1.0, 1.0e2, 1.0e4, 1.0e6];

/// `S T S`, where `T` is the Dirichlet one-dimensional discrete Laplacian
/// and the positive diagonal `S` has a prescribed squared contrast.
///
/// This keeps the nearest-neighbour coupling while separating diagonal
/// scaling from the spectrum that remains after Jacobi preconditioning.
#[derive(Debug)]
struct CongruenceScaledLaplacian {
    scales: Vec<f64>,
}

impl CongruenceScaledLaplacian {
    fn new(dimension: usize, diagonal_contrast: f64) -> Self {
        assert!(dimension >= 2);
        assert!(diagonal_contrast.is_finite() && diagonal_contrast >= 1.0);
        let denominator = (dimension - 1) as f64;
        let scales = (0..dimension)
            .map(|index| diagonal_contrast.powf(0.5 * index as f64 / denominator))
            .collect();
        Self { scales }
    }

    fn manufactured_solution(&self) -> Vec<f64> {
        let denominator = (self.scales.len() + 1) as f64;
        self.scales
            .iter()
            .enumerate()
            .map(|(index, scale)| {
                let coordinate = (index + 1) as f64 / denominator;
                let transformed = 1.0
                    + 0.2 * (5.0 * std::f64::consts::PI * coordinate).sin()
                    + 0.1 * (11.0 * std::f64::consts::PI * coordinate).cos();
                transformed / scale
            })
            .collect()
    }
}

impl LinearOperator for CongruenceScaledLaplacian {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.scales.len()
    }

    fn columns(&self) -> usize {
        self.scales.len()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.columns() || output.len() != self.rows() {
            return Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "congruence-scaled Laplacian action shape mismatch",
            ));
        }
        for index in 0..self.rows() {
            let scale = self.scales[index];
            let mut value = 2.0 * scale * scale * input[index];
            if index > 0 {
                value -= scale * self.scales[index - 1] * input[index - 1];
            }
            if index + 1 < self.rows() {
                value -= scale * self.scales[index + 1] * input[index + 1];
            }
            output[index] = value;
        }
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if output.len() != self.rows() {
            return Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "congruence-scaled Laplacian diagonal shape mismatch",
            ));
        }
        for (output, scale) in output.iter_mut().zip(&self.scales) {
            *output = 2.0 * scale * scale;
        }
        Ok(DiagonalAvailability::Available)
    }
}

#[derive(Debug, Clone, Copy)]
struct Outcome {
    identity_iterations: usize,
    jacobi_iterations: usize,
}

#[test]
fn jacobi_removes_controlled_diagonal_scaling_in_the_reference_oracle() {
    let outcomes = CONTRASTS.map(|contrast| {
        solve_pair(
            &REFERENCE_LINEAR_SOLVER,
            ReductionPolicy::Reproducible,
            contrast,
        )
    });

    let baseline = outcomes[0];
    let hardest = outcomes[outcomes.len() - 1];
    let jacobi_min = outcomes
        .iter()
        .map(|outcome| outcome.jacobi_iterations)
        .min()
        .unwrap();
    let jacobi_max = outcomes
        .iter()
        .map(|outcome| outcome.jacobi_iterations)
        .max()
        .unwrap();

    assert!(hardest.identity_iterations > baseline.identity_iterations);
    assert!(hardest.identity_iterations >= 2 * hardest.jacobi_iterations);
    assert!(jacobi_max - jacobi_min <= 2);
}

#[test]
fn faer_replays_the_same_preconditioner_policy_and_acceptance_contract() {
    let hardest = solve_pair(&FaerLinearSolver, ReductionPolicy::Fast, 1.0e6);

    assert!(hardest.identity_iterations >= 2 * hardest.jacobi_iterations);
}

fn solve_pair(
    backend: &dyn LinearSolverBackend,
    reduction: ReductionPolicy,
    contrast: f64,
) -> Outcome {
    let operator = CongruenceScaledLaplacian::new(DIMENSION, contrast);
    let exact = operator.manufactured_solution();
    let mut right_hand_side = vec![0.0; DIMENSION];
    operator.apply(&exact, &mut right_hand_side).unwrap();
    let problem = LinearProblem::new(
        &operator,
        &right_hand_side,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(4_096).unwrap(),
    )
    .unwrap()
    .with_reduction(reduction);

    let identity = backend.solve(
        &problem,
        plan.with_preconditioner(PreconditionerPolicy::Identity),
    );
    let jacobi = backend.solve(
        &problem,
        plan.with_preconditioner(PreconditionerPolicy::Jacobi),
    );
    let identity = identity
        .unwrap_or_else(|error| panic!("identity solve failed for contrast {contrast:e}: {error}"));
    let jacobi = jacobi
        .unwrap_or_else(|error| panic!("Jacobi solve failed for contrast {contrast:e}: {error}"));

    assert_accepted(&identity, backend, PreconditionerPolicy::Identity);
    assert_accepted(&jacobi, backend, PreconditionerPolicy::Jacobi);

    Outcome {
        identity_iterations: identity.report().completed_iterations(),
        jacobi_iterations: jacobi.report().completed_iterations(),
    }
}

fn assert_accepted(
    solution: &LinearSolution,
    backend: &dyn LinearSolverBackend,
    preconditioner: PreconditionerPolicy,
) {
    let report = solution.report();
    assert_eq!(report.backend(), backend.id());
    assert_eq!(report.preconditioner(), preconditioner);
    assert!(report.true_residual_norm() <= report.residual_target());
    assert!(solution.values().iter().all(|value| value.is_finite()));
}
