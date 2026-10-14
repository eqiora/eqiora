//! Shared initial residual and zero-iteration acceptance for real reference Krylov methods.

use super::*;

pub(super) struct InitialResidual {
    pub solution: Vec<f64>,
    pub applied: Vec<f64>,
    pub residual: Vec<f64>,
    pub initial_residual_norm: f64,
    pub target: f64,
    pub accepted_report: Option<SolveReport>,
}

pub(super) fn initial_residual(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<InitialResidual, Diagnostic> {
    let dimension = problem.operator().columns();
    let solution = problem
        .initial_guess()
        .map_or_else(|| vec![0.0; dimension], <[f64]>::to_vec);
    let mut applied = vec![0.0; dimension];
    execution.apply(problem.operator(), &solution, &mut applied)?;
    let residual = problem
        .right_hand_side()
        .iter()
        .zip(&applied)
        .map(|(right, applied)| right - applied)
        .collect::<Vec<_>>();
    require_finite(&residual, "initial residual")?;

    let right_hand_side_norm = norm(execution, problem.right_hand_side())?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(execution, &residual)?;
    let accepted_report = if initial_residual_norm <= target {
        Some(SolveReport::accepted(
            provider,
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            initial_residual_norm,
            initial_residual_norm,
            initial_residual_norm,
            target,
        )?)
    } else {
        None
    };

    Ok(InitialResidual {
        solution,
        applied,
        residual,
        initial_residual_norm,
        target,
        accepted_report,
    })
}

#[cfg(test)]
mod tests;
