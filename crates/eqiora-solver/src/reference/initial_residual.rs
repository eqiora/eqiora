//! Shared initial residual and zero-iteration acceptance for reference Krylov methods.

use super::*;

pub(super) struct InitialResidual<S> {
    pub solution: Vec<S>,
    pub applied: Vec<S>,
    pub residual: Vec<S>,
    pub initial_residual_norm: f64,
    pub target: f64,
    pub accepted_report: Option<SolveReport>,
}

pub(super) fn initial_residual<S: Scalar + ComplexFloat<Real = f64> + Sync>(
    provider: SolverProvider,
    problem: &LinearProblem<'_, S>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution<S>,
) -> Result<InitialResidual<S>, Diagnostic> {
    let dimension = problem.operator().columns();
    let solution = problem
        .initial_guess()
        .map_or_else(|| vec![S::zero(); dimension], <[S]>::to_vec);
    let mut applied = vec![S::zero(); dimension];
    execution.apply(problem.operator(), &solution, &mut applied)?;
    let mut residual = Vec::with_capacity(dimension);
    for (right, applied) in problem.right_hand_side().iter().zip(&applied) {
        residual.push(*right - *applied);
    }
    require_finite(&residual, "initial residual")?;

    let right_hand_side_norm = FixedOrderInnerProduct::norm(execution, problem.right_hand_side())?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = FixedOrderInnerProduct::norm(execution, &residual)?;
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
