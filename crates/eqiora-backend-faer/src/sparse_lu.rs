use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    ConvergenceReason, FixedOrderInnerProduct, LinearOperatorOrientation, LinearProblem,
    LinearSolution, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverPlan,
    accept_linear_solution,
};

use crate::FAER_SOLVER_PROVIDER;
use crate::sparse_lu_factor::{
    SparseLuNumericFactor, SparseLuSymbolicFactor, factor_numeric, factor_symbolic,
    solve_factored_oriented,
};

pub(super) fn solve_sparse_lu(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
) -> Result<LinearSolution, Diagnostic> {
    let system = problem.canonical_csr_system().ok_or_else(|| {
        if problem.operator().orientation() != LinearOperatorOrientation::Normal {
            invalid_realization(
                "faer sparse LU requires a normal-orientation canonical CSR problem or an oriented request with an exact canonical source",
            )
        } else {
            invalid_realization("faer sparse LU requires an exact canonical CSR coefficient source")
        }
    })?;
    if problem.operator().rows() != system.rows()
        || problem.operator().columns() != system.columns()
        || problem.properties() != system.properties()
    {
        return Err(invalid_realization(
            "faer sparse LU action, properties, and canonical CSR source disagree",
        ));
    }

    if let Some(initial) = accepted_initial(problem, plan)? {
        return Ok(initial);
    }
    let symbolic = factor_symbolic(system)?;
    let numeric = factor_numeric(&symbolic, system)?;
    solve_factored(problem, plan, &symbolic, &numeric)
}

pub(super) fn accepted_initial(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
) -> Result<Option<LinearSolution>, Diagnostic> {
    let initial = problem
        .initial_guess()
        .map_or_else(|| vec![0.0; problem.operator().columns()], <[f64]>::to_vec);
    let mut residual = vec![0.0; problem.operator().rows()];
    let norm = residual_norm(problem, &initial, &mut residual)?;
    if norm > plan.residual_target(fixed_norm(problem.right_hand_side())?)? {
        return Ok(None);
    }
    accept_linear_solution(
        problem,
        plan,
        FAER_SOLVER_PROVIDER,
        ConvergenceReason::InitialResidualSatisfied,
        0,
        norm,
        initial,
    )
    .map(Some)
}

/// Reuse the same factors for the original solve and bounded residual corrections.
/// Acceptance always replays the requested action, including its orientation.
pub(super) fn solve_factored(
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    symbolic: &SparseLuSymbolicFactor,
    numeric: &SparseLuNumericFactor,
) -> Result<LinearSolution, Diagnostic> {
    let orientation = problem.operator().orientation();
    let mut values =
        solve_factored_oriented(symbolic, numeric, problem.right_hand_side(), orientation)?;
    let target = plan.residual_target(fixed_norm(problem.right_hand_side())?)?;
    let mut residual = vec![0.0; problem.operator().rows()];
    let mut norm = residual_norm(problem, &values, &mut residual)?;
    let mut iterations = 1;
    while norm > target && iterations < plan.maximum_iterations().get() {
        let correction = solve_factored_oriented(symbolic, numeric, &residual, orientation)?;
        iterations += 1;
        let mut changed = false;
        for (value, correction) in values.iter_mut().zip(correction) {
            let candidate = *value + correction;
            changed |= candidate != *value;
            *value = candidate;
        }
        if !changed {
            break;
        }
        norm = residual_norm(problem, &values, &mut residual)?;
    }
    accept_linear_solution(
        problem,
        plan,
        FAER_SOLVER_PROVIDER,
        ConvergenceReason::ResidualToleranceSatisfied,
        iterations,
        norm,
        values,
    )
}

fn residual_norm(
    problem: &LinearProblem<'_>,
    values: &[f64],
    residual: &mut [f64],
) -> Result<f64, Diagnostic> {
    SERIAL_LINEAR_EXECUTION.apply(problem.operator(), values, residual)?;
    for (applied, right_hand_side) in residual.iter_mut().zip(problem.right_hand_side()) {
        *applied = right_hand_side - *applied;
    }
    fixed_norm(residual)
}

fn fixed_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let squared =
        SERIAL_LINEAR_EXECUTION.inner_product(FixedOrderInnerProduct::new(values, values)?)?;
    Ok(squared.sqrt())
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests;
