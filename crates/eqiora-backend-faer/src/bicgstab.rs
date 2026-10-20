//! Bounded Faer Krylov cycles with independently recomputed restart residuals.
use super::*;
use eqiora_solver::{FixedOrderInnerProduct, SERIAL_LINEAR_EXECUTION};
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};

pub(super) fn solve(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<f64>>,
) -> Result<LinearSolution, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let left = FaerPreconditioner::new(dimension, inverse_diagonal);
    let right = FaerPreconditioner::new(dimension, None);
    let parallelism = Par::Seq;
    let mut workspace = MemBuffer::new(bicgstab_scratch(
        left.clone(),
        right.clone(),
        operator.clone(),
        1,
        parallelism,
    ));
    let target = plan.residual_target(FixedOrderInnerProduct::norm(
        &SERIAL_LINEAR_EXECUTION,
        problem.right_hand_side(),
    )?)?;
    let mut best = output.col_as_slice(0).to_vec();
    let mut candidate = vec![0.; dimension];
    let mut residual = vec![0.; dimension];
    let mut best_norm = residual_norm(problem, &best, &mut residual)?;
    let mut iterations = 0;
    while best_norm > target && iterations < plan.maximum_iterations().get() {
        let before = best_norm;
        // One full-dimensional Krylov cycle is bounded independently of any
        // fixture or sampled convergence history. Reinitialization replaces the
        // recursive residual with b-Ax. Every attempted cycle consumes the same
        // original iteration budget, including a cycle with numerical breakdown.
        let limit = dimension.min(plan.maximum_iterations().get() - iterations);
        output.col_as_slice_mut(0).copy_from_slice(&best);
        let params = BicgParams {
            initial_guess: EXPLICIT_INITIAL_GUESS,
            abs_tolerance: target,
            rel_tolerance: 0.,
            max_iters: limit,
            ..Default::default()
        };
        let result = bicgstab(
            output.as_mut(),
            left.clone(),
            right.clone(),
            operator.clone(),
            right_hand_side.as_ref(),
            params,
            |point| {
                for (index, value) in candidate.iter_mut().enumerate() {
                    *value = point[(index, 0)];
                }
                retain_candidate(
                    problem,
                    &candidate,
                    &mut residual,
                    &mut best,
                    &mut best_norm,
                    &failure,
                );
            },
            parallelism,
            MemStack::new(&mut workspace),
        );
        // The library can return between callbacks after its first substep.
        retain_candidate(
            problem,
            output.col_as_slice(0),
            &mut residual,
            &mut best,
            &mut best_norm,
            &failure,
        );
        // A nonfinite search direction belongs to the solver recurrence.
        // An actual operator failure always propagates, even if an earlier
        // iterate happened to meet the tolerance.
        failure.take_breakdown()?;
        // The original guess was not accepted above. Faer's zero-RHS
        // shortcut can replace it while reporting zero Krylov iterations;
        // charge that attempted correction once instead of claiming initial
        // convergence or allowing work outside the original budget.
        iterations += result.as_ref().map_or(limit, |info| info.iter_count.max(1));
        if best_norm > target && best_norm >= before {
            return Err(solve_failed(
                "faer BiCGSTAB cycle made no true-residual progress",
            ));
        }
    }
    if best_norm > target {
        return Err(solve_failed(
            "faer BiCGSTAB exhausted its iteration budget before meeting the true-residual target",
        ));
    }
    accept_linear_solution(
        problem,
        plan,
        provider,
        convergence_reason(iterations),
        iterations,
        best_norm,
        best,
    )
}

fn residual_norm(
    problem: &LinearProblem<'_>,
    values: &[f64],
    residual: &mut [f64],
) -> Result<f64, Diagnostic> {
    problem.operator().apply(values, residual)?;
    for (value, rhs) in residual.iter_mut().zip(problem.right_hand_side()) {
        *value = rhs - *value;
    }
    FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, residual)
}

fn retain_candidate(
    problem: &LinearProblem<'_>,
    candidate: &[f64],
    residual: &mut [f64],
    best: &mut [f64],
    best_norm: &mut f64,
    failure: &ApplyFailure,
) {
    if candidate.iter().any(|value| !value.is_finite()) {
        return;
    }
    match residual_norm(problem, candidate, residual) {
        Ok(norm) if norm < *best_norm => {
            *best_norm = norm;
            best.copy_from_slice(candidate);
        }
        Ok(_) => {}
        Err(error) => failure.record(error),
    }
}

#[cfg(test)]
mod tests;
