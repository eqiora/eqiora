//! Bounded Faer Krylov cycles with independently recomputed restart residuals.
use super::*;
use crate::candidate::{CandidateAcceptance, ZERO_ONLY_TOLERANCE, residual_norm};
use eqiora_solver::{FixedOrderInnerProduct, SERIAL_LINEAR_EXECUTION};
mod kernel;

pub(super) fn solve<S: KrylovScalar>(
    provider: SolverProvider,
    problem: &LinearProblem<'_, S>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<S>>,
) -> Result<LinearSolution<S>, Diagnostic> {
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let left = FaerPreconditioner::new(dimension, inverse_diagonal);
    let right = FaerPreconditioner::new(dimension, None);
    let mut kernel = kernel::Kernel::new(operator, left, right);
    let target = plan.residual_target(FixedOrderInnerProduct::norm(
        &SERIAL_LINEAR_EXECUTION,
        problem.right_hand_side(),
    )?)?;
    let mut best = output.col_as_slice(0).to_vec();
    let mut candidate = vec![S::zero(); dimension];
    let mut residual = vec![S::zero(); dimension];
    let mut best_norm = residual_norm(problem, &best, &mut residual)?;
    let mut method = problem
        .has_candidate_check()
        .then(|| CandidateAcceptance::new(target));
    if let Some(method) = &mut method {
        method.consider(problem, &best, best_norm)?;
    }
    let mut iterations = 0;
    let mut cycle_limit = dimension;
    while !jointly_accepted(best_norm, target, &method)
        && iterations < plan.maximum_iterations().get()
    {
        let before = best_norm;
        // Begin with one full-dimensional cycle, then double the allowance:
        // finite-precision Krylov convergence may need more than n steps, so
        // repeatedly truncating at n can discard useful recurrence history.
        // Restart with b-Ax while charging all attempted work, including
        // numerical breakdown, to the original total iteration budget.
        let limit = cycle_limit.min(plan.maximum_iterations().get() - iterations);
        output.col_as_slice_mut(0).copy_from_slice(&best);
        let threshold = if method.is_some() {
            ZERO_ONLY_TOLERANCE
        } else {
            target
        };
        let result = kernel.solve(
            output.as_mut(),
            right_hand_side.as_ref(),
            threshold,
            limit,
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
                    &mut method,
                );
            },
        );
        // Also inspect the terminal point, including the zero-RHS correction.
        retain_candidate(
            problem,
            output.col_as_slice(0),
            &mut residual,
            &mut best,
            &mut best_norm,
            &failure,
            &mut method,
        );
        // A nonfinite search direction belongs to the solver recurrence.
        // An actual operator failure always propagates, even if an earlier
        // iterate happened to meet the tolerance.
        let breakdown = failure.take_breakdown()? || result.breakdown;
        // The original guess was not accepted above. Faer's zero-RHS
        // shortcut can replace it while reporting zero Krylov iterations;
        // charge that attempted correction once instead of claiming initial
        // convergence or allowing work outside the original budget.
        iterations += result.iter_count.max(1);
        if breakdown && !jointly_accepted(best_norm, target, &method) && best_norm >= before {
            return Err(solve_failed(
                "faer BiCGSTAB breakdown made no true-residual progress",
            ));
        }
        cycle_limit = cycle_limit.saturating_mul(2);
    }
    if let Some(method) = method {
        return method.finish(problem, plan, provider, iterations);
    }
    if best_norm > target {
        return Err(solve_failed(
            "faer BiCGSTAB exhausted its iteration budget before meeting the true-residual target",
        ));
    }
    eqiora_solver::LinearSolution::accept(
        problem,
        plan,
        provider,
        convergence_reason(iterations),
        iterations,
        best_norm,
        best,
    )
}

fn jointly_accepted<S: KrylovScalar>(
    norm: f64,
    target: f64,
    method: &Option<CandidateAcceptance<S>>,
) -> bool {
    method
        .as_ref()
        .map_or(norm <= target, CandidateAcceptance::accepted)
}

fn retain_candidate<S: KrylovScalar>(
    problem: &LinearProblem<'_, S>,
    candidate: &[S],
    residual: &mut [S],
    best: &mut [S],
    best_norm: &mut f64,
    failure: &ApplyFailure,
    method: &mut Option<CandidateAcceptance<S>>,
) {
    if candidate.iter().any(|value| !value.is_finite()) {
        return;
    }
    match residual_norm(problem, candidate, residual) {
        Ok(norm) => {
            if let Some(method) = method
                && let Err(error) = method.consider(problem, candidate, norm)
            {
                failure.record(error);
            }
            if norm < *best_norm {
                *best_norm = norm;
                best.copy_from_slice(candidate);
            }
        }
        Err(error) => failure.record(error),
    }
}

#[cfg(test)]
mod tests;
