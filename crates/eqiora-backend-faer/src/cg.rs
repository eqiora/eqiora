//! Faer CG invocation with optional method-owned candidate retention.
use super::*;
use faer::matrix_free::conjugate_gradient::{
    CgError, CgParams, conjugate_gradient, conjugate_gradient_scratch,
};

pub(super) fn solve(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<f64>>,
) -> Result<LinearSolution, Diagnostic> {
    if problem.has_candidate_check() {
        return solve_with_candidate_check(provider, problem, plan, inverse_diagonal);
    }
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let preconditioner = FaerPreconditioner::new(dimension, inverse_diagonal);
    let parallelism = Par::Seq;
    let scratch =
        conjugate_gradient_scratch(preconditioner.clone(), operator.clone(), 1, parallelism);
    let mut workspace = MemBuffer::new(scratch);
    let params = CgParams {
        initial_guess: EXPLICIT_INITIAL_GUESS,
        abs_tolerance: plan.absolute_tolerance(),
        rel_tolerance: plan.relative_tolerance(),
        max_iters: plan.maximum_iterations().get(),
        ..Default::default()
    };
    let result = conjugate_gradient(
        output.as_mut(),
        preconditioner,
        operator,
        right_hand_side.as_ref(),
        params,
        |_| {},
        parallelism,
        MemStack::new(&mut workspace),
    );
    failure.take()?;
    let info = result.map_err(|error| solve_failed(format!("faer CG failed: {error:?}")))?;
    eqiora_solver::LinearSolution::accept(
        problem,
        plan,
        provider,
        convergence_reason(info.iter_count),
        info.iter_count,
        info.abs_residual,
        output.col_as_slice(0).to_vec(),
    )
}

fn solve_with_candidate_check(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    inverse_diagonal: Option<Vec<f64>>,
) -> Result<LinearSolution, Diagnostic> {
    use crate::candidate::{CandidateAcceptance, ZERO_ONLY_TOLERANCE, residual_norm};
    use eqiora_solver::{FixedOrderInnerProduct, SERIAL_LINEAR_EXECUTION};
    let dimension = problem.operator().columns();
    let mut output = initial_matrix(problem);
    let mut candidate = output.col_as_slice(0).to_vec();
    let mut residual = vec![0.; dimension];
    let target = plan.residual_target(FixedOrderInnerProduct::norm(
        &SERIAL_LINEAR_EXECUTION,
        problem.right_hand_side(),
    )?)?;
    let mut method = CandidateAcceptance::new(target);
    method.consider(
        problem,
        &candidate,
        residual_norm(problem, &candidate, &mut residual)?,
    )?;
    if method.accepted() {
        return method.finish(problem, plan, provider, 0);
    }
    let right_hand_side = right_hand_side_matrix(problem);
    let failure = ApplyFailure::default();
    let operator = FaerOperator::new(problem.operator(), failure.clone());
    let preconditioner = FaerPreconditioner::new(dimension, inverse_diagonal);
    let parallelism = Par::Seq;
    let mut workspace = MemBuffer::new(conjugate_gradient_scratch(
        preconditioner.clone(),
        operator.clone(),
        1,
        parallelism,
    ));
    let params = CgParams {
        initial_guess: EXPLICIT_INITIAL_GUESS,
        abs_tolerance: ZERO_ONLY_TOLERANCE,
        rel_tolerance: 0.,
        max_iters: plan.maximum_iterations().get(),
        ..Default::default()
    };
    let mut observe = |values: &[f64]| {
        if values.iter().any(|value| !value.is_finite()) {
            return;
        }
        match residual_norm(problem, values, &mut residual)
            .and_then(|norm| method.consider(problem, values, norm))
        {
            Ok(()) => {}
            Err(error) => failure.record(error),
        }
    };
    let result = conjugate_gradient(
        output.as_mut(),
        preconditioner,
        operator,
        right_hand_side.as_ref(),
        params,
        |point| {
            for (index, value) in candidate.iter_mut().enumerate() {
                *value = point[(index, 0)];
            }
            observe(&candidate);
        },
        parallelism,
        MemStack::new(&mut workspace),
    );
    observe(output.col_as_slice(0));
    failure.take()?;
    let iterations = match result {
        // A zero-RHS shortcut can replace an initially rejected guess without
        // a Krylov iteration. Charge that correction, as the BiCG adapter does.
        Ok(info) => info.iter_count.max(1),
        Err(CgError::NoConvergence { .. }) => plan.maximum_iterations().get(),
        Err(error) => return Err(solve_failed(format!("faer CG failed: {error:?}"))),
    };
    method.finish(problem, plan, provider, iterations)
}
