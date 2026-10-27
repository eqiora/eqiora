//! Retain method acceptance separately from recurrence/minimum-residual state.
use super::*;
use eqiora_solver::{FixedOrderInnerProduct, SERIAL_LINEAR_EXECUTION};

// Faer compares norm < threshold. The smallest positive binary64 number accepts
// exactly representable zero, and no finite positive norm compares below it.
// Literal zero would miss alpha-only exact convergence and enter omega=0/0.
pub(super) const ZERO_ONLY_TOLERANCE: f64 = f64::from_bits(1);

pub(super) struct CandidateAcceptance {
    target: f64,
    accepted: Option<(Vec<f64>, f64)>,
}

impl CandidateAcceptance {
    pub(super) fn new(target: f64) -> Self {
        Self {
            target,
            accepted: None,
        }
    }

    pub(super) fn consider(
        &mut self,
        problem: &LinearProblem<'_>,
        values: &[f64],
        norm: f64,
    ) -> Result<(), Diagnostic> {
        // Freeze the first jointly accepted point. Faer cannot interrupt its call;
        // final shared acceptance checks that same retained point again.
        if self.accepted.is_none()
            && norm <= self.target
            && problem.candidate_is_accepted(values)?
        {
            self.accepted = Some((values.to_vec(), norm));
        }
        Ok(())
    }

    pub(super) fn accepted(&self) -> bool {
        self.accepted.is_some()
    }

    pub(super) fn finish(
        self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        provider: SolverProvider,
        iterations: usize,
    ) -> Result<LinearSolution, Diagnostic> {
        let (values, norm) = self.accepted.ok_or_else(|| {
            solve_failed(
                "faer invocation ended without a jointly accepted true-residual/method candidate",
            )
        })?;
        eqiora_solver::LinearSolution::accept(
            problem,
            plan,
            provider,
            convergence_reason(iterations),
            iterations,
            norm,
            values,
        )
    }
}

pub(super) fn residual_norm(
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

#[cfg(test)]
mod tests;
