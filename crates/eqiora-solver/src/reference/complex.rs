mod hpd;

use eqiora_core::Diagnostic;
use num_complex::Complex64;

use crate::{
    FixedOrderInnerProduct, LinearOperatorProperties, LinearProblem, LinearSolution,
    LinearSolverBackend, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverPlan,
};

use super::{ReferenceLinearSolver, invalid_realization, solve_failed};

impl LinearSolverBackend<Complex64> for ReferenceLinearSolver {
    fn provider(&self) -> crate::SolverProvider {
        Self::provider(self)
    }

    fn capabilities(&self) -> crate::SolverCapabilities {
        Self::capabilities(self)
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, Complex64>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution<Complex64>,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
        problem.require_candidate_check_execution(execution.report())?;
        self.capabilities().require_problem(
            plan,
            problem.scalar_domain(),
            problem.scalar_type(),
            problem.properties(),
        )?;
        if execution.provider() != SERIAL_LINEAR_EXECUTION.provider()
            || execution.report() != SERIAL_LINEAR_EXECUTION.report()
        {
            return Err(invalid_realization(
                "complex reference solves require the host-serial execution profile",
            ));
        }
        execution.require_reduction(plan.reduction())?;
        let source = problem.canonical_csr_system().ok_or_else(|| {
            invalid_realization(
                "complex reference solves require captured canonical complex coefficients",
            )
        })?;
        if problem.properties() == LinearOperatorProperties::HermitianPositiveDefinite {
            hpd::require_positive_pivots(source)?;
        }
        if plan.algorithm() == crate::LinearSolver::BiConjugateGradientStabilized {
            return super::bicgstab::solve_preconditioned_bicgstab(
                self.provider(),
                problem,
                plan,
                execution,
            );
        }
        solve_real_coordinates(problem, plan, execution, source)
    }
}

fn solve_real_coordinates(
    problem: &LinearProblem<'_, Complex64>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution<Complex64>,
    source: &crate::CanonicalCsrSystemView<Complex64>,
) -> Result<LinearSolution<Complex64>, Diagnostic> {
    let block = source
        .real_coordinate_system(problem.operator().orientation(), problem.right_hand_side())?;
    let mut block_problem = block.linear_problem()?;
    // Lower only coordinates: the method still assesses physical complex
    // candidates and the real recurrence retains the original total budget.
    let target = plan.residual_target(FixedOrderInnerProduct::norm(
        execution,
        problem.right_hand_side(),
    )?)?;
    let candidate_check = |values: &[f64]| {
        let values = values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| Complex64::new(pair[0], pair[1]))
            .collect::<Vec<_>>();
        let mut applied = vec![Complex64::new(0., 0.); values.len()];
        Ok(
            super::true_residual_norm(execution, problem, &values, &mut applied)? <= target
                && (!problem.has_candidate_check() || problem.candidate_is_accepted(&values)?),
        )
    };
    block_problem = block_problem.with_candidate_check(&candidate_check);
    let initial = problem.initial_guess().map(pack);
    if let Some(initial) = &initial {
        block_problem = block_problem.with_initial_guess(initial)?;
    }
    let real = if plan.algorithm() == crate::LinearSolver::MinimumResidual {
        super::minres::solve(
            ReferenceLinearSolver.provider(),
            &block_problem,
            plan,
            &SERIAL_LINEAR_EXECUTION,
        )?
    } else {
        <ReferenceLinearSolver as LinearSolverBackend<f64>>::solve_with_execution(
            &ReferenceLinearSolver,
            &block_problem,
            plan,
            &SERIAL_LINEAR_EXECUTION,
        )?
    };
    let values = real
        .values()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect();
    // Verify the original typed action, not the numerical lowering.
    LinearSolution::accept_with_execution(
        problem,
        plan,
        ReferenceLinearSolver.provider(),
        real.report().reason(),
        real.report().completed_iterations(),
        real.report().reported_residual_norm(),
        values,
        execution,
    )
}

fn pack(values: &[Complex64]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}

#[cfg(test)]
mod tests;
