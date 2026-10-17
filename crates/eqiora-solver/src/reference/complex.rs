mod bicgstab;
mod hpd;

use eqiora_core::Diagnostic;
use num_complex::Complex64;

use crate::{
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolverBackend,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolveReport, SolverPlan,
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
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
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
        let produced = if plan.algorithm() == crate::LinearSolver::BiConjugateGradientStabilized {
            bicgstab::solve(problem, plan)?
        } else {
            let block = source.real_coordinate_system(
                problem.operator().orientation(),
                problem.right_hand_side(),
            )?;
            let mut block_problem = block.linear_problem()?;
            let initial = problem.initial_guess().map(pack);
            if let Some(initial) = &initial {
                block_problem = block_problem.with_initial_guess(initial)?;
            }
            let real = <Self as LinearSolverBackend<f64>>::solve_with_execution(
                self,
                &block_problem,
                plan,
                execution,
            )?;
            bicgstab::Produced {
                values: real
                    .values()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| Complex64::new(pair[0], pair[1]))
                    .collect(),
                reason: real.report().reason(),
                iterations: real.report().completed_iterations(),
                residual: real.report().reported_residual_norm(),
            }
        };
        let values = produced.values;
        // Replay the original typed action, not the lowering used by production.
        let initial_values = problem.initial_guess().map_or_else(
            || vec![Complex64::new(0., 0.); source.columns()],
            <[Complex64]>::to_vec,
        );
        let initial_norm = residual_norm(problem, &initial_values)?;
        let true_norm = residual_norm(problem, &values)?;
        let rhs_norm = norm(problem.right_hand_side())?;
        let report = SolveReport::accepted(
            self.provider(),
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            produced.reason,
            produced.iterations,
            initial_norm,
            produced.residual,
            true_norm,
            plan.residual_target(rhs_norm)?,
        )?;
        LinearSolution::new(values, report)
    }
}

fn pack(values: &[Complex64]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}

fn norm(values: &[Complex64]) -> Result<f64, Diagnostic> {
    let coordinates = pack(values);
    SERIAL_LINEAR_EXECUTION
        .inner_product(crate::FixedOrderInnerProduct::new(
            &coordinates,
            &coordinates,
        )?)
        .map(f64::sqrt)
}

fn residual_norm(
    problem: &LinearProblem<'_, Complex64>,
    values: &[Complex64],
) -> Result<f64, Diagnostic> {
    let mut applied = vec![Complex64::new(0., 0.); values.len()];
    problem.operator().apply(values, &mut applied)?;
    for (applied, rhs) in applied.iter_mut().zip(problem.right_hand_side()) {
        *applied = *rhs - *applied;
    }
    norm(&applied)
}

#[cfg(test)]
mod tests;
