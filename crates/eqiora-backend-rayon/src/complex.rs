//! Typed complex delegation through the same pool, row actions and reductions.
use super::*;

#[cfg(test)]
mod tests;

impl LinearSolverBackend<Complex64> for ThreadedLinearSolver<'_> {
    fn default_execution_provider(&self) -> ExecutionProvider {
        RAYON_EXECUTION_PROVIDER
    }

    fn provider(&self) -> SolverProvider {
        self.backend.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        self.supported_capabilities()
            .expect("construction admitted an implemented reproducible backend tuple")
    }

    fn default_execution(&self) -> ExecutionReport {
        RayonLinearExecution { pool: self.pool }.report()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, Complex64>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution<Complex64>,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
        if execution.report() != SERIAL_LINEAR_EXECUTION.report() {
            return Err(invalid_realization(
                "a Rayon solver cannot be nested inside another replicated execution",
            ));
        }
        let backend = self.complex_implementation().ok_or_else(|| {
            invalid_realization(
                "the wrapped provider has no matching reproducible complex implementation",
            )
        })?;
        self.supported_capabilities()?.require_problem(
            plan,
            problem.scalar_domain(),
            problem.scalar_type(),
            problem.properties(),
        )?;
        backend.solve_with_execution(problem, plan, &RayonLinearExecution { pool: self.pool })
    }
}
