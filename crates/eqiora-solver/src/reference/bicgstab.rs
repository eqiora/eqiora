//! Deterministic right-preconditioned BiCGSTAB reference iteration.

use super::*;

pub(super) fn solve_preconditioned_bicgstab(
    provider: SolverProvider,
    problem: &LinearProblem<'_>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution,
) -> Result<LinearSolution, Diagnostic> {
    let InitialResidual {
        mut solution,
        mut applied,
        mut residual,
        initial_residual_norm,
        target,
        accepted_report,
    } = initial_residual(provider, problem, plan, execution)?;
    if let Some(report) = accepted_report {
        return LinearSolution::new(solution, report);
    }
    let dimension = solution.len();

    let inverse_diagonal = build_inverse_diagonal(
        problem,
        plan.preconditioner(),
        DiagonalRequirement::FiniteNonzero,
    )?;
    let mut recurrence = Recurrence::new(&residual);
    let mut preconditioned_direction = vec![0.0; dimension];
    let mut intermediate_residual = vec![0.0; dimension];
    let mut preconditioned_intermediate = vec![0.0; dimension];
    let mut intermediate_image = vec![0.0; dimension];
    let mut reported_residual_norm = initial_residual_norm;

    for iteration in 1..=plan.maximum_iterations().get() {
        let mut shadow_product = dot(execution, &recurrence.shadow_residual, &residual)?;
        if shadow_product == 0.0 {
            // Orthogonality to the old shadow is not singularity of the operator.
            // Restart the recurrence at the current residual; acceptance still
            // requires the independently reapplied true residual below.
            recurrence.reset(&residual);
            shadow_product = dot(execution, &recurrence.shadow_residual, &residual)?;
        }
        if shadow_product == 0.0 {
            return Err(solve_failed(
                "BiCGSTAB shadow-residual product broke down before convergence",
            ));
        }
        let fresh = recurrence.fresh_start;
        recurrence.prepare_direction(shadow_product, &residual)?;
        apply_preconditioner(
            &inverse_diagonal,
            &recurrence.direction,
            &mut preconditioned_direction,
        );
        require_finite(
            &preconditioned_direction,
            "BiCGSTAB preconditioned direction",
        )?;
        execution.apply(
            problem.operator(),
            &preconditioned_direction,
            &mut recurrence.direction_image,
        )?;
        require_finite(&recurrence.direction_image, "BiCGSTAB operator image")?;

        let alpha_denominator = dot(
            execution,
            &recurrence.shadow_residual,
            &recurrence.direction_image,
        )?;
        if alpha_denominator == 0.0 {
            if !fresh {
                recurrence.reset(&residual);
                continue;
            }
            return Err(solve_failed(
                "BiCGSTAB step denominator broke down before convergence",
            ));
        }
        recurrence.alpha = shadow_product / alpha_denominator;
        if !recurrence.alpha.is_finite() {
            return Err(solve_failed(
                "BiCGSTAB step produced a non-finite coefficient",
            ));
        }
        for index in 0..dimension {
            intermediate_residual[index] =
                residual[index] - recurrence.alpha * recurrence.direction_image[index];
        }
        require_finite(&intermediate_residual, "BiCGSTAB intermediate residual")?;

        let intermediate_residual_norm = norm(execution, &intermediate_residual)?;
        if intermediate_residual_norm <= target {
            for index in 0..dimension {
                solution[index] += recurrence.alpha * preconditioned_direction[index];
            }
            require_finite(&solution, "BiCGSTAB solution")?;
            let true_residual_norm =
                true_residual_norm(execution, problem, &solution, &mut applied)?;
            if true_residual_norm <= target {
                let report = SolveReport::accepted(
                    provider,
                    execution.provider(),
                    execution.report(),
                    problem.operator().orientation(),
                    plan,
                    ConvergenceReason::ResidualToleranceSatisfied,
                    iteration,
                    initial_residual_norm,
                    intermediate_residual_norm,
                    true_residual_norm,
                    target,
                )?;
                return LinearSolution::new(solution, report);
            }
            recurrence.restart(&applied, &mut residual);
            reported_residual_norm = true_residual_norm;
            continue;
        }

        apply_preconditioner(
            &inverse_diagonal,
            &intermediate_residual,
            &mut preconditioned_intermediate,
        );
        require_finite(
            &preconditioned_intermediate,
            "BiCGSTAB preconditioned intermediate residual",
        )?;
        execution.apply(
            problem.operator(),
            &preconditioned_intermediate,
            &mut intermediate_image,
        )?;
        require_finite(&intermediate_image, "BiCGSTAB stabilization image")?;
        let stabilization_denominator = dot(execution, &intermediate_image, &intermediate_image)?;
        if stabilization_denominator == 0.0 {
            return Err(solve_failed(
                "BiCGSTAB stabilization denominator broke down before convergence",
            ));
        }
        recurrence.omega = dot(execution, &intermediate_image, &intermediate_residual)?
            / stabilization_denominator;
        if !recurrence.omega.is_finite() {
            return Err(solve_failed(
                "BiCGSTAB stabilization produced a non-finite coefficient",
            ));
        }
        for index in 0..dimension {
            solution[index] += recurrence.alpha * preconditioned_direction[index]
                + recurrence.omega * preconditioned_intermediate[index];
            residual[index] =
                intermediate_residual[index] - recurrence.omega * intermediate_image[index];
        }
        require_finite(&solution, "BiCGSTAB solution")?;
        require_finite(&residual, "BiCGSTAB residual")?;
        reported_residual_norm = norm(execution, &residual)?;
        if reported_residual_norm <= target {
            let true_residual_norm =
                true_residual_norm(execution, problem, &solution, &mut applied)?;
            if true_residual_norm <= target {
                let report = SolveReport::accepted(
                    provider,
                    execution.provider(),
                    execution.report(),
                    problem.operator().orientation(),
                    plan,
                    ConvergenceReason::ResidualToleranceSatisfied,
                    iteration,
                    initial_residual_norm,
                    reported_residual_norm,
                    true_residual_norm,
                    target,
                )?;
                return LinearSolution::new(solution, report);
            }
            recurrence.restart(&applied, &mut residual);
            reported_residual_norm = true_residual_norm;
            continue;
        }
        if recurrence.omega == 0.0 {
            return Err(solve_failed(
                "BiCGSTAB stabilization coefficient broke down before convergence",
            ));
        }
        recurrence.previous_shadow_product = shadow_product;
    }

    let true_residual_norm = true_residual_norm(execution, problem, &solution, &mut applied)?;
    Err(solve_failed(format!(
        "BiCGSTAB reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
        plan.maximum_iterations()
    )))
}

struct Recurrence {
    shadow_residual: Vec<f64>,
    direction: Vec<f64>,
    direction_image: Vec<f64>,
    previous_shadow_product: f64,
    alpha: f64,
    omega: f64,
    fresh_start: bool,
}

impl Recurrence {
    fn new(initial_residual: &[f64]) -> Self {
        Self {
            shadow_residual: initial_residual.to_vec(),
            direction: vec![0.0; initial_residual.len()],
            direction_image: vec![0.0; initial_residual.len()],
            previous_shadow_product: 1.0,
            alpha: 1.0,
            omega: 1.0,
            fresh_start: true,
        }
    }

    fn prepare_direction(
        &mut self,
        shadow_product: f64,
        residual: &[f64],
    ) -> Result<(), Diagnostic> {
        if self.fresh_start {
            self.direction.copy_from_slice(residual);
            self.fresh_start = false;
            return Ok(());
        }
        if self.omega == 0.0 {
            return Err(solve_failed(
                "BiCGSTAB stabilization coefficient broke down before convergence",
            ));
        }
        let beta = (shadow_product / self.previous_shadow_product) * (self.alpha / self.omega);
        if !beta.is_finite() {
            return Err(solve_failed(
                "BiCGSTAB direction recurrence produced a non-finite coefficient",
            ));
        }
        for ((direction, direction_image), residual) in self
            .direction
            .iter_mut()
            .zip(&self.direction_image)
            .zip(residual)
        {
            *direction = residual + beta * (*direction - self.omega * direction_image);
        }
        require_finite(&self.direction, "BiCGSTAB direction")
    }

    fn restart(&mut self, true_residual: &[f64], residual: &mut [f64]) {
        residual.copy_from_slice(true_residual);
        self.reset(residual);
    }

    fn reset(&mut self, residual: &[f64]) {
        self.shadow_residual.copy_from_slice(residual);
        self.fresh_start = true;
        self.previous_shadow_product = 1.0;
        self.alpha = 1.0;
        self.omega = 1.0;
    }
}

#[cfg(test)]
mod tests;
