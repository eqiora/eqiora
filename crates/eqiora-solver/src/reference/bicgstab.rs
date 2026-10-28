//! One real/complex right-preconditioned BiCGSTAB recurrence and residual policy.

use super::correction::SolutionCorrection;
use super::*;

pub(super) fn solve_preconditioned_bicgstab<
    S: Scalar + ComplexFloat<Real = f64> + From<f64> + Sync,
>(
    provider: SolverProvider,
    problem: &LinearProblem<'_, S>,
    plan: SolverPlan,
    execution: &dyn ReplicatedLinearExecution<S>,
) -> Result<LinearSolution<S>, Diagnostic> {
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
    let accept = |values, iteration, reported, replayed| {
        let report = SolveReport::accepted(
            provider,
            execution.provider(),
            execution.report(),
            problem.operator().orientation(),
            plan,
            ConvergenceReason::ResidualToleranceSatisfied,
            iteration,
            initial_residual_norm,
            reported,
            replayed,
            target,
        )?;
        LinearSolution::new(values, report)
    };
    let dimension = solution.len();
    let mut correction = SolutionCorrection::new(&mut solution, target, initial_residual_norm);
    let mut recurrence_target = target;

    let inverse_diagonal = build_inverse_diagonal(
        problem,
        plan.preconditioner(),
        DiagonalRequirement::FiniteNonzero,
    )?;
    let mut recurrence = Recurrence::new(&residual);
    let mut preconditioned_direction = vec![S::zero(); dimension];
    let mut intermediate_residual = vec![S::zero(); dimension];
    let mut preconditioned_intermediate = vec![S::zero(); dimension];
    let mut intermediate_image = vec![S::zero(); dimension];
    let mut reported_residual_norm = initial_residual_norm;
    let mut cycle_length = dimension;
    let mut next_replacement = cycle_length;

    for iteration in 1..=plan.maximum_iterations().get() {
        // Finite precision can lose the biorthogonality that closes an exact
        // n-dimensional recurrence. Reapply b-Ax at bounded cycle boundaries;
        // growing cycles also allow useful history beyond n steps. Replacement
        // uses the existing correction owner and the original acceptance target.
        let replace_residual = iteration == next_replacement;
        if replace_residual {
            cycle_length = cycle_length.saturating_mul(2);
            next_replacement = iteration.saturating_add(cycle_length);
        }
        let mut shadow_product = dot(execution, &recurrence.shadow_residual, &residual)?;
        if shadow_product == S::zero() {
            // Orthogonality to the old shadow is not singularity of the operator.
            // Restart the recurrence at the current residual; acceptance still
            // requires the independently reapplied true residual below.
            recurrence.reset(&residual);
            shadow_product = dot(execution, &recurrence.shadow_residual, &residual)?;
        }
        if shadow_product == S::zero() {
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

        let mut alpha_denominator = dot(
            execution,
            &recurrence.shadow_residual,
            &recurrence.direction_image,
        )?;
        if alpha_denominator == S::zero() {
            if !fresh {
                recurrence.reset(&residual);
                continue;
            }
            // A fresh residual can be orthogonal to A M^-1 r even for a
            // nonsingular operator. Change the free shadow vector, not the
            // equation, preconditioner, stopping tolerance or iteration budget.
            (shadow_product, alpha_denominator) =
                recurrence.retarget_shadow(&residual, execution)?;
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

        let intermediate_residual_norm =
            FixedOrderInnerProduct::norm(execution, &intermediate_residual)?;
        let mut alpha_applied = false;
        if intermediate_residual_norm <= recurrence_target {
            for index in 0..dimension {
                solution[index] =
                    solution[index] + recurrence.alpha * preconditioned_direction[index];
            }
            alpha_applied = true;
            require_finite(&solution, "BiCGSTAB solution")?;
            let true_residual_norm = true_residual_norm(
                execution,
                problem,
                correction.candidate(&solution),
                &mut applied,
            )?;
            if true_residual_norm <= target
                && problem.candidate_is_accepted(correction.candidate(&solution))?
            {
                return accept(
                    correction.into_values(),
                    iteration,
                    intermediate_residual_norm,
                    true_residual_norm,
                );
            }
            if true_residual_norm == 0. {
                return Err(solve_failed(
                    "method-owned acceptance rejected a zero linear residual",
                ));
            }
            if true_residual_norm > target {
                recurrence_target = correction.restart(&mut solution, true_residual_norm);
                recurrence.restart(&applied, &mut residual);
                reported_residual_norm = true_residual_norm;
                continue;
            }
            // The global criterion passed but the method criterion did not.
            // Complete omega within this same iteration; alpha is already applied.
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
        if stabilization_denominator == S::zero() {
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
            solution[index] = if alpha_applied {
                solution[index] + recurrence.omega * preconditioned_intermediate[index]
            } else {
                solution[index]
                    + (recurrence.alpha * preconditioned_direction[index]
                        + recurrence.omega * preconditioned_intermediate[index])
            };
            residual[index] =
                intermediate_residual[index] - recurrence.omega * intermediate_image[index];
        }
        require_finite(&solution, "BiCGSTAB solution")?;
        require_finite(&residual, "BiCGSTAB residual")?;
        reported_residual_norm = FixedOrderInnerProduct::norm(execution, &residual)?;
        if reported_residual_norm <= recurrence_target || replace_residual {
            let true_residual_norm = true_residual_norm(
                execution,
                problem,
                correction.candidate(&solution),
                &mut applied,
            )?;
            if true_residual_norm <= target
                && problem.candidate_is_accepted(correction.candidate(&solution))?
            {
                return accept(
                    correction.into_values(),
                    iteration,
                    reported_residual_norm,
                    true_residual_norm,
                );
            }
            if true_residual_norm == 0. {
                return Err(solve_failed(
                    "method-owned acceptance rejected a zero linear residual",
                ));
            }
            recurrence_target = correction.restart(&mut solution, true_residual_norm);
            recurrence.restart(&applied, &mut residual);
            reported_residual_norm = true_residual_norm;
            continue;
        }
        if recurrence.omega == S::zero() {
            return Err(solve_failed(
                "BiCGSTAB stabilization coefficient broke down before convergence",
            ));
        }
        recurrence.previous_shadow_product = shadow_product;
    }

    let true_residual_norm = true_residual_norm(
        execution,
        problem,
        correction.candidate(&solution),
        &mut applied,
    )?;
    if true_residual_norm <= target
        && problem.candidate_is_accepted(correction.candidate(&solution))?
    {
        return accept(
            correction.into_values(),
            plan.maximum_iterations().get(),
            reported_residual_norm,
            true_residual_norm,
        );
    }
    Err(solve_failed(format!(
        "BiCGSTAB reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
        plan.maximum_iterations()
    )))
}

struct Recurrence<S> {
    shadow_residual: Vec<S>,
    direction: Vec<S>,
    direction_image: Vec<S>,
    previous_shadow_product: S,
    alpha: S,
    omega: S,
    fresh_start: bool,
}

impl<S: Scalar + ComplexFloat<Real = f64>> Recurrence<S> {
    fn new(initial_residual: &[S]) -> Self {
        Self {
            shadow_residual: initial_residual.to_vec(),
            direction: vec![S::zero(); initial_residual.len()],
            direction_image: vec![S::zero(); initial_residual.len()],
            previous_shadow_product: S::one(),
            alpha: S::one(),
            omega: S::one(),
            fresh_start: true,
        }
    }

    fn prepare_direction(&mut self, shadow_product: S, residual: &[S]) -> Result<(), Diagnostic> {
        if self.fresh_start {
            self.direction.copy_from_slice(residual);
            self.fresh_start = false;
            return Ok(());
        }
        if self.omega == S::zero() {
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
            *direction = *residual + beta * (*direction - self.omega * *direction_image);
        }
        require_finite(&self.direction, "BiCGSTAB direction")
    }

    fn retarget_shadow(
        &mut self,
        residual: &[S],
        execution: &dyn ReplicatedLinearExecution<S>,
    ) -> Result<(S, S), Diagnostic>
    where
        S: From<f64>,
    {
        let residual_norm = FixedOrderInnerProduct::norm(execution, residual)?;
        let image_norm = FixedOrderInnerProduct::norm(execution, &self.direction_image)?;
        if residual_norm == 0. || image_norm == 0. {
            return Err(solve_failed(
                "BiCGSTAB step denominator broke down before convergence",
            ));
        }
        // Called only for a fresh p=r with <r,v>=0, v=A M^-1 r.
        // With h=r/||r||+v/||v||, <h,r>=||r|| and <h,v>=||v||.
        // Both are nonzero for a nonzero residual and operator image. Scaling
        // keeps the shadow bounded; the ordinary true-residual check still owns
        // acceptance. No operator application or random perturbation is added.
        for ((shadow, r), v) in self
            .shadow_residual
            .iter_mut()
            .zip(residual)
            .zip(&self.direction_image)
        {
            *shadow = *r / <S as From<f64>>::from(residual_norm)
                + *v / <S as From<f64>>::from(image_norm);
        }
        require_finite(&self.shadow_residual, "BiCGSTAB replacement shadow")?;
        let rho = dot(execution, &self.shadow_residual, residual)?;
        let denominator = dot(execution, &self.shadow_residual, &self.direction_image)?;
        if rho == S::zero() || denominator == S::zero() {
            return Err(solve_failed(
                "BiCGSTAB replacement shadow broke down before convergence",
            ));
        }
        Ok((rho, denominator))
    }

    fn restart(&mut self, true_residual: &[S], residual: &mut [S]) {
        residual.copy_from_slice(true_residual);
        self.reset(residual);
    }

    fn reset(&mut self, residual: &[S]) {
        self.shadow_residual.copy_from_slice(residual);
        self.fresh_start = true;
        self.previous_shadow_product = S::one();
        self.alpha = S::one();
        self.omega = S::one();
    }
}

#[cfg(test)]
mod tests;
