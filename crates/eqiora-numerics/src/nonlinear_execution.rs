//! Damped Newton over an exact residual occurrence, independent of physical roles.

use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::SolveReport;

#[derive(Debug)]
pub(crate) struct NewtonEvidence {
    pub(crate) iterations: usize,
    pub(crate) initial_residual_norm: f64,
    pub(crate) residual_target: f64,
    pub(crate) linear_solves: Vec<SolveReport>,
}

/// A trial evaluator may retain an already computed linearization. Otherwise
/// only the accepted trial is linearized. Neither path publishes a State.
pub(crate) struct NewtonTrial<L> {
    pub(crate) residual_norm: f64,
    pub(crate) linearization: Option<L>,
}

/// The correction solves the caller's exact linearized equation. `trial` returns
/// `None` only for a recognized point-dependent admissibility failure; ordinary
/// diagnostics propagate. Physical acceptance remains the caller's obligation.
pub(crate) fn solve_damped_newton<L>(
    mut point: Vec<f64>,
    plan: NonlinearSolvePlan,
    mut linearize: impl FnMut(&[f64]) -> Result<L, Diagnostic>,
    mut residual_norm: impl FnMut(&L) -> Result<f64, Diagnostic>,
    mut correction: impl FnMut(&[f64], &L) -> Result<(Vec<f64>, SolveReport), Diagnostic>,
    mut trial: impl FnMut(&[f64]) -> Result<Option<NewtonTrial<L>>, Diagnostic>,
) -> Result<(L, NewtonEvidence), Diagnostic> {
    if point.is_empty() || point.iter().any(|value| !value.is_finite()) {
        return Err(failed("Newton requires a nonempty finite initial point"));
    }
    let mut current = {
        let _span = eqiora_execution::telemetry_span!(assembly("initial_linearization")).entered();
        linearize(&point)?
    };
    let initial_residual_norm = checked_norm(residual_norm(&current)?)?;
    let residual_target = plan
        .absolute_tolerance()
        .max(plan.relative_tolerance() * initial_residual_norm);
    let mut evidence = NewtonEvidence {
        iterations: 0,
        initial_residual_norm,
        residual_target,
        linear_solves: Vec::new(),
    };
    let mut norm = initial_residual_norm;
    eqiora_execution::telemetry_event!("newton", 0, norm, residual_target, norm <= residual_target);
    if norm <= residual_target {
        return Ok((current, evidence));
    }
    evidence
        .linear_solves
        .try_reserve_exact(plan.maximum_iterations().get())
        .map_err(|_| failed("Newton report allocation failed"))?;
    for iteration in 1..=plan.maximum_iterations().get() {
        // A residual-only trial and its subsequent linearization may differ
        // by rounding. Compare against the current assembled residual.
        let previous_norm = checked_norm(residual_norm(&current)?)?;
        let _span = eqiora_execution::telemetry_span!(nonlinear_iteration(
            "newton",
            iteration,
            previous_norm
        ))
        .entered();
        let (direction, report) = correction(&point, &current)?;
        if direction.len() != point.len() || direction.iter().any(|value| !value.is_finite()) {
            return Err(failed(
                "Newton correction differs from the finite point coordinates",
            ));
        }
        let mut accepted = None;
        let mut best_trial = (f64::INFINITY, 0.0);
        let mut scale = 1.0;
        for _ in 0..=plan.maximum_line_search_steps() {
            let candidate = point
                .iter()
                .zip(&direction)
                .map(|(point, direction)| point + scale * direction)
                .collect::<Vec<_>>();
            if candidate.iter().any(|value| !value.is_finite()) {
                return Err(failed("Newton trial coordinates are non-finite"));
            }
            let evaluated = {
                let _span =
                    eqiora_execution::telemetry_span!(assembly("line_search_trial")).entered();
                trial(&candidate)?
            };
            if let Some(evaluated) = evaluated {
                let trial_norm = checked_norm(evaluated.residual_norm)?;
                if trial_norm < best_trial.0 {
                    best_trial = (trial_norm, scale);
                }
                if trial_norm <= residual_target || trial_norm < previous_norm {
                    accepted = Some((candidate, evaluated, trial_norm));
                    break;
                }
            }
            scale *= 0.5;
        }
        let Some((candidate, evaluated, trial_norm)) = accepted else {
            return Err(failed(format!(
                "Newton line search found no residual-decreasing admissible point at iteration \
                 {iteration}: previous residual {previous_norm:e}, best trial residual {:e} at scale {:e}, \
                 target {residual_target:e}",
                best_trial.0, best_trial.1,
            )));
        };
        let _span = eqiora_execution::telemetry_span!(assembly("accepted_linearization")).entered();
        current = match evaluated.linearization {
            Some(linearization) => linearization,
            None => linearize(&candidate)?,
        };
        point = candidate;
        norm = trial_norm;
        evidence.iterations = iteration;
        evidence.linear_solves.push(report);
        eqiora_execution::telemetry_event!(
            "newton",
            iteration,
            norm,
            residual_target,
            norm <= residual_target
        );
        if norm <= residual_target {
            return Ok((current, evidence));
        }
    }
    Err(failed(format!(
        "Newton solve reached {} iterations above target {residual_target:e}",
        plan.maximum_iterations(),
    )))
}

fn checked_norm(norm: f64) -> Result<f64, Diagnostic> {
    if norm.is_finite() && norm >= 0.0 {
        Ok(norm)
    } else {
        Err(failed(
            "Newton residual norm must be finite and non-negative",
        ))
    }
}

fn failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests;
