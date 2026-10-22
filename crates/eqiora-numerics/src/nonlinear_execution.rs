//! Damped Newton over an exact residual occurrence, independent of physical roles.

use eqiora_core::{Diagnostic, diagnostic::codes};
mod stopping;
use eqiora_solver::SolveReport;
use stopping::NewtonConvergence;
pub(crate) use stopping::{NewtonOptions, NewtonStopping};

#[derive(Debug)]
pub(crate) struct NewtonEvidence {
    pub(crate) iterations: usize,
    pub(crate) initial_residual_norm: f64,
    pub(crate) convergence: NewtonConvergence,
    pub(crate) linear_solves: Vec<SolveReport>,
}

impl NewtonEvidence {
    pub(crate) fn residual_target(&self) -> Result<f64, Diagnostic> {
        match self.convergence {
            NewtonConvergence::Residual { target } => Ok(target),
            NewtonConvergence::Correction { .. } | NewtonConvergence::ExactResidualZero => Err(
                failed("correction convergence does not establish a residual target"),
            ),
        }
    }
}

/// Seed facts rederived at an accepted point, independent of a producer's history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NewtonInitialPoint {
    pub(crate) residual_norm: f64,
    pub(crate) at_seed: bool,
}

impl NewtonInitialPoint {
    /// The caller separately checks the Plan's maximum update count.
    pub(crate) fn validate_updates(self, target: f64, iterations: usize) -> Result<(), Diagnostic> {
        if (iterations == 0) != (self.residual_norm <= target) || (iterations == 0 && !self.at_seed)
        {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "nonlinear iteration history contradicts zero-update initial State acceptance",
            ));
        }
        Ok(())
    }
}

/// A trial evaluator may retain an already computed linearization. Otherwise
/// only the accepted trial is linearized. Neither path publishes a State.
pub(crate) struct NewtonTrial<L> {
    pub(crate) residual_norm: f64,
    pub(crate) linearization: Option<L>,
}

/// The correction solves the caller's exact linearized equation. `trial` returns
/// `None` only for a recognized point-dependent admissibility failure; ordinary
/// diagnostics propagate. Expression-domain violations are also rejected trials,
/// only at this speculative boundary. Initial and accepted linearizations still
/// propagate every error. The trial receives the current linearization so both
/// merits can use the same equation scaling. Correction convergence requires the
/// full Newton step, never just a small damped update. Physical acceptance remains
/// the caller's obligation.
pub(crate) fn solve_damped_newton<L>(
    mut point: Vec<f64>,
    options: NewtonOptions<'_>,
    mut linearize: impl FnMut(&[f64]) -> Result<L, Diagnostic>,
    mut residual_norm: impl FnMut(&L) -> Result<f64, Diagnostic>,
    mut correction: impl FnMut(&[f64], &L) -> Result<(Vec<f64>, SolveReport), Diagnostic>,
    mut trial: impl FnMut(&[f64], &L) -> Result<Option<NewtonTrial<L>>, Diagnostic>,
) -> Result<(L, NewtonEvidence), Diagnostic> {
    if point.is_empty() || point.iter().any(|value| !value.is_finite()) {
        return Err(failed("Newton requires a nonempty finite initial point"));
    }
    options.validate(point.len())?;
    let mut current = {
        let _span = eqiora_execution::telemetry_span!(assembly("initial_linearization")).entered();
        linearize(&point)?
    };
    let initial_residual_norm = checked_norm(residual_norm(&current)?)?;
    let residual_target = options.stopping.residual_target(initial_residual_norm)?;
    let mut evidence = NewtonEvidence {
        iterations: 0,
        initial_residual_norm,
        convergence: NewtonConvergence::Residual {
            target: residual_target,
        },
        linear_solves: Vec::new(),
    };
    eqiora_execution::telemetry_event!(
        "newton",
        0,
        initial_residual_norm,
        residual_target,
        initial_residual_norm <= residual_target
    );
    if initial_residual_norm <= residual_target {
        evidence.convergence =
            options
                .stopping
                .convergence(residual_target, 0., initial_residual_norm);
        return Ok((current, evidence));
    }
    evidence
        .linear_solves
        .try_reserve_exact(options.maximum_iterations.get())
        .map_err(|_| failed("Newton report allocation failed"))?;
    for iteration in 1..=options.maximum_iterations.get() {
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
        let correction_norm = options.stopping.correction_norm(&point, &direction)?;
        let mut accepted = None;
        let mut best_trial = (f64::INFINITY, 0.0);
        let mut scale = 1.0;
        for _ in 0..=options.maximum_line_search_steps {
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
                match trial(&candidate, &current) {
                    Err(error) if error.code() == codes::EXPRESSION_DOMAIN_VIOLATION => None,
                    evaluated => evaluated?,
                }
            };
            if let Some(evaluated) = evaluated {
                let trial_norm = checked_norm(evaluated.residual_norm)?;
                if trial_norm < best_trial.0 {
                    best_trial = (trial_norm, scale);
                }
                let small_full_correction =
                    scale == 1.0 && correction_norm.is_some_and(|n| n <= 1.);
                if small_full_correction
                    || trial_norm <= residual_target
                    || trial_norm < previous_norm
                {
                    accepted = Some((candidate, evaluated, trial_norm, small_full_correction));
                    break;
                }
            }
            scale *= 0.5;
        }
        let Some((candidate, evaluated, trial_norm, small_full_correction)) = accepted else {
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
        evidence.iterations = iteration;
        evidence.linear_solves.push(report);
        let converged = trial_norm <= residual_target || small_full_correction;
        match correction_norm {
            Some(norm) if trial_norm != 0. => {
                eqiora_execution::telemetry_event!(
                    "newton_correction",
                    iteration,
                    norm,
                    1.,
                    converged
                );
            }
            _ => {
                eqiora_execution::telemetry_event!(
                    "newton",
                    iteration,
                    trial_norm,
                    residual_target,
                    converged
                );
            }
        }
        if converged {
            evidence.convergence = options.stopping.convergence(
                residual_target,
                correction_norm.unwrap_or(0.),
                trial_norm,
            );
            return Ok((current, evidence));
        }
    }
    Err(failed(format!(
        "Newton solve reached {} iterations without meeting its stopping criterion",
        options.maximum_iterations,
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
