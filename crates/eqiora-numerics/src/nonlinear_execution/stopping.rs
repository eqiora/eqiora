//! Distinct residual and state-correction contracts share Newton globalization.
use super::{Diagnostic, failed};
use eqiora_realization::NonlinearSolvePlan;
use std::num::NonZeroUsize;

pub(crate) struct NewtonOptions<'a> {
    pub(crate) maximum_iterations: NonZeroUsize,
    pub(crate) maximum_line_search_steps: usize,
    pub(crate) stopping: NewtonStopping<'a>,
}

pub(crate) enum NewtonStopping<'a> {
    Residual { relative: f64, absolute: f64 },
    Correction { relative: f64, absolute: &'a [f64] },
}

#[derive(Debug)]
pub(crate) enum NewtonConvergence {
    Residual { target: f64 },
    Correction { weighted_norm: f64 },
    ExactResidualZero,
}

impl From<NonlinearSolvePlan> for NewtonOptions<'_> {
    fn from(plan: NonlinearSolvePlan) -> Self {
        Self {
            maximum_iterations: plan.maximum_iterations(),
            maximum_line_search_steps: plan.maximum_line_search_steps(),
            stopping: NewtonStopping::Residual {
                relative: plan.relative_tolerance(),
                absolute: plan.absolute_tolerance(),
            },
        }
    }
}

impl NewtonOptions<'_> {
    pub(super) fn validate(&self, dimension: usize) -> Result<(), Diagnostic> {
        let valid = match self.stopping {
            NewtonStopping::Residual { relative, absolute } => {
                relative.is_finite()
                    && relative >= 0.
                    && absolute.is_finite()
                    && absolute >= 0.
                    && (relative > 0. || absolute > 0.)
            }
            NewtonStopping::Correction { relative, absolute } => {
                relative.is_finite()
                    && relative > 0.
                    && absolute.len() == dimension
                    && absolute.iter().all(|v| v.is_finite() && *v > 0.)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(failed(
                "Newton stopping controls do not match finite point coordinates",
            ))
        }
    }
}

impl NewtonStopping<'_> {
    pub(super) fn residual_target(&self, initial: f64) -> Result<f64, Diagnostic> {
        let target = match *self {
            Self::Residual { relative, absolute } => absolute.max(relative * initial),
            // A correction tolerance is not a residual tolerance.
            Self::Correction { .. } => 0.,
        };
        if target.is_finite() {
            Ok(target)
        } else {
            Err(failed("Newton residual target is non-finite"))
        }
    }

    pub(super) fn correction_norm(
        &self,
        point: &[f64],
        direction: &[f64],
    ) -> Result<Option<f64>, Diagnostic> {
        let Self::Correction { relative, absolute } = self else {
            return Ok(None);
        };
        let mut norm = 0_f64;
        for ((value, delta), absolute) in point.iter().zip(direction).zip(*absolute) {
            let scale = absolute + relative * value.abs();
            if !scale.is_finite() || scale <= 0. {
                return Err(failed(
                    "Newton correction scale must be finite and positive",
                ));
            }
            norm = norm.hypot(delta / scale);
        }
        let norm = norm / (point.len() as f64).sqrt();
        if norm.is_finite() {
            Ok(Some(norm))
        } else {
            Err(failed("Newton weighted correction norm is non-finite"))
        }
    }

    pub(super) fn convergence(&self, target: f64, norm: f64, residual: f64) -> NewtonConvergence {
        match self {
            Self::Residual { .. } => NewtonConvergence::Residual { target },
            Self::Correction { .. } if residual == 0. => NewtonConvergence::ExactResidualZero,
            Self::Correction { .. } => NewtonConvergence::Correction {
                weighted_norm: norm,
            },
        }
    }
}
