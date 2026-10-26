//! Original residual and full-Newton-update acceptance have distinct meanings.
use super::{invalid_linearization, linearization_residual_norm};
use eqiora_core::Diagnostic;
use eqiora_solver::{CanonicalCsrSystemView, SolverPlan};
use std::sync::Arc;

#[cfg(test)]
mod tests;

/// Numerical criterion for the original equations at the selected point.
/// The provenance owner binds every witness to the same Model, inputs and point.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimalAcceptance {
    /// Bound on the original residual's Euclidean norm.
    Residual(f64),
    /// A full Newton correction accepted under physical coordinate controls.
    NewtonCorrection {
        /// Point at which the original correction equation was linearized.
        previous: Vec<f64>,
        /// Complete, undamped correction solving that equation.
        direction: Vec<f64>,
        /// Accepted point, exactly previous plus direction.
        accepted: Vec<f64>,
        /// Original correction matrix and right-hand side.
        system: Arc<CanonicalCsrSystemView>,
        /// Controls used to accept the correction's true linear residual.
        solve: SolverPlan,
        /// Relative physical-coordinate correction tolerance.
        relative: f64,
        /// Absolute physical-coordinate correction tolerances.
        absolute: Vec<f64>,
    },
    /// Contiguous equation blocks from separately accepted composed solves.
    /// Each entry gives its exact residual row count and acceptance criterion.
    Blocks(Vec<(usize, PrimalAcceptance)>),
}

impl PrimalAcceptance {
    /// Return a residual bound only when every block actually uses one.
    pub(super) fn validate(&self, residual: &[f64]) -> Result<Option<f64>, Diagnostic> {
        let norm = linearization_residual_norm(residual)?;
        match self {
            Self::Residual(tolerance) => {
                if !tolerance.is_finite() || *tolerance < 0. || norm > *tolerance {
                    return Err(invalid_linearization(format!(
                        "primal residual {norm:e} exceeds valid linearization tolerance {tolerance:e}"
                    )));
                }
                Ok(Some(*tolerance))
            }
            Self::NewtonCorrection {
                previous,
                direction,
                accepted,
                system,
                solve,
                relative,
                absolute,
            } => {
                let count = residual.len();
                if count == 0
                    || previous.len() != count
                    || direction.len() != count
                    || accepted.len() != count
                    || absolute.len() != count
                    || system.rows() != count
                    || system.columns() != count
                    || !relative.is_finite()
                    || *relative <= 0.
                    || absolute.iter().any(|v| !v.is_finite() || *v <= 0.)
                    || previous
                        .iter()
                        .chain(direction)
                        .chain(accepted)
                        .any(|v| !v.is_finite())
                {
                    return Err(invalid_linearization(
                        "Newton acceptance requires exact finite coordinates and physical controls",
                    ));
                }
                let mut weighted = 0_f64;
                for (((base, delta), end), absolute) in
                    previous.iter().zip(direction).zip(accepted).zip(absolute)
                {
                    let scale = absolute + relative * base.abs();
                    if !scale.is_finite()
                        || scale <= 0.
                        || (base + delta).to_bits() != end.to_bits()
                    {
                        return Err(invalid_linearization(
                            "Newton acceptance requires the complete undamped update",
                        ));
                    }
                    weighted = weighted.hypot(delta / scale);
                }
                weighted /= (count as f64).sqrt();
                if !weighted.is_finite() || weighted > 1. {
                    return Err(invalid_linearization(
                        "Newton correction exceeds its physical coordinate tolerance",
                    ));
                }
                let problem = system.linear_problem()?;
                let mut action = vec![0.; count];
                problem.operator().apply(direction, &mut action)?;
                for (value, rhs) in action.iter_mut().zip(system.right_hand_side()) {
                    *value -= rhs;
                }
                let target = solve
                    .residual_target(linearization_residual_norm(system.right_hand_side())?)?;
                if linearization_residual_norm(&action)? > target {
                    return Err(invalid_linearization(
                        "Newton correction fails its original linearized equation",
                    ));
                }
                Ok(None)
            }
            Self::Blocks(blocks) => {
                let mut offset = 0_usize;
                let mut tolerance = Some(0_f64);
                if blocks.is_empty() {
                    return Err(invalid_linearization(
                        "primal acceptance requires nonempty equation blocks",
                    ));
                }
                for (count, criterion) in blocks {
                    let end = offset
                        .checked_add(*count)
                        .filter(|end| *count > 0 && *end <= residual.len())
                        .ok_or_else(|| {
                            invalid_linearization(
                                "primal acceptance block has a foreign residual shape",
                            )
                        })?;
                    let bound = criterion.validate(&residual[offset..end])?;
                    tolerance = tolerance
                        .zip(bound)
                        .map(|(total, value)| total.hypot(value));
                    offset = end;
                }
                if offset != residual.len() || tolerance.is_some_and(|value| !value.is_finite()) {
                    return Err(invalid_linearization(
                        "primal acceptance blocks do not cover a finite complete residual",
                    ));
                }
                Ok(tolerance)
            }
        }
    }
}
