//! Time method selection independent of the spatial representation.
use super::*;

/// Deterministic time method and its validated method-specific controls.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTemporalPolicy(TemporalControls);

#[derive(Debug, Clone, PartialEq)]
enum TemporalControls {
    BackwardEuler(DynQuantity),
    Ode(CommonOdePolicy),
}

impl CommonTemporalPolicy {
    /// Construct one positive coherent-SI backward-Euler operator step.
    pub fn backward_euler(step_s: f64) -> Result<Self, Diagnostic> {
        if !step_s.is_finite() || step_s <= 0.0 || step_s.to_bits() == (-0.0_f64).to_bits() {
            return Err(invalid(
                "BackwardEuler step_s must be finite and strictly positive",
            ));
        }
        Ok(Self(TemporalControls::BackwardEuler(DynQuantity::new(
            step_s, TIME,
        ))))
    }

    /// Exact operator duration, only when backward Euler was requested.
    #[must_use]
    pub const fn as_backward_euler(&self) -> Option<DynQuantity> {
        match &self.0 {
            TemporalControls::BackwardEuler(step) => Some(*step),
            TemporalControls::Ode(_) => None,
        }
    }

    /// Controls for integration of continuous equations.
    #[must_use]
    pub const fn as_ode(&self) -> Option<&CommonOdePolicy> {
        match &self.0 {
            TemporalControls::Ode(policy) => Some(policy),
            TemporalControls::BackwardEuler(_) => None,
        }
    }

    pub(super) fn require_backward_euler(&self) -> Result<DynQuantity, Diagnostic> {
        self.as_backward_euler().ok_or_else(|| {
            invalid("discrete backward-Euler history requires a BackwardEuler policy")
        })
    }
}

impl From<CommonOdePolicy> for CommonTemporalPolicy {
    fn from(policy: CommonOdePolicy) -> Self {
        Self(TemporalControls::Ode(policy))
    }
}
