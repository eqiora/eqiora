//! Time method selection independent of the spatial representation.
use super::*;

/// Fixed-step backward-Euler controls requested from the common resolver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonBackwardEuler {
    step: DynQuantity,
}

impl CommonBackwardEuler {
    /// Construct one positive coherent-SI operator step.
    pub fn from_seconds(step_s: f64) -> Result<Self, Diagnostic> {
        if !step_s.is_finite() || step_s <= 0.0 || step_s.to_bits() == (-0.0_f64).to_bits() {
            return Err(invalid(
                "BackwardEuler step_s must be finite and strictly positive",
            ));
        }
        Ok(Self {
            step: DynQuantity::new(step_s, TIME),
        })
    }

    /// Exact physical duration of one operator construction.
    #[must_use]
    pub const fn step(self) -> DynQuantity {
        self.step
    }
}

/// Deterministic time method and its method-specific numerical controls.
#[derive(Debug, Clone, PartialEq)]
pub enum CommonTemporalPolicy {
    /// Fixed-step backward Euler, including exact discrete history.
    BackwardEuler(CommonBackwardEuler),
    /// Controlled integration of the continuous State/rate equations.
    Ode(CommonOdePolicy),
}

impl CommonTemporalPolicy {
    /// Fixed-step controls, only when backward Euler was requested.
    #[must_use]
    pub const fn as_backward_euler(&self) -> Option<CommonBackwardEuler> {
        match self {
            Self::BackwardEuler(policy) => Some(*policy),
            Self::Ode(_) => None,
        }
    }

    /// Controls for integration of continuous equations.
    #[must_use]
    pub const fn as_ode(&self) -> Option<&CommonOdePolicy> {
        match self {
            Self::Ode(policy) => Some(policy),
            Self::BackwardEuler(_) => None,
        }
    }

    pub(super) fn require_backward_euler(&self) -> Result<CommonBackwardEuler, Diagnostic> {
        self.as_backward_euler().ok_or_else(|| {
            invalid("discrete backward-Euler history requires a BackwardEuler policy")
        })
    }
}

impl From<CommonBackwardEuler> for CommonTemporalPolicy {
    fn from(policy: CommonBackwardEuler) -> Self {
        Self::BackwardEuler(policy)
    }
}

impl From<CommonOdePolicy> for CommonTemporalPolicy {
    fn from(policy: CommonOdePolicy) -> Self {
        Self::Ode(policy)
    }
}
