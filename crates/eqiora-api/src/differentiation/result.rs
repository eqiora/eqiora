//! One accepted primal/evidence envelope for each derivative role.

use super::*;

/// Accepted selected outputs, an optional derivative action, and producer evidence.
///
/// A primal result has the unit payload `()`. Forward and reverse results carry
/// a typed tangent and cotangent respectively. Only accepted evaluations can
/// construct this envelope; its derivative payload cannot change its evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableResult<T = ()> {
    pub(super) output: DifferentiableValues,
    pub(super) derivative: T,
    pub(super) evidence: DifferentiationEvidence,
}

impl<T> DifferentiableResult<T> {
    /// Complete selected outputs at the accepted point, in Program order.
    #[must_use]
    pub const fn output(&self) -> &DifferentiableValues {
        &self.output
    }

    /// Typed derivative action, or `()` for a primal occurrence.
    #[must_use]
    pub const fn derivative(&self) -> &T {
        &self.derivative
    }

    /// Evidence for this exact accepted occurrence and action.
    #[must_use]
    pub const fn evidence(&self) -> &DifferentiationEvidence {
        &self.evidence
    }

    /// Consume the occurrence without separating its values from their signatures.
    #[must_use]
    pub fn into_parts(self) -> (DifferentiableValues, T, DifferentiationEvidence) {
        (self.output, self.derivative, self.evidence)
    }
}
