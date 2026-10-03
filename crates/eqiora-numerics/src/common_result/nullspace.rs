//! Original equations and explicit reference evidence, distinct from solver convergence.
use super::*;
impl CommonResult {
    fn scalar_nullspace_evidence(&self) -> Option<&crate::nullspace::NullspaceEvidence> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::Scalar(evidence) => evidence.as_ref(),
                _ => None,
            },
            _ => None,
        }
    }
    /// Residual of the original scalar equations, excluding the gauge multiplier.
    #[must_use]
    pub fn scalar_original_residual_norm(&self) -> Option<f64> {
        self.scalar_nullspace_evidence()
            .map(|e| e.original_residual_norm)
    }
    /// Load balance against the admitted max-normalized constant null vector.
    #[must_use]
    pub fn scalar_compatibility_residual(&self) -> Option<f64> {
        self.scalar_nullspace_evidence()
            .map(|e| e.compatibility_residual)
    }
    /// Residual of the explicit spatial reference.
    #[must_use]
    pub fn scalar_gauge_residual(&self) -> Option<f64> {
        self.scalar_nullspace_evidence().map(|e| e.gauge_residual)
    }
    /// Numerical multiplier, kept separate from the physical scalar Field.
    #[must_use]
    pub fn scalar_gauge_multiplier(&self) -> Option<f64> {
        self.scalar_nullspace_evidence().map(|e| e.multiplier)
    }
}
