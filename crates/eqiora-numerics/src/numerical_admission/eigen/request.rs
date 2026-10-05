use super::*;
use eqiora_core::DynQuantity;
use std::num::NonZeroUsize;

/// Dense Hermitian spectral selection and dimensionless acceptance tolerances.
/// These are numerical Study controls, not additional Model equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonEigenRequest {
    count: NonZeroUsize,
    target: Option<DynQuantity>,
    interval: Option<[DynQuantity; 2]>,
    residual_tolerance: f64,
    normalization_tolerance: f64,
}

impl CommonEigenRequest {
    /// Request a finite number of modes from the dense Hermitian algorithm.
    pub fn dense(
        count: NonZeroUsize,
        residual_tolerance: f64,
        normalization_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        if [residual_tolerance, normalization_tolerance]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0. || *v >= 1.)
        {
            return Err(invalid(
                "spectral tolerances must be finite and strictly between zero and one",
            ));
        }
        Ok(Self {
            count,
            target: None,
            interval: None,
            residual_tolerance,
            normalization_tolerance,
        })
    }
    /// Prefer eigenvalues closest to this physically typed target.
    pub fn with_target(mut self, target: DynQuantity) -> Result<Self, Diagnostic> {
        if !target.value().is_finite() {
            return Err(invalid("spectral target must be finite"));
        }
        self.target = Some(target);
        Ok(self)
    }
    /// Restrict selection to this closed, physically typed interval.
    pub fn within_interval(mut self, interval: [DynQuantity; 2]) -> Result<Self, Diagnostic> {
        if interval.iter().any(|v| !v.value().is_finite())
            || interval[0].dim() != interval[1].dim()
            || interval[0].value() > interval[1].value()
        {
            return Err(invalid(
                "spectral interval requires finite ordered bounds of one dimension",
            ));
        }
        self.interval = Some(interval);
        Ok(self)
    }
    /// Requested number of modes.
    pub const fn count(self) -> NonZeroUsize {
        self.count
    }
    /// Optional target in coherent SI units.
    pub const fn target(self) -> Option<DynQuantity> {
        self.target
    }
    /// Optional inclusive interval in coherent SI units.
    pub const fn interval(self) -> Option<[DynQuantity; 2]> {
        self.interval
    }
    /// Relative original-pencil residual tolerance.
    pub const fn residual_tolerance(self) -> f64 {
        self.residual_tolerance
    }
    /// Dimensionless mass normalization and orthogonality tolerance.
    pub const fn normalization_tolerance(self) -> f64 {
        self.normalization_tolerance
    }

    pub(super) fn validate(self, problem: &HermitianEigenproblem<'_>) -> Result<(), Diagnostic> {
        if self.count.get() > problem.dimension() {
            return Err(invalid(
                "requested mode count exceeds the finite space dimension",
            ));
        }
        if self
            .target
            .into_iter()
            .chain(self.interval.into_iter().flatten())
            .any(|value| value.dim() != problem.eigenvalue_type().dimension())
        {
            return Err(invalid(
                "spectral target or interval has the wrong eigenvalue dimension",
            ));
        }
        Ok(())
    }
    pub(super) fn identity_bytes(self) -> Result<Vec<u8>, Diagnostic> {
        let quantity = |v: DynQuantity| (v.value(), v.dim().exponents());
        serde_json::to_vec(&(
            "dense-hermitian",
            self.count.get(),
            self.target.map(quantity),
            self.interval.map(|v| v.map(quantity)),
            self.residual_tolerance,
            self.normalization_tolerance,
        ))
        .map_err(|error| invalid(format!("cannot identify spectral request: {error}")))
    }
}
