//! Finite mathematical limits bind one exact coordinate factor, not moving geometry.
use super::*;
use eqiora_core::{ScalarDomain, ValueFrame};

impl ObservableMeasure {
    /// Check typed explicit limits independently of quadrature and current values.
    /// The initial profile is one complete coordinate interval with lumped real limits.
    ///
    /// # Errors
    /// Rejects physical/weighted measures, partial products, captured coordinates,
    /// non-real scalar limits or units different from the exact bound coordinate.
    pub fn validate_limits<I: Clone + PartialEq>(
        self,
        input: &SpatialSupport<I>,
        support: &SpatialSupport<I>,
        limits: [&ExpressionType<I>; 2],
    ) -> Result<(), Diagnostic> {
        let SpatialSupport::Coordinates { domain, factors } = support else {
            return Err(invalid(
                "explicit integral limits require an exact coordinate interval, not moving geometry",
            ));
        };
        let [(factor, dimension, 1)] = factors.as_slice() else {
            return Err(invalid(
                "explicit integral limits require one scalar coordinate factor",
            ));
        };
        if self != Self::Volume || input != support || factor != domain {
            return Err(invalid(
                "explicit integral limits require the complete exact interval and its Cartesian measure",
            ));
        }
        for limit in limits {
            let ty = &limit.value_type;
            if limit.support.is_some() {
                return Err(invalid(
                    "integral limits must be lumped; the bound coordinate cannot be captured",
                ));
            }
            if ty.scalar_domain() != ScalarDomain::Real
                || !ty.shape().is_scalar()
                || ty.array_rank() != 0
                || ty.frame() != ValueFrame::Invariant
                || ty.dimension() != *dimension
            {
                return Err(invalid(
                    "integral limits must be real scalar values with the exact coordinate unit",
                ));
            }
        }
        Ok(())
    }
}
