//! Mathematical population moments, distinct from sample-estimator conventions.

use super::invalid;
use eqiora_core::{Diagnostic, ScalarDomain, ValueFrame, ValueType};

mod reduction;

/// A moment under an exact declared probability law, not a finite-sample estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityMoment {
    /// Integral of a random quantity against the normalized law.
    Expectation,
    /// Centered second moment of one real scalar random quantity.
    Variance,
    /// Centered paired moment of two real scalar random quantities under the same law.
    Covariance,
}

impl ProbabilityMoment {
    /// Number of ordered random-variable expressions consumed by this moment.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Expectation | Self::Variance => 1,
            Self::Covariance => 2,
        }
    }

    /// Derive the moment's units from its integrands, independently of evaluated values.
    ///
    /// The initial moment profile admits real invariant scalar quantities. Complex
    /// covariance conventions, tensor contractions and sampled estimators are separate
    /// meanings and cannot be inferred from component storage.
    pub fn output_type(self, values: &[&ValueType]) -> Result<ValueType, Diagnostic> {
        if values.len() != self.arity()
            || values.iter().any(|value| {
                value.scalar_domain() != ScalarDomain::Real
                    || !value.shape().is_scalar()
                    || value.array_rank() != 0
                    || value.frame() != ValueFrame::Invariant
                    || value.finite_bases().next().is_some()
            })
        {
            return Err(invalid(
                "probability moments require their ordered real scalar quantities",
            ));
        }
        let left = values[0].dimension();
        let dimension = match self {
            Self::Expectation => Some(left),
            Self::Variance => left.mul(left),
            Self::Covariance => left.mul(values[1].dimension()),
        }
        .ok_or_else(|| invalid("probability moment dimension exceeds its exact representation"))?;
        ValueType::scalar(ScalarDomain::Real, dimension)
            .map_err(|_| invalid("probability moment requires a real scalar result"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::DimExponents;

    #[test]
    fn paired_moments_keep_both_quantity_units() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
        let x = ValueType::scalar(ScalarDomain::Real, length).unwrap();
        let y = ValueType::scalar(ScalarDomain::Real, time).unwrap();
        assert_eq!(
            ProbabilityMoment::Expectation.output_type(&[&x]).unwrap(),
            x
        );
        assert_eq!(
            ProbabilityMoment::Variance
                .output_type(&[&x])
                .unwrap()
                .dimension(),
            DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap()
        );
        assert_eq!(
            ProbabilityMoment::Covariance
                .output_type(&[&x, &y])
                .unwrap()
                .dimension(),
            DimExponents::from_integers([0, 1, 1, 0, 0, 0, 0]).unwrap()
        );
        assert!(ProbabilityMoment::Variance.output_type(&[&x, &y]).is_err());
        assert!(ProbabilityMoment::Covariance.output_type(&[&x]).is_err());
        assert!(
            ProbabilityMoment::Expectation
                .output_type(&[&ValueType::boolean()])
                .is_err()
        );
        let complex = ValueType::scalar(ScalarDomain::Complex, length).unwrap();
        assert!(
            ProbabilityMoment::Variance
                .output_type(&[&complex])
                .is_err()
        );
    }
}
