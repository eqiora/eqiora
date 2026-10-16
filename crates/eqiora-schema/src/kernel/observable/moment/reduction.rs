//! One bounded weighted-moment arithmetic owner; law/sample provenance stays with callers.

use super::*;
use eqiora_core::ValueLiteral;

impl ProbabilityMoment {
    /// Admit proportional weights with finite nonnegative entries and positive total mass.
    /// This checks normalization, not probability-law or sample provenance. It does
    /// not sum in floating point, so a finite ratio need not have a representable sum.
    /// # Errors
    /// Rejects empty/all-zero weights, negative values, NaN and infinity.
    pub fn validate_weights(weights: impl IntoIterator<Item = f64>) -> Result<(), Diagnostic> {
        let mut positive = false;
        for weight in weights {
            validate_weight(weight)?;
            positive |= weight > 0.0;
        }
        if !positive {
            return Err(invalid("weighted moment has zero normalization"));
        }
        Ok(())
    }

    /// Evaluate a finite weighted moment, normalizing the nonnegative weights explicitly.
    ///
    /// Each row contains this moment's ordered quantities. This arithmetic operation
    /// alone does not establish that rows exhaust a distribution, form an ensemble,
    /// are independent, or yield an unbiased estimator. Those claims require the
    /// caller's admitted law or sample-design context.
    ///
    /// Binary64 inputs are accumulated exactly within the shared rational work bound
    /// and rounded once on output. Overflow or exhausted arithmetic work rejects;
    /// no cancellation tolerance, clipping, or order-dependent floating sum is used.
    pub fn evaluate(
        self,
        rows: &[Vec<ValueLiteral>],
        weights: &[f64],
    ) -> Result<ValueLiteral, Diagnostic> {
        let Some(first) = rows.first() else {
            return Err(invalid(
                "a probability moment requires nonempty weighted support",
            ));
        };
        if rows.len() != weights.len() {
            return Err(invalid("each moment row requires its exact weight"));
        }
        let types = first
            .iter()
            .map(|value| value.value_type().clone())
            .collect::<Vec<_>>();
        let mut moments = super::WeightedMoments::new(self, &types)?;
        for (row, &weight) in rows.iter().zip(weights) {
            moments.push(row, weight)?;
        }
        moments.value(0)
    }
}

pub(super) fn validate_weight(weight: f64) -> Result<(), Diagnostic> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(invalid("moment weights must be finite and nonnegative"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::DimExponents;

    fn quantity(value: f64, exponents: [i32; 7]) -> ValueLiteral {
        ValueLiteral::from_real(
            ValueType::scalar(
                ScalarDomain::Real,
                DimExponents::from_integers(exponents).unwrap(),
            )
            .unwrap(),
            value,
        )
        .unwrap()
    }

    #[test]
    fn named_law_reference_moments_have_independent_values_and_paired_units() {
        let length = [0, 1, 0, 0, 0, 0, 0];
        let time = [0, 0, 1, 0, 0, 0, 0];
        let x = vec![vec![quantity(0.0, length)], vec![quantity(2.0, length)]];
        let weights = [0.25, 0.75];
        assert_eq!(
            ProbabilityMoment::Expectation
                .evaluate(&x, &weights)
                .unwrap(),
            quantity(1.5, length)
        );
        assert_eq!(
            ProbabilityMoment::Variance.evaluate(&x, &weights).unwrap(),
            quantity(0.75, [0, 2, 0, 0, 0, 0, 0])
        );
        // Y = 3 s - (s/m) X, so Cov(X,Y) = -(s/m) Var(X).
        let paired = vec![
            vec![quantity(0.0, length), quantity(3.0, time)],
            vec![quantity(2.0, length), quantity(1.0, time)],
        ];
        assert_eq!(
            ProbabilityMoment::Covariance
                .evaluate(&paired, &weights)
                .unwrap(),
            quantity(-0.75, [0, 1, 1, 0, 0, 0, 0])
        );
        assert_eq!(
            ProbabilityMoment::Expectation
                .evaluate(&x, &[1.0, 3.0])
                .unwrap(),
            quantity(1.5, length)
        );
    }

    #[test]
    fn normalization_and_cancellation_do_not_overflow_or_erase_a_small_mean() {
        let unit = [0; 7];
        let rows = vec![vec![quantity(f64::MAX, unit)]; 2];
        assert_eq!(
            ProbabilityMoment::Expectation
                .evaluate(&rows, &[f64::MAX; 2])
                .unwrap(),
            quantity(f64::MAX, unit)
        );
        assert_eq!(
            ProbabilityMoment::Variance
                .evaluate(&rows, &[1.0; 2])
                .unwrap(),
            quantity(0.0, unit)
        );
        let rows = vec![
            vec![quantity(1e16, unit)],
            vec![quantity(1.0, unit)],
            vec![quantity(-1e16, unit)],
        ];
        assert_eq!(
            ProbabilityMoment::Expectation
                .evaluate(&rows, &[1.0; 3])
                .unwrap(),
            quantity(1.0 / 3.0, unit)
        );
        for weights in [[0.0, 0.0, 0.0], [1.0, -1.0, 1.0], [1.0, f64::NAN, 1.0]] {
            assert!(
                ProbabilityMoment::Expectation
                    .evaluate(&rows, &weights)
                    .is_err()
            );
        }
        assert!(
            ProbabilityMoment::Expectation
                .evaluate(&rows, &[1.0])
                .is_err()
        );
        assert!(ProbabilityMoment::Expectation.evaluate(&[], &[]).is_err());
    }
}
