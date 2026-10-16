//! One bounded weighted-moment arithmetic owner; law/sample provenance stays with callers.

use super::*;
use crate::kernel::pure_operator::{ExactRational, PureOperatorError};
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
            if !weight.is_finite() || weight < 0.0 {
                return Err(invalid("moment weights must be finite and nonnegative"));
            }
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
        Self::validate_weights(weights.iter().copied())?;
        let types = first
            .iter()
            .map(ValueLiteral::value_type)
            .collect::<Vec<_>>();
        let output = self.output_type(&types)?;
        let mut total = ExactRational::integer(0);
        let mut sums = vec![ExactRational::integer(0); self.arity()];
        let mut product = ExactRational::integer(0);
        for (row, &weight) in rows.iter().zip(weights) {
            if row.len() != self.arity()
                || row
                    .iter()
                    .zip(&types)
                    .any(|(value, ty)| value.value_type() != *ty)
            {
                return Err(invalid(
                    "weighted moments require matching quantity types and finite nonnegative weights",
                ));
            }
            let weight = ExactRational::from_binary64(weight).expect("checked finite weight");
            total = total.checked_add(weight.clone()).map_err(arithmetic)?;
            let values = row
                .iter()
                .map(|value| {
                    ExactRational::from_binary64(
                        value.real_scalar_value().expect("checked scalar").value(),
                    )
                    .expect("finite ValueLiteral")
                })
                .collect::<Vec<_>>();
            for (sum, value) in sums.iter_mut().zip(&values) {
                *sum = sum
                    .checked_add(weight.checked_mul(value.clone()).map_err(arithmetic)?)
                    .map_err(arithmetic)?;
            }
            if self != Self::Expectation {
                let right = if self == Self::Variance {
                    &values[0]
                } else {
                    &values[1]
                };
                let pair = values[0].checked_mul(right.clone()).map_err(arithmetic)?;
                product = product
                    .checked_add(weight.checked_mul(pair).map_err(arithmetic)?)
                    .map_err(arithmetic)?;
            }
        }
        let mean = sums[0].checked_div(total.clone()).map_err(arithmetic)?;
        let result = if self == Self::Expectation {
            mean
        } else {
            let other = if self == Self::Variance {
                mean.clone()
            } else {
                sums[1].checked_div(total.clone()).map_err(arithmetic)?
            };
            let correction = mean
                .checked_mul(other)
                .map_err(arithmetic)?
                .checked_neg()
                .map_err(arithmetic)?;
            product
                .checked_div(total)
                .map_err(arithmetic)?
                .checked_add(correction)
                .map_err(arithmetic)?
        };
        ValueLiteral::from_real(output, result.to_f64().map_err(arithmetic)?)
            .map_err(|error| invalid(&error.to_string()))
    }
}

fn arithmetic(error: PureOperatorError) -> Diagnostic {
    invalid(&format!("exact weighted-moment arithmetic: {error}"))
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
