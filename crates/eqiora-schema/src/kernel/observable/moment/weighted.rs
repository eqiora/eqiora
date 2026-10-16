//! Streaming exact weighted arithmetic shared by laws and finite collections.
use super::*;
use crate::kernel::pure_operator::{ExactRational, PureOperatorError};
use eqiora_core::ValueLiteral;
#[cfg(test)]
mod tests;

/// Bounded exact sufficient statistics for one ordered scalar moment.
///
/// Storage is independent of occurrence count. Every binary64 input is exact;
/// only requested results and first derivatives are rounded to binary64.
/// The shared rational bound applies to every intermediate operation. No sample
/// identity, independence, population or unbiasedness claim follows from this
/// arithmetic owner; the caller must bind those meanings separately.
#[derive(Debug, Clone)]
pub struct WeightedMoments {
    moment: ProbabilityMoment,
    types: Vec<ValueType>,
    output: ValueType,
    count: i64,
    first_weight: Option<f64>,
    equal_weights: bool,
    total: ExactRational,
    sums: Vec<ExactRational>,
    product: ExactRational,
}

impl WeightedMoments {
    /// Establish the ordered integrand types before consuming occurrences.
    /// # Errors
    /// Rejects types or arity outside the moment's admitted scalar profile.
    pub fn new(moment: ProbabilityMoment, types: &[ValueType]) -> Result<Self, Diagnostic> {
        let output = moment.output_type(&types.iter().collect::<Vec<_>>())?;
        Ok(Self {
            moment,
            types: types.to_vec(),
            output,
            count: 0,
            first_weight: None,
            equal_weights: true,
            total: ExactRational::integer(0),
            sums: vec![ExactRational::integer(0); moment.arity()],
            product: ExactRational::integer(0),
        })
    }

    /// Consume one occurrence, including duplicate values and zero weights.
    /// # Errors
    /// Rejects wrong types, nonfinite/negative weights, count overflow or an
    /// exhausted arithmetic bound. Failure leaves the accumulator unchanged.
    pub fn push(&mut self, row: &[ValueLiteral], weight: f64) -> Result<(), Diagnostic> {
        let values = self.row(row, weight)?;
        let exact_weight = exact(weight);
        let count = self
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("moment occurrence count overflow"))?;
        let total = self
            .total
            .checked_add(exact_weight.clone())
            .map_err(arithmetic)?;
        let sums = self
            .sums
            .iter()
            .zip(&values)
            .map(|(sum, value)| {
                sum.checked_add(
                    exact_weight
                        .checked_mul(value.clone())
                        .map_err(arithmetic)?,
                )
                .map_err(arithmetic)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let right = if self.moment == ProbabilityMoment::Covariance {
            &values[1]
        } else {
            &values[0]
        };
        let product = self
            .product
            .checked_add(
                exact_weight
                    .checked_mul(values[0].checked_mul(right.clone()).map_err(arithmetic)?)
                    .map_err(arithmetic)?,
            )
            .map_err(arithmetic)?;
        self.count = count;
        self.total = total;
        self.sums = sums;
        self.product = product;
        self.equal_weights &= self.first_weight.is_none_or(|first| first == weight);
        self.first_weight.get_or_insert(weight);
        Ok(())
    }

    /// Exact normalized moment, with an explicit finite-count correction.
    ///
    /// Zero correction uses the total weight. A positive correction requires
    /// equal positive weights and divides centered moments by n-correction,
    /// instead of n. This defines arithmetic only, never an IID claim.
    /// # Errors
    /// Rejects empty/zero mass, invalid correction, or unrepresentable output.
    pub fn value(&self, correction: usize) -> Result<ValueLiteral, Diagnostic> {
        let factor = self.correction(correction)?;
        let value = if self.moment == ProbabilityMoment::Expectation {
            self.mean(0)?
        } else {
            self.centered()?.checked_mul(factor).map_err(arithmetic)?
        };
        self.literal(&self.output, value)
    }

    /// Unnormalized weighted sum of a single quantity, with its original units.
    /// # Errors
    /// Rejects empty/zero mass, paired/centered moments or unrepresentable output.
    pub fn sum(&self) -> Result<ValueLiteral, Diagnostic> {
        self.correction(0)?;
        if self.moment != ProbabilityMoment::Expectation {
            return Err(invalid("a weighted sum requires one first-moment quantity"));
        }
        self.literal(&self.output, self.sums[0].clone())
    }

    /// Estimated standard error of an equal-weight mean, in the original units.
    ///
    /// Computes sqrt(sum((x-mean)^2)/(n*(n-1))) with an exact radicand and
    /// one correctly rounded square root. The caller must establish IID sampling
    /// and assume a finite population second moment; this is arithmetic only.
    /// # Errors
    /// Rejects other moments, unequal/nonpositive weights, n < 2 and range errors.
    pub fn mean_standard_error(&self) -> Result<ValueLiteral, Diagnostic> {
        if self.moment != ProbabilityMoment::Expectation
            || self.count < 2
            || !self.equal_weights
            || self.first_weight.is_none_or(|w| w <= 0.0)
        {
            return Err(invalid(
                "mean standard error requires at least two equally positive weighted occurrences",
            ));
        }
        let squared = self
            .centered()?
            .checked_div(ExactRational::integer(self.count - 1))
            .map_err(arithmetic)?;
        ValueLiteral::from_real(
            self.output.clone(),
            squared.sqrt_to_f64().map_err(arithmetic)?,
        )
        .map_err(|error| invalid(&error.to_string()))
    }

    /// Derivative of this normalized finite statistic with respect to one row.
    ///
    /// Weights are fixed. Results are coherent-SI coordinate coefficients in
    /// ordered integrand order; coefficient j has statistic-unit/input-unit-j.
    /// The caller owns membership and must supply an actual accepted occurrence.
    /// # Errors
    /// Rejects invalid rows, weights, normalization or unrepresentable coefficients.
    pub fn derivative(
        &self,
        row: &[ValueLiteral],
        weight: f64,
        correction: usize,
    ) -> Result<Vec<f64>, Diagnostic> {
        let factor = self.correction(correction)?;
        let values = self.row(row, weight)?;
        let normalized = exact(weight)
            .checked_div(self.total.clone())
            .map_err(arithmetic)?
            .checked_mul(factor)
            .map_err(arithmetic)?;
        let differences = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .checked_add(self.mean(index)?.checked_neg().map_err(arithmetic)?)
                    .map_err(arithmetic)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let coefficients = match self.moment {
            ProbabilityMoment::Expectation => vec![normalized],
            ProbabilityMoment::Variance => vec![
                normalized
                    .checked_mul(ExactRational::integer(2))
                    .map_err(arithmetic)?
                    .checked_mul(differences[0].clone())
                    .map_err(arithmetic)?,
            ],
            ProbabilityMoment::Covariance => vec![
                normalized
                    .checked_mul(differences[1].clone())
                    .map_err(arithmetic)?,
                normalized
                    .checked_mul(differences[0].clone())
                    .map_err(arithmetic)?,
            ],
        };
        coefficients
            .iter()
            .map(|value| value.to_f64().map_err(arithmetic))
            .collect()
    }

    fn row(&self, row: &[ValueLiteral], weight: f64) -> Result<Vec<ExactRational>, Diagnostic> {
        super::reduction::validate_weight(weight)?;
        if row.len() != self.types.len()
            || row
                .iter()
                .zip(&self.types)
                .any(|(value, ty)| value.value_type() != ty)
        {
            return Err(invalid("weighted moments require matching quantity types"));
        }
        Ok(row
            .iter()
            .map(|value| exact(value.real_scalar_value().expect("checked scalar").value()))
            .collect())
    }

    fn correction(&self, correction: usize) -> Result<ExactRational, Diagnostic> {
        if self.total <= ExactRational::integer(0) {
            return Err(invalid("weighted moment has zero normalization"));
        }
        if correction == 0 {
            return Ok(ExactRational::integer(1));
        }
        let correction = i64::try_from(correction)
            .map_err(|_| invalid("moment correction exceeds occurrence count"))?;
        if self.moment == ProbabilityMoment::Expectation
            || !self.equal_weights
            || self.first_weight.is_none_or(|weight| weight <= 0.0)
            || correction >= self.count
        {
            return Err(invalid(
                "centered moment correction requires equal positive weights and count greater than correction",
            ));
        }
        ExactRational::new(self.count, self.count - correction).map_err(arithmetic)
    }

    fn mean(&self, index: usize) -> Result<ExactRational, Diagnostic> {
        self.sums[index]
            .checked_div(self.total.clone())
            .map_err(arithmetic)
    }

    fn centered(&self) -> Result<ExactRational, Diagnostic> {
        let right = if self.moment == ProbabilityMoment::Covariance {
            1
        } else {
            0
        };
        self.product
            .checked_div(self.total.clone())
            .map_err(arithmetic)?
            .checked_add(
                self.mean(0)?
                    .checked_mul(self.mean(right)?)
                    .map_err(arithmetic)?
                    .checked_neg()
                    .map_err(arithmetic)?,
            )
            .map_err(arithmetic)
    }

    fn literal(&self, ty: &ValueType, value: ExactRational) -> Result<ValueLiteral, Diagnostic> {
        ValueLiteral::from_real(ty.clone(), value.to_f64().map_err(arithmetic)?)
            .map_err(|error| invalid(&error.to_string()))
    }
}

fn exact(value: f64) -> ExactRational {
    ExactRational::from_binary64(value).expect("finite typed component or checked weight")
}

fn arithmetic(error: PureOperatorError) -> Diagnostic {
    invalid(&format!("exact weighted-moment arithmetic: {error}"))
}
