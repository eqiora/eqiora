//! Exact coefficients with bounded arithmetic work, independent of machine word size.
mod sqrt;

use super::PureOperatorError;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

// Every binary64 datum needs at most 1075 bits. A difference of two such
// values needs at most 2099, and a ratio of differences at most 4198.
// This bounds input and intermediate allocation, not the floating-point range.
const MAX_BITS: u64 = 8192;
// 30103/100000 is an upper bound on log10(2); exact bit limits follow parsing.
const MAX_DECIMAL_DIGITS: usize = (MAX_BITS * 30103).div_ceil(100000) as usize;

/// Exact reduced rational used by operator definitions and proofs.
///
/// Coefficients admit every finite binary64 value, including subnormals. Each
/// numerator and denominator is limited to 8192 bits. Checked arithmetic rejects
/// larger operands/results; executable arithmetic retains its original node order.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "CanonicalRational", into = "CanonicalRational")]
pub struct ExactRational(pub(super) BigRational);

impl TryFrom<BigRational> for ExactRational {
    type Error = PureOperatorError;

    /// Admit an existing exact ratio through the same canonical work bound.
    fn try_from(value: BigRational) -> Result<Self, Self::Error> {
        Self::check_parts(&value)?;
        // BigRational::new_raw permits a zero or negative denominator and an
        // unreduced numerator. Use the canonical decoder's admission for it.
        Self::from_canonical_parts(&value.numer().to_string(), &value.denom().to_string())
    }
}

impl From<ExactRational> for BigRational {
    fn from(value: ExactRational) -> Self {
        value.0
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRational {
    numerator: String,
    denominator: String,
}

impl From<ExactRational> for CanonicalRational {
    fn from(value: ExactRational) -> Self {
        let (numerator, denominator) = value.canonical_parts();
        Self {
            numerator,
            denominator,
        }
    }
}

impl TryFrom<CanonicalRational> for ExactRational {
    type Error = PureOperatorError;

    fn try_from(value: CanonicalRational) -> Result<Self, Self::Error> {
        Self::from_canonical_parts(&value.numerator, &value.denominator)
    }
}

impl ExactRational {
    /// Construct and reduce an exact ratio of signed integers.
    ///
    /// # Errors
    /// Rejects a zero denominator.
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, PureOperatorError> {
        if denominator == 0 {
            return Err(PureOperatorError::InvalidRational);
        }
        Self::checked(BigRational::new(numerator.into(), denominator.into()))
    }

    /// Reconstruct reduced canonical decimal parts without leading zeros or `+`.
    ///
    /// # Errors
    /// Rejects noncanonical spelling, a nonpositive denominator, unreduced ratios,
    /// and operands outside the arithmetic work budget before normalization.
    pub fn from_canonical_parts(
        numerator: &str,
        denominator: &str,
    ) -> Result<Self, PureOperatorError> {
        for (text, signed) in [(numerator, true), (denominator, false)] {
            let digits = if signed {
                text.strip_prefix('-').unwrap_or(text)
            } else {
                text
            };
            if digits.len() > MAX_DECIMAL_DIGITS {
                return Err(PureOperatorError::RationalOverflow);
            }
            if digits.is_empty()
                || !digits.bytes().all(|b| b.is_ascii_digit())
                || (digits.starts_with('0') && digits.len() != 1)
                || text == "-0"
            {
                return Err(PureOperatorError::InvalidRational);
            }
        }
        let n = numerator
            .parse()
            .map_err(|_| PureOperatorError::InvalidRational)?;
        let d = denominator
            .parse()
            .map_err(|_| PureOperatorError::InvalidRational)?;
        let raw = BigRational::new_raw(n, d);
        if raw.denom().is_zero() {
            return Err(PureOperatorError::InvalidRational);
        }
        Self::check_parts(&raw)?;
        let reduced = BigRational::new(raw.numer().clone(), raw.denom().clone());
        if reduced.numer() != raw.numer() || reduced.denom() != raw.denom() {
            return Err(PureOperatorError::InvalidRational);
        }
        Ok(Self(reduced))
    }

    /// Canonical base-ten numerator and positive denominator.
    #[must_use]
    pub fn canonical_parts(&self) -> (String, String) {
        (self.0.numer().to_string(), self.0.denom().to_string())
    }

    /// Encode the same length-prefixed signed numerator and positive denominator
    /// for every exact-calculus identity owner.
    pub fn append_canonical_bytes(&self, bytes: &mut Vec<u8>) {
        for part in [self.0.numer(), self.0.denom()] {
            let encoded = part.to_signed_bytes_be();
            bytes.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&encoded);
        }
    }

    /// Exact integer.
    #[must_use]
    pub fn integer(value: i64) -> Self {
        Self(BigRational::from_integer(value.into()))
    }

    /// Exact nonnegative decimal coefficient times a signed power of ten.
    ///
    /// # Errors
    /// Rejects malformed coefficients or input/result sizes outside the work budget.
    pub fn from_decimal_parts(
        coefficient: &str,
        exponent: i64,
        negative: bool,
    ) -> Result<Self, PureOperatorError> {
        if coefficient.len() > MAX_DECIMAL_DIGITS {
            return Err(PureOperatorError::RationalOverflow);
        }
        if coefficient.is_empty() || !coefficient.bytes().all(|b| b.is_ascii_digit()) {
            return Err(PureOperatorError::InvalidRational);
        }
        let mut value = BigRational::from_integer(
            coefficient
                .parse()
                .map_err(|_| PureOperatorError::InvalidRational)?,
        );
        Self::check_parts(&value)?;
        if value.is_zero() {
            return Ok(Self::integer(0));
        }
        if exponent.unsigned_abs() > MAX_DECIMAL_DIGITS as u64 {
            return Err(PureOperatorError::RationalOverflow);
        }
        value *= BigRational::from_integer(10.into()).pow(exponent as i32);
        Self::checked(if negative { -value } else { value })
    }

    /// Checked conversion of both parts to signed 32-bit integers.
    #[must_use]
    pub fn to_i32_ratio(&self) -> Option<(i32, i32)> {
        Some((self.0.numer().to_i32()?, self.0.denom().to_i32()?))
    }

    /// Correctly rounded binary64 coefficient for executable lowering.
    ///
    /// # Errors
    /// Rejects overflow and rounding a nonzero exact coefficient to zero.
    pub fn to_f64(&self) -> Result<f64, PureOperatorError> {
        let value = self.0.to_f64().ok_or(PureOperatorError::CoefficientRange)?;
        if !value.is_finite() || (value == 0.0 && !self.is_zero()) {
            return Err(PureOperatorError::CoefficientRange);
        }
        Ok(value)
    }

    /// Whether this value is exactly zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Exact negation within the coefficient work budget.
    pub fn checked_neg(&self) -> Result<Self, PureOperatorError> {
        Self::checked(-&self.0)
    }

    /// Exact sum within the coefficient work budget.
    pub fn checked_add(&self, other: Self) -> Result<Self, PureOperatorError> {
        Self::checked(&self.0 + other.0)
    }

    /// Exact product within the coefficient work budget.
    pub fn checked_mul(&self, other: Self) -> Result<Self, PureOperatorError> {
        Self::checked(&self.0 * other.0)
    }

    /// Exact quotient within the coefficient work budget; rejects division by zero.
    pub fn checked_div(&self, other: Self) -> Result<Self, PureOperatorError> {
        if other.is_zero() {
            return Err(PureOperatorError::InvalidRational);
        }
        Self::checked(&self.0 / other.0)
    }

    fn check_parts(value: &BigRational) -> Result<(), PureOperatorError> {
        if value.numer().bits() > MAX_BITS || value.denom().bits() > MAX_BITS {
            return Err(PureOperatorError::RationalOverflow);
        }
        Ok(())
    }

    pub(super) fn checked(value: BigRational) -> Result<Self, PureOperatorError> {
        Self::check_parts(&value)?;
        Ok(Self(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_ratios_reuse_canonical_and_bounded_admission() {
        let exact = BigRational::new(1.into(), 3.into());
        assert_eq!(
            BigRational::from(ExactRational::try_from(exact.clone()).unwrap()),
            exact
        );
        for raw in [
            BigRational::new_raw(2.into(), 4.into()),
            BigRational::new_raw(1.into(), (-3).into()),
            BigRational::new_raw(1.into(), 0.into()),
        ] {
            assert!(ExactRational::try_from(raw).is_err());
        }
        let oversized = BigRational::from_integer(2.into()).pow(8192);
        assert!(ExactRational::try_from(oversized).is_err());
    }

    #[test]
    fn exact_arithmetic_crosses_machine_words_without_rounding() {
        let maximum = ExactRational::integer(i64::MAX);
        assert_eq!(
            maximum
                .checked_mul(maximum.clone())
                .unwrap()
                .canonical_parts(),
            ("85070591730234615847396907784232501249".into(), "1".into()),
        );
        let large = ExactRational::from_binary64(f64::MAX).unwrap();
        let small = ExactRational::from_binary64(f64::from_bits(1)).unwrap();
        let difference = large.checked_add(small.checked_neg().unwrap()).unwrap();
        assert_eq!(
            difference.checked_div(difference.clone()).unwrap(),
            ExactRational::integer(1)
        );
        assert_eq!(difference.checked_add(small).unwrap(), large);
        assert_eq!(
            ExactRational::integer(1).checked_div(ExactRational::integer(0)),
            Err(PureOperatorError::InvalidRational)
        );
        assert!(ExactRational::new(1, 3).unwrap() < ExactRational::new(1, 2).unwrap());
    }

    #[test]
    fn canonical_serialization_admits_one_spelling_and_bounds_before_reduction() {
        let value = ExactRational::from_binary64(1e-6).unwrap();
        let bytes = serde_json::to_string(&value).unwrap();
        assert_eq!(
            bytes,
            r#"{"numerator":"4722366482869645","denominator":"4722366482869645213696"}"#
        );
        assert_eq!(
            serde_json::from_str::<ExactRational>(&bytes).unwrap(),
            value
        );
        for input in [
            r#"{"numerator":1,"denominator":2}"#,
            r#"{"numerator":"1","denominator":"2","extra":0}"#,
            r#"{"numerator":"2","denominator":"4"}"#,
            r#"{"numerator":"01","denominator":"2"}"#,
            r#"{"numerator":"+1","denominator":"2"}"#,
            r#"{"numerator":"-0","denominator":"1"}"#,
            r#"{"numerator":"0","denominator":"2"}"#,
            r#"{"numerator":"1","denominator":"0"}"#,
            r#"{"numerator":"1","denominator":"-2"}"#,
        ] {
            assert!(
                serde_json::from_str::<ExactRational>(input).is_err(),
                "{input}"
            );
        }
        let overlong = "1".repeat(MAX_DECIMAL_DIGITS + 1);
        assert_eq!(
            ExactRational::from_canonical_parts(&overlong, &overlong),
            Err(PureOperatorError::RationalOverflow)
        );
        let too_wide = BigRational::from_integer(2.into())
            .pow(MAX_BITS as i32)
            .numer()
            .to_string();
        assert_eq!(
            ExactRational::from_canonical_parts(&too_wide, &too_wide),
            Err(PureOperatorError::RationalOverflow)
        );
        let limit =
            ExactRational::checked(BigRational::from_integer(2.into()).pow(MAX_BITS as i32 - 1))
                .unwrap();
        assert_eq!(
            limit.checked_mul(ExactRational::integer(2)),
            Err(PureOperatorError::RationalOverflow)
        );
    }

    #[test]
    fn executable_conversion_rejects_nonzero_underflow_and_overflow() {
        for exponent in [-1000, 1000] {
            let value = ExactRational::from_decimal_parts("1", exponent, false).unwrap();
            assert_eq!(value.to_f64(), Err(PureOperatorError::CoefficientRange));
        }
        assert_eq!(ExactRational::integer(0).to_f64(), Ok(0.0));
    }

    #[test]
    fn decimal_and_binary_measurements_keep_their_distinct_exact_meaning() {
        let decimal = ExactRational::from_decimal_parts("1", -6, false).unwrap();
        let binary = ExactRational::from_binary64(1e-6).unwrap();
        assert_ne!(decimal, binary);
        assert_eq!(decimal.canonical_parts(), ("1".into(), "1000000".into()));
        assert_eq!(
            decimal.to_f64().unwrap().to_bits(),
            binary.to_f64().unwrap().to_bits()
        );
        assert_eq!(
            ExactRational::from_decimal_parts("125", -2, true).unwrap(),
            ExactRational::new(-5, 4).unwrap()
        );
        assert_eq!(
            ExactRational::from_decimal_parts("1", i64::MAX, false),
            Err(PureOperatorError::RationalOverflow)
        );
        assert_eq!(
            ExactRational::from_decimal_parts("0", i64::MAX, false).unwrap(),
            ExactRational::integer(0)
        );
    }
}
