//! Exact binary64-to-rational conversion shared by tables and mathematical classifiers.
use super::ExactRational;

impl ExactRational {
    /// Represent a finite binary64 value exactly as a reduced dyadic rational.
    /// Returns `None` when the numerator or denominator does not fit.
    #[must_use]
    pub fn from_binary64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::integer(0));
        }
        let bits = value.to_bits();
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mut significand, mut exponent) = if encoded_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1_u64 << 52), encoded_exponent - 1023 - 52)
        };
        let shift = significand.trailing_zeros();
        significand >>= shift;
        exponent += shift as i32;
        let signed = if bits >> 63 == 0 {
            i128::from(significand)
        } else {
            -i128::from(significand)
        };
        let (numerator, denominator) = if exponent >= 0 {
            let scale = 1_i128
                .checked_shl(exponent as u32)
                .filter(|scale| *scale > 0)?;
            let numerator = signed.checked_mul(scale)?;
            (i64::try_from(numerator).ok()?, 1)
        } else {
            let denominator = 1_u64.checked_shl((-exponent) as u32)?;
            (signed as i64, denominator)
        };
        Self::from_canonical_parts(numerator, denominator).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_data_is_exact_and_bounded() {
        assert_eq!(
            ExactRational::from_binary64(0.5),
            Some(ExactRational::new(1, 2).unwrap())
        );
        assert_eq!(
            ExactRational::from_binary64(0.1).unwrap(),
            ExactRational::from_canonical_parts(3602879701896397, 36028797018963968).unwrap()
        );
        assert_eq!(
            ExactRational::from_binary64(-0.0).unwrap(),
            ExactRational::integer(0)
        );
        assert_eq!(
            ExactRational::from_binary64(-9223372036854775808.0).unwrap(),
            ExactRational::integer(i64::MIN)
        );
        assert_eq!(
            ExactRational::from_binary64(2_f64.powi(-63))
                .unwrap()
                .denominator(),
            1_u64 << 63
        );
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            2_f64.powi(-64),
            2_f64.powi(63),
            2_f64.powi(127),
        ] {
            assert_eq!(ExactRational::from_binary64(value), None);
        }
    }
}
