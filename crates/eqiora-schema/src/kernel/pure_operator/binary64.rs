//! Exact binary64 projection shared by bounded mathematical classifiers.
use super::ExactRational;

impl ExactRational {
    /// Preserve a finite binary64 value exactly as a reduced dyadic rational.
    /// Returns `None` when its numerator or denominator exceeds this representation.
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
