//! Correct rounding of a nonnegative exact square root without a squared f64.
use super::{BigRational, ExactRational, PureOperatorError};

impl ExactRational {
    pub(crate) fn sqrt_to_f64(&self) -> Result<f64, PureOperatorError> {
        if self < &Self::integer(0) {
            return Err(PureOperatorError::InvalidRational);
        }
        if self.is_zero() {
            return Ok(0.0);
        }
        // Positive finite binary64 encodings are ordered. Bracket the root using
        // exact squares, then compare with the exact midpoint for ties-to-even.
        // The infinity sentinel's arithmetic endpoint is 2^1024, so rounding
        // overflow follows the same midpoint rule as every finite binade.
        let endpoint = |bits| {
            if bits == f64::INFINITY.to_bits() {
                BigRational::from_integer(2.into()).pow(1024)
            } else {
                Self::from_binary64(f64::from_bits(bits))
                    .expect("finite endpoint")
                    .0
            }
        };
        let mut lower = 0;
        let mut upper = f64::INFINITY.to_bits();
        while upper - lower > 1 {
            let middle = lower + (upper - lower) / 2;
            let value = endpoint(middle);
            if &value * &value <= self.0 {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let midpoint = (endpoint(lower) + endpoint(upper)) / BigRational::from_integer(2.into());
        let threshold = &midpoint * &midpoint;
        let bits = if self.0 < threshold || (self.0 == threshold && lower % 2 == 0) {
            lower
        } else {
            upper
        };
        let value = f64::from_bits(bits);
        if !value.is_finite() || value == 0.0 {
            return Err(PureOperatorError::CoefficientRange);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_root_rounding_covers_range_and_even_midpoints() {
        for x in [
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1.0,
            2.0,
            f64::MAX,
        ] {
            let x_exact = ExactRational::from_binary64(x).unwrap();
            assert_eq!(
                x_exact.checked_mul(x_exact.clone()).unwrap().sqrt_to_f64(),
                Ok(x)
            );
        }
        assert_eq!(ExactRational::integer(2).sqrt_to_f64(), Ok(2.0f64.sqrt()));
        for lower in [0.0, f64::from_bits(1), 1.0, 1.0f64.next_up()] {
            let l = ExactRational::from_binary64(lower).unwrap();
            let u = ExactRational::from_binary64(lower.next_up()).unwrap();
            let mid = l
                .checked_add(u)
                .unwrap()
                .checked_div(ExactRational::integer(2))
                .unwrap();
            let rounded = if lower.to_bits() % 2 == 0 {
                lower
            } else {
                lower.next_up()
            };
            let result = mid.checked_mul(mid.clone()).unwrap().sqrt_to_f64();
            if rounded == 0.0 {
                assert_eq!(result, Err(PureOperatorError::CoefficientRange));
            } else {
                assert_eq!(result, Ok(rounded));
            }
        }
        assert_eq!(
            ExactRational::integer(-1).sqrt_to_f64(),
            Err(PureOperatorError::InvalidRational)
        );
        let huge = ExactRational::from_decimal_parts("1", 1000, false).unwrap();
        assert_eq!(huge.sqrt_to_f64(), Err(PureOperatorError::CoefficientRange));
    }
}
