//! Exact binary64 admission shared by tables and mathematical classifiers.
use super::ExactRational;

impl ExactRational {
    /// Represent every finite binary64 value exactly as a reduced dyadic rational.
    /// Signed zeros have the same canonical rational value; nonfinite values reject.
    #[must_use]
    pub fn from_binary64(value: f64) -> Option<Self> {
        num_rational::BigRational::from_float(value).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_data_is_exact_across_the_complete_finite_range() {
        assert_eq!(
            ExactRational::from_binary64(0.1).unwrap(),
            ExactRational::from_canonical_parts("3602879701896397", "36028797018963968").unwrap()
        );
        assert_eq!(
            ExactRational::from_binary64(-0.0).unwrap(),
            ExactRational::integer(0)
        );
        assert_eq!(
            ExactRational::from_binary64(1e-6).unwrap(),
            ExactRational::from_canonical_parts("4722366482869645", "4722366482869645213696")
                .unwrap()
        );
        // Every exponent, both signs, and significand boundaries. The expected bits
        // are the authored IEEE-754 input, independent of rational conversion.
        for exponent in 0..2047_u64 {
            for fraction in [0, 1, (1_u64 << 51), (1_u64 << 52) - 1] {
                for sign in [0, 1_u64 << 63] {
                    let bits = sign | (exponent << 52) | fraction;
                    let value = f64::from_bits(bits);
                    let actual = ExactRational::from_binary64(value)
                        .unwrap()
                        .to_f64()
                        .unwrap();
                    assert_eq!(actual.to_bits(), if value == 0.0 { 0 } else { bits });
                }
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(ExactRational::from_binary64(value), None);
        }
    }
}
