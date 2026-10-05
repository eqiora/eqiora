use core::fmt;

const LIMIT: i128 = i32::MAX as i128;

/// Canonical rational SI exponents, in kg, m, s, A, K, mol, cd order.
/// Numerator magnitudes and positive denominators are at most `i32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimExponents {
    values: [(i32, i32); 7],
}

impl DimExponents {
    /// The dimensionless exponent vector.
    pub const DIMENSIONLESS: Self = Self {
        values: [(0, 1); 7],
    };

    /// Normalize seven rational exponents; reject zero denominators and oversized inputs.
    pub const fn from_rationals(values: [(i32, i32); 7]) -> Option<Self> {
        let mut result = Self::DIMENSIONLESS;
        let mut index = 0;
        while index < 7 {
            let (numerator, denominator) = values[index];
            if numerator == i32::MIN || denominator == i32::MIN {
                return None;
            }
            result.values[index] = match reduced(numerator as i128, denominator as i128) {
                Some(value) => value,
                None => return None,
            };
            index += 1;
        }
        Some(result)
    }

    /// Construct integral exponents with the same finite bounds as rational input.
    pub const fn from_integers(values: [i32; 7]) -> Option<Self> {
        let mut fractions = [(0, 1); 7];
        let mut index = 0;
        while index < 7 {
            fractions[index] = (values[index], 1);
            index += 1;
        }
        Self::from_rationals(fractions)
    }

    /// Return reduced numerator/positive-denominator pairs in SI base-dimension order.
    pub const fn exponents(self) -> [(i32, i32); 7] {
        self.values
    }

    /// Add exponents for a product; reject an out-of-range reduced result.
    pub const fn mul(self, other: Self) -> Option<Self> {
        self.combine(other, false)
    }

    /// Subtract exponents for a quotient; reject an out-of-range reduced result.
    pub const fn div(self, other: Self) -> Option<Self> {
        self.combine(other, true)
    }

    /// Apply an exact rational dimension power, independently of value evaluation.
    pub const fn pow(self, numerator: i32, denominator: i32) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        // The power is an operation parameter, not a persisted dimension.
        // Its unreduced magnitude need not fit the result's canonical bounds.
        // i128 safely holds every product of two admitted i32 components.
        let power_n = numerator as i128;
        let power_d = denominator as i128;
        let mut result = Self::DIMENSIONLESS;
        let mut index = 0;
        while index < 7 {
            let (n, d) = self.values[index];
            let cancel_left = gcd(n as i128, power_d);
            let cancel_right = gcd(power_n, d as i128);
            let numerator = match (n as i128 / cancel_left).checked_mul(power_n / cancel_right) {
                Some(value) => value,
                None => return None,
            };
            let denominator = match (d as i128 / cancel_right).checked_mul(power_d / cancel_left) {
                Some(value) => value,
                None => return None,
            };
            result.values[index] = match reduced(numerator, denominator) {
                Some(value) => value,
                None => return None,
            };
            index += 1;
        }
        Some(result)
    }

    const fn combine(self, other: Self, subtract: bool) -> Option<Self> {
        let mut result = Self::DIMENSIONLESS;
        let mut index = 0;
        while index < 7 {
            let (a, b) = self.values[index];
            let (c, d) = other.values[index];
            let common = gcd(b as i128, d as i128);
            let left = match (a as i128).checked_mul(d as i128 / common) {
                Some(value) => value,
                None => return None,
            };
            let right = match (c as i128).checked_mul(b as i128 / common) {
                Some(value) => value,
                None => return None,
            };
            let numerator = match if subtract {
                left.checked_sub(right)
            } else {
                left.checked_add(right)
            } {
                Some(value) => value,
                None => return None,
            };
            let denominator = match (b as i128 / common).checked_mul(d as i128) {
                Some(value) => value,
                None => return None,
            };
            result.values[index] = match reduced(numerator, denominator) {
                Some(value) => value,
                None => return None,
            };
            index += 1;
        }
        Some(result)
    }
}

impl Default for DimExponents {
    fn default() -> Self {
        Self::DIMENSIONLESS
    }
}

// Operands have at most 31 magnitude bits. Cross-cancelled products and sums
// have at most 63 magnitude bits, so all intermediate arithmetic fits i128.
const fn reduced(mut numerator: i128, mut denominator: i128) -> Option<(i32, i32)> {
    if denominator == 0 {
        return None;
    }
    if denominator < 0 {
        numerator = -numerator;
        denominator = -denominator;
    }
    let divisor = gcd(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
    if numerator < -LIMIT || numerator > LIMIT || denominator > LIMIT {
        return None;
    }
    Some((numerator as i32, denominator as i32))
}

const fn gcd(mut left: i128, mut right: i128) -> i128 {
    if left < 0 {
        left = -left;
    }
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

impl fmt::Display for DimExponents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote = false;
        for (symbol, (numerator, denominator)) in ["M", "L", "T", "I", "Θ", "N", "J"]
            .into_iter()
            .zip(self.values)
        {
            if numerator == 0 {
                continue;
            }
            if wrote {
                write!(f, "·")?;
            }
            write!(f, "{symbol}")?;
            if denominator != 1 {
                write!(f, "^({numerator}/{denominator})")?;
            } else if numerator != 1 {
                write!(f, "^{numerator}")?;
            }
            wrote = true;
        }
        if !wrote {
            write!(f, "1")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DimExponents;
    use std::collections::HashSet;

    fn length(n: i32, d: i32) -> DimExponents {
        DimExponents::from_rationals([(0, 1), (n, d), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])
            .unwrap()
    }

    #[test]
    fn fractions_have_one_equality_hash_and_zero() {
        assert_eq!(
            HashSet::from([length(1, 2), length(2, 4), length(-1, -2)]).len(),
            1
        );
        assert_eq!(length(0, -7), DimExponents::DIMENSIONLESS);
        assert_eq!(length(0, 7).exponents(), [(0, 1); 7]);
        assert_eq!(length(1, -2).to_string(), "L^(-1/2)");
        assert!(DimExponents::from_rationals([(0, 0); 7]).is_none());
        assert!(DimExponents::from_rationals([(i32::MIN, 2); 7]).is_none());
        assert!(DimExponents::from_rationals([(1, i32::MIN); 7]).is_none());
    }

    #[test]
    fn exact_algebra_closes_roots_and_normalized_wave_dimensions() {
        assert_eq!(length(2, 1).pow(1, 2), Some(length(1, 1)));
        let density = length(-1, 2).pow(2, 1).unwrap();
        assert_eq!(density.mul(length(1, 1)), Some(DimExponents::DIMENSIONLESS));
        assert_ne!(
            length(-1, 1).pow(2, 1).unwrap().mul(length(1, 1)),
            Some(DimExponents::DIMENSIONLESS)
        );
        assert_eq!(length(1, 3).div(length(1, 2)), Some(length(-1, 6)));
    }

    #[test]
    fn dimension_powers_bound_reduced_results_not_operation_parameters() {
        // 0 * (-2^31) is exactly zero; (1/2) * (-2^31) = -2^30.
        assert_eq!(
            DimExponents::DIMENSIONLESS.pow(i32::MIN, 1),
            Some(DimExponents::DIMENSIONLESS)
        );
        assert_eq!(length(1, 2).pow(i32::MIN, 1), Some(length(-(1 << 30), 1)));
        // A negative denominator changes the sign before canonicalization.
        assert_eq!(length(2, 1).pow(1, i32::MIN), Some(length(-1, 1 << 30)));
        assert_eq!(length(1, 1).pow(i32::MIN, i32::MIN), Some(length(1, 1)));
        // The final nonzero exponent still must satisfy the persisted bounds.
        assert!(length(1, 1).pow(i32::MIN, 1).is_none());
        assert!(length(1, 1).pow(1, i32::MIN).is_none());
        assert!(DimExponents::DIMENSIONLESS.pow(0, 0).is_none());
    }

    #[test]
    fn reduced_bounds_reject_overflow_without_rejecting_cancellation() {
        let limit = i32::MAX;
        assert_eq!(
            length(1, limit).mul(length(limit - 1, limit)),
            Some(length(1, 1))
        );
        assert_eq!(
            length(limit, limit - 1).pow(limit - 1, limit),
            Some(length(1, 1))
        );
        assert!(length(limit, 1).mul(length(1, 1)).is_none());
        assert!(length(-limit, 1).div(length(1, 1)).is_none());
        assert!(length(1, limit).pow(1, 2).is_none());
        assert!(length(1, 1).pow(1, 0).is_none());
    }
}
