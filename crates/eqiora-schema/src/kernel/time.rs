//! Exact, canonical model-time values.

use core::cmp::Ordering;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath};

/// A non-negative rational number of seconds in reduced form.
///
/// Model clocks use exact rational time so multi-rate coincidences never
/// depend on floating-point equality. Physical integration time may still use
/// floating-point values after realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RationalTime {
    numerator: u64,
    denominator: u64,
}

impl RationalTime {
    /// Zero seconds.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// Construct a canonical non-negative rational number of seconds.
    ///
    /// # Errors
    /// Returns `EQ0305` when `denominator` is zero.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, Diagnostic> {
        if denominator == 0 {
            return Err(Diagnostic::error(
                codes::INVALID_CLOCK,
                "rational model time denominator must be non-zero",
            )
            .with_graph_path(GraphPath::new(["semantic", "clock-domain"])));
        }
        if numerator == 0 {
            return Ok(Self::ZERO);
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// Numerator of the reduced seconds fraction.
    #[must_use]
    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    /// Denominator of the reduced seconds fraction.
    #[must_use]
    pub const fn denominator(self) -> u64 {
        self.denominator
    }

    /// Convert to seconds for numerical realization.
    #[must_use]
    pub fn as_seconds_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Whether the exact time is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.numerator == 0
    }

    /// Add two exact model-time values without silently overflowing.
    ///
    /// # Errors
    /// Returns `EQ0305` when the reduced result cannot be represented by the
    /// `u64` numerator/denominator wire contract.
    pub fn checked_add(self, other: Self) -> Result<Self, Diagnostic> {
        let numerator = u128::from(self.numerator)
            .checked_mul(u128::from(other.denominator))
            .and_then(|left| {
                u128::from(other.numerator)
                    .checked_mul(u128::from(self.denominator))
                    .and_then(|right| left.checked_add(right))
            })
            .ok_or_else(time_overflow)?;
        let denominator = u128::from(self.denominator)
            .checked_mul(u128::from(other.denominator))
            .ok_or_else(time_overflow)?;
        let divisor = gcd_u128(numerator, denominator);
        let reduced_numerator = u64::try_from(numerator / divisor).map_err(|_| time_overflow())?;
        let reduced_denominator =
            u64::try_from(denominator / divisor).map_err(|_| time_overflow())?;
        Self::new(reduced_numerator, reduced_denominator)
    }
}

impl PartialOrd for RationalTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalTime {
    fn cmp(&self, other: &Self) -> Ordering {
        (u128::from(self.numerator) * u128::from(other.denominator))
            .cmp(&(u128::from(other.numerator) * u128::from(self.denominator)))
    }
}

const fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn time_overflow() -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_CLOCK,
        "exact model-time arithmetic exceeds the u64 rational representation",
    )
    .with_graph_path(GraphPath::new(["semantic", "clock-domain"]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_are_reduced_and_order_exactly() {
        let ten_ms = RationalTime::new(10, 1_000).expect("valid");
        let one_hundredth = RationalTime::new(1, 100).expect("valid");
        let twenty_ms = RationalTime::new(1, 50).expect("valid");

        assert_eq!(ten_ms, one_hundredth);
        assert_eq!((ten_ms.numerator(), ten_ms.denominator()), (1, 100));
        assert!(ten_ms < twenty_ms);
    }

    #[test]
    fn zero_denominator_is_rejected() {
        let diagnostic = RationalTime::new(1, 0).expect_err("invalid rational");
        assert_eq!(diagnostic.code(), codes::INVALID_CLOCK);
    }

    #[test]
    fn exact_addition_reduces_before_returning() {
        let one_third = RationalTime::new(1, 3).expect("valid");
        let one_sixth = RationalTime::new(1, 6).expect("valid");

        assert_eq!(
            one_third.checked_add(one_sixth).expect("representable"),
            RationalTime::new(1, 2).expect("valid")
        );
    }
}
