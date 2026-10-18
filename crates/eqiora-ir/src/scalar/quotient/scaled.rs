//! Binary64 significands with a separate exponent for one quotient action.
use super::super::binary_scale::{exponent, scale};

#[derive(Clone, Copy)]
pub(super) struct Scaled {
    significand: f64,
    exponent: i32,
}

impl Scaled {
    pub(super) const ZERO: Self = Self {
        significand: 0.,
        exponent: 0,
    };

    pub(super) fn new(value: f64) -> Self {
        Self::normalize(value, 0)
    }

    fn normalize(value: f64, offset: i32) -> Self {
        if value == 0. {
            return Self::ZERO;
        }
        let shift = exponent(value);
        Self {
            significand: scale(value, -shift),
            exponent: offset + shift,
        }
    }

    pub(super) fn value(self) -> f64 {
        scale(self.significand, self.exponent)
    }

    pub(super) fn sub(self, other: Self) -> Self {
        if other.significand == 0. {
            return self;
        }
        if self.significand == 0. {
            return Self {
                significand: -other.significand,
                exponent: other.exponent,
            };
        }
        let common = self.exponent.max(other.exponent);
        Self::normalize(
            scale(self.significand, self.exponent - common)
                - scale(other.significand, other.exponent - common),
            common,
        )
    }

    pub(super) fn mul(self, other: Self) -> Self {
        Self::normalize(
            self.significand * other.significand,
            self.exponent + other.exponent,
        )
    }

    pub(super) fn div(self, other: Self) -> Self {
        Self::normalize(
            self.significand / other.significand,
            self.exponent - other.exponent,
        )
    }
}
