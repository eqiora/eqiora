//! Binary64 significands with retained exponents through scalar and map arithmetic.
use super::binary_scale::{exponent, scale};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Scaled {
    significand: f64,
    exponent: i64,
}

impl Scaled {
    pub(super) const ZERO: Self = Self {
        significand: 0.,
        exponent: 0,
    };

    pub(super) fn new(value: f64) -> Self {
        Self::normalize(value, 0)
    }

    fn normalize(value: f64, offset: i64) -> Self {
        if value == 0. {
            return Self {
                significand: value,
                exponent: 0,
            };
        }
        let shift = exponent(value);
        match offset.checked_add(i64::from(shift)) {
            Some(exponent) => Self {
                significand: scale(value, -shift),
                exponent,
            },
            None => Self {
                significand: f64::NAN,
                exponent: 0,
            },
        }
    }

    pub(super) fn value(self) -> f64 {
        scale(self.significand, self.exponent.clamp(-4096, 4096) as i32)
    }

    pub(super) fn sub(self, other: Self) -> Self {
        if other.significand == 0. {
            return if self.significand == 0. {
                Self::new(self.significand - other.significand)
            } else {
                self
            };
        }
        if self.significand == 0. {
            return Self {
                significand: -other.significand,
                exponent: other.exponent,
            };
        }
        let common = self.exponent.max(other.exponent);
        Self::normalize(
            scale(
                self.significand,
                self.exponent.saturating_sub(common).clamp(-4096, 0) as i32,
            ) - scale(
                other.significand,
                other.exponent.saturating_sub(common).clamp(-4096, 0) as i32,
            ),
            common,
        )
    }

    pub(super) fn mul(self, other: Self) -> Self {
        self.exponent
            .checked_add(other.exponent)
            .map_or(Self::new(f64::NAN), |e| {
                Self::normalize(self.significand * other.significand, e)
            })
    }

    pub(super) fn div(self, other: Self) -> Self {
        self.exponent
            .checked_sub(other.exponent)
            .map_or(Self::new(f64::NAN), |e| {
                Self::normalize(self.significand / other.significand, e)
            })
    }
}

impl Scaled {
    pub(super) fn is_finite(self) -> bool {
        self.significand.is_finite()
    }
    pub(super) fn is_zero(self) -> bool {
        self.significand == 0.
    }
    pub(super) fn signum(self) -> f64 {
        self.significand.signum()
    }
    pub(super) fn is_positive(self) -> bool {
        self.significand > 0.
    }
    pub(super) fn neg(self) -> Self {
        Self {
            significand: -self.significand,
            ..self
        }
    }
    pub(super) fn add(self, other: Self) -> Self {
        self.sub(other.neg())
    }
    pub(super) fn powi(self, power: i32) -> Self {
        let mut remaining = power.unsigned_abs();
        let mut base = if power < 0 {
            Self::new(1.).div(self)
        } else {
            self
        };
        let mut result = Self::new(1.);
        while remaining != 0 {
            if remaining & 1 != 0 {
                result = result.mul(base);
            }
            remaining >>= 1;
            if remaining != 0 {
                base = base.mul(base);
            }
        }
        result
    }
}
impl From<f64> for Scaled {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}
impl From<Scaled> for f64 {
    fn from(value: Scaled) -> Self {
        value.value()
    }
}
impl std::ops::Neg for Scaled {
    type Output = Self;
    fn neg(self) -> Self {
        self.neg()
    }
}
impl std::ops::AddAssign for Scaled {
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}
macro_rules! arithmetic {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait for Scaled {
            type Output = Self;
            fn $method(self, other: Self) -> Self {
                self.$method(other)
            }
        }
        impl std::ops::$trait<f64> for Scaled {
            type Output = Self;
            fn $method(self, other: f64) -> Self {
                self.$method(Self::new(other))
            }
        }
        impl std::ops::$trait<Scaled> for f64 {
            type Output = Scaled;
            fn $method(self, other: Scaled) -> Scaled {
                Scaled::new(self).$method(other)
            }
        }
    };
}
arithmetic!(Add, add);
arithmetic!(Sub, sub);
arithmetic!(Mul, mul);
arithmetic!(Div, div);
impl std::iter::Sum for Scaled {
    fn sum<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ZERO, Self::add)
    }
}

impl Scaled {
    pub(super) fn abs(self) -> Self {
        Self {
            significand: self.significand.abs(),
            ..self
        }
    }
    pub(super) fn abs_cmp(self, other: Self) -> std::cmp::Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self
                .exponent
                .cmp(&other.exponent)
                .then_with(|| self.significand.abs().total_cmp(&other.significand.abs())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn derivative_cancellation_keeps_primal_domains_and_output_range_checks() {
        use eqiora_core::{DimExponents, DynQuantity};
        let big = 2f64.powi(700);
        // f(p)=big*p-big*p=0. With direction/cotangent big, the two
        // derivative contributions exceed binary64 but their sum is zero.
        let instructions = [
            Instruction::Read(SymbolSlot(0)),
            Instruction::Constant(DynQuantity::new(big, DimExponents::DIMENSIONLESS)),
            Instruction::Mul(ValueId(0), ValueId(1)),
            Instruction::Neg(ValueId(1)),
            Instruction::Mul(ValueId(0), ValueId(3)),
            Instruction::Add(ValueId(2), ValueId(4)),
        ];
        let root = [ValueId(5)];
        let roles = [DifferentiationRole::Parameter];
        for point in [0., 1.] {
            let action =
                ScalarLinearization::bind(&instructions, &root, 1, &[point], &roles).unwrap();
            let mut output = [1.];
            action
                .jvp(RelationTangent::Parameter(&[big]), &mut output)
                .unwrap();
            assert_eq!(output, [0.]);
            action
                .vjp(&[big], RelationCotangent::Parameter(&mut output))
                .unwrap();
            assert_eq!(output, [0.]);
        }
        let invalid = ScalarLinearization::bind(&instructions, &root, 1, &[big], &roles).unwrap();
        let mut output = [0.];
        assert!(
            invalid
                .jvp(RelationTangent::Parameter(&[0.]), &mut output)
                .is_err()
        );
        assert!(
            invalid
                .vjp(&[0.], RelationCotangent::Parameter(&mut output))
                .is_err()
        );

        // Requesting only the first term has an unrepresentable derivative.
        let root = [ValueId(2)];
        let action = ScalarLinearization::bind(&instructions, &root, 1, &[0.], &roles).unwrap();
        assert!(
            action
                .jvp(RelationTangent::Parameter(&[big]), &mut output)
                .is_err()
        );
        assert!(
            action
                .vjp(&[big], RelationCotangent::Parameter(&mut output))
                .is_err()
        );
    }
}
