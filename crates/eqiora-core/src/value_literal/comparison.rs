use super::{InvalidValueLiteral, ValueLiteral};
use crate::ScalarDomain;
use core::cmp::Ordering;

impl ValueLiteral {
    /// Exact scalar equality without numeric Boolean or integer/real coercion.
    /// Real and complex quantities share equality only at equal dimensions and frames;
    /// nominal indexes require the same complete type. Arrays and counts reject.
    pub fn checked_equal(&self, other: &Self) -> Result<bool, InvalidValueLiteral> {
        self.comparable_scalar(other)?;
        match (
            self.value_type().scalar_domain(),
            other.value_type().scalar_domain(),
        ) {
            (ScalarDomain::Enum, ScalarDomain::Enum) => {
                if self.value_type() != other.value_type() {
                    return Err(InvalidValueLiteral::ScalarDomain);
                }
                Ok(self.enum_tag() == other.enum_tag())
            }
            (ScalarDomain::Boolean, ScalarDomain::Boolean) => Ok(self.as_bool() == other.as_bool()),
            (ScalarDomain::Integer, ScalarDomain::Integer) => {
                if self.value_type() != other.value_type() {
                    return Err(InvalidValueLiteral::ScalarDomain);
                }
                Ok(self.integer_component(0) == other.integer_component(0))
            }
            (
                ScalarDomain::Real | ScalarDomain::Complex,
                ScalarDomain::Real | ScalarDomain::Complex,
            ) => Ok(self.component(0) == other.component(0)),
            _ => Err(InvalidValueLiteral::ScalarDomain),
        }
    }

    /// Exact ordering of ordinary real or integer scalars, without cross-domain coercion.
    /// Complex values, Booleans, nominal indexes, counts and arrays reject.
    pub fn checked_order(&self, other: &Self) -> Result<Ordering, InvalidValueLiteral> {
        self.comparable_scalar(other)?;
        if self.value_type() != other.value_type() || self.value_type().index_set().is_some() {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        match self.value_type().scalar_domain() {
            ScalarDomain::Integer => self
                .integer_scalar_value()
                .zip(other.integer_scalar_value())
                .map(|(left, right)| left.cmp(&right))
                .ok_or(InvalidValueLiteral::ScalarDomain),
            ScalarDomain::Real => self
                .real_scalar_value()
                .zip(other.real_scalar_value())
                .and_then(|(left, right)| left.value().partial_cmp(&right.value()))
                .ok_or(InvalidValueLiteral::ScalarDomain),
            ScalarDomain::Boolean | ScalarDomain::Complex | ScalarDomain::Enum => {
                Err(InvalidValueLiteral::ScalarDomain)
            }
        }
    }

    fn comparable_scalar(&self, other: &Self) -> Result<(), InvalidValueLiteral> {
        let left = self.value_type();
        let right = other.value_type();
        if !left.shape().is_scalar()
            || !right.shape().is_scalar()
            || left.dimension() != right.dimension()
            || left.frame() != right.frame()
        {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DimExponents, Id, ValueType};

    fn integer(n: i64) -> ValueLiteral {
        ValueLiteral::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            n,
        )
        .unwrap()
    }

    fn real(n: f64, dimension: DimExponents) -> ValueLiteral {
        ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, dimension).expect("checked scalar type"),
            n,
        )
        .unwrap()
    }

    #[test]
    fn comparisons_preserve_exact_domains_dimensions_and_nominal_identity() {
        let low = integer(9_007_199_254_740_992);
        let high = integer(9_007_199_254_740_993);
        assert!(!low.checked_equal(&high).unwrap());
        assert_eq!(low.checked_order(&high).unwrap(), Ordering::Less);
        assert_eq!(high.checked_order(&low).unwrap(), Ordering::Greater);
        assert_eq!(high.checked_order(&high).unwrap(), Ordering::Equal);
        assert_eq!(
            integer(i64::MIN).checked_order(&integer(i64::MAX)).unwrap(),
            Ordering::Less
        );
        let dimension = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let one = real(1.0, dimension);
        let adjacent = real(f64::from_bits(1.0_f64.to_bits() + 1), dimension);
        assert!(!one.checked_equal(&adjacent).unwrap());
        assert_eq!(one.checked_order(&adjacent).unwrap(), Ordering::Less);
        let complex = ValueLiteral::new(
            ValueType::scalar(ScalarDomain::Complex, dimension).expect("checked scalar type"),
            [(1.0, 0.0)],
        )
        .unwrap();
        let imaginary = ValueLiteral::new(
            ValueType::scalar(ScalarDomain::Complex, dimension).expect("checked scalar type"),
            [(1.0, 1.0)],
        )
        .unwrap();
        assert!(one.checked_equal(&complex).unwrap());
        assert!(complex.checked_equal(&one).unwrap());
        assert!(!one.checked_equal(&imaginary).unwrap());
        assert!(imaginary.checked_equal(&imaginary).unwrap());
        assert!(one.checked_order(&complex).is_err());
        assert!(complex.checked_order(&complex).is_err());
        assert!(
            one.checked_equal(&real(1.0, DimExponents::DIMENSIONLESS))
                .is_err()
        );
        assert!(
            one.checked_order(&real(1.0, DimExponents::DIMENSIONLESS))
                .is_err()
        );
        let ordinary_real = real(1.0, DimExponents::DIMENSIONLESS);
        assert!(ordinary_real.checked_equal(&integer(1)).is_err());
        assert!(integer(1).checked_order(&ordinary_real).is_err());
        for left in [false, true] {
            for right in [false, true] {
                assert_eq!(
                    ValueLiteral::boolean(left)
                        .checked_equal(&ValueLiteral::boolean(right))
                        .unwrap(),
                    left == right
                );
                assert!(
                    ValueLiteral::boolean(left)
                        .checked_order(&ValueLiteral::boolean(right))
                        .is_err()
                );
            }
            assert!(
                ValueLiteral::boolean(left)
                    .checked_equal(&integer(i64::from(left)))
                    .is_err()
            );
        }
        let set = Id::new();
        let index = ValueLiteral::from_integer(ValueType::index(set, 2).unwrap(), 1).unwrap();
        let zero = ValueLiteral::from_integer(ValueType::index(set, 2).unwrap(), 0).unwrap();
        let foreign =
            ValueLiteral::from_integer(ValueType::index(Id::new(), 2).unwrap(), 1).unwrap();
        assert!(index.checked_equal(&index).unwrap());
        assert!(!index.checked_equal(&zero).unwrap());
        assert!(index.checked_equal(&foreign).is_err());
        assert!(index.checked_equal(&integer(1)).is_err());
        assert!(index.checked_order(&zero).is_err());
        let space = Id::new();
        for value_type in [
            ValueType::counts(space, 1).unwrap(),
            ValueType::coordinates(
                crate::FiniteBasis::new(space, 1).unwrap(),
                crate::ScalarDomain::Integer,
                crate::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type")
                .array(1)
                .unwrap(),
        ] {
            let value = ValueLiteral::integer(value_type, [1]).unwrap();
            assert!(value.checked_equal(&value).is_err());
            assert!(value.checked_order(&value).is_err());
        }
    }
}
