use super::{ExpressionType, TypeViolation, additive};
use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueType};

impl<I: Clone + Eq> ExpressionType<I> {
    /// Type arithmetic addition, including a count plus same-basis signed change.
    pub fn sum(self, other: Self) -> Result<Self, TypeViolation<I>> {
        if matches!(
            self.value_type.scalar_domain(),
            ScalarDomain::Boolean | ScalarDomain::Enum
        ) || matches!(
            other.value_type.scalar_domain(),
            ScalarDomain::Boolean | ScalarDomain::Enum
        ) || self.value_type.index_set().is_some()
            || other.value_type.index_set().is_some()
        {
            return Err(TypeViolation::ScalarDomainMismatch);
        }
        if self.value_type.is_count() {
            if other.value_type.is_count()
                || self.value_type.finite_space() != other.value_type.finite_space()
                || self.shape() != other.shape()
                || self.dimension() != other.dimension()
                || self.frame() != other.frame()
            {
                return Err(TypeViolation::AdditiveTypeMismatch {
                    left: Box::new(self),
                    right: Box::new(other),
                });
            }
            let support = super::combine_additive_support(&self.support, &other.support)?;
            return Ok(Self::new(self.value_type, support));
        }
        additive(&self, &other)
    }

    /// Explicitly project a nominal index to its dimensionless integer ordinal.
    pub fn ordinal(self) -> Result<Self, TypeViolation<I>> {
        if self.value_type.index_set().is_none() {
            return Err(TypeViolation::ScalarDomainMismatch);
        }
        Ok(Self::new(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            self.support,
        ))
    }

    /// Type a checked integer quotient or remainder, without real promotion.
    pub fn integer_quotient(self, other: Self) -> Result<Self, TypeViolation<I>> {
        integer_scalar(&self)?;
        integer_scalar(&other)?;
        additive(&self, &other)
    }

    /// Explicitly convert an integer scalar to a dimensionless real scalar.
    pub fn to_real(self) -> Result<Self, TypeViolation<I>> {
        integer_scalar(&self)?;
        Ok(Self::new(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            self.support,
        ))
    }

    /// Admit the type of a checked real-to-integer conversion.
    /// The integral value and exact range are checked when evaluating it.
    pub fn to_integer(self) -> Result<Self, TypeViolation<I>> {
        if self.value_type.scalar_domain() != ScalarDomain::Real
            || !self.shape().is_scalar()
            || self.dimension() != DimExponents::DIMENSIONLESS
        {
            return Err(TypeViolation::ScalarDomainMismatch);
        }
        Ok(Self::new(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            self.support,
        ))
    }
}

fn integer_scalar<I>(value: &ExpressionType<I>) -> Result<(), TypeViolation<I>> {
    if value.value_type.index_set().is_some()
        || value.value_type.scalar_domain() != ScalarDomain::Integer
        || !value.shape().is_scalar()
        || value.frame() != ValueFrame::Invariant
        || value.dimension() != DimExponents::DIMENSIONLESS
    {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::typing::{additive, divide, multiply, time_derivative};
    #[test]
    fn discrete_domains_require_explicit_conversion_and_have_no_derivative() {
        let integer: ExpressionType<()> = ExpressionType::new(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            None,
        );
        let real = integer.clone().to_real().unwrap();
        assert_eq!(real.clone().to_integer().unwrap(), integer);
        assert_eq!(additive(&integer, &integer).unwrap(), integer);
        assert_eq!(multiply(&integer, &integer).unwrap(), integer);
        assert!(additive(&integer, &real).is_err());
        assert!(multiply(&real, &integer).is_err());
        assert!(divide(&integer, &integer).is_err());
        assert!(time_derivative(&integer).is_err());
        assert!(integer.clone().to_integer().is_err());
        assert_eq!(
            integer.clone().integer_quotient(integer.clone()).unwrap(),
            integer
        );
    }
}

#[cfg(test)]
mod nominal_tests {
    use super::*;
    use eqiora_core::{Id, entity::kinds};
    #[test]
    fn count_changes_and_ordinals_preserve_nominal_boundaries() {
        let species = Id::<kinds::FiniteSpace>::new();
        let counts: ExpressionType<()> =
            ExpressionType::new(ValueType::counts(species, 2).unwrap(), None);
        let changes = ExpressionType::new(
            ValueType::coordinates(
                eqiora_core::FiniteBasis::new(species, 2).unwrap(),
                eqiora_core::ScalarDomain::Integer,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            None,
        );
        assert_eq!(counts.clone().sum(changes.clone()).unwrap(), counts);
        assert!(counts.clone().sum(counts.clone()).is_err());
        assert!(changes.sum(counts.clone()).is_err());
        let foreign = ExpressionType::new(
            ValueType::coordinates(
                eqiora_core::FiniteBasis::new(Id::new(), 2).unwrap(),
                eqiora_core::ScalarDomain::Integer,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            None,
        );
        assert!(counts.sum(foreign).is_err());
        let index: ExpressionType<()> =
            ExpressionType::new(ValueType::index(Id::new(), 2).unwrap(), None);
        assert!(index.clone().sum(index.clone()).is_err());
        assert!(index.clone().to_real().is_err());
        assert_eq!(
            index.ordinal().unwrap().value_type,
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type")
        );
    }
}
