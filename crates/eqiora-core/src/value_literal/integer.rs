use super::{InvalidValueLiteral, Payload, ValueLiteral};
use crate::{DimExponents, ScalarDomain, ValueFrame, ValueType};

impl ValueLiteral {
    /// Construct exact integer components in row-major order, without real conversion.
    /// All-zero input uses compact storage; at most one excess input is consumed.
    pub fn integer(
        value_type: ValueType,
        components: impl IntoIterator<Item = i64>,
    ) -> Result<Self, InvalidValueLiteral> {
        check_integer_type(&value_type)?;
        let count = value_type.shape().component_count().expect("checked shape");
        let mut input = components.into_iter();
        let (lower, upper) = input.size_hint();
        if lower > count || upper.is_some_and(|upper| upper < count) {
            return Err(InvalidValueLiteral::ComponentCount);
        }
        let mut values = Vec::new();
        for index in 0..count {
            let value = input.next().ok_or(InvalidValueLiteral::ComponentCount)?;
            check_range(&value_type, value)?;
            if !values.is_empty() || value != 0 {
                values
                    .try_reserve(index + 1 - values.len())
                    .map_err(|_| InvalidValueLiteral::Allocation)?;
                values.resize(index, 0);
                values.push(value);
            }
        }
        if input.next().is_some() {
            return Err(InvalidValueLiteral::ComponentCount);
        }
        Ok(Self {
            value_type,
            payload: if values.is_empty() {
                Payload::Zero
            } else {
                Payload::Integers(values.into_boxed_slice())
            },
        })
    }

    /// Construct an integer scalar or compact contextual integer zero.
    pub fn from_integer(value_type: ValueType, value: i64) -> Result<Self, InvalidValueLiteral> {
        check_integer_type(&value_type)?;
        if value == 0 {
            return Ok(Self {
                value_type,
                payload: Payload::Zero,
            });
        }
        if !value_type.shape().is_scalar() {
            return Err(InvalidValueLiteral::NonzeroShape);
        }
        Self::integer(value_type, [value])
    }

    /// Exact integer component, absent for other domains or out-of-range positions.
    #[must_use]
    pub fn integer_component(&self, index: usize) -> Option<i64> {
        if self.value_type.scalar_domain() != ScalarDomain::Integer {
            return None;
        }
        match &self.payload {
            Payload::Zero => (index < self.component_count()).then_some(0),
            Payload::Integers(values) => values.get(index).copied(),
            Payload::Components(_) | Payload::Boolean(_) | Payload::Enum(_) => None,
        }
    }

    /// Exact ordered integer components, absent for other domains.
    pub fn integer_components(
        &self,
    ) -> Option<impl ExactSizeIterator<Item = i64> + DoubleEndedIterator + '_> {
        (self.value_type.scalar_domain() == ScalarDomain::Integer).then(|| {
            (0..self.component_count())
                .map(|index| self.integer_component(index).expect("in-range integer"))
        })
    }

    /// Extract only an exact invariant dimensionless integer scalar.
    #[must_use]
    pub fn integer_scalar_value(&self) -> Option<i64> {
        if !self.value_type.shape().is_scalar() || self.value_type.index_set().is_some() {
            return None;
        }
        self.integer_component(0)
    }

    /// Explicitly forget a checked index's nominal set, retaining its exact ordinal.
    pub fn ordinal(&self) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.index_set().is_none() {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        Self::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            self.integer_component(0)
                .ok_or(InvalidValueLiteral::ScalarDomain)?,
        )
    }

    /// Checked exact integer addition, with equal complete types and no broadcasting.
    pub fn checked_add(&self, other: &Self) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.is_count() {
            if other.value_type.is_count()
                || self.value_type.finite_space() != other.value_type.finite_space()
                || self.value_type.shape() != other.value_type.shape()
            {
                return Err(InvalidValueLiteral::ScalarDomain);
            }
            return self.integer_results(other, i64::checked_add);
        }
        self.integer_binary(other, i64::checked_add)
    }
    /// Checked exact integer subtraction.
    pub fn checked_sub(&self, other: &Self) -> Result<Self, InvalidValueLiteral> {
        self.integer_binary(other, i64::checked_sub)
    }
    /// Checked exact integer multiplication.
    pub fn checked_mul(&self, other: &Self) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.finite_space().is_some() {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        self.integer_binary(other, i64::checked_mul)
    }
    /// Checked exact integer negation.
    pub fn checked_neg(&self) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.scalar_domain() != ScalarDomain::Integer
            || self.value_type.is_count()
            || self.value_type.index_set().is_some()
        {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        if self.is_zero() {
            return Ok(self.clone());
        }
        let values = self
            .integer_components()
            .ok_or(InvalidValueLiteral::ScalarDomain)?
            .map(|n| n.checked_neg().ok_or(InvalidValueLiteral::IntegerOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Self::integer(self.value_type.clone(), values)
    }
    /// Integer quotient truncating toward zero; zero and MIN / -1 reject.
    pub fn checked_quotient(&self, other: &Self) -> Result<Self, InvalidValueLiteral> {
        self.integer_division(other, i64::checked_div)
    }
    /// Integer remainder with the dividend's sign; zero and MIN % -1 reject.
    pub fn checked_remainder(&self, other: &Self) -> Result<Self, InvalidValueLiteral> {
        self.integer_division(other, i64::checked_rem)
    }
    /// Explicit integer scalar to binary64, rounded once to nearest, ties to even.
    pub fn to_real(&self) -> Result<Self, InvalidValueLiteral> {
        let value = self
            .integer_scalar_value()
            .ok_or(InvalidValueLiteral::ScalarDomain)?;
        Self::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            value as f64,
        )
    }
    /// Convert only a finite dimensionless integral real scalar in the signed range.
    pub fn to_integer(&self) -> Result<Self, InvalidValueLiteral> {
        let value = self
            .real_scalar_value()
            .filter(|v| v.dim() == DimExponents::DIMENSIONLESS)
            .ok_or(InvalidValueLiteral::ScalarDomain)?
            .value();
        // MAX rounds up to 2^63 as binary64: the upper endpoint must be exclusive.
        if value.fract() != 0.0 || !(-9223372036854775808.0..9223372036854775808.0).contains(&value)
        {
            return Err(InvalidValueLiteral::IntegerConversion);
        }
        Self::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            value as i64,
        )
    }

    fn integer_binary(
        &self,
        other: &Self,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.is_count()
            || self.value_type.index_set().is_some()
            || self.value_type != other.value_type
        {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        self.integer_results(other, operation)
    }
    fn integer_results(
        &self,
        other: &Self,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<Self, InvalidValueLiteral> {
        if self.value_type.scalar_domain() != ScalarDomain::Integer
            || other.value_type.scalar_domain() != ScalarDomain::Integer
        {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        if self.is_zero() && other.is_zero() {
            return Ok(self.clone());
        }
        let left = self
            .integer_components()
            .ok_or(InvalidValueLiteral::ScalarDomain)?;
        let right = other
            .integer_components()
            .ok_or(InvalidValueLiteral::ScalarDomain)?;
        let values = left
            .zip(right)
            .map(|(a, b)| operation(a, b).ok_or(InvalidValueLiteral::IntegerOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Self::integer(self.value_type.clone(), values)
    }
    fn integer_division(
        &self,
        other: &Self,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<Self, InvalidValueLiteral> {
        if self.integer_scalar_value().is_none() || other.integer_scalar_value().is_none() {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        if other.integer_scalar_value() == Some(0) {
            return Err(InvalidValueLiteral::ZeroDivisor);
        }
        self.integer_binary(other, operation)
    }
}

fn check_integer_type(value_type: &ValueType) -> Result<(), InvalidValueLiteral> {
    if value_type.scalar_domain() != ScalarDomain::Integer
        || value_type.dimension() != DimExponents::DIMENSIONLESS
        || value_type.frame() != ValueFrame::Invariant
    {
        return Err(InvalidValueLiteral::ScalarDomain);
    }
    Ok(())
}

fn check_range(value_type: &ValueType, value: i64) -> Result<(), InvalidValueLiteral> {
    if (value_type.is_count() && value < 0)
        || value_type
            .index_extent()
            .is_some_and(|extent| value < 0 || value >= i64::from(extent))
    {
        return Err(InvalidValueLiteral::NominalRange);
    }
    Ok(())
}
