use eqiora_core::{DimExponents, ScalarDomain, ValueShape};

use eqiora_core::{InvalidValueType, ValueFrame, ValueType};

use super::{SpatialSupport, TypeViolation};

/// Complete static type of one residual-expression value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressionType<I> {
    /// Checked mathematical type, independent of execution choices.
    pub value_type: ValueType,
    /// Exact nominal spatial support, absent for global scalars.
    pub support: Option<SpatialSupport<I>>,
}

impl<I> ExpressionType<I> {
    pub(super) fn checked(
        scalar_domain: ScalarDomain,
        dimension: DimExponents,
        shape: ValueShape,
        frame: ValueFrame,
        support: Option<SpatialSupport<I>>,
    ) -> Result<Self, TypeViolation<I>> {
        let value_type = ValueType::shaped(scalar_domain, dimension, shape, frame).map_err(
            |error| match error {
                InvalidValueType::EnumType
                | InvalidValueType::BooleanType
                | InvalidValueType::FiniteSpaceType
                | InvalidValueType::FiniteSpaceShape
                | InvalidValueType::ScalarFrame => TypeViolation::IncompatibleFrame,
                InvalidValueType::ComponentCountOverflow | InvalidValueType::ArrayExtent => {
                    TypeViolation::SpatialExtentInvalid
                }
            },
        )?;
        Ok(Self::new(value_type, support))
    }
    /// Attach exact support to an already checked mathematical type.
    #[must_use]
    pub fn new(value_type: ValueType, support: Option<SpatialSupport<I>>) -> Self {
        Self {
            value_type,
            support,
        }
    }

    /// An invariant real scalar with the supplied dimension and support.
    #[must_use]
    pub fn scalar(dimension: DimExponents, support: Option<SpatialSupport<I>>) -> Self {
        Self::new(
            ValueType::scalar(ScalarDomain::Real, dimension).expect("checked scalar type"),
            support,
        )
    }

    /// A real value with an exact mathematical shape, frame and support.
    ///
    /// # Errors
    /// Rejects an invalid shape/frame combination or unrepresentable component count.
    pub fn shaped(
        dimension: DimExponents,
        shape: ValueShape,
        frame: ValueFrame,
        support: Option<SpatialSupport<I>>,
    ) -> Result<Self, InvalidValueType> {
        Ok(Self::new(
            ValueType::shaped(ScalarDomain::Real, dimension, shape, frame)?,
            support,
        ))
    }

    /// Exact physical dimension.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.value_type.dimension()
    }

    /// Mathematical component shape.
    #[must_use]
    pub const fn shape(&self) -> &ValueShape {
        self.value_type.shape()
    }

    /// Component-frame meaning.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.value_type.frame()
    }
}
