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
                InvalidValueType::DimensionOverflow => TypeViolation::DimensionOverflow {
                    operation: "value type derivation",
                },
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

impl<I: Clone + PartialEq> ExpressionType<I> {
    /// Type one exact factor axis, retaining the projection's declared support.
    pub fn coordinate(
        factor: &I,
        axis: usize,
        relation: Option<&SpatialSupport<I>>,
    ) -> Result<ExpressionType<I>, TypeViolation<I>> {
        let support = relation.ok_or(TypeViolation::CoordinateRequiresSpatialScope)?;
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length dimension");
        let (unit, dimensions) = match support {
            SpatialSupport::Volume { domain, dimensions } if domain == factor => {
                (length, *dimensions)
            }
            SpatialSupport::Boundary {
                parent, dimensions, ..
            } if parent == factor => (length, *dimensions),
            SpatialSupport::Coordinates { factors, .. } => factors
                .iter()
                .find(|(id, _, _)| id == factor)
                .map(|(_, unit, axes)| (*unit, *axes))
                .ok_or(TypeViolation::CoordinateFactorMismatch)?,
            _ => return Err(TypeViolation::CoordinateFactorMismatch),
        };
        if axis >= dimensions {
            return Err(TypeViolation::CoordinateAxisOutOfRange { axis, dimensions });
        }
        Ok(ExpressionType::scalar(unit, Some(support.clone())))
    }
}
