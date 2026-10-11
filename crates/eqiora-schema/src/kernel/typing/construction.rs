use super::{ExpressionType, TypeViolation, additive};
use eqiora_core::{ScalarDomain, ValueShape, ValueType};

impl<I: Clone + Eq> ExpressionType<I> {
    /// Construct one outer channel axis with complete elements and compatible exact supports.
    pub fn array(elements: &[ExpressionType<I>]) -> Result<ExpressionType<I>, TypeViolation<I>> {
        let Some(first) = elements.first() else {
            return Err(TypeViolation::EmptyArray);
        };
        let mut element = first.clone();
        for next in &elements[1..] {
            element = additive(&element, next)?;
        }
        let extent =
            u32::try_from(elements.len()).map_err(|_| TypeViolation::SpatialExtentInvalid)?;
        element.value_type = element
            .value_type
            .array(extent)
            .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
        Ok(element)
    }
}

impl<I: Clone> ExpressionType<I> {
    /// Select an exact outer channel; Cartesian component axes are not channel axes.
    pub fn index(self, index: u32) -> Result<ExpressionType<I>, TypeViolation<I>> {
        let value = self;
        let channels = value.value_type.array_rank();
        if channels == 0 {
            return Err(TypeViolation::IndexRequiresArray);
        }
        let extents = value.shape().extents();
        if index >= extents[0].get() {
            return Err(TypeViolation::IndexOutOfBounds);
        }
        let shape = ValueShape::new(extents[channels..].iter().map(|n| n.get()))
            .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
        let mut element = ValueType::shaped(
            value.value_type.scalar_domain(),
            value.dimension(),
            shape,
            value.frame(),
        )
        .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
        for extent in extents[1..channels].iter().rev() {
            element = element
                .array(extent.get())
                .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
        }
        Ok(ExpressionType::new(element, value.support))
    }
}

impl<I: Clone + Eq> ExpressionType<I> {
    /// Construct a complex scalar, preserving physical dimension and compatible support.
    pub fn complex(self, imag: ExpressionType<I>) -> Result<ExpressionType<I>, TypeViolation<I>> {
        let real = self;
        if [&real, &imag].iter().any(|value| {
            !value.shape().is_scalar() || value.value_type.scalar_domain() != ScalarDomain::Real
        }) {
            return Err(TypeViolation::ComplexRequiresRealScalars);
        }
        let result = additive(&real, &imag)?;
        Ok(ExpressionType::new(
            ValueType::scalar(ScalarDomain::Complex, result.dimension())
                .expect("checked scalar type"),
            result.support,
        ))
    }
}

impl<I: Clone + Eq> ExpressionType<I> {
    /// Type a real scalar partial derivative with respect to an exact coordinate.
    ///
    /// The caller must establish that the selector is a coordinate symbol; this
    /// rule checks its admitted support, component types, and resulting units.
    pub fn coordinate_partial(&self, selected: &Self) -> Result<Self, TypeViolation<I>> {
        use super::{SpatialSupport, combine_additive_support};
        if !matches!(
            selected.support,
            Some(SpatialSupport::Volume { .. } | SpatialSupport::Coordinates { .. })
        ) {
            return Err(TypeViolation::CoordinatePartialRequiresCoordinate);
        }
        if [self, selected].iter().any(|ty| {
            !ty.shape().is_scalar()
                || ty.value_type.array_rank() != 0
                || ty.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        }) {
            return Err(TypeViolation::RootRequiresRealScalar);
        }
        match combine_additive_support(&self.support, &selected.support) {
            Err(error) => Err(error),
            Ok(support) => self
                .dimension()
                .div(selected.dimension())
                .map(|dimension| ExpressionType::scalar(dimension, support))
                .ok_or(TypeViolation::DimensionOverflow {
                    operation: "coordinate partial",
                }),
        }
    }
}
