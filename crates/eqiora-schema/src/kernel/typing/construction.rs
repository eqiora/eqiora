use super::{ExpressionType, TypeViolation, additive};
use eqiora_core::{ScalarDomain, ValueShape, ValueType};

impl<I: Clone + Eq> ExpressionType<I> {
    /// Construct an outer channel axis from complete elements with compatible supports.
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
    /// Select an outer channel; Cartesian component axes are not channel axes.
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
    /// Combine matching real components, preserving their complete type and support.
    pub fn complex(self, imag: ExpressionType<I>) -> Result<ExpressionType<I>, TypeViolation<I>> {
        let real = self;
        if [&real, &imag]
            .iter()
            .any(|value| value.value_type.scalar_domain() != ScalarDomain::Real)
        {
            return Err(TypeViolation::ComplexRequiresRealComponents);
        }
        let result = additive(&real, &imag)?;
        Ok(ExpressionType::new(
            result
                .value_type
                .with_scalar_domain(ScalarDomain::Complex)
                .expect("checked real components"),
            result.support,
        ))
    }
}

impl<I: Clone + Eq> ExpressionType<I> {
    /// Type a real-linear forward action with a complete input tangent block.
    /// Input identity and executable differentiability belong to the caller.
    ///
    /// # Errors
    /// Rejects nonnumeric values, a direction with a different complete input
    /// contract, or incompatible supports.
    pub fn jvp(&self, selected: &Self, direction: &Self) -> Result<Self, TypeViolation<I>> {
        let support = self.action_support(selected)?;
        if direction != selected {
            return Err(TypeViolation::DerivativeDirectionTypeMismatch {
                expected: Box::new(selected.clone()),
                actual: Box::new(direction.clone()),
            });
        }
        Ok(Self::new(self.value_type.clone(), support))
    }

    /// Type a real-adjoint action using a dimensionless real dual pairing.
    /// The seed retains the output's shape/domain and reciprocal units; the
    /// returned block retains the selected input's shape/domain and reciprocal units.
    ///
    /// # Errors
    /// Rejects nonnumeric values, wrong complete dual contracts, incompatible
    /// supports, or unrepresentable reciprocal dimensions.
    pub fn vjp(&self, selected: &Self, cotangent: &Self) -> Result<Self, TypeViolation<I>> {
        let support = self.action_support(selected)?;
        let dual = self.dual()?;
        if cotangent != &dual {
            return Err(TypeViolation::DerivativeCotangentTypeMismatch {
                expected: Box::new(dual),
                actual: Box::new(cotangent.clone()),
            });
        }
        Ok(Self::new(selected.dual()?.value_type, support))
    }

    fn action_support(
        &self,
        selected: &Self,
    ) -> Result<Option<super::SpatialSupport<I>>, TypeViolation<I>> {
        if [self, selected].iter().any(|value| {
            !matches!(
                value.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
            )
        }) {
            return Err(TypeViolation::ScalarDomainMismatch);
        }
        super::combine_additive_support(&self.support, &selected.support)
    }

    /// Retain numeric shape, domain, meaning and support with reciprocal units
    /// for the dimensionless real dual pairing.
    ///
    /// # Errors
    /// Rejects nonnumeric domains and unrepresentable reciprocal dimensions.
    pub fn dual(&self) -> Result<Self, TypeViolation<I>> {
        self.action_support(self)?;
        self.dimension()
            .pow(-1, 1)
            .and_then(|dimension| self.value_type.clone().with_dimension(dimension).ok())
            .map(|value_type| Self::new(value_type, self.support.clone()))
            .ok_or(TypeViolation::DimensionOverflow {
                operation: "derivative dual",
            })
    }

    /// Type a numeric value's partial derivative with respect to one real coordinate.
    ///
    /// The caller must establish that the selector is a coordinate symbol; this rule
    /// checks its support, the component types, and the resulting units.
    pub fn coordinate_partial(&self, selected: &Self) -> Result<Self, TypeViolation<I>> {
        use super::SpatialSupport;
        if !matches!(
            selected.support,
            Some(SpatialSupport::Volume { .. } | SpatialSupport::Coordinates { .. })
        ) {
            return Err(TypeViolation::CoordinatePartialRequiresCoordinate);
        }
        self.partial(selected)
    }

    /// Preserve a numeric value's component meaning under differentiation by an
    /// independent real scalar. Selector identity and executable calculus remain
    /// the caller's responsibility; this is the common type and unit rule.
    ///
    /// # Errors
    /// Rejects nonnumeric values, non-real or shaped selectors, incompatible
    /// supports and unrepresentable result dimensions.
    pub fn partial(&self, selected: &Self) -> Result<Self, TypeViolation<I>> {
        if !selected.shape().is_scalar()
            || selected.value_type.array_rank() != 0
            || selected.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        {
            return Err(TypeViolation::RootRequiresRealScalar);
        }
        if !matches!(
            self.value_type.scalar_domain(),
            eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
        ) {
            return Err(TypeViolation::RootRequiresRealScalar);
        }
        match super::combine_additive_support(&self.support, &selected.support) {
            Err(error) => Err(error),
            Ok(support) => self
                .dimension()
                .div(selected.dimension())
                .and_then(|dimension| self.value_type.clone().with_dimension(dimension).ok())
                .map(|value_type| ExpressionType::new(value_type, support))
                .ok_or(TypeViolation::DimensionOverflow {
                    operation: "partial derivative",
                }),
        }
    }
}
