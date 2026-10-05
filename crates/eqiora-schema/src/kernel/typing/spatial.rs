//! Spatial differential and support-boundary typing rules.
use super::*;

/// Type a physical-space gradient.
pub fn gradient<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let support = operand
        .support
        .as_ref()
        .ok_or(TypeViolation::GradientRequiresSpatialSupport)?;
    if !matches!(support, SpatialSupport::Volume { .. }) {
        return Err(TypeViolation::GradientRequiresVolume);
    }
    if operand.value_type.array_rank() != 0
        || (operand.shape().is_scalar() && operand.frame() != ValueFrame::Invariant)
        || (!operand.shape().is_scalar() && operand.frame() != ValueFrame::SpatialCartesian)
    {
        return Err(TypeViolation::IncompatibleFrame);
    }
    let extent = u32::try_from(
        support
            .ambient_dimensions()
            .ok_or(TypeViolation::GradientRequiresVolume)?,
    )
    .ok()
    .filter(|extent| *extent > 0)
    .ok_or(TypeViolation::SpatialExtentInvalid)?;
    let shape = operand
        .shape()
        .appended(extent)
        .map_err(|_| TypeViolation::SpatialExtentInvalid)?;
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        spatial_derivative_dimension(operand.dimension())?,
        shape,
        ValueFrame::SpatialCartesian,
        operand.support.clone(),
    )
}

/// Type a physical-space divergence.
pub fn divergence<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let support = operand
        .support
        .as_ref()
        .ok_or(TypeViolation::DivergenceRequiresSpatialSupport)?;
    if !matches!(support, SpatialSupport::Volume { .. }) {
        return Err(TypeViolation::DivergenceRequiresVolume);
    }
    let Some((shape, last)) = operand.shape().remove_last() else {
        return Err(TypeViolation::DivergenceRequiresTensor);
    };
    if operand.frame() != ValueFrame::SpatialCartesian || operand.value_type.array_rank() != 0 {
        return Err(TypeViolation::IncompatibleFrame);
    }
    if usize::try_from(last.get()).ok() != support.ambient_dimensions() {
        return Err(TypeViolation::DivergenceRequiresTensor);
    }
    let frame = if shape.is_scalar() {
        ValueFrame::Invariant
    } else {
        ValueFrame::SpatialCartesian
    };
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        spatial_derivative_dimension(operand.dimension())?,
        shape,
        frame,
        operand.support.clone(),
    )
}

/// Type the symmetric part of an exact square Cartesian tensor.
pub fn symmetric_part<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    let Some(SpatialSupport::Volume { dimensions, .. }) = operand.support.as_ref() else {
        return Err(TypeViolation::SymmetricPartRequiresVolume);
    };
    let extents = operand.shape().extents();
    if operand.frame() != ValueFrame::SpatialCartesian
        || operand.value_type.array_rank() != 0
        || extents.len() != 2
        || usize::try_from(extents[0].get()).ok() != Some(*dimensions)
        || usize::try_from(extents[1].get()).ok() != Some(*dimensions)
    {
        return Err(TypeViolation::SymmetricPartRequiresSquareSpatialTensor);
    }
    Ok(operand.clone())
}

/// Type an isotropic lift whose tensor extent comes solely from volume
/// support.
pub fn isotropic_lift<I: Clone>(
    operand: &ExpressionType<I>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    let Some(SpatialSupport::Volume { dimensions, .. }) = operand.support.as_ref() else {
        return Err(TypeViolation::IsotropicLiftRequiresVolume);
    };
    if !operand.shape().is_scalar() || operand.frame() != ValueFrame::Invariant {
        return Err(TypeViolation::IsotropicLiftRequiresInvariantScalar);
    }
    let extent = u32::try_from(*dimensions)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or(TypeViolation::SpatialExtentInvalid)?;
    let shape =
        ValueShape::new([extent, extent]).map_err(|_| TypeViolation::SpatialExtentInvalid)?;
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        operand.dimension(),
        shape,
        ValueFrame::SpatialCartesian,
        operand.support.clone(),
    )
}

/// Type a boundary trace.
pub fn trace<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    boundary_operator(operand, relation, false)
}

/// Type an outward-normal contraction.
pub fn normal<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    boundary_operator(operand, relation, true)
}

/// Divide a dimension by the exact positive time power of a Field derivative.
pub fn time_derivative<I: Clone>(
    operand: &ExpressionType<I>,
    order: std::num::NonZeroU32,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    if matches!(
        operand.value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
    ) {
        return Err(TypeViolation::ScalarDomainMismatch);
    }

    if operand.value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
        return Err(TypeViolation::ScalarDomainMismatch);
    }
    let overflow = || TypeViolation::DimensionOverflow {
        operation: "Field derivative",
    };
    let mut exponents = operand.dimension().exponents();
    let (numerator, denominator) = exponents[2];
    // Subtract before narrowing: the final exponent may be representable even
    // when the intermediate time power is not. Subtracting an integer preserves
    // the coprimality of the canonical numerator and denominator.
    exponents[2].0 =
        i32::try_from(i128::from(numerator) - i128::from(order.get()) * i128::from(denominator))
            .map_err(|_| overflow())?;
    let dimension = DimExponents::from_rationals(exponents).ok_or_else(overflow)?;
    Ok(ExpressionType::new(
        operand
            .value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|_| TypeViolation::ScalarDomainMismatch)?,
        operand.support.clone(),
    ))
}

fn boundary_operator<I: Clone + Eq>(
    operand: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
    normal_component: bool,
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    let Some(SpatialSupport::Boundary {
        parent,
        domain,
        dimensions,
    }) = relation
    else {
        return Err(TypeViolation::BoundaryOperatorRequiresBoundaryScope);
    };
    let operand_is_parent_volume =
        operand.support.as_ref().map(SpatialSupport::domain) == Some(parent);
    let operand_is_this_boundary = normal_component
        && operand
            .support
            .as_ref()
            .is_some_and(|support| support == relation.expect("boundary scope was matched"));
    if !operand_is_parent_volume && !operand_is_this_boundary {
        return Err(TypeViolation::BoundaryOperandSupportMismatch);
    }
    if !normal_component {
        return Ok(ExpressionType::new(
            operand.value_type.clone(),
            relation.cloned(),
        ));
    }
    let Some((shape, last)) = operand.shape().remove_last() else {
        return Err(TypeViolation::NormalRequiresTensor);
    };
    if usize::try_from(last.get()).ok() != Some(*dimensions) {
        return Err(TypeViolation::NormalRequiresTensor);
    }
    if operand.frame() != ValueFrame::SpatialCartesian || operand.value_type.array_rank() != 0 {
        return Err(TypeViolation::IncompatibleFrame);
    }
    let frame = if shape.is_scalar() {
        ValueFrame::Invariant
    } else {
        operand.frame()
    };
    ExpressionType::checked(
        operand.value_type.scalar_domain(),
        operand.dimension(),
        shape,
        frame,
        Some(SpatialSupport::Boundary {
            domain: domain.clone(),
            parent: parent.clone(),
            dimensions: *dimensions,
        }),
    )
}

fn spatial_derivative_dimension<I>(
    dimension: DimExponents,
) -> Result<DimExponents, TypeViolation<I>> {
    dimension
        .div(DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"))
        .ok_or(TypeViolation::DimensionOverflow {
            operation: "spatial derivative",
        })
}
