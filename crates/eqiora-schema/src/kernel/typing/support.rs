//! Exact support combination for arithmetic and equation operands.
use super::*;

pub(super) fn combine_support<I: Clone + Eq>(
    left: &Option<SpatialSupport<I>>,
    right: &Option<SpatialSupport<I>>,
) -> Result<Option<SpatialSupport<I>>, TypeViolation<I>> {
    match (left, right) {
        (None, support) | (support, None) => Ok(support.clone()),
        (Some(left), Some(right)) if left == right => Ok(Some(left.clone())),
        (
            Some(SpatialSupport::Volume { domain, dimensions }),
            Some(
                boundary @ SpatialSupport::Boundary {
                    parent,
                    dimensions: boundary_dimensions,
                    ..
                },
            ),
        )
        | (
            Some(
                boundary @ SpatialSupport::Boundary {
                    parent,
                    dimensions: boundary_dimensions,
                    ..
                },
            ),
            Some(SpatialSupport::Volume { domain, dimensions }),
        ) if domain == parent && dimensions == boundary_dimensions => Ok(Some(boundary.clone())),
        (Some(left), Some(right)) => Err(TypeViolation::IncompatibleSupport {
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
        }),
    }
}

pub(super) fn combine_additive_support<I: Clone + Eq>(
    left: &Option<SpatialSupport<I>>,
    right: &Option<SpatialSupport<I>>,
) -> Result<Option<SpatialSupport<I>>, TypeViolation<I>> {
    match (left, right) {
        (None, support) | (support, None) => Ok(support.clone()),
        (Some(left), Some(right)) if left == right => Ok(Some(left.clone())),
        (Some(left), Some(right)) => Err(TypeViolation::IncompatibleSupport {
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
        }),
    }
}
