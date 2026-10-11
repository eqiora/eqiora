//! Relation and activation root admission.
use super::*;

/// Check one residual root against its Relation scope.
pub fn residual<I: Clone + Eq>(
    root: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<(), TypeViolation<I>> {
    if root.support.as_ref().map(SpatialSupport::domain) != relation.map(SpatialSupport::domain) {
        return Err(TypeViolation::ResidualSupportMismatch {
            residual: Box::new(root.support.clone()),
            relation: Box::new(relation.cloned()),
        });
    }
    Ok(())
}

/// Check one activation root, which must remain a real invariant scalar.
pub fn scalar_root<I: Clone + Eq>(
    root: &ExpressionType<I>,
    relation: Option<&SpatialSupport<I>>,
) -> Result<(), TypeViolation<I>> {
    if !root.shape().is_scalar()
        || root.frame() != ValueFrame::Invariant
        || root.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
    {
        return Err(TypeViolation::RootRequiresRealScalar);
    }
    residual(root, relation)
}
