//! Finite algebra owns nominal endpoints independently of storage extents and spatial frames.
use super::{ExpressionType, TypeViolation, combine_additive_support};
use crate::kernel::{FiniteBinaryOperation, FiniteUnaryOperation};
use eqiora_core::{FiniteBasis, ScalarDomain, ValueType};

fn continuous<I>(value: &ValueType) -> Result<(), TypeViolation<I>> {
    if matches!(
        value.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) {
        Ok(())
    } else {
        Err(TypeViolation::ScalarDomainMismatch)
    }
}

impl<I: Clone + Eq> ExpressionType<I> {
    /// Admit coordinate dualization or a map transpose/orthonormal adjoint.
    pub fn finite_unary(self, operation: FiniteUnaryOperation) -> Result<Self, TypeViolation<I>> {
        continuous(&self.value_type)?;
        let value = &self.value_type;
        if let FiniteUnaryOperation::PermuteFactors(order) = operation {
            let permute = |basis: FiniteBasis| -> Result<FiniteBasis, TypeViolation<I>> {
                let [left, right] = basis.factors().ok_or(TypeViolation::FiniteBasisMismatch)?;
                match order {
                    [0, 1] => Ok(basis),
                    [1, 0] => FiniteBasis::product(right, left)
                        .map_err(|_| TypeViolation::FiniteBasisMismatch),
                    _ => Err(TypeViolation::FiniteBasisMismatch),
                }
            };
            let output = if let Some(basis) = value.coordinate_basis() {
                ValueType::coordinates(permute(basis)?, value.scalar_domain(), value.dimension())
            } else if let Some((source, target)) = value.map_bases() {
                ValueType::linear_map(
                    permute(source)?,
                    permute(target)?,
                    value.scalar_domain(),
                    value.dimension(),
                )
            } else {
                return Err(TypeViolation::FiniteBasisMismatch);
            }
            .map_err(|_| TypeViolation::FiniteBasisMismatch)?;
            return Ok(Self::new(output, self.support));
        }
        let output = if let Some(basis) = value.coordinate_basis() {
            ValueType::coordinates(basis.dual(), value.scalar_domain(), value.dimension())
        } else if let Some((source, target)) = value.map_bases() {
            let (source, target) = if operation == FiniteUnaryOperation::Transpose {
                (target.dual(), source.dual())
            } else {
                (target, source)
            };
            ValueType::linear_map(source, target, value.scalar_domain(), value.dimension())
        } else {
            return Err(TypeViolation::FiniteBasisMismatch);
        }
        .map_err(|_| TypeViolation::FiniteBasisMismatch)?;
        Ok(Self::new(output, self.support))
    }

    /// Admit application, composition or bilinear pairing with exact endpoint identities.
    pub fn finite_binary(
        self,
        operation: FiniteBinaryOperation,
        right: Self,
    ) -> Result<Self, TypeViolation<I>> {
        continuous(&self.value_type)?;
        continuous(&right.value_type)?;
        let (a, b) = (&self.value_type, &right.value_type);
        let domain = a
            .scalar_domain()
            .common(b.scalar_domain())
            .ok_or(TypeViolation::ScalarDomainMismatch)?;
        let dimension =
            a.dimension()
                .mul(b.dimension())
                .ok_or(TypeViolation::DimensionOverflow {
                    operation: "finite contraction",
                })?;
        let output = match operation {
            FiniteBinaryOperation::Apply => {
                let (source, target) = a.map_bases().ok_or(TypeViolation::FiniteBasisMismatch)?;
                if b.coordinate_basis() != Some(source) {
                    return Err(TypeViolation::FiniteBasisMismatch);
                }
                ValueType::coordinates(target, domain, dimension)
            }
            FiniteBinaryOperation::Compose => {
                let (middle, target) = a.map_bases().ok_or(TypeViolation::FiniteBasisMismatch)?;
                let (source, other_middle) =
                    b.map_bases().ok_or(TypeViolation::FiniteBasisMismatch)?;
                if middle != other_middle {
                    return Err(TypeViolation::FiniteBasisMismatch);
                }
                ValueType::linear_map(source, target, domain, dimension)
            }
            FiniteBinaryOperation::TensorProduct => {
                let product = |a, b| {
                    FiniteBasis::product(a, b).map_err(|_| TypeViolation::FiniteBasisMismatch)
                };
                if let (Some(left), Some(right)) = (a.coordinate_basis(), b.coordinate_basis()) {
                    ValueType::coordinates(product(left, right)?, domain, dimension)
                } else if let (Some((a_source, a_target)), Some((b_source, b_target))) =
                    (a.map_bases(), b.map_bases())
                {
                    ValueType::linear_map(
                        product(a_source, b_source)?,
                        product(a_target, b_target)?,
                        domain,
                        dimension,
                    )
                } else {
                    return Err(TypeViolation::FiniteBasisMismatch);
                }
            }
            FiniteBinaryOperation::Pair => {
                let dual = a
                    .coordinate_basis()
                    .ok_or(TypeViolation::FiniteBasisMismatch)?;
                let primal = b
                    .coordinate_basis()
                    .ok_or(TypeViolation::FiniteBasisMismatch)?;
                if dual != primal.dual() {
                    return Err(TypeViolation::FiniteBasisMismatch);
                }
                ValueType::scalar(domain, dimension)
            }
        }
        .map_err(|_| TypeViolation::FiniteBasisMismatch)?;
        Ok(Self::new(
            output,
            combine_additive_support(&self.support, &right.support)?,
        ))
    }
}

pub(super) fn scaled_type<I>(
    value: &ValueType,
    scalar: &ValueType,
    divide: bool,
) -> Result<ValueType, TypeViolation<I>> {
    continuous(value)?;
    continuous(scalar)?;
    if !scalar.shape().is_scalar() || scalar.frame() != eqiora_core::ValueFrame::Invariant {
        return Err(TypeViolation::MultiplicationRequiresScalar);
    }
    let domain = value
        .scalar_domain()
        .common(scalar.scalar_domain())
        .ok_or(TypeViolation::ScalarDomainMismatch)?;
    let dimension = if divide {
        value.dimension().div(scalar.dimension())
    } else {
        value.dimension().mul(scalar.dimension())
    }
    .ok_or(TypeViolation::DimensionOverflow {
        operation: "finite scaling",
    })?;
    if let Some(basis) = value.coordinate_basis() {
        ValueType::coordinates(basis, domain, dimension)
    } else if let Some((source, target)) = value.map_bases() {
        ValueType::linear_map(source, target, domain, dimension)
    } else {
        return Err(TypeViolation::FiniteBasisMismatch);
    }
    .map_err(|_| TypeViolation::FiniteBasisMismatch)
}
